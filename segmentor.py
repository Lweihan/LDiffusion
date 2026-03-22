import os
import sys
import tempfile
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torchvision import transforms
from .model.loss import CombinedLoss
from torch.utils.data import TensorDataset, DataLoader
from diffusers import UNet2DConditionModel, StableDiffusionImg2ImgPipeline
from .utils import extract_topk_points, generate_multi_class_heatmaps, mean_iou_and_per_class, create_nnunet_dataset, load_image_to_numpy, prepare_image_for_predictor
from PIL import Image

class Segmentor:
    def __init__(self, train_loader, val_loader, level, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.level = level
        self.num_classes = num_classes
        self.model = None
        self.train_loader, self.val_loader = train_loader, val_loader
        self.ldiffusion_proj = None

    def _resolve_ldiffusion_dir(self, ldiffusion_weight):
        if os.path.isdir(ldiffusion_weight):
            return ldiffusion_weight
        return os.path.dirname(ldiffusion_weight)

    def _ensure_ldiffusion_proj(self, pipeline, unet, ldiffusion_weight=None):
        text_hidden_size = pipeline.text_encoder.config.hidden_size
        cross_attention_dim = unet.config.cross_attention_dim

        need_rebuild = (
            self.ldiffusion_proj is None
            or self.ldiffusion_proj.in_features != text_hidden_size
            or self.ldiffusion_proj.out_features != cross_attention_dim
        )

        if need_rebuild:
            self.ldiffusion_proj = nn.Linear(text_hidden_size, cross_attention_dim).to(self.device, dtype=torch.float32)

        if ldiffusion_weight is not None:
            proj_weight_path = os.path.join(self._resolve_ldiffusion_dir(ldiffusion_weight), "proj_weights.pt")
            if os.path.exists(proj_weight_path):
                state_dict = torch.load(proj_weight_path, map_location=self.device)
                self.ldiffusion_proj.load_state_dict(state_dict, strict=True)

        self.ldiffusion_proj = self.ldiffusion_proj.to(self.device, dtype=torch.float32)
        self.ldiffusion_proj.eval()
        return self.ldiffusion_proj

    @torch.no_grad()
    def _get_text_embeddings(self, prompt, batch_size, pipeline, unet):
        proj = self._ensure_ldiffusion_proj(pipeline, unet)
        input_ids = pipeline.tokenizer([prompt] * batch_size)["input_ids"]
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        text_embeddings = pipeline.text_encoder(input_ids)["last_hidden_state"].to(dtype=torch.float32)
        return proj(text_embeddings).to(dtype=torch.float32)

    def initialize_model(self, level, num_classes):
        if level == "tissue":
            # Initialize tissue-level model (SAM + DeepLab)
            from .model.conductor import TissueSegNet
            model = TissueSegNet(num_classes)
        elif level == "cell":
            from .model.conductor import CellSegClassifier
            model = CellSegClassifier(num_classes)
        else:
            raise ValueError("Invalid level specified. Choose 'tissue' or 'cell'.")

        model.to(self.device)
        return model

    def load_ldiffusion(self, ldiffusion_weight, diffusion_path):
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(diffusion_path, torch_dtype=torch.float32).to(self.device)
        pipeline.text_encoder.to(self.device, dtype=torch.float32)
        unet = UNet2DConditionModel.from_pretrained(ldiffusion_weight).eval().to(self.device, dtype=torch.float32)
        vae = pipeline.vae.to(self.device, dtype=torch.float32)

        self._ensure_ldiffusion_proj(pipeline, unet, ldiffusion_weight=ldiffusion_weight)

        return pipeline, unet, vae

    def ldiffusion_augment(self, inputs, pipeline, unet, vae):
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),  # 调整大小以匹配模型的输入要求
            transforms.ToTensor(),
        ])

        self._ensure_ldiffusion_proj(pipeline, unet)
        text_embeddings = self._get_text_embeddings("A pathological slide", 1, pipeline, unet)

        decoded_image_list = []
        for index in range(len(inputs)):
            image = inputs[index].unsqueeze(0).to(self.device, dtype=torch.float32)
            with torch.no_grad():
                latents = vae.encode(image).latent_dist.mean.to(dtype=torch.float32)
            pipeline.scheduler.set_timesteps(1, device=self.device)
            for i, t in enumerate(pipeline.scheduler.timesteps):
                latents = pipeline.scheduler.scale_model_input(latents, t).to(dtype=torch.float32)
                output = unet(latents, t, text_embeddings)
                latents = pipeline.scheduler.step(output[0], t, latents).prev_sample.to(self.device)
                with torch.no_grad():
                    decoded_images = pipeline.decode_latents(latents.detach())
            decoded_image = pipeline.numpy_to_pil(decoded_images)[0]
            decoded_image_tensor = transform(decoded_image).unsqueeze(0)
            decoded_image_list.append(decoded_image_tensor)

        decoded_images_tensor = torch.cat(decoded_image_list, dim=0)
        return decoded_images_tensor.to(self.device)

    def micro_dice(self, predicted_labels, true_labels, num_classes=7):

        predicted_labels = torch.argmax(predicted_labels, dim=1)  # shape: (batch_size, height, width)

        true_labels = true_labels.view(-1)  # shape: (batch_size * height * width,)
        predicted_labels = predicted_labels.view(-1)  # shape: (batch_size * height * width,)

        dice_scores = torch.zeros(num_classes, device=true_labels.device)

        for class_id in range(num_classes):
            true_class = (true_labels == class_id).float()
            predicted_class = (predicted_labels == class_id).float()

            if torch.sum(true_class) == 0 and torch.sum(predicted_class) == 0:
                dice_scores[class_id] = 1
            else:
                # 计算TP, FP, FN
                TP = torch.sum(true_class * predicted_class)
                FP = torch.sum((1 - true_class) * predicted_class)
                FN = torch.sum(true_class * (1 - predicted_class))

                if TP + FP + FN == 0:
                    dice_scores[class_id] = 0
                else:
                    dice_scores[class_id] = 2 * TP / (2 * TP + FP + FN)

        average_dice = torch.mean(dice_scores)

        return dice_scores, average_dice

    def build_augmented_dataloader(self, dataloader, augment_fn, pipeline, unet, vae, device, batch_size, category, num_workers=0):
        all_aug_inputs = []
        all_masks = []

        for inputs, masks, _ in tqdm(dataloader, desc=f"Cache L-Diffusion Augmented {category} Inputs"):
            inputs, masks = inputs.to(device), masks.to(device)
            with torch.no_grad():
                aug_inputs = augment_fn(inputs, pipeline, unet, vae)
            all_aug_inputs.append(aug_inputs.cpu())
            all_masks.append(masks.cpu())

        aug_inputs_tensor = torch.cat(all_aug_inputs, dim=0)
        masks_tensor = torch.cat(all_masks, dim=0)

        dataset = TensorDataset(aug_inputs_tensor, masks_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        return loader

    def train_tissue_model_nnUNetv2(self, epochs, ldiffusion_weight, diffusion_path, train_images=None, train_labels=None, test_images=None, test_labels=None, num_classes=None):
        from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry
        from nnunetv2.run.run_training import get_trainer_from_args

        print("\033[32m[LDiffusion] Preparing data by L-Diffusion...\033[0m")
        pipeline, unet, _ = self.load_ldiffusion(ldiffusion_weight, diffusion_path)

        if num_classes is None:
            num_classes = self.num_classes
        if train_images is None or train_labels is None or test_images is None or test_labels is None:
            if self.train_loader is None or self.val_loader is None:
                raise ValueError("train_loader/val_loader are required when explicit nnUNet data lists are not provided")
            train_images = self.train_loader.dataset.image_dir
            train_labels = self.train_loader.dataset.label_dir
            test_images = self.val_loader.dataset.image_dir
            test_labels = self.val_loader.dataset.label_dir

        self._ensure_ldiffusion_proj(pipeline, unet, ldiffusion_weight=ldiffusion_weight)
        cached_text_embeddings = self._get_text_embeddings("A pathological slide", 1, pipeline, unet)

        class _UNetTextAlignWrapper(nn.Module):
            def __init__(self, base_unet, default_text_embeddings):
                super().__init__()
                self.base_unet = base_unet
                self.default_text_embeddings = default_text_embeddings
                self.cross_attention_dim = base_unet.config.cross_attention_dim

            def forward(self, sample, timestep, encoder_hidden_states, *args, **kwargs):
                use_fallback = (
                    encoder_hidden_states is None
                    or encoder_hidden_states.shape[-1] != self.cross_attention_dim
                )

                if use_fallback:
                    bsz = sample.shape[0]
                    text_embeddings = self.default_text_embeddings
                    if text_embeddings.shape[0] != bsz:
                        text_embeddings = text_embeddings[:1].expand(bsz, -1, -1)
                    encoder_hidden_states = text_embeddings.to(sample.device, dtype=torch.float32)
                else:
                    encoder_hidden_states = encoder_hidden_states.to(sample.device, dtype=torch.float32)

                return self.base_unet(sample, timestep, encoder_hidden_states, *args, **kwargs)

        aligned_unet = _UNetTextAlignWrapper(unet, cached_text_embeddings).to(self.device).eval()

        new_num, new_dataset_name = create_nnunet_dataset(
            train_images,
            train_labels,
            test_images,
            test_labels,
            num_classes,
            pipeline,
            aligned_unet,
        )

        print("\033[32m[Segmentor-nnUNet] Data preprocessing and plan generation in progress...\033[0m")
        sys.argv = [
            "nnUNetv2_plan_and_preprocess",  # argv[0] 通常是脚本名，可以随便写
            "-d", str(new_num),
            "--verify_dataset_integrity",
            "--clean",
        ]

        plan_and_preprocess_entry()

        configuration = "2d"
        fold = 0
        device = "cuda"

        print("\033[32m[Segmentor-nnUNet] Training is starting...\033[0m")
        trainer = get_trainer_from_args(
            dataset_name_or_id=new_dataset_name,
            configuration=configuration,
            fold=fold,
            plans_identifier="nnUNetPlans",
        )
        trainer.num_epochs = epochs
        trainer.run_training()

    def train_cell_model(self, epochs, ldiffusion_weight, diffusion_path):
        print("Training cell-level model...")
        if self.model is None:
            self.model = self.initialize_model("cell", self.num_classes)
        criterion, checkpoint = CombinedLoss(num_classes=self.num_classes), 0.0
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        pipeline, unet, vae = self.load_ldiffusion(ldiffusion_weight, diffusion_path)
        current_date = datetime.now().strftime("%y_%m_%d")

        aug_train_loader = self.build_augmented_dataloader(self.train_loader, self.ldiffusion_augment, pipeline, unet,
                                                           vae, self.device, batch_size=1, category="Train",
                                                           num_workers=4)
        aug_val_loader = self.build_augmented_dataloader(self.val_loader, self.ldiffusion_augment, pipeline, unet, vae,
                                                         self.device, batch_size=1, category="Validation",
                                                         num_workers=4)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, masks in tqdm(aug_train_loader, desc=f"Epoch {epoch + 11}/{epochs + 10} - Train"):
                masks = masks.to(self.device)
                image_np = inputs[0].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]

                optimizer.zero_grad()
                outputs = self.model(image_np)["out"]
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            avg_running_loss = running_loss / len(aug_train_loader)
            print(f"Epoch {epoch + 11}/{epochs + 10}, Loss: {avg_running_loss:.4f}")

            self.model.eval()
            total_dice = 0.0
            with torch.no_grad():
                for images, labels in tqdm(aug_val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Val"):
                    image_np = images[0].cpu().permute(1, 2, 0).numpy()
                    labels = labels.to(self.device)
                    outputs = self.model(image_np)["out"]
                    dice_scores, avg_dice = self.micro_dice(
                        predicted_labels=outputs,
                        true_labels=labels,
                        num_classes=self.num_classes
                    )
                    total_dice += avg_dice.item()

            total_dice = total_dice / len(aug_val_loader)
            if total_dice > checkpoint:
                checkpoint = total_dice
                os.makedirs(f"./LDiffusion/train_save/cellclassifier/{current_date}", exist_ok=True)
                torch.save(self.model.state_dict(),
                           f"./LDiffusion/train_save/cellclassifier/{current_date}/cellclassifier.pth")
                print(f"\033[32mNew Best Validation Dice Score: {total_dice:.4f}\033[0m")
            else:
                print(f"Validation Dice Score: {total_dice:.4f}")

    @torch.no_grad()
    def ldiffusion_augment_for_multimodal(self, rgb, dtm, pipeline, unet, vae, controlnet, batch_size, device):
        # 模型加载
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder.to(device)
        controlnet = controlnet.to(device)
        proj = self._ensure_ldiffusion_proj(pipeline, unet)

        unet.eval()
        controlnet.eval()
        vae.eval()
        text_encoder.eval()

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        transform_canny = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        rgb = F.interpolate(rgb, size=(256, 256), mode='bilinear', align_corners=False).to(
            device)
        dtm = F.interpolate(dtm, size=(256, 256), mode='bilinear', align_corners=False).to(
            device)

        recon_batch_np = []

        for i in range(dtm.shape[0]):
            dtm_i = dtm[i]
            rgb_i = rgb[i].unsqueeze(0)

            depth = dtm_i.unsqueeze(0).to(device)
            depth_condition = depth.repeat(1, 3, 1, 1)

            # VAE encode
            latents = vae.encode(rgb_i).latent_dist.sample() * 0.18215
            depth_resized = F.interpolate(dtm_i.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False)
            depth_resized = depth_resized.repeat(1, latents.shape[1], 1, 1)

            # 添加训练时一致的 Laplace noise
            noise = torch.distributions.laplace.Laplace(0.0, 1.0).sample(latents.shape).to(device)
            latents_noisy = latents + noise * depth_resized

            # 文本编码
            input_ids = tokenizer(["A remote sense image"], padding="max_length", max_length=77, return_tensors="pt").input_ids.to(
                device)
            text_embeddings = text_encoder(input_ids)["last_hidden_state"].to(dtype=torch.float32)
            text_embeddings = proj(text_embeddings).to(dtype=torch.float32)

            # ControlNet 推理
            pipeline.scheduler.set_timesteps(1, device=device)
            for timestep in pipeline.scheduler.timesteps:
                timestep = timestep.to(device)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    sample=latents_noisy,
                    timestep=timestep,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=depth_condition,
                    return_dict=False
                )

                # UNet 预测 noise
                noise_pred = unet(
                    latents_noisy,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

            # 反噪
            latents_denoised = latents_noisy - noise_pred * depth_resized

            # 解码为图像
            with torch.no_grad():
                recon_image = vae.decode(latents_denoised / 0.18215).sample  # or .sample

            recon_image = recon_image.squeeze(0).cpu()
            dtm_np = dtm.squeeze(0).cpu().numpy()
            recon_np = recon_image.permute(1, 2, 0).numpy()
            recon_batch_np.append(recon_np)

        return recon_batch_np

    def inference_tissue_model_nnUNetv2(self, image_path, diffusion_path, ldiffusion_weight, segmentor_weight, output_path=None):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        print("Running inference on tissue-level model...")
        if self.model is None:
            self.model = self.initialize_model("tissue", self.num_classes)

        # Load pipeline and models
        pipeline, unet, _ = self.load_ldiffusion(ldiffusion_weight, diffusion_path)

        # Case 1
        if os.path.isdir(image_path):
            if not output_path:
                raise ValueError("When image_path is a folder, output_path must be specified!")

            predictor = nnUNetPredictor(verbose=True)
            predictor.initialize_from_trained_model_folder(
                segmentor_weight,
                use_folds=(0,),
                checkpoint_name='checkpoint_best.pth'
            )
            predictor.predict_from_files(
                image_path,
                output_path,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
                folder_with_segs_from_prev_stage=None,
                num_parts=1, part_id=0
            )
            return None, None  # 批量模式不返回单张 mask

        # Case 2
        image = Image.open(image_path).convert("RGB")
        if image.mode != "RGB":
            raise ValueError(f"Input image is not in RGB mode: {image.mode}")

        width, height = image.size
        if width == height:
            transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(image).unsqueeze(0).to(self.device)
            self.model.eval()

            with torch.no_grad():
                latents = pipeline.vae.encode(image).latent_dist.mean.to(dtype=torch.float32)
                pipeline.scheduler.set_timesteps(1, device=self.device)
                self._ensure_ldiffusion_proj(pipeline, unet)
                text_embeddings = self._get_text_embeddings("A pathological slide", 1, pipeline, unet)

                for i, t in enumerate(pipeline.scheduler.timesteps):
                    latents = pipeline.scheduler.scale_model_input(latents, t).to(dtype=torch.float32)
                    output = unet(latents, t, text_embeddings)
                    latents = pipeline.scheduler.step(output[0], t, latents).prev_sample.to(self.device)
                    with torch.no_grad():
                        decoded_images = pipeline.decode_latents(latents.detach())
                decoded_image = pipeline.numpy_to_pil(decoded_images)[0]
        else:
            decoded_image = image

        # --- 临时目录保存输入图像 ---
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "imagesTs")
            output_dir = os.path.join(tmpdir, "pred")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            input_path = os.path.join(input_dir, "case_0000.png")
            decoded_image.save(input_path)

            # --- nnUNet Predictor ---
            predictor = nnUNetPredictor(verbose=True)
            predictor.initialize_from_trained_model_folder(
                segmentor_weight,
                use_folds=(0,),
                checkpoint_name='checkpoint_best.pth'
            )
            predictor.predict_from_files(
                input_dir,
                output_dir,
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=2,
                num_processes_segmentation_export=2,
                folder_with_segs_from_prev_stage=None,
                num_parts=1, part_id=0
            )

            # --- 读取预测结果 PNG ---
            pred_files = [f for f in os.listdir(output_dir) if f.endswith(".png")]
            if not pred_files:
                raise RuntimeError("No prediction PNG mask generated!")

            pred_path = os.path.join(output_dir, pred_files[0])
            mask = np.array(Image.open(pred_path).convert("L"), dtype=np.uint8)

            return decoded_image, mask

    def inference_cell_model(self, image_path, diffusion_path, ldiffusion_weight, segmentor_weight):
        print("Running inference on cell-level model...")
        if self.model is None:
            self.model = self.initialize_model("cell", self.num_classes)

        # Load pipeline and models
        pipeline, unet, _ = self.load_ldiffusion(ldiffusion_weight, diffusion_path)
        self.model.load_state_dict(torch.load(os.path.join(segmentor_weight, "cellclassifier.pth"), weights_only=True))
        self.model.to(self.device)

        # Load and preprocess the input image
        image = Image.open(image_path).convert("RGB")
        if image.mode != "RGB":
            raise ValueError(f"Input image is not in RGB mode: {image.mode}")
        width, height = image.size
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(self.device)
        self.model.eval()

        # Ensure image dimensions are correct
        if image.dim() != 4:
            raise ValueError(f"Input image tensor has invalid dimensions: {image.dim()} (expected 4).")

        # Perform inference
        with torch.no_grad():
            latents = pipeline.vae.encode(image).latent_dist.mean.to(dtype=torch.float32)
            pipeline.scheduler.set_timesteps(1, device=self.device)
            self._ensure_ldiffusion_proj(pipeline, unet)
            text_embeddings = self._get_text_embeddings("A pathological slide", 1, pipeline, unet)

            for i, t in enumerate(pipeline.scheduler.timesteps):
                latents = pipeline.scheduler.scale_model_input(latents, t).to(dtype=torch.float32)
                output = unet(latents, t, text_embeddings)
                latents = pipeline.scheduler.step(output[0], t, latents).prev_sample.to(self.device)
                with torch.no_grad():
                    decoded_images = pipeline.decode_latents(latents.detach())
            decoded_image = pipeline.numpy_to_pil(decoded_images)[0]

            # Pass the decoded image through the model
            model_input = transform(decoded_image).unsqueeze(0).to(self.device)
            model_input = model_input[0].cpu().permute(1, 2, 0).numpy()
            outputs = self.model(model_input)["out"]
            pred_mask = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            pred_mask = pred_mask.squeeze(0).cpu().numpy()

            pred_mask_resized = Image.fromarray(pred_mask.astype(np.uint8)).resize((width, height),
                                                                                   resample=Image.NEAREST)
            pred_mask_resized = np.array(pred_mask_resized)

            decoded_image = decoded_image.resize((width, height), Image.BILINEAR)

            return decoded_image, pred_mask_resized

