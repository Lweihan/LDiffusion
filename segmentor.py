import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import torchvision.transforms.functional as TF
from datetime import datetime
from torchvision import transforms
from .model.loss import CombinedLoss, FocalLoss, KLDivLossMultiChannel
from .utils import extract_topk_points, generate_multi_class_heatmaps, mean_iou_and_per_class, create_nnunet_dataset, load_image_to_numpy, prepare_image_for_predictor
from torch.utils.data import TensorDataset, DataLoader
from diffusers import UNet2DConditionModel, StableDiffusionImg2ImgPipeline, ControlNetModel
from segment_anything import sam_model_registry
from PIL import Image

class Segmentor:
    def __init__(self, train_loader, val_loader, level, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.level = level
        self.num_classes = num_classes
        self.model = None
        self.train_loader, self.val_loader = train_loader, val_loader

    def initialize_model(self, level, num_classes):
        if level == "tissue":
            # Initialize tissue-level model (ConvNext + DeepLab)
            from .model.conductor import TissueSegNet
            model = TissueSegNet(num_classes)
        elif level == "cell":
            from .model.conductor import CellSegClassifier
            model = CellSegClassifier(num_classes)
        elif level == "remote sense":
            from .model.conductor import TissueSegWithDepthHeatmap
            model = TissueSegWithDepthHeatmap(num_classes)
        else:
            raise ValueError("Invalid level specified. Choose 'tissue' or 'cell'.")

        model.to(self.device)
        return model

    def load_ldiffusion(self, ldiffusion_weight, diffusion_path):
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(diffusion_path).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(ldiffusion_weight).eval().to(self.device)
        vae = pipeline.vae.to(self.device)
        return pipeline, unet, vae

    def ldiffusion_augment(self, inputs, pipeline, unet, vae, batch_size=1):
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),  # 调整大小以匹配模型的输入要求
            transforms.ToTensor(),
        ])

        linear_layer = nn.Linear(768, 1280).to(self.device)
        input_ids = pipeline.tokenizer(["A pathological slide"] * batch_size)["input_ids"]
        input_ids = torch.tensor(input_ids).to(self.device)
        text_embeddings = pipeline.text_encoder(input_ids)['last_hidden_state']
        text_embeddings = linear_layer(text_embeddings)
        text_embeddings = text_embeddings.clone().detach().to(self.device)

        decoded_image_list = []
        for index in range(len(inputs)):
            image = inputs[index].unsqueeze(0).to(self.device)
            with torch.no_grad():
                latents = vae.encode(image).latent_dist.mean
            pipeline.scheduler.set_timesteps(1, device=self.device)
            for i, t in enumerate(pipeline.scheduler.timesteps):
                latents = pipeline.scheduler.scale_model_input(latents, t)
                output = unet(latents, t, text_embeddings)
                latents = pipeline.scheduler.step(output[0], t, latents).prev_sample.to(self.device)
                with torch.no_grad():
                    decoded_images = pipeline.decode_latents(latents.detach())
            decoded_image = pipeline.numpy_to_pil(decoded_images)[0]
            decoded_image_tensor = transform(decoded_image).unsqueeze(0)
            decoded_image_list.append(decoded_image_tensor)

        decoded_images_tensor = torch.cat(decoded_image_list, dim=0)
        return decoded_images_tensor.to(self.device)

    @torch.no_grad()
    def ldiffusion_augment_for_multimodal(self, rgb, dtm, pipeline, unet, vae, controlnet, batch_size, device):
        # 模型加载
        linear_layer = torch.nn.Linear(768, 1280).to(device)
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder.to(device)
        controlnet = controlnet.to(device)

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
            text_embeddings = text_encoder(input_ids)["last_hidden_state"]

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

                # 线性层调整文本维度
                text_embeddings = linear_layer(text_embeddings)

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

    def build_augmented_dataloader(self, dataloader, augment_fn, pipeline, unet, vae, controlnet, device, batch_size, category, num_workers=0):
        all_aug_inputs = []
        all_masks = []
        all_dtms = []

        if self.level == "tissue" or self.level == "cell":
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
        elif self.level == "remote sense":
            for batch in tqdm(dataloader, desc=f"Cache L-Diffusion Augmented {category} Inputs"):
                rgb = batch["rgb"]
                dtm = batch["dtm"]
                mask = batch["mask"]
                recon_images_np = augment_fn(rgb, dtm, pipeline, unet, vae, controlnet, batch_size, device)
                for recon_image_np in recon_images_np:
                    recon_tensor = transforms.ToTensor()(recon_image_np).unsqueeze(0)
                    all_aug_inputs.append(recon_tensor)
                all_masks.append(mask)
                all_dtms.append(dtm)
            aug_inputs_tensor = torch.cat(all_aug_inputs, dim=0)
            masks_tensor = torch.cat(all_masks, dim=0)
            dtms_tensor = torch.cat(all_dtms, dim=0)
            dataset = TensorDataset(aug_inputs_tensor, masks_tensor, dtms_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            return loader
        else:
            assert "Level must be either 'tissue', 'cell', or 'remote sense'."
            raise NotImplementedError

    def train_tissue_model(self, epochs, ldiffusion_weight, diffusion_path):
        print("Training tissue-level model...")
        if self.model is None and self.level == "tissue":
            self.model = self.initialize_model("tissue", self.num_classes)
        criterion, checkpoint = CombinedLoss(num_classes=self.num_classes), 0.0
        optimizer = torch.optim.AdamW([
            {'params': self.model.backbone.parameters(), 'lr': 1e-5},
            {'params': self.model.decoder.parameters(), 'lr': 1e-4},
        ])
        pipeline, unet, vae = self.load_ldiffusion(ldiffusion_weight, diffusion_path)
        current_date = datetime.now().strftime("%y_%m_%d")

        aug_train_loader = self.build_augmented_dataloader(self.train_loader, self.ldiffusion_augment, pipeline, unet, vae, None, self.device, batch_size=2, category="Train", num_workers=4)
        aug_val_loader = self.build_augmented_dataloader(self.val_loader, self.ldiffusion_augment, pipeline, unet, vae, None, self.device, batch_size=1, category="Validation", num_workers=4)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, masks in tqdm(aug_train_loader, desc=f"Epoch {epoch + 11}/{epochs + 10} - Train"):
                if inputs.size(0) == 1:
                    continue
                inputs, masks = inputs.to(self.device), masks.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)["out"]
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_running_loss = running_loss / len(aug_train_loader)
            print(f"Epoch {epoch + 11}/{epochs + 10}, Loss: {avg_running_loss:.4f}")

            self.model.eval()
            total_dice = 0.0
            with torch.no_grad():
                for imgs, masks in tqdm(aug_val_loader, desc=f"Epoch {epoch + 11}/{epochs + 10} - Validation"):
                    imgs, masks = imgs.to(self.device), masks.to(self.device)
                    outputs = self.model(imgs)['out']
                    dice_categories, average_dice = self.micro_dice(outputs, masks, num_classes=self.num_classes)
                    total_dice += average_dice

            total_dice = total_dice / len(aug_val_loader)
            if total_dice > checkpoint:
                checkpoint = total_dice
                os.makedirs(f"./LDiffusion/train_save/convnext_tiny/{current_date}", exist_ok=True)
                torch.save(self.model.state_dict(),f"./LDiffusion/train_save/convnext_tiny/{current_date}/convnext_tiny.pth")
                print(f"\033[32mNew Best Validation Dice Score: {total_dice:.4f}\033[0m")
            else:
                print(f"Validation Dice Score: {total_dice:.4f}")

    def train_tissue_model_nnUNetv2(self, epochs, ldiffusion_weight, diffusion_path, train_images, train_labels, test_images, test_labels, num_classes):
        from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry
        from nnunetv2.run.run_training import get_trainer_from_args

        print("\033[32m[LDiffusion] Preparing data by L-Diffusion...\033[0m")
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(diffusion_path).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(ldiffusion_weight).to(self.device)
        new_num, new_dataset_name = create_nnunet_dataset(train_images, train_labels, test_images, test_labels, num_classes, pipeline, unet)

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
                                                           vae, None, self.device, batch_size=1, category="Train",
                                                           num_workers=4)
        aug_val_loader = self.build_augmented_dataloader(self.val_loader, self.ldiffusion_augment, pipeline, unet, vae,
                                                         None, self.device, batch_size=1, category="Validation",
                                                         num_workers=4)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, masks in tqdm(aug_train_loader, desc=f"Epoch {epoch + 11}/{epochs + 10} - Train"):
                if inputs.size(0) == 1:
                    continue
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

    def train_remote_sense_model(self, epochs, ldiffusion_weight, diffusion_path, controlnet_path, batch_size):
        print("Training remote sensing model...")
        if self.model is None:
            self.model = self.initialize_model("remote sense", self.num_classes)
        criterion, checkpoint = CombinedLoss(num_classes=self.num_classes), 0.0
        kl_loss_fn = KLDivLossMultiChannel()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        pipeline, unet, vae = self.load_ldiffusion(ldiffusion_weight, diffusion_path)
        controlnet = ControlNetModel.from_pretrained(controlnet_path)
        current_date = datetime.now().strftime("%y_%m_%d")

        aug_train_loader = self.build_augmented_dataloader(self.train_loader, self.ldiffusion_augment_for_multimodal, pipeline, unet,
                                                           vae, controlnet, self.device, batch_size=batch_size, category="Train",
                                                           num_workers=4)
        aug_val_loader = self.build_augmented_dataloader(self.val_loader, self.ldiffusion_augment_for_multimodal, pipeline, unet, vae,
                                                         controlnet, self.device, batch_size=1, category="Validation",
                                                         num_workers=4)

        for epoch in range(epochs):
            self.model.train()
            running_loss, avg_heatmap_loss, avg_dice_loss = 0.0, 0.0, 0.0

            for inputs, masks, dtms in tqdm(aug_train_loader, desc=f"Epoch {epoch + 11}/{epochs + 10} - Train"):
                if inputs.size(0) == 1:
                    continue
                inputs, masks, dtms = inputs.to(self.device), masks.to(self.device), dtms.to(self.device)
                optimizer.zero_grad()
                results = self.model(inputs, dtms)
                pred_mask, pred_heatmap = results["seg"], results["heatmap"]
                masks = masks.squeeze(1).long()
                gt_heatmaps = generate_multi_class_heatmaps(
                    masks.cpu().numpy(),
                    num_classes=self.num_classes
                ).to(self.device)
                heat_loss = kl_loss_fn(pred_heatmap, gt_heatmaps)
                seg_loss = criterion(pred_mask, masks)
                loss = 0.25 * heat_loss + 1.75 * seg_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_running_loss = running_loss / len(aug_train_loader)
            print(f"Epoch {epoch + 11}/{epochs + 10}, Loss: {avg_running_loss:.4f}")

            self.model.eval()
            total_IoU, total_iou_dict = 0.0, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
            with torch.no_grad():
                for imgs, masks, dtms in tqdm(aug_val_loader, desc=f"Epoch {epoch + 11}/{epochs + 10} - Validation"):
                    imgs, masks, dtms = imgs.to(self.device), masks.to(self.device), dtms.to(self.device)
                    pred_mask = self.model(imgs, dtms)['seg']
                    masks = masks.squeeze(1).long()
                    mIoU, iou_dict = mean_iou_and_per_class(pred_mask, masks, num_classes=self.num_classes)
                    for i in range(self.num_classes):
                        if iou_dict[i] is not None:
                            total_iou_dict[i] += iou_dict[i]
                        else:
                            total_iou_dict[i] += 1
                    total_IoU += mIoU

            total_IoU = total_IoU / len(aug_val_loader)
            for k in total_iou_dict:
                total_iou_dict[k] /= len(aug_val_loader)
            if total_IoU > checkpoint:
                checkpoint = total_IoU
                os.makedirs(f"./LDiffusion/train_save/convnext_tiny/{current_date}", exist_ok=True)
                torch.save(self.model.state_dict(),
                           f"./LDiffusion/train_save/convnext_tiny/{current_date}/convnext_tiny.pth")
                print(f"\033[32mNew Best Validation mIoU Score: {total_IoU:.4f} | Background: {total_iou_dict[0]:.4f} | Hydrology: {total_iou_dict[1]:.4f} | Mound: {total_iou_dict[2]:.4f} | Temple: {total_iou_dict[3]:4f}\033[0m")
            else:
                print(f"Validation mIoU Score: {total_IoU:.4f}")

    def inference_tissue_model(self, image_path, diffusion_path, ldiffusion_weight, segmentor_weight):
        print("Running inference on tissue-level model...")
        if self.model is None:
            self.model = self.initialize_model("tissue", self.num_classes)

        # Load pipeline and models
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(diffusion_path).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(ldiffusion_weight).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(segmentor_weight, "convnext_tiny.pth"), weights_only=True))
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

        # Perform inference
        with torch.no_grad():
            latents = pipeline.vae.encode(image).latent_dist.mean
            pipeline.scheduler.set_timesteps(1, device=self.device)
            linear_layer = nn.Linear(768, 1280).to(self.device)
            input_ids = pipeline.tokenizer(["A pathological slide"] * 1)["input_ids"]
            input_ids = torch.tensor(input_ids).to(self.device)
            text_embeddings = pipeline.text_encoder(input_ids)['last_hidden_state']
            text_embeddings = linear_layer(text_embeddings)
            text_embeddings = text_embeddings.clone().detach().to(self.device)

            for i, t in enumerate(pipeline.scheduler.timesteps):
                latents = pipeline.scheduler.scale_model_input(latents, t)
                output = unet(latents, t, text_embeddings)
                latents = pipeline.scheduler.step(output[0], t, latents).prev_sample.to(self.device)
                with torch.no_grad():
                    decoded_images = pipeline.decode_latents(latents.detach())
            decoded_image = pipeline.numpy_to_pil(decoded_images)[0]

            # Pass the decoded image through the model
            model_input = transform(decoded_image).unsqueeze(0).to(self.device)
            outputs = self.model(model_input)["out"]
            pred_mask = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            pred_mask = pred_mask.squeeze(0).cpu().numpy()

            pred_mask_resized = Image.fromarray(pred_mask.astype(np.uint8)).resize((width, height), resample=Image.NEAREST)
            pred_mask_resized = np.array(pred_mask_resized)

            decoded_image = decoded_image.resize((width, height), Image.BILINEAR)

            return decoded_image, pred_mask_resized

    def inference_tissue_model_nnUNetv2(self, image_path, output_path, diffusion_path, ldiffusion_weight, segmentor_weight):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        print("Running inference on tissue-level model...")
        if self.model is None:
            self.model = self.initialize_model("tissue", self.num_classes)

        # Load pipeline and models
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(diffusion_path).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(ldiffusion_weight).to(self.device)

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
                latents = pipeline.vae.encode(image).latent_dist.mean
                pipeline.scheduler.set_timesteps(1, device=self.device)
                linear_layer = nn.Linear(768, 1280).to(self.device)
                input_ids = pipeline.tokenizer(["A pathological slide"] * 1)["input_ids"]
                input_ids = torch.tensor(input_ids).to(self.device)
                text_embeddings = pipeline.text_encoder(input_ids)['last_hidden_state']
                text_embeddings = linear_layer(text_embeddings)
                text_embeddings = text_embeddings.clone().detach().to(self.device)

                for i, t in enumerate(pipeline.scheduler.timesteps):
                    latents = pipeline.scheduler.scale_model_input(latents, t)
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
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(diffusion_path).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(ldiffusion_weight).to(self.device)
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
            latents = pipeline.vae.encode(image).latent_dist.mean
            pipeline.scheduler.set_timesteps(1, device=self.device)
            linear_layer = nn.Linear(768, 1280).to(self.device)
            input_ids = pipeline.tokenizer(["A pathological slide"] * 1)["input_ids"]
            input_ids = torch.tensor(input_ids).to(self.device)
            text_embeddings = pipeline.text_encoder(input_ids)['last_hidden_state']
            text_embeddings = linear_layer(text_embeddings)
            text_embeddings = text_embeddings.clone().detach().to(self.device)

            for i, t in enumerate(pipeline.scheduler.timesteps):
                latents = pipeline.scheduler.scale_model_input(latents, t)
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

    def inference_remote_sense_model(self, image_path, dtm_path, diffusion_path, ldiffusion_weight, segmentor_weight):
        print("Running inference on remote sense model...")
        if self.model is None:
            self.model = self.initialize_model("remote sense", self.num_classes)

        # Load pipeline and models
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(diffusion_path).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(os.path.join(ldiffusion_weight, "diffusion_unet")).to(self.device)
        controlnet = ControlNetModel.from_pretrained(os.path.join(ldiffusion_weight, "diffusion_controlnet")).to(self.device)
        linear_layer = torch.nn.Linear(768, 1280).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(segmentor_weight, "convnext_tiny.pth"), weights_only=True))
        self.model.to(self.device)

        # Load and preprocess the input image
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(image_path.replace("rgb", "masks")).convert("L")
        dtm = Image.open(dtm_path).convert("L")
        if image.mode != "RGB":
            raise ValueError(f"Input image is not in RGB mode: {image.mode}")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_dtm = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        transform_canny = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        image = transform(image).unsqueeze(0).to(self.device)
        self.model.eval()

        # Perform inference
        with torch.no_grad():
            dtm_np = np.array(dtm)
            dtm_min, dtm_max = dtm_np.min() / 255, dtm_np.max() / 255
            dtm = transform_dtm(dtm)
            canny_img = cv2.Canny((dtm.permute(1, 2, 0).numpy() * 255).astype(np.uint8), 100, 200)
            canny_img = transforms.Resize(image.shape[-2:])(Image.fromarray(canny_img))
            canny_tensor = transform_canny(canny_img)
            control_img = canny_tensor.expand(3, -1, -1)
            control_img = control_img.unsqueeze(0).to(self.device)
            depth = dtm.unsqueeze(0).to(self.device)

            # VAE encode
            latents = pipeline.vae.encode(image).latent_dist.sample() * 0.18215
            depth_resized = F.interpolate(depth, size=(32, 32), mode='bilinear', align_corners=False)
            depth_resized = depth_resized.repeat(1, latents.shape[1], 1, 1)

            # 添加训练时一致的 Laplace noise
            noise = torch.distributions.laplace.Laplace(0.0, 1.0).sample(latents.shape).to(self.device)
            latents_noisy = latents + noise * depth_resized

            # 文本编码
            input_ids = pipeline.tokenizer(["a remote sense image"], padding="max_length", max_length=77, return_tensors="pt").input_ids.to(
                self.device)
            text_embeddings = pipeline.text_encoder(input_ids)["last_hidden_state"]

            # ControlNet 推理
            pipeline.scheduler.set_timesteps(1, device=self.device)
            for t in tqdm(pipeline.scheduler.timesteps):
                t = t.to(self.device)

                noise = torch.distributions.laplace.Laplace(0.0, 1.0).sample(latents.shape).to(self.device)
                latents_noisy = latents + noise * depth_resized

                down_block_res_samples, mid_block_res_sample = controlnet(
                    sample=latents_noisy,
                    timestep=t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=control_img,
                    return_dict=False
                )

                text_embeddings_proj = linear_layer(text_embeddings)
                noise_pred = unet(
                    latents_noisy,
                    t,
                    encoder_hidden_states=text_embeddings_proj,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

            # 反噪
            latents_denoised = latents_noisy - noise_pred * depth_resized

            # 解码为图像
            with torch.no_grad():
                recon_image = pipeline.vae.decode(latents_denoised / 0.18215).sample  # or .sample

            recon_image = recon_image.squeeze(0).cpu()
            dtm_np = dtm.squeeze(0).cpu().numpy()
            recon_np = recon_image.permute(1, 2, 0).numpy()

            num_bins = 4
            bin_edges = np.linspace(dtm_min, dtm_max, num_bins + 1)
            bin_indices = np.digitize(dtm_np, bins=bin_edges)
            bin_areas = [np.sum(bin_indices == b) for b in range(1, num_bins + 1)]
            largest_bin = np.argmax(bin_areas) + 1

            recon_smoothed = recon_np.copy()
            for b in range(1, num_bins + 1):
                if b == largest_bin:
                    continue

                mask = (bin_indices == b)
                if np.sum(mask) == 0:
                    continue

                mean_color = recon_np[mask].mean(axis=0)
                recon_smoothed[mask] = mean_color

            recon_smoothed = np.clip(recon_smoothed, 0, 1)
            recon_image_np = (recon_smoothed * 255).astype(np.uint8)
            input_tensor = torch.from_numpy(recon_image_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            dtm_tensor = torch.from_numpy(dtm_np).unsqueeze(0).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                result = self.model(input_tensor, dtm_tensor)
                pred_mask, pred_heatmap = result["seg"], result["heatmap"]
                print("Outputs stats:", pred_mask.shape, pred_mask.min().item(), pred_mask.max().item())
                channel_max = pred_mask.mean(dim=(0, 2, 3))
                print("Average logit per class:", channel_max)
                pred_mask = torch.argmax(torch.softmax(pred_mask, dim=1), dim=1)
                pred_mask = pred_mask.squeeze(0).cpu().numpy()

            return recon_np, pred_mask, pred_heatmap


