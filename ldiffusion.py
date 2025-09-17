import random
import csv
import argparse
import os
import kornia
import time
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.optim import Adam
import torch.nn.functional as F
from torchvision import transforms
from datetime import datetime
from tqdm import tqdm
from .segmentor import Segmentor
from .model.loss import InfoNceLoss
from torch.utils.data import DataLoader
from .dataset import MedicalSegmentationDataset, RgbDtmMaskDataset
from diffusers import UNet2DConditionModel, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline, ControlNetModel

def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion model training parameters")
    parser.add_argument("--diffusion-path", type=str, required=True, help="stable diffusion base model path")
    parser.add_argument("--controlnet-path", type=str, required=False, help="controlnet model path")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--label-dir", type=str, required=False)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    return parser.parse_args()

class LDiffusionModel:
    def __init__(self, diffusion_path, level):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.level = level
        self.diffusion_path = diffusion_path
        self.pipeline, self.vae = None, None
        self.info_nce_loss = InfoNceLoss()

    def load_model(self, diffusion_model_path=None, controlnet_model_path=None):
        if controlnet_model_path is not None:
            controlnet = ControlNetModel.from_pretrained(controlnet_model_path).to(self.device)
        else:
            controlnet = None
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(diffusion_model_path)
        vae = pipeline.vae.to(self.device)
        pipeline.text_encoder.to(self.device)
        return pipeline, vae, controlnet

    def load_data(self, image_dir, label_dir, batch_size, train_ratio=0.9):
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        labels = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])

        assert len(images) == len(labels), "原始图片和标签图片的数量不一致"

        indices = list(range(len(images)))
        random.shuffle(indices)

        split_index = int(len(images) * train_ratio)
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        train_images, train_labels = [images[i] for i in train_indices], [labels[i] for i in train_indices]
        test_images, test_labels = [images[i] for i in test_indices], [labels[i] for i in test_indices]

        train_dataset = MedicalSegmentationDataset(train_images, train_labels, transform, self.level)
        val_dataset = MedicalSegmentationDataset(test_images, test_labels, transform, self.level)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

        return train_loader, val_loader, train_images, train_labels, test_images, test_labels

    def load_remote_sense_data(self, root_dir, batch_size):
        transform_rgb = transforms.Compose([
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

        train_dataset = RgbDtmMaskDataset(root_dir, split='train', transform_rgb=transform_rgb,
                                    transform_dtm=transform_dtm, transform_mask=None, transform_canny=transform_canny)
        test_dataset = RgbDtmMaskDataset(root_dir, split='test', transform_rgb=transform_rgb,
                                    transform_dtm=transform_dtm, transform_mask=None, transform_canny=transform_canny)
        val_dataset = RgbDtmMaskDataset(root_dir, split='val', transform_rgb=transform_rgb,
                                    transform_dtm=transform_dtm, transform_mask=None, transform_canny=transform_canny)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

        return train_loader, test_loader

    def train_ldiffusion(self, args, train_loader, val_loader):

        def rgb_to_grayscale(img):
            # img: [B, 3, H, W], float 0~1
            return kornia.color.rgb_to_grayscale(img)

        def edge_map(img_gray):
            if img_gray.ndim == 3:
                img_gray = img_gray.unsqueeze(1)
            edges = kornia.filters.sobel(img_gray)  # 默认计算边缘
            return edges

        batch_size = args.batch_size
        num_inference_steps = args.num_inference_steps
        if self.level == "tissue" or self.level == "cell":
            self.pipeline, self.vae, _ = self.load_model(args.diffusion_path, None)
        elif self.level == "remote sense":
            self.pipeline, self.vae, controlnet = self.load_model(args.diffusion_path, args.controlnet_path)

        current_date = datetime.now().strftime("%y_%m_%d")
        csv_file = f'train_save/loss/{current_date}/contrast_loss.csv'
        header = ['epoch', 'loss']

        os.makedirs(f'train_save/loss/{current_date}', exist_ok=True)

        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

        if self.level == "tissue" or self.level == "cell":
            num_epochs = 10
            text_prompts = ["A pathological slide"] * batch_size
        elif self.level == "remote sense":
            num_epochs = 10
            text_prompts = ["A remote sense image"] * batch_size
        unet = UNet2DConditionModel().to(self.device)
        if self.level == "remote sense":
            optimizer = Adam(list(unet.parameters()) + list(controlnet.parameters()), lr=1e-5)
        else:
            optimizer = Adam(unet.parameters(), lr=1e-5)
        save_path = f"LDiffusion/train_save/unet/{current_date}"
        checkpoint = float('inf')

        num_inference_steps = min(int(num_inference_steps / 5), len(self.pipeline.scheduler.alphas_cumprod))

        resize_transform = transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR)

        for epoch in range(num_epochs):
            unet.train()
            if self.level == "remote sense":
                controlnet.train()
            total_combined_loss = 0.0
            start_time = time.time()
            if self.level == "tissue" or self.level == "cell":
                for image, _, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Warming up", leave=False):
                    image = torch.stack([resize_transform(img) for img in image], dim=0)
                    linear_layer = nn.Linear(768, 1280).to(self.device)
                    input_ids = self.pipeline.tokenizer(text_prompts)["input_ids"]
                    input_ids = torch.tensor(input_ids).to(self.device)
                    text_embeddings = self.pipeline.text_encoder(input_ids)['last_hidden_state']
                    text_embeddings = linear_layer(text_embeddings)
                    text_embeddings = text_embeddings.clone().detach().to(self.device)
                    del input_ids  # 清理无用变量
                    torch.cuda.empty_cache()

                    image, label = image.to(self.device), label.to(self.device)
                    label_float = label.to(torch.float32)
                    label = F.interpolate(label_float, size=(64, 64), mode='bilinear', align_corners=False)
                    label = label.to(torch.uint8)
                    latents = self.vae.encode(image).latent_dist.mean
                    self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)

                    optimizer.zero_grad()
                    decoded_image_rgb = None
                    for i, t in enumerate(self.pipeline.scheduler.timesteps):
                        latents = self.pipeline.scheduler.scale_model_input(latents, t)
                        scale_factor = torch.sqrt(1 - self.pipeline.scheduler.alphas_cumprod[t])
                        laplace_dist = torch.distributions.Laplace(0, scale_factor)
                        noise = laplace_dist.sample(latents.shape).to(self.device)
                        noisy_latents = latents + noise
                        pred_noise = unet(noisy_latents, t, text_embeddings).sample
                        denoised_latents = self.pipeline.scheduler.step(
                            model_output=pred_noise,
                            timestep=t,
                            sample=noisy_latents
                        ).prev_sample

                        decoded_image_rgb = F.interpolate(self.vae.decode(denoised_latents).sample, size=(64, 64), mode='bilinear', align_corners=False)
                        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=decoded_image_rgb.device).view(1, 3, 1, 1)
                        decoded_image_gray_new = (decoded_image_rgb * weights).sum(dim=1, keepdim=True)

                        if i == 0:
                            decoded_image_gray = decoded_image_gray_new
                        else:
                            decoded_image_gray = torch.cat([decoded_image_gray, decoded_image_gray_new], dim=1)

                        del decoded_image_gray_new

                    decoded_image_rgb = F.interpolate(decoded_image_rgb, size=(1024, 1024), mode='bilinear', align_corners=False)
                    loss = self.info_nce_loss.compute_loss(image, decoded_image_rgb, decoded_image_gray, label)

                    loss.backward()
                    optimizer.step()
                    total_combined_loss += loss.item()

                current_loss = total_combined_loss / len(train_loader)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {current_loss:.4f}, Elapsed Time: {elapsed_time}s")
                if current_loss < checkpoint:
                    unet.save_pretrained(save_path)
                    checkpoint = current_loss

                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch + 1, total_combined_loss / len(train_loader)])
            elif self.level == "remote sense":
                for batch in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]"):

                    rgb = batch["rgb"].to(self.device)
                    depth = batch["dtm"].to(self.device)
                    mask = batch["mask"].to(self.device)
                    control_img = batch["canny"].to(self.device)

                    with torch.no_grad():
                        latents = self.vae.encode(rgb).latent_dist.sample() * 0.18215

                    depth_resized = F.interpolate(depth, size=(32, 32), mode='bilinear', align_corners=False)
                    depth_resized = depth_resized.repeat(1, latents.shape[1], 1, 1)
                    depth_condition = depth.repeat(1, 3, 1, 1)

                    linear_layer = nn.Linear(768, 1280).to(self.device)
                    input_ids = self.pipeline.tokenizer(text_prompts)["input_ids"]
                    input_ids = torch.tensor(input_ids).to(self.device)
                    text_embeddings = self.pipeline.text_encoder(input_ids)['last_hidden_state']

                    self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)

                    batch_loss = 0
                    for t in self.pipeline.scheduler.timesteps:
                        t = t.to(self.device)

                        noise = torch.distributions.laplace.Laplace(0.0, 1.0).sample(latents.shape).to(self.device)
                        latents_noisy = latents + noise * depth_resized

                        down_block_res_samples, mid_block_res_sample = controlnet(
                            sample=latents_noisy,
                            timestep=t,
                            encoder_hidden_states=text_embeddings,
                            controlnet_cond=depth_condition,
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

                        loss = F.mse_loss(noise_pred, noise)
                        batch_loss += loss

                        with torch.no_grad():
                            latents_pred = self.pipeline.scheduler.step(
                                model_output=noise_pred,
                                timestep=t,
                                sample=latents_noisy
                            ).prev_sample

                    batch_loss = batch_loss / len(self.pipeline.scheduler.timesteps)  # timestep 平均损失

                    generated_img = self.vae.decode(latents_pred).sample  # [B, 3, H, W], float0~1

                    # 灰度图 + 边缘
                    gen_gray = rgb_to_grayscale(torch.nan_to_num(generated_img, nan=0.0, posinf=1.0, neginf=0.0))
                    gen_edge = edge_map(gen_gray)

                    # mask 归一化防止除零
                    max_val = mask.max()
                    if max_val > 0:
                        mask_float = mask.float() / max_val
                    else:
                        mask_float = mask.float()

                    mask_float = torch.nan_to_num(mask_float, nan=0.0, posinf=1.0, neginf=0.0)
                    mask_edge = edge_map(mask_float)

                    # L1 loss 忽略 NaN
                    diff = gen_edge - mask_edge
                    diff = diff[~torch.isnan(diff)]
                    shape_loss = diff.abs().mean() if diff.numel() > 0 else torch.tensor(0.0, device=gen_edge.device)

                    # 总loss加权合成
                    alpha_shape = 1.0

                    total_loss = batch_loss + alpha_shape * shape_loss
                    total_combined_loss += total_loss.item()

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                end_time = time.time()
                elapsed_time = end_time - start_time
                avg_loss = total_combined_loss / len(train_loader)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {elapsed_time:.1f}s")

                if avg_loss < checkpoint:
                    unet.save_pretrained(os.path.join(save_path, "diffusion_unet"))
                    controlnet.save_pretrained(os.path.join(save_path, "diffusion_controlnet"))
                    checkpoint = avg_loss
        # 清理显存中的模型
        del self.pipeline
        del self.vae
        del unet
        torch.cuda.empty_cache()

        return save_path

    def train(self, args, component="all", ldiffusion_weight=None, controlnet_weight=None):
        if self.level == "tissue" or self.level == "cell":
            train_loader, val_loader, train_images, train_labels, test_images, test_labels = self.load_data(args.image_dir, args.label_dir, args.batch_size)
        elif self.level == "remote sense":
            train_loader, val_loader = self.load_remote_sense_data(args.image_dir, args.batch_size)
        segmentor = Segmentor(train_loader, val_loader, self.level, args.num_classes)
        if component == "all" or component == "ldiffusion":
            print("Starting LDiffusion warming up...")
            ldiffusion_weight = self.train_ldiffusion(args, train_loader, val_loader)
            torch.cuda.empty_cache()  # Clear GPU memory after LDiffusion training
        if component == "all" or component == "segmentor":
            print("Starting Segmentor training...")
            if component == "segmentor":
                ldiffusion_weight = ldiffusion_weight
            if segmentor.level == "tissue":
                # segmentor.train_tissue_model(args.num_epochs - 10, ldiffusion_weight, args.diffusion_path)
                segmentor.train_tissue_model_nnUNetv2(args.num_epochs, ldiffusion_weight, args.diffusion_path, train_images, train_labels, test_images, test_labels, args.num_classes)
            elif segmentor.level == "cell":
                segmentor.train_cell_model(args.num_epochs - 10, ldiffusion_weight, args.diffusion_path)
            elif segmentor.level == "remote sense":
                segmentor.train_remote_sense_model(args.num_epochs - 10, ldiffusion_weight, args.diffusion_path, controlnet_weight, args.batch_size)
            else:
                raise ValueError("Invalid level specified. Choose 'tissue' or 'cell'.")

    def inference(self, image_path, dtm_path, ldiffusion_weight, segmentor_weight, num_classes, output_path=None):
        segmentor = Segmentor(train_loader=None, val_loader=None, level=self.level, num_classes=num_classes)
        if self.level == "tissue":
            return segmentor.inference_tissue_model_nnUNetv2(image_path, output_path, self.diffusion_path, ldiffusion_weight, segmentor_weight)
        elif self.level == "cell":
            return segmentor.inference_cell_model(image_path, self.diffusion_path, ldiffusion_weight, segmentor_weight)
        elif self.level == "remote sense":
            return segmentor.inference_remote_sense_model(image_path, dtm_path, self.diffusion_path, ldiffusion_weight, segmentor_weight)
        else:
            raise ValueError("Invalid level specified. Choose 'tissue' or 'cell'.")

if __name__ == "__main__":
    args = parse_args()
    print("\033[35m" + str(vars(args)) + "\033[0m")
    trainer = LDiffusionModel(args.diffusion_path, level="tissue")
    trainer.train(args, component="segmentor",
                  ldiffusion_weight='/home/disk3/lwh/image_process/evaluation/LDiffusion/train_save/unet/25_09_13',
                  controlnet_weight=None)
