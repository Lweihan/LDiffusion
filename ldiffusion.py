import random
import csv
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from datetime import datetime
from tqdm import tqdm
from .segmentor import Segmentor
from .model.loss import InfoNceLoss
from torch.utils.data import DataLoader, DistributedSampler
from .dataset import MedicalSegmentationDataset
import deepspeed
from diffusers import StableDiffusionImg2ImgPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion model training parameters")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)))
    parser.add_argument("--diffusion-path", type=str, required=True, help="stable diffusion base model path")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--label-dir", type=str, required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    return parser.parse_args()

class LDiffusionModel:
    def __init__(self, diffusion_path, level, local_rank=-1):
        self.local_rank = local_rank
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.is_distributed = self.world_size > 1
        if self.is_distributed and not deepspeed.comm.is_initialized():
            deepspeed.init_distributed()

        if torch.cuda.is_available():
            if self.local_rank is None or self.local_rank < 0:
                self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        self.level = level
        self.diffusion_path = diffusion_path
        self.pipeline, self.vae = None, None
        self.info_nce_loss = InfoNceLoss()
        self.linear_layer = None  # 文本嵌入投影层，随 UNet 一起学习

    def _is_main_process(self):
        return self.global_rank == 0

    def _reduce_mean(self, value):
        if not self.is_distributed:
            return value

        reduced = torch.tensor(value, device=self.device, dtype=torch.float32)
        deepspeed.comm.all_reduce(reduced, op=deepspeed.comm.ReduceOp.SUM)
        reduced = reduced / self.world_size
        return reduced.item()

    def load_model(self, model_path):
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
        vae = pipeline.vae.to(self.device, dtype=torch.float32)
        pipeline.text_encoder.to(self.device, dtype=torch.float32)
        return pipeline, vae

    def load_data(self, image_dir, label_dir, batch_size, train_ratio=0.7):
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

        train_sampler = None
        val_sampler = None
        if self.is_distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=True, drop_last=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=False, drop_last=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=4,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            sampler=val_sampler,
            num_workers=4,
        )

        return train_loader, val_loader, train_sampler

    def train_ldiffusion(self, args, train_loader, val_loader):
        num_epochs = 10
        batch_size = args.batch_size
        num_inference_steps = args.num_inference_steps
        self.pipeline, self.vae = self.load_model(args.diffusion_path)

        current_date = datetime.now().strftime("%y_%m_%d")
        csv_file = f'train_save/loss/{current_date}/contrast_loss.csv'
        header = ['epoch', 'loss']

        os.makedirs(f'train_save/loss/{current_date}', exist_ok=True)

        if self._is_main_process():
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)

        # 从预训练 pipeline 加载 UNet 微调
        unet = self.pipeline.unet.to(self.device, dtype=torch.float32)

        text_hidden_size = self.pipeline.text_encoder.config.hidden_size
        cross_attention_dim = self.pipeline.unet.config.cross_attention_dim

        # 初始化或复用持久化的投影层
        if (
            self.linear_layer is None
            or self.linear_layer.in_features != text_hidden_size
            or self.linear_layer.out_features != cross_attention_dim
        ):
            self.linear_layer = nn.Linear(text_hidden_size, cross_attention_dim)
        self.linear_layer = self.linear_layer.to(self.device, dtype=torch.float32)

        # 封装为单一 nn.Module，供 DeepSpeed ZeRO-3 统一管理参数分片与梯度聚合
        class _TrainWrapper(nn.Module):
            def __init__(self, unet, proj):
                super().__init__()
                self.unet = unet
                self.proj = proj
            def forward(self, sample, timestep, encoder_hidden_states):
                return self.unet(sample, timestep, encoder_hidden_states)

        wrapper = _TrainWrapper(unet, self.linear_layer).to(self.device, dtype=torch.float32)

        # DeepSpeed ZeRO-3 配置：通过对比损失拉大不同标签像素间的扩散表示差距
        ds_config = {
            "train_micro_batch_size_per_gpu": batch_size,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "offload_param": {"device": "cpu", "pin_memory": True},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
            },
            "fp16": {"enabled": False},
            "gradient_clipping": 1.0,
        }
        engine, _, _, _ = deepspeed.initialize(
            model=wrapper,
            model_parameters=wrapper.parameters(),
            config=ds_config
        )

        save_path = f"LDiffusion/train_save/unet/{current_date}"
        checkpoint = 100

        num_inference_steps = min(int(num_inference_steps / 5), len(self.pipeline.scheduler.alphas_cumprod))

        resize_transform = transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR)

        for epoch in range(num_epochs):
            if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            engine.train()
            total_combined_loss = 0.0
            start_time = time.time()
            for image, _, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Warming up", leave=False):
                current_batch_size = image.shape[0]
                text_prompts = ["A pathological slide"] * current_batch_size
                image = torch.stack([resize_transform(img) for img in image], dim=0).to(self.device, dtype=torch.float32)
                input_ids = self.pipeline.tokenizer(text_prompts)["input_ids"]
                input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    text_embeddings = self.pipeline.text_encoder(input_ids)['last_hidden_state'].to(dtype=torch.float32)
                proj_device = engine.module.proj.weight.device
                text_embeddings = text_embeddings.to(device=proj_device, dtype=torch.float32)
                text_embeddings = engine.module.proj(text_embeddings)
                del input_ids  # 清理无用变量
                torch.cuda.empty_cache()

                image, label = image.to(self.device, dtype=torch.float32), label.to(self.device)
                label_float = label.to(torch.float32)
                label = F.interpolate(label_float, size=(64, 64), mode='bilinear', align_corners=False)
                label = label.to(torch.uint8)
                with torch.no_grad():
                    latents = self.vae.encode(image).latent_dist.mean.to(dtype=torch.float32)
                self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)

                decoded_image_rgb = None
                for i, t in enumerate(self.pipeline.scheduler.timesteps):
                    latents = self.pipeline.scheduler.scale_model_input(latents, t).to(dtype=torch.float32)
                    scale_factor = torch.sqrt(1 - self.pipeline.scheduler.alphas_cumprod[t]).to(device=self.device, dtype=torch.float32)
                    laplace_dist = torch.distributions.Laplace(0, scale_factor)
                    noise = laplace_dist.sample(latents.shape).to(self.device, dtype=torch.float32)
                    noisy_latents = (latents + noise).to(dtype=torch.float32)
                    denoised_latents = engine(noisy_latents, t, text_embeddings).sample

                    decoded_image_rgb = F.interpolate(self.vae.decode(denoised_latents.to(dtype=torch.float32)).sample, size=(64, 64), mode='bilinear', align_corners=False).to(dtype=torch.float32)
                    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=decoded_image_rgb.device, dtype=torch.float32).view(1, 3, 1, 1)
                    decoded_image_gray_new = (decoded_image_rgb * weights).sum(dim=1, keepdim=True)

                    if i == 0:
                        decoded_image_gray = decoded_image_gray_new
                    else:
                        decoded_image_gray = torch.cat([decoded_image_gray, decoded_image_gray_new], dim=1)

                    del decoded_image_gray_new

                decoded_image_rgb = F.interpolate(decoded_image_rgb, size=(1024, 1024), mode='bilinear', align_corners=False).to(dtype=torch.float32)
                loss = self.info_nce_loss.compute_loss(image, decoded_image_rgb, decoded_image_gray, label)

                engine.backward(loss)
                engine.step()
                total_combined_loss += loss.item()

            current_loss = total_combined_loss / len(train_loader)
            current_loss = self._reduce_mean(current_loss)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if self._is_main_process():
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {current_loss:.4f}, Elapsed Time: {elapsed_time}s")

            should_save = current_loss < checkpoint
            if should_save:
                if self._is_main_process():
                    os.makedirs(save_path, exist_ok=True)

                # ZeRO-3 参数聚合需要所有 rank 进入同一通信路径，避免在 epoch 切换处卡住
                with deepspeed.zero.GatheredParameters(engine.module.parameters(), modifier_rank=0):
                    if self._is_main_process():
                        engine.module.unet.save_pretrained(save_path)
                        torch.save(
                            engine.module.proj.state_dict(),
                            os.path.join(save_path, "proj_weights.pt")
                        )

                checkpoint = current_loss

            if self.is_distributed:
                deepspeed.comm.barrier()

            if self._is_main_process():
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch + 1, current_loss])

        # 清理显存中的模型
        del self.pipeline
        del self.vae
        del engine
        torch.cuda.empty_cache()

        return save_path

    def train(self, args, component="all", ldiffusion_weight=None):
        train_loader, val_loader, _ = self.load_data(args.image_dir, args.label_dir, args.batch_size)
        segmentor = Segmentor(train_loader, val_loader, self.level, args.num_classes)
        if component == "all" or component == "ldiffusion":
            if self._is_main_process():
                print("Starting LDiffusion warming up...")
            ldiffusion_weight = self.train_ldiffusion(args, train_loader, val_loader)
            torch.cuda.empty_cache()  # Clear GPU memory after LDiffusion training
        if component == "all" or component == "segmentor":
            if self._is_main_process():
                print("Starting Segmentor training...")
            if component == "segmentor":
                ldiffusion_weight = ldiffusion_weight
            if segmentor.level == "tissue":
                segmentor.train_tissue_model_nnUNetv2(args.num_epochs - 10, ldiffusion_weight, args.diffusion_path)
            elif segmentor.level == "cell":
                segmentor.train_cell_model(args.num_epochs - 10, ldiffusion_weight, args.diffusion_path)
            else:
                raise ValueError("Invalid level specified. Choose 'tissue' or 'cell'.")

    def inference(self, image_path, ldiffusion_weight, segmentor_weight, num_classes):
        segmentor = Segmentor(train_loader=None, val_loader=None, level=self.level, num_classes=num_classes)
        if self.level == "tissue":
            return segmentor.inference_tissue_model_nnUNetv2(image_path, self.diffusion_path, ldiffusion_weight, segmentor_weight)
        elif self.level == "cell":
            return segmentor.inference_cell_model(image_path, self.diffusion_path, ldiffusion_weight, segmentor_weight)
        else:
            raise ValueError("Invalid level specified. Choose 'tissue' or 'cell'.")

if __name__ == "__main__":
    args = parse_args()
    if int(os.environ.get("RANK", "0")) == 0:
        print("\033[35m" + str(vars(args)) + "\033[0m")
    trainer = LDiffusionModel(args.diffusion_path, level="tissue", local_rank=args.local_rank)
    trainer.train(args, component="segmentor", ldiffusion_weight='/c23227/lwh/code/LDiffusion/train_save/unet/26_03_22')
