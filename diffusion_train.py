import csv
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torchvision import transforms
from diffusers import UNet2DConditionModel, StableDiffusionImg2ImgPipeline, LMSDiscreteScheduler
from safetensors.torch import save_file
from datetime import datetime
from tqdm import tqdm
from model.loss import info_nce_loss, save_class_distances_to_csv
from dataloader import MedicalSegmentationDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion model training parameters")
    parser.add_argument("--diffusion-path", type=str, required=True, help="stable diffusion base model path")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--label-dir", type=str, required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--visualization", action="store_true")
    return parser.parse_args()

def load_model(model_path, device):
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_path)
    vae = pipeline.vae.to(device)
    pipeline.text_encoder.to(device)
    return pipeline, vae

def load_data(image_dir, label_dir):
    transform = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.ToTensor(),          
    ])
    dataset = MedicalSegmentationDataset(image_dir, label_dir, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return data_loader

def diffusion_train(pipeline, vae, data_loader, args):
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_inference_steps = args.num_inference_steps
    visualization = args.visualization

    current_date = datetime.now().strftime("%y_%m_%d")
    csv_file = f'train_save/loss/{current_date}/contrast_loss.csv'
    header = ['epoch', 'loss']

    os.makedirs(f'train_save/loss/{current_date}', exist_ok=True)
        
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)


    text_prompts = ["A pathological slide"] * batch_size  # 假设 batch_size 为 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNet2DConditionModel().to(device)
    optimizer = Adam(unet.parameters(), lr=1e-5)
    checkpoint = 100 

    for epoch in range(num_epochs):
        unet.train()
        total_combined_loss = 0.0
        start_time = time.time()
        for image, label, img_name, label_name in tqdm(data_loader):
            linear_layer = nn.Linear(768, 1280).to(device)
            input_ids = pipeline.tokenizer(text_prompts)["input_ids"]
            input_ids = torch.tensor(input_ids).to(device)
            text_embeddings = pipeline.text_encoder(input_ids)['last_hidden_state']
            text_embeddings = linear_layer(text_embeddings)
            text_embeddings = torch.tensor(text_embeddings).to(device)
            del input_ids  # 清理无用变量
            torch.cuda.empty_cache()

            image, label = image.to(device), label.to(device)
            # 将 label 转换为 Float 类型进行插值缩小到64*64
            label_float = label.to(torch.float32)
            label = F.interpolate(label_float, size=(64, 64), mode='bilinear', align_corners=False)
            label = label.to(torch.uint8)
            latents = vae.encode(image).latent_dist.mean
            pipeline.scheduler.set_timesteps(num_inference_steps)

            optimizer.zero_grad()
            for i, t in enumerate(pipeline.scheduler.timesteps):
                if str(i) == str(num_inference_steps):
                    break
                latents = pipeline.scheduler.scale_model_input(latents, t)
                output = unet(latents, t, text_embeddings)
                latents = pipeline.scheduler.step(output[0], t, latents).prev_sample

                decoded_image_rgb = F.interpolate(vae.decode(latents).sample, size=(64, 64), mode='bilinear', align_corners=False)
                weights = torch.tensor([0.2989, 0.5870, 0.1140], device=decoded_image_rgb.device).view(1, 3, 1, 1)
                decoded_image_gray_new = (decoded_image_rgb * weights).sum(dim=1, keepdim=True)
                
                if i == 0:
                    decoded_image_gray = decoded_image_gray_new
                else:
                    decoded_image_gray = torch.cat([decoded_image_gray, decoded_image_gray_new], dim=1)

                del output, decoded_image_rgb, decoded_image_gray_new  # 清理中间变量

            loss = info_nce_loss(decoded_image_gray, label)

            if visualization == True:
                save_class_distances_to_csv(decoded_image_gray, label, epoch, current_date)
                visualization = False
            del decoded_image_gray, latents, image, label
            torch.cuda.empty_cache()
            
            loss.backward()
            optimizer.step()
            total_combined_loss += loss.item()

        current_loss = total_combined_loss/len(data_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {current_loss:.4f}, Elapsed Time: {elapsed_time}s")
        if current_loss < checkpoint:
            unet.save_pretrained(f"train_save/unet/{current_date}")
            checkpoint = current_loss

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, total_combined_loss/len(data_loader)])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    print("\033[35m" + str(vars(args)) + "\033[0m")
    data_loader = load_data(args.image_dir, args.label_dir)
    pipeline, vae = load_model(args.diffusion_path, device)
    diffusion_train(pipeline=pipeline, vae=vae, data_loader=data_loader, args=args)
