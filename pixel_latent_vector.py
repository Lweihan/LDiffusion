import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import LMSDiscreteScheduler
from diffusers import UNet2DConditionModel
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from dataset import CustomDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Pixel latent vector construction parameters")
    parser.add_argument("--diffusion-path", type=str, required=True, help="stable diffusion base model path")
    parser.add_argument("--save-model", type=str, required=True, help="diffusion train save model path")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--label-dir", type=str, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    return parser.parse_args()

def load_data(image_dir, label_dir):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),         
    ])
    train_dataset = CustomDataset(image_dir, label_dir, train=True, transform=transform)
    non_train_dataset = CustomDataset(image_dir, label_dir, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    non_train_loader = DataLoader(non_train_dataset, batch_size=1, shuffle=False)
    return train_loader, non_train_loader

def load_model(model_path, save_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_path)
    vae = pipeline.vae.to(device)
    pipeline.text_encoder.to(device)
    unet = UNet2DConditionModel.from_pretrained(save_model)
    unet.eval()
    unet.to(device)
    return pipeline, vae, unet

def generate_title(n):
    title = ['Pixel No.']
    for i in range(n):
        index = i + 1
        col = 'Sample ' + str(index)
        title.append(col)
    title.append('Category')
    return title

def pixel_latent_vector(pipeline, vae, unet, num_inference_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_date = datetime.now().strftime("%y_%m_%d")
    os.makedirs(f'eval/vector_set/{current_date}', exist_ok=True)
    image_index = 0
    for image, label in tqdm(train_loader):
        linear_layer = nn.Linear(768, 1280).to(device)
        input_ids = pipeline.tokenizer(["A pathological slide"] * 1)["input_ids"]
        input_ids = torch.tensor(input_ids).to(device)
        text_embeddings = pipeline.text_encoder(input_ids)['last_hidden_state']
        text_embeddings = linear_layer(text_embeddings)
        text_embeddings = torch.tensor(text_embeddings).to(device)
        
        image, label = image.to(device), label.to(device)
        with torch.no_grad():
            latents = vae.encode(image).latent_dist.mean
        pipeline.scheduler.set_timesteps(num_inference_steps-1, device=device)
        decoded_image_list = []
        for i, t in enumerate(pipeline.scheduler.timesteps):
            latents = pipeline.scheduler.scale_model_input(latents, t)
            output = unet(latents, t, text_embeddings)
            latents = pipeline.scheduler.step(output[0], t, latents).prev_sample.to(device)
            with torch.no_grad():
                decoded_images = pipeline.decode_latents(latents.detach())
            decoded_image = pipeline.numpy_to_pil(decoded_images)[0]
            decoded_image_list.append(decoded_image)
        pixel_dict = {}
        grayscale_image_list = [image.convert("L") for image in decoded_image_list]
        grayscale_images = [np.array(img) for img in grayscale_image_list]
        height, width = grayscale_images[0].shape
        pixel_values = np.array(label[0].cpu())
        for px_i in range(height):
            for px_j in range(width):
                pixel_vector = [grayscale_images[k][px_i, px_j] for k in range(num_inference_steps)]
                pixel_vector.append(pixel_values[0][px_i, px_j])
                pixel_dict[(px_i, px_j)] = pixel_vector
        
        with open(f'eval/vector_set/{current_date}/pixel_dict_{image_index}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            title = generate_title(num_inference_steps)
            writer.writerow(title)
            for key, values in pixel_dict.items():
                # 将键和列表值写入 CSV，每个列表的元素将分别填充到 Value1, Value2, ... 中
                writer.writerow([key] + values)
        image_index += 1

if __name__ == "__main__":
    args = parse_args()
    print("\033[35m" + str(vars(args)) + "\033[0m")
    train_loader, non_train_loader = load_data(args.image_dir, args.label_dir)
    pipeline, vae, unet = load_model(args.diffusion_path, args.save_model)
    pixel_latent_vector(pipeline=pipeline, vae=vae, unet=unet, num_inference_steps=args.num_inference_steps)
