import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MedicalSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_names = [f for f in sorted(os.listdir(image_dir)) if f.endswith('.tif')]
        self.label_names = [f for f in sorted(os.listdir(label_dir)) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        label_name = os.path.join(self.label_dir, self.label_names[idx])
        image = Image.open(img_name).convert("RGB")  # 转换为 RGB 格式
        label = Image.open(label_name).convert("L").resize((512, 512)) # 标签通常是单通道
        
        if self.transform:
            image = self.transform(image)

        # 将像素值为 255 的位置转成 5
        image_array = np.array(label)
        image_array[image_array == 255] = 0
        pixel_values = image_array // 50
        image_tensor = torch.tensor(pixel_values, dtype=torch.uint8)
        # 将数组调整为适当的形状，并增加一个维度以符合 [1, 512, 512] 形状
        label = image_tensor.view(512, 512).unsqueeze(0)  # 在最前面添加一个维度
        return image, label, img_name, label_name

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, train=True, train_ratio=0.7, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('jpg', 'png', 'tif'))])
        random.shuffle(self.image_files)  # 随机打乱
        
        train_size = int(len(self.image_files) * train_ratio)
        
        if train:
            self.image_files = self.image_files[:train_size]
        else:
            self.image_files = self.image_files[train_size:]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".tif", ".png"))
        
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L").resize((512, 512))
        
        if self.transform:
            image = self.transform(image)

        image_array = np.array(label)
        image_array[image_array == 255] = 0
        pixel_values = image_array // 50
        # pixel_values = image_array // 25
        image_tensor = torch.tensor(pixel_values, dtype=torch.uint8)
        label = image_tensor.view(512, 512).unsqueeze(0)
        return image, label

class ConvNeXTDataset(Dataset):
    def __init__(self, features, labels, target_size=(32, 32)):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.target_size = target_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx].unsqueeze(-1).unsqueeze(-1)  
        feature = feature.expand(-1, *self.target_size)         
        return feature, self.labels[idx]