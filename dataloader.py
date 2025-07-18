import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

pixel_to_label = {
    0: 0,
    100: 1,
    150: 2,
    50: 3,
    200: 4,
    250: 5,
    255: 6
}

pixel_to_label_cell = {
    0: 0,
    25: 1,
    50: 2,
    75: 3,
    100: 4,
    125: 5,
    150: 6,
    175: 7,
    200: 8,
    225: 9,
    250: 10
}

def convert_labels(img_path, level):
    img = Image.open(img_path).convert("L")
    img_array = np.array(img, dtype=np.uint8)

    label_img = np.zeros_like(img_array, dtype=np.uint8)

    if level == "tissue":
        for orig_pixel, new_label in pixel_to_label.items():
            label_img[img_array == orig_pixel] = new_label
        return label_img
    elif level == "cell":
        for orig_pixel, new_label in pixel_to_label_cell.items():
            label_img[img_array == orig_pixel] = new_label
        return label_img
    else:
        raise ValueError("Unsupported level. Use 'tissue' or 'cell'.")

# 数据集类
class MedicalSegmentationDataset(Dataset):
    def __init__(self, image_dir_list, label_dir_list, transform=None, level=None):
        self.image_dir = image_dir_list
        self.label_dir = label_dir_list
        self.transform = transform
        self.level = level

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        # 图像路径和标签路径
        image_path = self.image_dir[idx]
        label_path = self.label_dir[idx]

        # 加载 RGB 图像并 resize
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        mask = convert_labels(label_path, self.level)
        label = torch.tensor(np.array(mask), dtype=torch.uint8).unsqueeze(0)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask, label

