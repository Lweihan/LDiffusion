import os
import cv2
import glob
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

ID_TO_CLASS = {
    0: 0,   # background
    60: 1,   # hydrology
    120: 2,  # mound
    180: 3,  # temple
    255: 0  # void
}

def map_mask(mask_np):
    mapped = np.zeros_like(mask_np, dtype=np.int64)
    for k, v in ID_TO_CLASS.items():
        mapped[mask_np == k] = v
    return mapped

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

class MedicalSegmentationDataset(Dataset):
    def __init__(self, image_dir_list, label_dir_list, transform=None, level=None):
        self.image_dir = image_dir_list
        self.label_dir = label_dir_list
        self.transform = transform
        self.level = level

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        image_path = self.image_dir[idx]
        label_path = self.label_dir[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        mask = convert_labels(label_path, self.level)
        label = torch.tensor(np.array(mask), dtype=torch.uint8).unsqueeze(0)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask, label

class RgbDtmMaskDataset(Dataset):
    def __init__(self, root_dir, split='train', transform_rgb=None, transform_dtm=None, transform_mask=None, transform_canny=None):
        self.rgb_dir = os.path.join(root_dir, split, 'rgb')
        self.dtm_dir = os.path.join(root_dir, split, 'dtm')
        self.mask_dir = os.path.join(root_dir, split, 'masks')

        # 按文件名排序，确保对应关系
        self.rgb_files = sorted(glob.glob(os.path.join(self.rgb_dir, '*.tif')))
        self.dtm_files = sorted(glob.glob(os.path.join(self.dtm_dir, '*.tif')))
        self.mask_files = sorted(glob.glob(os.path.join(self.mask_dir, '*.tif')))

        rgb_size = len(self.rgb_files)
        dtm_size = len(self.dtm_files)
        mask_size = len(self.mask_files)
        assert rgb_size == dtm_size == mask_size, f"文件数量不匹配{rgb_size}, {dtm_size}, {mask_size}"

        self.transform_rgb = transform_rgb
        self.transform_dtm = transform_dtm
        self.transform_mask = transform_mask
        self.transform_canny = transform_canny

        # 预定义基本转换
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # 读取 RGB
        rgb = Image.open(self.rgb_files[idx]).convert('RGB')
        # 读取 DTM (灰度)
        dtm = Image.open(self.dtm_files[idx]).convert('L')
        # 读取 mask
        mask = Image.open(self.mask_files[idx]).convert('L').resize((256, 256), resample=Image.NEAREST)

        # 转 numpy 映射 mask
        mask_np = np.array(mask)
        mask_mapped = map_mask(mask_np)

        # 变换
        if self.transform_rgb:
            rgb = self.transform_rgb(rgb)
        else:
            rgb = self.to_tensor(rgb)  # [3, H, W], float32, 0~1

        if self.transform_dtm:
            dtm = self.transform_dtm(dtm)
        else:
            dtm = self.to_tensor(dtm)  # [1, H, W], float32, 0~1

        if self.transform_mask:
            mask = self.transform_mask(mask)
        else:
            mask = torch.from_numpy(mask_mapped).long()  # [H, W], int64

        canny_img = cv2.Canny((dtm.permute(1, 2, 0).numpy() * 255).astype(np.uint8), 100, 200)
        canny_img = transforms.Resize(rgb.shape[-2:])(Image.fromarray(canny_img))
        canny_tensor = self.transform_canny(canny_img)
        canny_tensor = canny_tensor.expand(3, -1, -1)

        return {
            "rgb": rgb,
            "dtm": dtm,
            "mask": mask,
            "canny": canny_tensor
        }

