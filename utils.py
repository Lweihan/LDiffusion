import tifffile
import torch.nn as nn
from torchvision import transforms
import json
import numpy as np
import torch
import scipy.ndimage
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from nnunetv2.paths import nnUNet_raw
from LDiffusion.dataset import pixel_to_label

warned = False  # 全局标记，只提示一次

def fix_label_size(img, lbl, img_name, lbl_name):
    global warned
    if img.size != lbl.size:
        if not warned:
            print("\033[91m[WARNING] Some image/label sizes are inconsistent. Labels will be resized to match images.\033[0m")
            warned = True
        lbl = lbl.resize(img.size, Image.NEAREST)
    return lbl

def generate_multi_class_heatmaps(masks, num_classes, sigma=5):
    """
    masks: (B, H, W) 长整型，每像素值为类别编号
    returns: (B, num_classes, H, W) float32 heatmaps
    """
    B, H, W = masks.shape
    heatmaps = np.zeros((B, num_classes, H, W), dtype=np.float32)

    for b in range(B):
        for cls in range(num_classes):  # 包含背景 0
            mask = (masks[b] == cls).astype(np.uint8)
            labeled, num = scipy.ndimage.label(mask)
            for region_idx in range(1, num + 1):
                region_mask = (labeled == region_idx).astype(np.uint8)
                if region_mask.sum() == 0:
                    continue
                y, x = scipy.ndimage.center_of_mass(region_mask)
                if np.isnan(x) or np.isnan(y):
                    continue
                heatmaps[b, cls] += generate_gaussian(H, W, x, y, sigma)
                heatmaps[b, cls] = np.clip(heatmaps[b, cls], 0, 1)
    return torch.from_numpy(heatmaps)

def generate_gaussian(H, W, x, y, sigma):
    """生成以(x,y)为中心的高斯热力图"""
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return heatmap

def micro_dice(predicted_labels, true_labels, num_classes=7):
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

def mean_iou_and_per_class(pred, target, num_classes):
    pred_labels = torch.argmax(pred, dim=1)
    ious = []
    iou_dict = {}

    for cls in range(num_classes):
        pred_inds = (pred_labels == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            iou_dict[cls] = None
            continue
        iou = intersection / union
        ious.append(iou)
        iou_dict[cls] = iou

    mean_iou = sum(ious) / len(ious) if ious else 1.0
    return mean_iou, iou_dict

def extract_topk_points(heatmaps, k=5, ignore_class=0, score_threshold=0.5):
    B, C, H, W = heatmaps.shape
    device = heatmaps.device
    coords_list = []
    labels_list = []

    class_range = list(range(C))
    if ignore_class in class_range:
        class_range.remove(ignore_class)

    for b in range(B):
        scores = []
        coords = []
        labels = []

        for cls in class_range:
            heatmap = heatmaps[b, cls]  # [H, W]
            flat_heatmap = heatmap.view(-1)
            topk_vals, topk_idxs = torch.topk(flat_heatmap, k, largest=True)

            # 跳过该类（最大值低于阈值）
            if topk_vals.max().item() < score_threshold:
                continue

            y_coords = topk_idxs // W
            x_coords = topk_idxs % W
            point_coords = torch.stack([x_coords, y_coords], dim=1)  # [k, 2]

            scores.append(topk_vals)
            coords.append(point_coords)
            labels.append(torch.full((k,), cls, dtype=torch.int64, device=device))

        if len(scores) == 0:
            # 没有任何类别满足条件，返回空值
            coords_list.append(torch.empty((0, 2), dtype=torch.int64, device=device))
            labels_list.append(torch.empty((0,), dtype=torch.int64, device=device))
            continue

        all_scores = torch.cat(scores, dim=0)
        all_coords = torch.cat(coords, dim=0)
        all_labels = torch.cat(labels, dim=0)

        topk_final = min(k, all_scores.size(0))
        topk_final_scores, indices = torch.topk(all_scores, topk_final, largest=True)
        coords_list.append(all_coords[indices])
        labels_list.append(all_labels[indices])

    return coords_list, labels_list

def check_images_same_size(image_paths):
    """检查所有图片尺寸是否一致"""
    sizes = set()
    for path in image_paths:
        with Image.open(path) as img:
            sizes.add(img.size)
            if len(sizes) > 1:  # 提前结束
                return False
    return True

def convert_and_save_label(lbl, dst_path, mapping):
    """读取灰度图，按照 mapping 转换像素值并保存"""

    arr = np.array(lbl)

    converted = np.zeros_like(arr, dtype=np.uint8)
    for k, v in mapping.items():
        converted[arr == k] = v

    Image.fromarray(converted).save(dst_path)

def copy_or_convert_image(img, src_path, dst_path, pipeline=None, unet=None, use_diffusion=True):
    """拷贝或转换图像（tif→png）"""
    device = 'cuda'
    if use_diffusion:
        width, height = img.size
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(img).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            latents = pipeline.vae.encode(image).latent_dist.mean
            pipeline.scheduler.set_timesteps(1, device=device)
            linear_layer = nn.Linear(768, 1280).to(device)
            input_ids = pipeline.tokenizer(["A pathological slide"] * 1)["input_ids"]
            input_ids = torch.tensor(input_ids).to(device)
            text_embeddings = pipeline.text_encoder(input_ids)['last_hidden_state']
            text_embeddings = linear_layer(text_embeddings)
            text_embeddings = text_embeddings.clone().detach().to(device)

            for i, t in enumerate(pipeline.scheduler.timesteps):
                latents = pipeline.scheduler.scale_model_input(latents, t)
                output = unet(latents, t, text_embeddings)
                latents = pipeline.scheduler.step(output[0], t, latents).prev_sample.to(device)
                with torch.no_grad():
                    decoded_images = pipeline.decode_latents(latents.detach())
            decoded_image = pipeline.numpy_to_pil(decoded_images)[0]
        decoded_image.save(dst_path)  # 转成 PNG
    else:
        shutil.copy(src_path, dst_path)

def create_nnunet_dataset(train_images, train_labels, test_images, test_labels, num_classes, pipeline, unet):
    nnUNet_raw_path = Path(nnUNet_raw)  # 🔹 保证是 Path 对象
    assert nnUNet_raw_path.exists(), f"{nnUNet_raw} 不存在"
    use_diffusion = check_images_same_size(train_images) and check_images_same_size(test_images)
    # 找到现有 Dataset 最大序号
    existing_datasets = [d.name for d in nnUNet_raw_path.iterdir() if d.is_dir() and d.name.startswith("Dataset")]
    max_num = 0
    for ds in existing_datasets:
        try:
            num = int(ds[7:10])  # Dataset001 -> 001
            max_num = max(max_num, num)
        except:
            continue

    new_num = max_num + 1
    new_dataset_name = f"Dataset{new_num:03d}_Custom"
    new_dataset_path = nnUNet_raw_path / new_dataset_name
    new_dataset_path.mkdir(parents=True, exist_ok=False)

    # 创建子目录
    (new_dataset_path / "imagesTr").mkdir()
    (new_dataset_path / "labelsTr").mkdir()
    (new_dataset_path / "imagesTs").mkdir()
    (new_dataset_path / "labelsTs").mkdir()

    # ---- 重命名并拷贝 ----
    # 训练集
    training_pairs = []
    for idx, (img_path, lbl_path) in tqdm(enumerate(zip(train_images, train_labels)), desc="Creating Training pairs", total=len(train_images)):
        case_id = f"case_{idx:03d}"
        new_img_name = f"{case_id}_0000.png"  # 统一转成 png
        new_lbl_name = f"{case_id}.png"

        img_path = Path(img_path)
        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path).convert("L")  # 灰度图

        lbl = fix_label_size(img, lbl, img_path.name, lbl_path)

        copy_or_convert_image(img, img_path, new_dataset_path / "imagesTr" / new_img_name, pipeline, unet, use_diffusion)
        convert_and_save_label(lbl, new_dataset_path / "labelsTr" / new_lbl_name, pixel_to_label)

        training_pairs.append({
            "image": f"./imagesTr/{new_img_name}",
            "label": f"./labelsTr/{new_lbl_name}"
        })

    # 测试集
    test_list = []
    for idx, (img_path, lbl_path) in tqdm(enumerate(zip(test_images, test_labels)), desc="Creating Testing pairs", total=len(test_images)):
        case_id = f"caseTs_{idx:03d}"
        new_img_name = f"{case_id}_0000.png"
        new_lbl_name = f"{case_id}.png"

        img_path = Path(img_path)
        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path).convert("L")  # 灰度图

        lbl = fix_label_size(img, lbl, img_path.name, lbl_path)

        copy_or_convert_image(img, img_path, new_dataset_path / "imagesTs" / new_img_name, pipeline, unet, use_diffusion)
        convert_and_save_label(lbl, new_dataset_path / "labelsTs" / new_lbl_name, pixel_to_label)

        test_list.append(f"./imagesTs/{new_img_name}")

    # ---- 生成 dataset.json ----
    dataset_json = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "labels": {
            "background": 0,
            **{f"class{i}": i for i in range(1, num_classes)}
        },
        "numTraining": len(training_pairs),
        "file_ending": ".png",
    }

    with open(new_dataset_path / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"create new nnUNet Dataset：{new_dataset_name}")
    return new_num, new_dataset_name

def load_image_to_numpy(img_input):
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, Image.Image):
        img = img_input.convert("RGB")
    elif isinstance(img_input, np.ndarray):
        if img_input.ndim == 2:  # 灰度图
            img_input = np.expand_dims(img_input, -1)
        return np.transpose(img_input, (2, 0, 1)).astype(np.float32)
    else:
        raise TypeError(f"Unsupported input type: {type(img_input)}")

    img_np = np.array(img, dtype=np.float32)

    img_np = np.transpose(img_np, (2, 0, 1))

    return img_np


def prepare_image_for_predictor(arr):
    """
    Prepare a single 2D/3-channel image for nnUNet predictor.

    Parameters
    ----------
    arr : np.ndarray
        Input image array, shape (H, W), (H, W, 3) or (3, H, W)

    Returns
    -------
    np.ndarray
        Shape (1, 3, H, W) ready for predict_single_npy_array
    """
    # If channels last (H, W, 3), transpose to (3, H, W)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr.transpose(2, 0, 1)  # -> (3, H, W)

    # If 2D (H, W), expand to 3 channels
    elif arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=0)  # -> (3, H, W)

    # Ensure now shape is (3, H, W)
    if arr.shape[0] != 3:
        raise ValueError(f"Unexpected image shape after conversion: {arr.shape}")

    # Add batch dimension: (1, 3, H, W)
    arr = arr[np.newaxis, :, :, :]

    return arr


