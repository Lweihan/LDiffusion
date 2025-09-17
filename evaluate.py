import os
import sys
import glob
import datetime
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from LDiffusion.utils import micro_dice, mean_iou_and_per_class

def pixel_accuracy(pred, target, num_classes):
    pred_labels = torch.argmax(pred, dim=1)
    acc_list = []

    for cls in range(num_classes):
        pred_inds = (pred_labels == cls)
        target_inds = (target == cls)

        tp = (pred_inds & target_inds).sum().item()
        total = target_inds.sum().item()

        if total == 0:
            acc_list.append(1.0)  # 如果该类不存在，认为准确率=1
        else:
            acc_list.append(tp / total)

    return sum(acc_list) / len(acc_list), acc_list

def frequency_weighted_iou(pred, target, num_classes, ignore_background=False):
    pred_labels = torch.argmax(pred, dim=1)

    hist = torch.zeros((num_classes, num_classes), dtype=torch.float)
    for i in range(num_classes):
        for j in range(num_classes):
            hist[i, j] = ((target == i) & (pred_labels == j)).sum().item()

    freq = hist.sum(1) / hist.sum()
    iu = torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist) + 1e-10)

    if ignore_background:
        freq = freq[1:]
        iu = iu[1:]

    fwiou = (freq * iu).sum().item()
    return fwiou


def evaluate(image_dir, label_dir, num_classes, save_dir="./eval_results"):
    os.makedirs(save_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.png")))

    if len(image_files) != len(label_files):
        raise ValueError(f"The number of images: {len(image_files)}, The number of labels: {len(label_files)}, they must be equal.")

    all_dice, all_iou, all_pa, all_fwiou = [], [], [], []
    per_class_dice, per_class_iou, per_class_pa = [], [], []

    for img_path, lbl_path in tqdm(zip(image_files, label_files), total=len(image_files)):
        pred = np.array(Image.open(img_path))
        gt = np.array(Image.open(lbl_path))

        if pred.shape != gt.shape:
            raise ValueError(f"尺寸不一致: {img_path} vs {lbl_path}")

        pred_tensor = torch.from_numpy(pred).long().unsqueeze(0)  # (1, H, W)
        gt_tensor = torch.from_numpy(gt).long()

        pred_onehot = torch.nn.functional.one_hot(pred_tensor, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # Dice
        dice_scores, avg_dice = micro_dice(pred_onehot, gt_tensor, num_classes=num_classes)
        fg_dice_scores = dice_scores[1:]
        all_dice.append(torch.mean(fg_dice_scores).item())
        per_class_dice.append(fg_dice_scores.cpu().numpy())

        # IoU
        mean_iou, iou_dict = mean_iou_and_per_class(pred_onehot, gt_tensor, num_classes)
        iou_vals = [iou_dict[c] for c in range(1, num_classes) if iou_dict.get(c) is not None]
        all_iou.append(sum(iou_vals) / len(iou_vals) if iou_vals else 1.0)
        per_class_iou.append(
            [iou_dict.get(c, 1.0) if iou_dict.get(c) is not None else 1.0 for c in range(1, num_classes)])

        # mPA
        mean_pa, pa_list = pixel_accuracy(pred_onehot, gt_tensor, num_classes)
        fg_pa_list = pa_list[1:]
        all_pa.append(np.mean(fg_pa_list))
        per_class_pa.append(fg_pa_list)

        # FWIoU
        fwiou = frequency_weighted_iou(pred_onehot, gt_tensor, num_classes, ignore_background=True)
        all_fwiou.append(fwiou)

    mean_dice = np.mean(all_dice)
    mean_iou = np.mean(all_iou)
    mean_pa = np.mean(all_pa)
    mean_fwiou = np.mean(all_fwiou)

    per_class_dice = np.mean(per_class_dice, axis=0)
    per_class_iou = np.mean(per_class_iou, axis=0)
    per_class_pa = np.mean(per_class_pa, axis=0)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"metrics_{timestamp}.txt")

    with open(save_path, "w") as f:
        f.write("=== Segmentation Evaluation Results ===\n")
        f.write(f"Image dir: {image_dir}\n")
        f.write(f"Label dir: {label_dir}\n")
        f.write(f"Classes: {num_classes}\n\n")
        f.write(f"The number of images: {len(image_files)}\n\n")

        f.write(f"Mean Dice:  {mean_dice:.4f}\n")
        f.write(f"Mean IoU:   {mean_iou:.4f}\n")
        f.write(f"Mean PA:    {mean_pa:.4f}\n")
        f.write(f"Mean FWIoU: {mean_fwiou:.4f}\n\n")

        f.write("Per-class metrics:\n")
        for c in range(1, num_classes):
            idx = c - 1
            f.write(
                f"Class {c}: Dice={per_class_dice[idx]:.4f}, IoU={per_class_iou[idx]:.4f}, PA={per_class_pa[idx]:.4f}\n"
            )

    print(f"✅ Evaluation complete! Results saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate segmentation results.")
    parser.add_argument("--image-dir", type=str, required=True, help="predicted images folder")
    parser.add_argument("--label-dir", type=str, required=True, help="labels folder")
    parser.add_argument("--num-classes", type=int, required=True, help="num-classes")
    parser.add_argument("--save-dir", type=str, default="./LDiffusion/eval/eval_report", help="results save folder")

    args = parser.parse_args()
    evaluate(args.image_dir, args.label_dir, args.num_classes, args.save_dir)
