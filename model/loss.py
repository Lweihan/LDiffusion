import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import LMSDiscreteScheduler
from diffusers import UNet2DConditionModel

def info_nce_loss(features, labels, temperature=0.5, eps=1e-8):
    # 获取 batch_size, 类别数和图像的宽度、高度
    batch_size, num_vectors, height, width = features.shape
    _, _, h, w = labels.shape

    # 将 features 转换为形状 [1, 64*64, 10]
    features = features.view(batch_size, num_vectors, -1).permute(0, 2, 1)  # [1, 64*64, 10]

    # 将 labels 转换为形状 [1, 64*64]
    labels = labels.view(batch_size, -1)  # [1, 64*64]

    # 计算保留的样本数量（30%）
    retain_num = int(num_vectors * 0.5)
    
    # 随机选择 retain_num 个索引
    # indices = torch.randperm(num_vectors)[:retain_num]

    # features_selected = features[:, indices, :]
    # labels_selected = labels[:, indices]

    # 获取相似性矩阵
    similarity_matrix = torch.bmm(features, features.transpose(1, 2)) / temperature  # [1, 64*64, 64*64]

    # 构建标签掩码矩阵
    mask = (labels.unsqueeze(1) == labels.unsqueeze(2)).float()  # [1, 64*64, 64*64]

    # 对比损失计算
    exp_sim = torch.exp(similarity_matrix)  # [1, 64*64, 64*64]
    positive_sim = (mask * exp_sim).sum(dim=2)  # [1, 64*64]
    negative_sim = ((1 - mask) * exp_sim).sum(dim=2)  # [1, 64*64]

    # 计算对比损失
    loss = -torch.log((positive_sim + eps) / (positive_sim + negative_sim + eps)).mean()

    # loss = loss.to(device).to(torch.float16)

    # 检查是否有nan
    if torch.isnan(loss):
        print("Warning: Loss is NaN!")
    
    return loss

def save_class_distances_to_csv(features, labels, epoch, current_date, save_path):
    # 确保输入形状匹配
    batch_size, num_vectors, height, width = features.shape
    features = features.view(batch_size, num_vectors, -1).permute(0, 2, 1)  # [1, 64*64, 10]
    labels = labels.view(batch_size, -1)  # [1, 64*64]
    
    full_save_path = f"{save_path}/{current_date}"
    num_classes = labels.max().item() + 1  # 计算类别数

    os.makedirs(full_save_path, exist_ok=True)
    
    # 计算每个类别的特征中心
    class_centers = []
    for cls in range(num_classes):
        cls_mask = (labels == cls).float().unsqueeze(-1)  # [1, 64*64, 1]
        cls_features = (features * cls_mask).sum(dim=1) / (cls_mask.sum(dim=1) + 1e-8)  # [1, 10]
        class_centers.append(cls_features.squeeze(0).detach().cpu().numpy())

        # 保存当前类别的所有特征
        cls_indices = (labels[0] == cls).nonzero(as_tuple=True)[0]  # 获取属于当前类别的索引
        all_features = features[0, cls_indices].detach().cpu().numpy()  # 提取所有特征
        cls_features_df = pd.DataFrame(all_features, columns=[f"Feature_{i}" for i in range(num_vectors)])
        cls_features_file_path = f"{save_path}/{current_date}/class_{cls}_features_epoch_{epoch}.csv"
        cls_features_df.to_csv(cls_features_file_path, index=False)
    
    class_centers = np.array(class_centers)  # [num_classes, embedding_dim]
    
    # 计算类别中心之间的距离矩阵
    dist_matrix = np.linalg.norm(class_centers[:, None] - class_centers[None, :], axis=-1)  # [num_classes, num_classes]
    
    # 保存距离矩阵到 CSV
    dist_df = pd.DataFrame(dist_matrix, columns=[f"Class_{i}" for i in range(num_classes)],
                           index=[f"Class_{i}" for i in range(num_classes)])
    dist_file_path = f"{save_path}/{current_date}/distance_matrix_epoch_{epoch}.csv"
    dist_df.to_csv(dist_file_path, index=True)
    
    # 保存类别中心到 CSV
    centers_df = pd.DataFrame(class_centers, columns=[f"Feature_{i}" for i in range(class_centers.shape[1])],
                              index=[f"Class_{i}" for i in range(num_classes)])
    centers_file_path = f"{save_path}/{current_date}/class_centers_epoch_{epoch}.csv"
    centers_df.to_csv(centers_file_path, index=True)
    
    print(f"Epoch {epoch}: Distance matrix and class centers saved to {save_path}")