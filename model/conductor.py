import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import contextlib
import sys
import os
from torchvision import models as tv_models, transforms
from torchvision.ops import roi_align
from PIL import Image
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from segment_anything import sam_model_registry  # 确保你已安装 segment-anything

# CBAM模块定义
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        return self.sigmoid(x_out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)
        return out

# ASPP模块定义
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList()
        for dilation in dilations:
            if dilation == 1:
                self.convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ))
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        aspp_outs = [conv(x) for conv in self.convs]
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        aspp_outs.append(global_feat)
        x = torch.cat(aspp_outs, dim=1)
        return self.project(x)

# TissueSegNet定义
class TissueSegNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 加载官方预训练ConvNeXt Tiny，weights可以选择 pretrained 或 None
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        backbone = convnext_tiny(weights=weights)
        # 去掉分类头，只保留feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # 输出通道默认384

        self.cbam = CBAM(768)          # ConvNeXt Tiny最后stage输出channels=384
        self.aspp = ASPP(768, 256)    # ASPP 输入384 输出256
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)        # (B, 384, H/32, W/32)
        feat = self.cbam(feat)         # 加入注意力
        feat = self.aspp(feat)         # ASPP多尺度特征
        out = self.decoder(feat)       # 输出类别数通道
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)  # 恢复到输入尺寸
        return {"out": out}

class CellSegClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # 延迟加载 Cellpose 模型，避免导入时执行
        self.cellpose_model = None

        # 分类模型
        backbone = tv_models.resnet152(weights=tv_models.ResNet152_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # [B, 2048, H/32, W/32]
        self.adapter = nn.Conv2d(2048, 256, kernel_size=3, padding=1)
        self.classifier = nn.Linear(256, num_classes)

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),  # 比小 patch 更适合分类
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def load_cellpose_model(self):
        if self.cellpose_model is None:
            with self.suppress_stdout_stderr():
                from cellpose import models
                self.cellpose_model = models.CellposeModel(pretrained_model='cyto2', gpu=True)

    @contextlib.contextmanager
    def suppress_stdout_stderr(self):
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    def forward(self, image_np):
        self.load_cellpose_model()
        device = next(self.parameters()).device

        with self.suppress_stdout_stderr():
            masks, _, _ = self.cellpose_model.eval(image_np, diameter=None, channels=[0, 0])

        instance_ids = np.unique(masks)
        instance_ids = instance_ids[instance_ids != 0]

        if len(instance_ids) == 0:
            return {"out": torch.zeros((1, self.num_classes, *image_np.shape[:2]), device=device, requires_grad=True)}

        patches = []
        valid_instance_ids = []
        boxes = []

        for inst_id in instance_ids:
            mask = (masks == inst_id)
            ys, xs = np.where(mask)
            y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()

            if y2 - y1 < 4 or x2 - x1 < 4:
                continue

            patch = image_np[y1:y2 + 1, x1:x2 + 1]  # 裁剪原图
            patch_pil = Image.fromarray((patch * 255).astype(np.uint8))  # 转为 PIL 图像
            patch_tensor = self.preprocess(patch_pil)  # 预处理：ToTensor → Resize → Normalize
            patches.append(patch_tensor)
            valid_instance_ids.append(inst_id)
            boxes.append((x1, y1, x2, y2))

        if not patches:
            print("未检测到有效细胞")
            return {"out": torch.zeros((1, self.num_classes, *image_np.shape[:2]), device=device, requires_grad=True)}

        batch = torch.stack(patches).to(device)  # [N, 3, H, W]

        # 特征提取并分类
        with torch.no_grad():  # 避免保存 encoder 的中间梯度
            feats = self.encoder(batch)  # [N, 2048, h, w]
            feats = self.adapter(feats)  # [N, 256, h, w]
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)  # [N, 256]

        logits = self.classifier(feats)  # [N, num_classes]
        probs = F.softmax(logits, dim=1)[:, 1:]  # 只考虑类 1 ~ num_classes-1
        _, pred_labels = torch.topk(probs, k=1, dim=1)
        pred_labels = pred_labels + 1  # shift 回 [1, num_classes-1]

        # 构造输出 mask
        final_mask = torch.zeros((1, self.num_classes, *image_np.shape[:2]), device=device, requires_grad=True)

        for i, inst_id in enumerate(valid_instance_ids):
            mask = torch.tensor(masks == inst_id, dtype=torch.float32, device=device)
            class_id = pred_labels[i].item()
            one_hot_class_id = F.one_hot(torch.tensor(class_id, device=device),
                                         num_classes=self.num_classes).float().unsqueeze(-1).unsqueeze(-1)
            final_mask = final_mask + one_hot_class_id * mask.unsqueeze(0)

        return {"out": final_mask}


