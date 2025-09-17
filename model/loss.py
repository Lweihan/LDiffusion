import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import LMSDiscreteScheduler
from diffusers import UNet2DConditionModel
from torchvision.models import vgg19, VGG19_Weights


class InfoNceLoss:
    def __init__(self, temperature=0.5, num_negatives=1024, eps=1e-8):
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.eps = eps
        self.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()  # Load VGG19 for content loss
        for param in self.vgg.parameters():
            param.requires_grad = False

    def compute_content_loss(self, original_image, generated_image):
        """
        Compute content loss based on VGG19 features.

        Args:
            original_image (torch.Tensor): Original image tensor [B, C, H, W].
            generated_image (torch.Tensor): Generated image tensor [B, C, H, W].

        Returns:
            torch.Tensor: Content loss.
        """
        device = original_image.device
        self.vgg.to(device)

        # Ensure input images are resized to 224x224 for VGG19
        resize_transform = torch.nn.functional.interpolate
        original_image = resize_transform(original_image, size=(224, 224), mode='bilinear', align_corners=False)
        generated_image = resize_transform(generated_image, size=(224, 224), mode='bilinear', align_corners=False)

        original_features = self.vgg(original_image)
        generated_features = self.vgg(generated_image)

        content_loss = F.mse_loss(original_features, generated_features)
        return content_loss

    def compute_contrastive_loss(self, features, labels):
        """
        Compute contrastive loss for segmentation tasks.

        Args:
            features (torch.Tensor): [1, n, 64, 64] feature map.
            labels (torch.Tensor): [1, 1, 64, 64] label map with values 0-6.

        Returns:
            torch.Tensor: Contrastive loss.
        """
        batch_size, num_vectors, height, width = features.shape
        device = features.device

        # Flatten spatial dimensions
        features = features.view(batch_size, num_vectors, -1).permute(0, 2, 1)  # [1, 64*64, C]
        labels = labels.view(batch_size, -1)  # [1, 64*64]

        total_loss = 0.0
        valid_count = 0

        for b in range(batch_size):
            feat = features[b]  # [H*W, C]
            label = labels[b]  # [H*W]
            unique_labels = torch.unique(label)

            positive_pairs = []

            for lbl in unique_labels:
                mask = (label == lbl)
                pos_idx = torch.nonzero(mask).squeeze(-1)
                neg_idx = torch.nonzero(~mask).squeeze(-1)

                if len(pos_idx) > 1 and len(neg_idx) > self.num_negatives:
                    sampled = torch.randperm(len(pos_idx))[:max(1, int(0.01 * len(pos_idx)))]
                    for idx in sampled:
                        anchor_idx = pos_idx[idx].item()
                        positive_pool = pos_idx[pos_idx != anchor_idx]
                        if len(positive_pool) == 0:
                            continue

                        pos_sample = positive_pool[torch.randint(0, len(positive_pool), (1,))].item()
                        neg_sample = neg_idx[torch.randperm(len(neg_idx))[:self.num_negatives]].tolist()
                        positive_pairs.append((anchor_idx, pos_sample, neg_sample))

            if not positive_pairs:
                continue

            for anchor_idx, pos_idx, neg_indices in positive_pairs:
                anchor_feat = feat[anchor_idx].unsqueeze(0)  # [1, C]
                positive_feat = feat[pos_idx].unsqueeze(0)  # [1, C]
                negative_feats = feat[neg_indices]  # [N, C]

                # Similarity
                positive_sim = torch.matmul(anchor_feat, positive_feat.t()) / self.temperature  # [1,1]
                negative_sim = torch.matmul(anchor_feat, negative_feats.t()) / self.temperature  # [1,N]
                logits = torch.cat([positive_sim, negative_sim], dim=-1)  # [1, 1+N]

                target = torch.tensor([0], dtype=torch.long, device=device)
                total_loss += F.cross_entropy(logits, target)
                valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, requires_grad=True, device=device)

        return total_loss / valid_count

    def compute_loss(self, original_image, generated_image, features, labels):
        """
        Compute combined loss: content loss + contrastive loss.

        Args:
            original_image (torch.Tensor): Original image tensor [B, C, H, W].
            generated_image (torch.Tensor): Generated image tensor [B, C, H, W].
            features (torch.Tensor): [1, n, 64, 64] feature map.
            labels (torch.Tensor): [1, 1, 64, 64] label map with values 0-6.

        Returns:
            torch.Tensor: Combined loss.
        """
        content_loss = self.compute_content_loss(original_image, generated_image)
        contrastive_loss = self.compute_contrastive_loss(features, labels)
        return content_loss + contrastive_loss

class MicroDiceLoss(nn.Module):
    def __init__(self, num_classes=2, smooth=1e-5, class_weights=None):
        super(MicroDiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.class_weights = class_weights if class_weights is not None else torch.ones(num_classes)

    def forward(self, preds, targets):
        """
        preds: Tensor of shape (B, C, H, W)
        targets: Tensor of shape (B, H, W)
        """
        if isinstance(preds, dict):
            raise TypeError("Expected preds to be a Tensor, but got a dict.")

        # 如果 targets 尺寸与 preds 不一致，进行 resize
        if targets.shape[-2:] != preds.shape[-2:]:
            targets = F.interpolate(targets.unsqueeze(1).float(), size=preds.shape[2:], mode='nearest').squeeze(1).long()

        # 获取预测类别索引
        preds = torch.argmax(preds, dim=1)  # (B, H, W)

        # 展平
        targets = targets.view(-1)
        preds = preds.view(-1)

        dice_scores = torch.zeros(self.num_classes, device=targets.device)

        for class_id in range(self.num_classes):
            true_class = (targets == class_id).float()
            predicted_class = (preds == class_id).float()

            if torch.sum(true_class) == 0 and torch.sum(predicted_class) == 0:
                dice_scores[class_id] = 0
            else:
                TP = torch.sum(true_class * predicted_class)
                FP = torch.sum((1 - true_class) * predicted_class)
                FN = torch.sum(true_class * (1 - predicted_class))
                dice_scores[class_id] = 2 * TP / (2 * TP + 0.3 * FP + 0.7 * FN + self.smooth)

        weighted_dice = dice_scores * torch.tensor(self.class_weights, device=dice_scores.device)
        average_dice = torch.mean(weighted_dice)

        return 1 - average_dice

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1, ce_weight=1, num_classes=7):
        super(CombinedLoss, self).__init__()
        self.dice_loss = MicroDiceLoss(num_classes=num_classes, class_weights=[1.0, 2.0, 2.0, 1.0])
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):

        dice = self.dice_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)

        return self.dice_weight * dice + self.ce_weight * ce

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class KLDivLossMultiChannel(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(KLDivLossMultiChannel, self).__init__()
        self.reduction = reduction

    def forward(self, pred_logits, target_logits):
        pred_probs = F.log_softmax(pred_logits, dim=1)  # log p
        target_probs = F.softmax(target_logits, dim=1)  # q
        return F.kl_div(pred_probs, target_probs, reduction=self.reduction)
