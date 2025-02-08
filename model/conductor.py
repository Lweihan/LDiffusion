import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.atrous2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.atrous3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.atrous4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.concat_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        size = x.size()[2:]

        atrous1 = self.atrous1(x)
        atrous2 = self.atrous2(x)
        atrous3 = self.atrous3(x)
        atrous4 = self.atrous4(x)

        global_avg = self.global_avg_pool(x)
        global_avg = self.conv1x1(global_avg)
        global_avg = F.upsample(global_avg, size=size, mode='bilinear', align_corners=False)

        out = torch.cat([atrous1, atrous2, atrous3, atrous4, global_avg], dim=1)
        out = self.concat_conv(out)
        return out

class DeepLabPixelClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(DeepLabPixelClassifier, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(512, 256)  
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)  

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        return x