import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 下采样层
        self.down1 = DoubleConv(in_channels, features[0])
        self.down2 = DoubleConv(features[0], features[1])
        self.down3 = DoubleConv(features[1], features[2])
        self.down4 = DoubleConv(features[2], features[3])

        # Bottleneck
        self.bottleneck = DoubleConv(features[3], features[3] * 2)

        # 上采样层
        self.up1 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(features[3] * 2, features[3])
        self.up2 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(features[2] * 2, features[2])
        self.up3 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(features[1] * 2, features[1])
        self.up4 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(features[0] * 2, features[0])

        # 最终卷积层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # 下采样
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        # Bottleneck
        x5 = self.bottleneck(self.pool(x4))

        # 上采样
        x = self.up1(x5)
        x = self.conv_up1(torch.cat((x, x4), dim=1))

        x = self.up2(x)
        x = self.conv_up2(torch.cat((x, x3), dim=1))

        x = self.up3(x)
        x = self.conv_up3(torch.cat((x, x2), dim=1))

        x = self.up4(x)
        x = self.conv_up4(torch.cat((x, x1), dim=1))

        return self.final_conv(x)


class GestureNet(nn.Module):
    def __init__(self, num_heatmaps, num_gesture_types, image_channels=3, UNet_output_dim=512):
        super(GestureNet, self).__init__()
        self.num_heatmaps = num_heatmaps

        # U-Net 结构
        self.unet = UNet(in_channels=image_channels + num_heatmaps, out_channels=num_heatmaps)

        # U-Net 输出尺寸调整层
        self.unet_output_adjust = nn.Linear(UNet_output_dim, 256)

        # 关键点信息处理 MLP
        self.keypoint_mlp = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # 综合特征处理 MLP
        self.combined_mlp = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, num_gesture_types)
        )

    def forward(self, image, heatmaps, gesture_types):
        batch_size = image.shape[0] 

        # 将原始图像和 Heatmaps 合并并通过 U-Net 处理
        combined_input = torch.cat([image, heatmaps], dim=1)
        unet_output = self.unet(combined_input)
        unet_features = self.unet_output_adjust(unet_output.view(batch_size, -1))

        # 处理 gesture_types 中的每个关键点
        keypoint_features = [self.keypoint_mlp(gesture_types[:, i, :]) for i in range(gesture_types.shape[1])]
        keypoint_features = torch.stack(keypoint_features, dim=1).sum(dim=1)

        # 结合 U-Net 输出和 gesture_types
        combined_features = torch.cat([unet_features, keypoint_features], dim=1)

        # 通过 MLP 处理
        output = self.combined_mlp(combined_features)

        return output



if __name__ == "__main__":
    # 示例：创建网络实例
    net = GestureNet(n_channels=3, n_classes=21, img_size=256, gesture_types=5).to('cuda')
    summary(net, input_size=[(3, 256, 256), (21, 256, 256)])