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
    def __init__(self, num_heatmaps, num_gesture_types):
        super(GestureNet, self).__init__()
        self.unet = UNet(in_channels=1, out_channels=1)
        self.num_heatmaps = num_heatmaps

        self.keypoint_mlp = nn.Sequential(
            nn.Linear(5, 128),  # 5 个元素：手的类别、关键点 ID、x、y、手势的类别
            nn.ReLU(),
            nn.Linear(128, 64)  # 输出一个较小的特征向量
        )

        self.combined_mlp = nn.Sequential(
            nn.Linear(num_heatmaps * 64 + 64, 256),  # 加上一个关键点的特征向量大小
            nn.ReLU(),
            nn.Linear(256, num_gesture_types)
        )

    def forward(self, heatmaps, gesture_types):
        batch_size = heatmaps.shape[0]

        # 处理 Heatmaps
        unet_features = [self.unet(heatmaps[:, i, :, :].unsqueeze(1)).view(batch_size, -1) for i in range(self.num_heatmaps)]
        unet_features = torch.cat(unet_features, dim=1)

        # 确保 unet_features 的大小正确
        # 例如，如果 unet_features 的大小不是期望的，可以进行必要的调整
        # unet_features = unet_features[:, :期望的大小]

        # 处理 gesture_types 中的每个关键点
        if gesture_types.shape[2] != 5:
            raise ValueError("gesture_types 的每个元素应该有 5 个维度 (手的类别, 关键点 ID, x, y, 手势的类别)")
        keypoint_features = [self.keypoint_mlp(gesture_types[:, i, :]) for i in range(gesture_types.shape[1])]
        keypoint_features = torch.stack(keypoint_features, dim=1).sum(dim=1)

        # 结合 Heatmap 特征和关键点特征
        combined = torch.cat([unet_features, keypoint_features], dim=1)

        # 确保 combined 的大小与 combined_mlp 的第一个线性层的期望输入大小匹配
        # combined = combined[:, :期望的大小]

        # 通过 MLP 进行预测
        output = self.combined_mlp(combined)

        return output

if __name__ == "__main__":
    # 示例：创建网络实例
    net = GestureNet(n_channels=3, n_classes=21, img_size=256, gesture_types=5).to('cuda')
    summary(net, input_size=[(3, 256, 256), (21, 256, 256)])