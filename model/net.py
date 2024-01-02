import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        # 编码器
        self.enc_conv0 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # 解码器
        self.dec_upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # 最终卷积层
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码器
        enc0 = F.relu(self.enc_conv0(x))
        enc1 = F.relu(self.enc_conv1(self.pool(enc0)))
        enc2 = F.relu(self.enc_conv2(self.pool(enc1)))
        enc3 = F.relu(self.enc_conv3(self.pool(enc2)))

        # 解码器
        dec3 = F.relu(self.dec_upconv3(enc3))
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = F.relu(self.dec_conv3(dec3))

        dec2 = F.relu(self.dec_upconv2(dec3))
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = F.relu(self.dec_conv2(dec2))

        dec1 = F.relu(self.dec_upconv1(dec2))
        dec1 = torch.cat((dec1, enc0), dim=1)
        dec1 = F.relu(self.dec_conv1(dec1))

        return self.final_conv(dec1)


class GestureNet(nn.Module):
    
    def __init__(self, n_channels, n_classes, img_size, gesture_types):
        super(GestureNet, self).__init__()
        self.unet = UNet(n_channels, n_classes)

        # 计算展平后的尺寸
        unet_output_size = img_size * img_size * n_classes
        heatmaps_size = img_size * img_size * n_classes  # 假设 heatmaps 与 U-Net 输出尺寸相同
        total_feature_size = unet_output_size + heatmaps_size

        self.mlp = nn.Sequential(
            nn.Linear(total_feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, gesture_types * 3)  # 每种手势类型对应3个值（ID, x, y）
        )

    def forward(self, image, heatmaps):
        unet_output = self.unet(image)
        combined = torch.cat([unet_output, heatmaps], dim=1)
        combined = combined.view(combined.size(0), -1)  # 展平
        keypoints = self.mlp(combined)
        return keypoints


if __name__ == "__main__":
    # 示例：创建网络实例
    net = GestureNet(n_channels=3, n_classes=21, img_size=256, gesture_types=5).to('cuda')
    summary(net, input_size=[(3, 256, 256), (21, 256, 256)])