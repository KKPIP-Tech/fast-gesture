import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Encoder(nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()
        self.enc_blocks = nn.ModuleList(
            [ConvBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        block_outputs = []
        for block in self.enc_blocks:
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)
        return block_outputs


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=2, stride=2) for i in range(len(channels) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [ConvBlock(channels[i] + channels[i + 1], channels[i + 1]) for i in range(len(channels) - 1)]
        )

    def forward(self, x, enc_features):
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            enc_feat = self.crop(enc_features[i], x)
            x = torch.cat([x, enc_feat], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_features, x):
        _, _, H, W = x.shape
        enc_features = F.interpolate(enc_features, size=(H, W), mode='bilinear', align_corners=True)
        return enc_features


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.encoder = Encoder([n_channels, 64, 128, 256, 512])
        self.decoder = Decoder([512, 256, 128, 64, n_classes])

    def forward(self, x):
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features[-1], enc_features[:-1][::-1])
        return dec_features


class GestureNet(nn.Module):
    def __init__(self, n_channels, n_classes, img_size, gesture_types):
        super(GestureNet, self).__init__()
        self.unet = UNet(n_channels, n_classes)
        self.mlp = nn.Sequential(
            nn.Linear(img_size * img_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, gesture_types * 3)  # 每种手势类型对应3个值：ID, x, y
        )

    def forward(self, image, heatmaps):
        # U-Net 处理图像
        unet_output = self.unet(image)
        # 将 U-Net 的输出和 Heatmaps 进行拼接
        combined = torch.cat([unet_output, heatmaps], dim=1)
        combined = combined.view(combined.size(0), -1)  # 展平以供 MLP 处理
        # MLP 处理关键点信息
        keypoints = self.mlp(combined)
        return keypoints


if __name__ == "__main__":
    # 示例：创建网络实例
    net = GestureNet(n_channels=3, n_classes=21, img_size=256, gesture_types=5).to('cuda')
    summary(net, input_size=[(3, 256, 256), (21, 256, 256)])