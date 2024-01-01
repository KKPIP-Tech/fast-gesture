import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    

class HeatmapCNN(nn.Module):
    """用于处理热力图的卷积神经网络"""
    def __init__(self, n_keypoints, n_features):
        super().__init__()
        self.conv1 = nn.Conv2d(n_keypoints, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 假设经过卷积操作后，特征图的尺寸减半
        self.fc = nn.Linear(128 * 28 * 28, n_features)  # 调整特征图大小

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class GestureNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_keypoints, img_size, gesture_types):
        super(GestureNet, self).__init__()
        self.unet = UNet(n_channels, n_classes)
        self.heatmap_cnn = HeatmapCNN(n_keypoints, img_size * img_size)
        # 假设全连接层的输入尺寸需要动态计算
        # 这里先初始化为 None，实际尺寸将在第一次前向传播时计算
        self.fc1 = None
        self.fc2 = nn.Linear(512, gesture_types * 5)  # 每个手势类型的 x_min, y_min, x_max, y_max, hand_label
        self.gesture_types = gesture_types

    def forward(self, image, heatmaps):
        img_features = self.unet(image)  # 假设UNet输出尺寸是 [batch_size, 64, 56, 56]

        if self.fc1 is None:
            _, c, h, w = img_features.size()
            img_features_size = c * h * w
            heatmap_features_size = 128 * 28 * 28  # 调整HeatmapCNN输出尺寸
            total_features_size = img_features_size + heatmap_features_size
            self.fc1 = nn.Linear(total_features_size, 512).to(img_features.device)

        img_features = img_features.view(img_features.size(0), -1)
        heatmap_features = self.heatmap_cnn(heatmaps)  # 确保heatmap_features的尺寸正确
        combined_features = torch.cat([img_features, heatmap_features], dim=1)

        
        gesture_output = self.fc2(combined_features)
        gesture_output = gesture_output.view(-1, self.gesture_types, 5)
        return gesture_output




if __name__ == "__main__":
    from torchsummary import summary

    # 检测是否有可用的GPU，如果有，使用GPU；否则使用CPU
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    # 假设的输入维度
    n_channels = 3  # 三通道BGR图像
    n_classes = 5   # 当前有5个物体类别
    n_keypoints = 21 # 21个关键点类别
    keypoint_dim = 2 # 假设每个关键点的维度为2（x, y坐标）

    # 创建网络实例，并将其移至正确的设备
    model = GestureNet(n_channels, n_classes, n_keypoints, keypoint_dim).to(device)

    # 假设的输入大小
    image_size = (n_channels, 224, 224)  # 例如，224x224的图像
    keypoints_size = (n_keypoints, keypoint_dim)  # 例如，21个关键点，每个有x, y坐标

    # 使用torchsummary打印网络结构和参数
    # 注意：torchsummary 需要模型在CPU上才能正确显示
    model.to('cpu')
    summary(model, input_size=[image_size, keypoints_size])
    model.to(device)  # 如果需要继续使用GPU，将模型再次移回GPU

