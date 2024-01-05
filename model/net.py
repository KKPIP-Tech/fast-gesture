import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.n_channels = 3
        self.n_classes = 21  # 根据实际情况调整

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = F.max_pool2d(x1, 2)
        x2 = self.down1(x2)
        x3 = F.max_pool2d(x2, 2)
        x3 = self.down2(x3)
        x4 = F.max_pool2d(x3, 2)
        x4 = self.down3(x4)
        x5 = F.max_pool2d(x4, 2)
        x5 = self.down4(x5)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.upconv1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.upconv4(x)

        logits = self.outc(x)
        return logits


class MLP(nn.Module):
    """
    多层感知器，用于类别标签的分类任务。
    """
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=21):  # 假设有21个手势类别
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: torch.Size([4, 1])
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HandGestureNetwork(nn.Module):
    def __init__(self, max_hand_num, num_keypoints=21, num_classes=20):
        super(HandGestureNetwork, self).__init__()
        self.max_hand_num = max_hand_num
        self.unet = UNet()
        self.mlps = nn.ModuleList([MLP(1, 128, num_classes) for _ in range(max_hand_num)])
        self.detector_heads = nn.ModuleList([DetectorHead(21 + num_keypoints, num_keypoints) for _ in range(max_hand_num)])  # 加入关键点信息

    def forward(self, image, keypoints, class_labels):
        feature_map = self.unet(image)
        gesture_outputs = []
        keypoint_outputs = []

        for i in range(self.max_hand_num):
            # 处理类别标签
            class_label = class_labels[:, i, :].squeeze(-1)
            class_label = class_label.view(-1, 1)  # 确保 class_label 是二维的，形状为 [batch_size, 1]
            gesture_output = self.mlps[i](class_label.float())

            # 将关键点信息与特征图结合
            combined_feature_map = torch.cat((feature_map, keypoints[:, i, :, :, :]), dim=1)

            # 使用检测头定位关键点
            keypoint_output = self.detector_heads[i](combined_feature_map)

            gesture_outputs.append(gesture_output)
            keypoint_outputs.append(keypoint_output)

        return gesture_outputs, keypoint_outputs



class DetectorHead(nn.Module):
    """检测头，用于定位关键点"""
    
    def __init__(self, in_channels, num_keypoints):
        super(DetectorHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(128, num_keypoints, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

if __name__ == "__main__":
    # 网络实例化和使用示例
    max_hand_num = 5  # 假设最大手部数量为5
    net = HandGestureNetwork(max_hand_num)

    # 示例输入
    image = torch.rand(1, 3, 256, 256)  # 假设的原图
    keypoints = torch.rand(1, max_hand_num, 21, 256, 256)  # 假设的关键点标签
    class_labels = torch.randint(0, 20, (1, max_hand_num, 1))  # 假设的类别标签

    # 前向传播
    gesture_outputs, keypoint_outputs = net(image, keypoints, class_labels)
