import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super(UNetBlock, self).__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.LeakyReLU(),
            )

        # if use_dropout:
        #     self.block = nn.Sequential(self.block, nn.Dropout(0.5))

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = UNetBlock(3, 32, down=True, use_dropout=False)
        self.down2 = UNetBlock(32, 64, down=True, use_dropout=False)
        self.down3 = UNetBlock(64, 128, down=True, use_dropout=False)
        self.down4 = UNetBlock(128, 256, down=True, use_dropout=True)
        self.up1 = UNetBlock(256, 256, down=False, use_dropout=True)
        self.up2 = UNetBlock(256, 128, down=False, use_dropout=False)
        self.up3 = UNetBlock(128, 64, down=False, use_dropout=False)
        self.final = nn.Conv2d(64, 21, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        return self.final(torch.cat([u3, d1], 1))


class MLPHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),  # 调整输入维度以匹配UNet的输出
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


class HandGestureNet(nn.Module):
    def __init__(self, max_hand_num, device):
        super(HandGestureNet, self).__init__()
        self.max_hand_num = max_hand_num
        self.unet = UNet()
        self.unet_output_dim = 537600  # UNet输出的扁平化维度
        self.mlp = MLPHead(self.unet_output_dim, max_hand_num)
        # self.gesture_heads = nn.ModuleList([nn.Linear(1, 1) for _ in range(max_hand_num)])
        # 调整关键点检测头的输入维度为整个gesture_logits的输出
        self.keypoint_heads = nn.ModuleList([nn.Linear(max_hand_num, 3) for _ in range(1 * 21)])
        self.device = device
        self.kc = 21
        
    def forward(self, x):
        # 使用 U-Net 提取特征
        features = self.unet(x)  # 假设的 U-Net 结构
        flat_features = features.view(features.size(0), -1)

        # 使用 MLP 进行手势识别
        gesture_logits = self.mlp(flat_features)
        gesture_probs = F.softmax(gesture_logits, dim=1)
        gesture_values = torch.argmax(gesture_probs, dim=1)

        # # 解决MPS不支持的操作
        # class_labels = gesture_values.view(-1, 1, 1).repeat(1, self.max_hand_num, 1).float()
        # class_labels_cpu = class_labels.to('cpu')
        # class_labels_cpu[class_labels_cpu == 0] = 19
        # class_labels = class_labels_cpu.to(class_labels.device)
        
        # 调整手势类别标签的形状以匹配数据集的输出
        class_labels = gesture_values.view(-1, 1, 1).repeat(1, self.max_hand_num, 1).float()
        class_labels[class_labels == 0] = 19  # 将空白手势类别填充为 19


        # 关键点检测
        keypoints = torch.stack([head(gesture_logits) for head in self.keypoint_heads], dim=1)
        keypoints = torch.clamp((keypoints + 1) / 2, 0, 1)  # 将其中的每一个元素控制在 [0, 1] 之间

        # 调整keypoints形状以匹配Datasets输出
        batch_size = keypoints.shape[0]
        total_kps = self.max_hand_num * self.kc * 2
        current_size = keypoints.numel()

        # 如果keypoints数量不足，则用0填充
        if current_size < total_kps * batch_size:
            padding_size = total_kps * batch_size - current_size
            padding = torch.zeros(padding_size, device=keypoints.device)
            keypoints = torch.cat([keypoints.flatten(), padding], dim=0)

        keypoints = keypoints.view(batch_size, self.max_hand_num, self.kc, 2)

        return class_labels.requires_grad_(), keypoints.requires_grad_()

    
if __name__ == "__main__":
    # 实例化网络
    max_hand_num = 5  # 可以根据实际需求调整
    net = HandGestureNet(max_hand_num)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # 打印网络摘要
    summary(net, input_size=(3, 256, 256), batch_size=8)
