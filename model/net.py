import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super(UNetBlock, self).__init__()
        self.down = down

        # 主要卷积或转置卷积块
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            # 残差连接中的降采样
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(),
            )
            # 残差连接中的升采样
            self.residual = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels // 2, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels // 2)
            )

        # 可选的 Dropout 层
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        identity = self.residual(x)
        out = self.conv(x)
        out = out + identity  # 残差连接

        if self.use_dropout:
            out = self.dropout(out)

        return out

'''
深度可分离卷积（Depthwise Separable Convolution）是一种卷积神经网络中的操作，
它将标准卷积操作分解为两个步骤：
    1. 深度卷积（Depthwise Convolution）
    2. 逐点卷积（Pointwise Convolution）
'''
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积（Depthwise Convolution）
        self.DSC = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels),
            nn.BatchNorm2d(out_channels),  # 批次归一化
            nn.ReLU(inplace=True),  # 激活函数，用于输出
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),  # 批次归一化
            nn.ReLU(inplace=True),  # 激活函数，用于输出
        )
    
    def forward(self, x):
        x = self.DSC(x)
        return x


class UNetFinal(nn.Module):
    def __init__(self, in_channel=64, out_channel=21) -> None:
        super(UNetFinal, self).__init__()
        
        self.finalConv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.finalConv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 下采样层
        self.down1 = UNetBlock(3, 32, down=True, use_dropout=False)
        self.down2 = UNetBlock(32, 64, down=True, use_dropout=False)
        self.down3 = UNetBlock(64, 128, down=True, use_dropout=False)
        self.down4 = UNetBlock(128, 256, down=True, use_dropout=True)

        self.conv_layer = DepthwiseSeparableConv(256, 256)
        
        # 上采样层
        self.up1 = UNetBlock(256, 256, down=False, use_dropout=True)
        self.up2 = UNetBlock(256, 128, down=False, use_dropout=False)
        self.up3 = UNetBlock(128, 64, down=False, use_dropout=False)

        # 最终卷积层
        # self.final = nn.Conv2d(64, 21, kernel_size=1)
        self.final = UNetFinal(in_channel=64, out_channel=21)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        cl1 = self.conv_layer(d4)
        cl2 = self.conv_layer(cl1)
        cl3 = self.conv_layer(cl2)
        cl4 = self.conv_layer(cl3)
        cl5 = self.conv_layer(cl4)
        
        u1 = self.up1(cl5)
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
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class HandGestureNet(nn.Module):
    def __init__(self, max_hand_num, device):
        super(HandGestureNet, self).__init__()
        self.max_hand_num = max_hand_num
        self.unet = UNet()
        self.unet_output_dim = 537600  # UNet输出的扁平化维度
        self.mlp = MLPHead(self.unet_output_dim, self.max_hand_num)  # 修改 MLPHead 的输出维度
        self.kc = 21  # 假设每只手有21个关键点
        self.keypoint_heads = nn.ModuleList(
            [nn.Linear(self.max_hand_num, 2) for _ in range(self.kc)]
        )
        self.device = device
        
    def forward(self, x):
        # x shape: [batch_size, image_channel, size, size]
        tensor_19 = torch.tensor(6, device=self.device) 
        class_pred = []
        keypoints_pred = []

        for one_batch in x:
            one_batch = one_batch.unsqueeze(0)
            features = self.unet(one_batch)  # [1, keypoints_number, size/2, size/2]
            flat_features = features.view(features.size(0), -1)
            gesture_logits = self.mlp(flat_features)

            gesture_probs = F.softmax(gesture_logits, dim=1)
            gesture_values = torch.argmax(gesture_probs, dim=1)

            keypoints = [head(gesture_logits) for head in self.keypoint_heads]

            gesture_batch = []
            keypoint_batch = []
            for i in range(self.max_hand_num):
                hand_data = []
                for kp in keypoints:
                    kp_data = kp[i] if i < kp.size(0) else torch.zeros(2, device=self.device)
                    # 保持张量形式，但不使用原地操作
                    # kp_data_normalized = (kp_data + 1) / 2
                    hand_data.append(kp_data)

                gesture_value = gesture_values[i] if i < gesture_values.size(0) else tensor_19
                gesture_batch.append(gesture_value.unsqueeze(0))
                keypoint_batch.append(torch.stack(hand_data))

            class_pred.append(torch.stack(gesture_batch))
            keypoints_pred.append(torch.stack(keypoint_batch))

        # 将所有结果转换为张量
        gesture_values_tensor = torch.stack(class_pred).to(device=self.device, dtype=torch.float)
        keypoints_pred_tensor = torch.stack(keypoints_pred).to(device=self.device)

        return gesture_values_tensor, keypoints_pred_tensor


if __name__ == "__main__":
    # 实例化网络
    max_hand_num = 2  # 可以根据实际需求调整
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = HandGestureNet(max_hand_num, device=device)

    # 设置设备
    net = net.to(device)

    # 打印网络摘要
    summary(net, input_size=(3, 320, 320), batch_size=4)
