import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.use_res_connect = in_channels == out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MLP(nn.Module):
    def __init__(self, channel_size):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel_size, channel_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channel_size // 2, channel_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)  # reshape for linear layer
        x = self.fc(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels) -> None:
        super(DownSample, self).__init__()

        # self.Down = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, 3, 2, 1),
        #     nn.ReLU()
        # )
        
        self.Down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.Down(x)


class UpSample(nn.Module):
    def __init__(self,  in_channels) -> None:
        super(UpSample, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        # self.Up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # self.conv = DoubleConv(in_channels, in_channels*2, channel_reduce=True)
        self.Up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, r):
        # print("X Shape", x.shape)
        x = self.Up(x)

        # print("X & R Shape", x.shape, r.shape)
        # 拼接，当前上采样的，和之前下采样过程中的
        cat = torch.cat((x, r), 1)
        return cat

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


class DetectHead(nn.Module):
    def __init__(self,head_nums, in_channles=32) -> None:
        super(DetectHead, self).__init__()

        self.heads = nn.ModuleList()
        
        for _ in range(head_nums):
            head = nn.Sequential(
                nn.Conv2d(in_channles, in_channles//2, kernel_size=(1, 1), padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channles//2, in_channles//4, kernel_size=(1, 1), padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channles//4, 1, kernel_size=(1, 1), padding=0),
                # nn.ReLU(inplace=True),
                nn.Sigmoid()
            )
            self.heads.append(head)
    def forward(self, x):

        heatmaps = [head(x) for head in self.heads]

        return heatmaps


class SPPF(nn.Module):
    def __init__(self, in_channels):
        super(SPPF, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.conv = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        x = torch.cat([x, x1, x2, x3], dim=1)
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class MLPUNET(nn.Module):
    def __init__(self, detect_num=22):  # 只是定义网络中需要用到的方法
        super(MLPUNET, self).__init__()

        # 下采样
        self.DownConv1 = Conv(3, 16)
        self.DownSample1 = DownSample(16)
        self.DownConv2 = Conv(16, 32)
        self.DownSample2 = DownSample(32)
        self.DownConv3 = Conv(32, 64)
        self.DownSample3 = DownSample(64)
        self.DownConv4 = Conv(64, 128)
        self.DownSample4 = DownSample(128)
        self.DownConv5 = Conv(128, 256)
        # self.DownSample5 = DownSample(512)
        # self.DownConv6 = Conv(512, 1024)
        
        # 添加 SPPF 层
        self.sppf = SPPF(256)

        self.conv_layer = DepthwiseSeparableConv(256, 256)
        
        # 通道注意力和空间注意力
        self.ca = ChannelAttention(32)
        self.sa = SpatialAttention()
        
        self.mlp1 = MLP(16)
        self.mlp2 = MLP(32)
        self.mlp3 = MLP(64)
        self.mlp4 = MLP(128)
        self.mlp5 = MLP(256)
        # self.mlp6 = MLP(1024)
        
        # self.UpSample2 = UpSample(1024)
        # self.UpConv1 = Conv(1024, 512)
        self.UpSample3 = UpSample(256)
        self.UpConv2 = Conv(256, 128)
        self.UpSample4 = UpSample(128)
        self.UpConv3 = Conv(128, 64)
        self.UpSample5 = UpSample(64)
        self.UpConv4 = Conv(64, 32)
        self.UpSample6 = UpSample(32)
        self.UpConv5 = Conv(32, 16)

        # 最后一层
        # self.conv = nn.Conv2d(32, 9, kernel_size=(1, 1), padding=0)
        self.head = DetectHead(head_nums=detect_num, in_channles=16)

    def forward(self, x):
        
        # Down Sample
        R1 = self.DownConv1(x)            # [BatchSize, 32, 320, 320]
        R1 = self.mlp1(R1)
        R2 = self.DownConv2(self.DownSample1(R1))  # [BatchSize, 64, 160, 160]
        R2 = self.ca(R2) * R2
        R2 = self.sa(R2) * R2
        R2 = self.mlp2(R2)
        R3 = self.DownConv3(self.DownSample2(R2))  # [BatchSize, 128, 80, 80]
        R3 = self.mlp3(R3)
        R4 = self.DownConv4(self.DownSample3(R3))  # [BatchSize, 256, 40, 40]
        R4 = self.mlp4(R4)
        R5 = self.DownConv5(self.DownSample4(R4))  # [BatchSize, 512, 20, 20]
        # R5 = self.mlp5(R5)
        # R6 = self.DownConv6(self.DownSample5(R5))  # [BatchSize, 512, 10, 10]
        
        # 使用 SPPF 层
        
        R5 = self.conv_layer(R5)
        R5 = self.conv_layer(R5)
        R5 = self.conv_layer(R5)  # [BatchSize, 512, 20, 20]
        
        # 使用注意力机制
        
        # R5 = self.sppf(R5)
        R5 = self.mlp5(R5)
        
        # 应用MLP模块
        # R6 = self.mlp6(R6)

        # O2 = self.UpConv1(self.UpSample2(R6, R5))  # [BatchSize, 512, 40, 40]
        O3 = self.UpConv2(self.UpSample3(R5, R4))  # [BatchSize, 256, 40, 40]
        O4 = self.UpConv3(self.UpSample4(O3, R3))  # [BatchSize, 128, 80, 80]
        O5 = self.UpConv4(self.UpSample5(O4, R2))  # [BatchSize, 64, 160, 160]
        a = self.UpConv5(self.UpSample6(O5, R1))  # [BatchSize, 32, 320, 320]
        # print(a.shape)
        
        # # 最后一层，隐射到3个特征图
        output = self.head(a)  # 输出两张 Heatmap

        return output



class FastGesture(nn.Module):
    def __init__(self, detect_num:int=22, max_hand_num:int=2) -> None:
        super(FastGesture, self).__init__()
        self.unet = MLPUNET(detect_num=detect_num)
        self.max_hand_num = max_hand_num

        # 添加额外的处理层
        self.bbox_layer = nn.Conv2d(detect_num, 4, kernel_size=1)  # 用于提取手势的识别框
        self.classification_layer = nn.Conv2d(detect_num, 1, kernel_size=1)  # 用于手势类别识别

    def forward(self, x):
        batch_size = x.shape[0]
        heatmaps = self.unet(x)

        # 假设heatmaps的最后一个元素是最终的特征图
        last_heatmap = heatmaps[-1]  # 取最后一个heatmap，假设其形状为 [batch, channels, height, width]

        # 提取手势识别框
        bboxes = self.bbox_layer(last_heatmap)  # [batch, 4, height, width]
        bboxes = bboxes.permute(0, 2, 3, 1)  # [batch, height, width, 4]
        bboxes = bboxes.contiguous().view(batch_size, -1, 4)  # [batch, height*width, 4]

        # 提取手势类别
        classifications = self.classification_layer(last_heatmap)  # [batch, 1, height, width]
        classifications = classifications.squeeze(1)  # [batch, height, width]
        classifications = classifications.view(batch_size, -1)  # [batch, height*width]

        # 关键点提取
        keypoints = self.extract_keypoints(heatmaps, batch_size)

        # 构建最终的输出
        final_output = {
            "bboxes": bboxes,
            "classifications": classifications,
            "keypoints": keypoints
        }
        return heatmaps, final_output
    
    def extract_keypoints(self, heatmaps, batch_size):
        """
        提取关键点的位置。
        :param heatmaps: 输入的特征图 [detect_num, batch, height, width]
        :param batch_size: 批次大小
        :return: 关键点的坐标列表
        """
        keypoints = []
        for heatmap in heatmaps:
            # 应用3x3最大池化来找到关键点，保留边缘
            pooled_heatmaps = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
            # 找到最大值点，即关键点位置
            keypoints_mask = (heatmap == pooled_heatmaps)

            # 对于每个batch中的每张heatmap，提取关键点位置
            for b in range(batch_size):
                keypoint_coords = torch.nonzero(keypoints_mask[b], as_tuple=False)
                # 将关键点坐标归一化
                normalized_keypoints = keypoint_coords.float() / torch.tensor([heatmap.shape[2], heatmap.shape[3]])
                keypoints.append(normalized_keypoints)

        # 将所有batch的关键点坐标汇总到一个列表中
        keypoints = torch.cat(keypoints, dim=0)
        return keypoints

    
if __name__ == "__main__":
    net = MLPUNET(detect_num=22).to('cpu')
    summary(net,  input_size=(3, 320, 320), batch_size=1, device='cpu')
