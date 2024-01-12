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
    def __init__(self,head_nums) -> None:
        super(DetectHead, self).__init__()

        self.heads = nn.ModuleList()
        
        for _ in range(head_nums):
            head = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=(1, 1), padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 8, kernel_size=(1, 1), padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=(1, 1), padding=0),
                # nn.ReLU(inplace=True),
                nn.Sigmoid()
            )
            self.heads.append(head)
    def forward(self, x):

        heatmaps = [head(x) for head in self.heads]

        return heatmaps


class MLPUNET(nn.Module):
    def __init__(self, detect_num=22):  # 只是定义网络中需要用到的方法
        super(MLPUNET, self).__init__()

        # 下采样
        self.DownConv1 = Conv(3, 32)
        self.DownSample1 = DownSample(32)
        self.DownConv2 = Conv(32, 64)
        self.DownSample2 = DownSample(64)
        self.DownConv3 = Conv(64, 128)
        self.DownSample3 = DownSample(128)
        self.DownConv4 = Conv(128, 256)
        self.DownSample4 = DownSample(256)
        self.DownConv5 = Conv(256, 512)
        # self.DownSample5 = DownSample(512)
        # self.DownConv6 = Conv(512, 1024)

        self.conv_layer = DepthwiseSeparableConv(512, 512)
        self.mlp1 = MLP(32)
        self.mlp2 = MLP(64)
        self.mlp3 = MLP(128)
        self.mlp4 = MLP(256)
        self.mlp5 = MLP(512)
        # self.mlp6 = MLP(1024)
        
        # self.UpSample2 = UpSample(1024)
        # self.UpConv1 = Conv(1024, 512)
        self.UpSample3 = UpSample(512)
        self.UpConv2 = Conv(512, 256)
        self.UpSample4 = UpSample(256)
        self.UpConv3 = Conv(256, 128)
        self.UpSample5 = UpSample(128)
        self.UpConv4 = Conv(128, 64)
        self.UpSample6 = UpSample(64)
        self.UpConv5 = Conv(64, 32)

        # 最后一层
        # self.conv = nn.Conv2d(32, 9, kernel_size=(1, 1), padding=0)
        self.head = DetectHead(head_nums=detect_num)

    def forward(self, x):
        
        # Down Sample
        R1 = self.DownConv1(x)            # [BatchSize, 32, 320, 320]
        R1 = self.mlp1(R1)
        R2 = self.DownConv2(self.DownSample1(R1))  # [BatchSize, 64, 160, 160]
        R2 = self.mlp2(R2)
        R3 = self.DownConv3(self.DownSample2(R2))  # [BatchSize, 128, 80, 80]
        R3 = self.mlp3(R3)
        R4 = self.DownConv4(self.DownSample3(R3))  # [BatchSize, 256, 40, 40]
        R4 = self.mlp4(R4)
        R5 = self.DownConv5(self.DownSample4(R4))  # [BatchSize, 512, 20, 20]
        # R5 = self.mlp5(R5)
        # R6 = self.DownConv6(self.DownSample5(R5))  # [BatchSize, 512, 10, 10]
        
        R5 = self.conv_layer(R5)
        R5 = self.conv_layer(R5)
        R5 = self.conv_layer(R5)  # [BatchSize, 512, 20, 20]
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
    def __init__(self, detect_num:int=22) -> None:
        super(FastGesture, self).__init__()
        self.unet = MLPUNET(detect_num=detect_num)
        # 分类模块
        self.classifier = nn.Sequential(
            nn.Conv2d(3 + detect_num - 1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, detect_num - 1, kernel_size=1)  # 输出每个关键点的分类
        )

    def forward(self, x):
        unet_feature = self.unet(x)  # 特征提取
        
        print(f"unet feature: {len(unet_feature)}")
        # keypoint_maps = unet_feature[:-1]  # 取出前21个热图

        # 从每个热图中提取关键点位置
        # keypoint_positions = [self.extract_keypoints(map) for map in keypoint_maps]
        
        

        # # 结合原图和关键点信息
        # combined = torch.cat([x] + keypoint_maps, dim=1)

        # # 对每个关键点进行分类
        # classified_keypoints = self.classifier(combined)
        
        # print(classified_keypoints.shape)

        # return unet_feature, classified_keypoints, keypoint_positions
        return unet_feature

    @staticmethod
    def extract_keypoints(heatmap, threshold=120):
        """
        从热图中提取关键点位置。使用非极大值抑制（NMS）来识别局部最大值。
        """
        
        print(f"heatmap.size: {heatmap.size()}")
        b, _, h, w = heatmap.size()

        # # 应用阈值
        # heatmap = heatmap * (heatmap > threshold).float()

        # # 执行NMS
        # pool = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
        # heatmap = heatmap * (heatmap == pool).float()

        # # 提取关键点位置
        # keypoints = []
        # for batch in range(b):
        #     keypoint = []
        #     for y in range(h):
        #         for x in range(w):
        #             if heatmap[batch, 0, y, x] > 0:
        #                 keypoint.append((x, y))
        #     keypoints.append(keypoint)

        # return keypoints
    
if __name__ == "__main__":
    net = FastGesture(detect_num=22).to('cpu')
    summary(net,  input_size=(3, 320, 320), batch_size=8, device='cpu')
