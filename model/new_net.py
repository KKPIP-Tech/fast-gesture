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

'''
下采样模块
'''
class DownSample(nn.Module):
    def __init__(self, in_channels) -> None:
        super(DownSample, self).__init__()

        self.Down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.Down(x)


'''
上采样模块
'''
class UpSample(nn.Module):
    def __init__(self,  in_channels) -> None:
        super(UpSample, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # self.conv = DoubleConv(in_channels, in_channels*2, channel_reduce=True)
        
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
                nn.ReLU(inplace=True),
                # nn.Sigmoid()
            )
            self.heads.append(head)
    def forward(self, x):

        heatmaps = [head(x) for head in self.heads]

        return heatmaps


class U_Net(nn.Module):
    
    def __init__(self, detect_num=21):  # 只是定义网络中需要用到的方法
        super(U_Net, self).__init__()

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
        self.DownSample5 = DownSample(512)
        self.DownConv6 = Conv(512, 1024)

        self.conv_layer = DepthwiseSeparableConv(1024, 1024)
        self.mlp1 = MLP(32)
        self.mlp2 = MLP(64)
        self.mlp3 = MLP(128)
        self.mlp4 = MLP(256)
        self.mlp5 = MLP(512)
        self.mlp6 = MLP(1024)
        
        self.UpSample2 = UpSample(1024)
        self.UpConv1 = Conv(1024, 512)
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
        # R1 = self.mlp1(R1)
        R2 = self.DownConv2(self.DownSample1(R1))  # [BatchSize, 64, 160, 160]
        # R2 = self.mlp2(R2)
        R3 = self.DownConv3(self.DownSample2(R2))  # [BatchSize, 128, 80, 80]
        # R3 = self.mlp3(R3)
        R4 = self.DownConv4(self.DownSample3(R3))  # [BatchSize, 256, 40, 40]
        # R4 = self.mlp4(R4)
        R5 = self.DownConv5(self.DownSample4(R4))  # [BatchSize, 512, 20, 20]
        # R5 = self.mlp5(R5)
        R6 = self.DownConv6(self.DownSample5(R5))  # [BatchSize, 512, 10, 10]
        
        R6 = self.conv_layer(R6)  # [BatchSize, 512, 20, 20]
        
        # 应用MLP模块
        R6 = self.mlp6(R6)

        O2 = self.UpConv1(self.UpSample2(R6, R5))  # [BatchSize, 512, 40, 40]
        O3 = self.UpConv2(self.UpSample3(O2, R4))  # [BatchSize, 256, 40, 40]
        O4 = self.UpConv3(self.UpSample4(O3, R3))  # [BatchSize, 128, 80, 80]
        O5 = self.UpConv4(self.UpSample5(O4, R2))  # [BatchSize, 64, 160, 160]
        a = self.UpConv5(self.UpSample6(O5, R1))  # [BatchSize, 32, 320, 320]
        # print(a.shape)
        
        # # 最后一层，隐射到3个特征图
        output = self.head(a)  # 输出两张 Heatmap

        return output
    
    
if __name__ == "__main__":
    net = U_Net(detect_num=21).to('cpu')
    summary(net,  input_size=(3, 320, 320), batch_size=-1)
                

