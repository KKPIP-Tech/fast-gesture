import cv2
import numpy as np

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
            nn.Dropout(0.1)   
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
            nn.ReLU(inplace=True),
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

        self.mlp = MLP(in_channles)
        
        self.heads = nn.ModuleList()
        
        for _ in range(head_nums):
            head = nn.Sequential(
                nn.Conv2d(in_channles, in_channles//2, kernel_size=(1, 1), padding=0),
                # nn.BatchNorm2d(in_channles//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channles//2, in_channles//4, kernel_size=(1, 1), padding=0),
                # nn.BatchNorm2d(in_channles//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channles//4, 1, kernel_size=(1, 1), padding=0),
                # nn.ReLU(inplace=True),
                nn.Sigmoid()
            )
            self.heads.append(head)
    def forward(self, x):
        x = self.mlp(x)
        heatmaps = [head(x) for head in self.heads]

        return heatmaps


class MLPUNET(nn.Module):
    def __init__(self, detect_num=22):  # 只是定义网络中需要用到的方法
        super(MLPUNET, self).__init__()

        # 修改后的下采样层
        self.DownConv1 = Conv(1, 8)
        self.DownSample1 = DownSample(8)
        self.DownConv2 = Conv(8, 8)  # 修改通道数为8
        self.DownSample2 = DownSample(8)
        self.DownConv3 = Conv(8, 16)  # 修改通道数为16
        self.DownSample3 = DownSample(16)
        self.DownConv4 = Conv(16, 16)  # 修改通道数为16
        self.DownSample4 = DownSample(16)
        self.DownConv5 = Conv(16, 32)  # 修改通道数为32

        # 中间层
        self.conv_layer = DepthwiseSeparableConv(32, 32)

        # MLP模块
        self.mlp1 = MLP(8)
        self.mlp2 = MLP(8)  # 修改MLP的通道数
        self.mlp3 = MLP(16)
        self.mlp4 = MLP(16)  # 修改MLP的通道数
        self.mlp5 = MLP(32)

        # 修改后的上采样层
        self.UpSample3 = UpSample(32)
        self.UpConv2 = Conv(32, 16)  # 融合后的通道数为80 (64 + 16)
        self.UpSample4 = UpSample(16)
        self.UpConv3 = Conv(24, 8)  # 融合后的通道数为32 (16 + 16)
        self.UpSample5 = UpSample(8)
        self.UpConv4 = Conv(12, 8)  # 融合后的通道数为16 (8 + 8)
        self.UpSample6 = UpSample(8)
        self.UpConv5 = Conv(12, 8)  # 融合后的通道数为16 (8 + 8)
        # 最后一层
        # self.conv = nn.Conv2d(32, 9, kernel_size=(1, 1), padding=0)
        self.head = DetectHead(head_nums=detect_num, in_channles=8)

    def forward(self, x):
        # print(f"Input shape {x.shape}")
        # Down Sample
        R1 = self.DownConv1(x)            # [BatchSize, 32, 320, 320]
        # print(f"R1 shape {R1.shape}")
        # R1m = self.mlp1(R1)
        R2 = self.DownConv2(self.DownSample1(R1))  # [BatchSize, 64, 160, 160]
        # print(f"R2 shape {R2.shape}")
        # R2m = self.mlp2(R2)
        R3 = self.DownConv3(self.DownSample2(R2))  # [BatchSize, 128, 80, 80]
        # print(f"R3 shape {R3.shape}")
        # R3m = self.mlp3(R3)
        R4 = self.DownConv4(self.DownSample3(R3))  # [BatchSize, 256, 40, 40]
        # print(f"R4 shape {R4.shape}")
        R4m = self.mlp4(R4)
        R5 = self.DownConv5(self.DownSample4(R4m))  # [BatchSize, 512, 20, 20]
        # print(f"R5 shape {R5.shape}")
        # R5 = self.mlp5(R5)
        # R6 = self.DownConv6(self.DownSample5(R5))  # [BatchSize, 512, 10, 10]

        
        R5 = self.conv_layer(R5)
        R5 = self.conv_layer(R5)
        R5 = self.conv_layer(R5)  # [BatchSize, 512, 20, 20]
        
        # R4 = self.mlp4(R4)
        # R5 = self.conv_layer(R5)
        # R5 = self.conv_layer(R5)
        R5 = self.mlp5(R5)

        # 应用MLP模块
        # R6 = self.mlp6(R6)

        # O2 = self.UpConv1(self.UpSample2(R6, R5))  # [BatchSize, 512, 40, 40]
        O3 = self.UpConv2(self.UpSample3(R5, R4))  # [BatchSize, 256, 40, 40]
        O3 = O3 + R4m
        # print(f"O3 shape {O3.shape}")
        O4 = self.UpConv3(self.UpSample4(O3, R3))  # [BatchSize, 128, 80, 80]
        # print(f"O4 shape {O4.shape}")
        O5 = self.UpConv4(self.UpSample5(O4, R2))  # [BatchSize, 64, 160, 160]
        
        O5 = O5 + R2
        # print(f"O5 shape {O5.shape}")
        a = self.UpConv5(self.UpSample6(O5, R1))  # [BatchSize, 32, 320, 320]
        # print(f"a shape {a.shape}")
        
        # cv2.imshow("Final a img", a[0][-1].cpu().detach().numpy().astype(np.float32))
        # cv2.waitKey()
        
        # print(a.shape)
        
        a = a + R1
        
        # # 最后一层，隐射到3个特征图
        output = self.head(a)  # 输出两张 Heatmap

        return output


"""
class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_repeats):
        super(RepBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        in_channels = out_channels

        for _ in range(num_repeats - 1):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        self.rep_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.rep_block(x)



class LabelAssignNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LabelAssignNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BBoxAssignNet(nn.Module):
    def __init__(self, in_channels):
        super(BBoxAssignNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 4)  # 4 for bounding box coordinates

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ClassNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ClassNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.classifier = nn.Conv2d(256, num_classes, 1)  # 逐像素分类

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.classifier(x)
        return x


class BoxNet(nn.Module):
    def __init__(self, in_channels, num_boxes_max, box_features):
        super(BoxNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_boxes_max * box_features)  # 每个框4个特征(cx, cy, w, h)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class ObjNet(nn.Module):
    def __init__(self, in_channels):
        super(ObjNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.obj_predictor = nn.Conv2d(256, 1, 1)  # 预测置信度

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.obj_predictor(x)
        return x



class FastGesture(nn.Module):
    def __init__(self, detect_num:int=22, heatmap_channels:int=1, num_classes:int=5) -> None:
        super(FastGesture, self).__init__()
        self.unet = MLPUNET(detect_num=detect_num)
        
        self.rep_block = RepBlock(heatmap_channels, 32, num_repeats=3)
        self.class_net = ClassNet(32, num_classes)
        N=10  # max boxes number
        self.box_net = BoxNet(32, N, 4)
        self.obj_net = ObjNet(32)
        
    def forward(self, x):
        
        heatmaps = self.unet(x)

        # 假设heatmaps的最后一个元素是最终的特征图
        last_heatmap = heatmaps[-1]  # 取最后一个heatmap，假设其形状为 [batch, channels, height, width]
        
        # print(f"last heatmaps group shape {last_heatmap.shape}")
        
        features = self.rep_block(last_heatmap)
        
        class_scores = self.class_net(features)
        bboxes = self.box_net(features)
        N=10
        bboxes = bboxes.view(-1, 10, 4)
        obj_scores = self.obj_net(features)
        
        # print(f"class scores shape: {class_scores.shape}")
        # print(f"bboxes shape: {bboxes.shape}")
        # print(f"obj shape: {obj_scores.shape}")
        # class scores shape: torch.Size([2, 5, 320, 320])
        # bboxes shape: torch.Size([2, 4, 320, 320])
        # obj shape: torch.Size([2, 1, 320, 320])
        
        return heatmaps, class_scores, bboxes, obj_scores
"""
    
if __name__ == "__main__":
    net = MLPUNET(detect_num=22).to('cuda')
    summary(net,  input_size=(1, 320, 320), batch_size=1, device='cuda')
