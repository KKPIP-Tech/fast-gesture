import torch.nn as nn


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
