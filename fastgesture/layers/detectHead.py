import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from fastgesture.layers.dsc import DepthwiseSeparableConv


class KeyPointsDH(nn.Module):
    def __init__(self,head_nums, in_channles=32) -> None:
        super(KeyPointsDH, self).__init__()        
        self.heads = nn.ModuleList()
        
        for _ in range(head_nums):
            head = nn.Sequential(
                nn.Conv2d(in_channles, in_channles//2, kernel_size=(1, 1), padding=0),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channles//2, in_channles//4, kernel_size=(1, 1), padding=0),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channles//4, in_channles//8, kernel_size=(1, 1), padding=0),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channles//8, 1, kernel_size=(1, 1), padding=0),

                nn.Sigmoid()
            )
            self.heads.append(head)
    def forward(self, x) -> list:
        heatmaps = [head(x) for head in self.heads]
        return heatmaps


class AscriptionDH(nn.Module):
    def __init__(self, keypoints_number=11) -> None:
        super(AscriptionDH, self).__init__()
        
        self.heads = nn.ModuleList()
        
        for _ in range(keypoints_number*2+2):
            head = nn.Sequential(
                
                nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8),
                nn.BatchNorm2d(8),  # 批次归一化
                nn.LeakyReLU(inplace=False, negative_slope=0.5),  # 激活函数/，用于输出
                nn.Conv2d(8, 8, kernel_size=1),
                nn.BatchNorm2d(8),  # 批次归一化
                nn.LeakyReLU(inplace=False, negative_slope=0.5),  # 激活函数，用于输出
                
                nn.Conv2d(8, 4, kernel_size=(1, 1), padding=0),
                nn.LeakyReLU(inplace=False, negative_slope=0.5),
                
                nn.Conv2d(4, 1, kernel_size=(1, 1), padding=0),
                nn.LeakyReLU(inplace=False, negative_slope=0.5),
            )
            self.heads.append(head)
        
    def forward(self, x) -> list:
        fields = [head(x) for head in self.heads]
        return fields


class CommonDH(nn.Module):
    def __init__(self, in_channel, keypoints_number=11) -> None:
        super(CommonDH, self).__init__()
        
        self.dsc = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=8),
            nn.BatchNorm2d(in_channel),  # 批次归一化
            nn.LeakyReLU(inplace=False, negative_slope=0.5),  # 激活函数/，用于输出
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),  # 批次归一化
            nn.LeakyReLU(inplace=False, negative_slope=0.5),  # 激活函数，用于输出
            
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=8),
            nn.BatchNorm2d(in_channel),  # 批次归一化
            nn.LeakyReLU(inplace=False, negative_slope=0.5),  # 激活函数/，用于输出
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),  # 批次归一化
            nn.LeakyReLU(inplace=False, negative_slope=0.5),  # 激活函数，用于输出
            
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=8),
            nn.BatchNorm2d(in_channel),  # 批次归一化
            nn.LeakyReLU(inplace=False, negative_slope=0.5),  # 激活函数/，用于输出
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),  # 批次归一化
            nn.LeakyReLU(inplace=False, negative_slope=0.5),  # 激活函数，用于输出
            
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=8),
            nn.BatchNorm2d(in_channel),  # 批次归一化
            nn.LeakyReLU(inplace=False, negative_slope=0.5),  # 激活函数/，用于输出
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),  # 批次归一化
            nn.LeakyReLU(inplace=False, negative_slope=0.5),  # 激活函数，用于输出
            
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=8),
            nn.BatchNorm2d(in_channel),  # 批次归一化
            nn.LeakyReLU(inplace=False, negative_slope=0.5),  # 激活函数/，用于输出
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),  # 批次归一化
            nn.LeakyReLU(inplace=False, negative_slope=0.5),  # 激活函数，用于输出
            
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=8),
            nn.BatchNorm2d(in_channel),  # 批次归一化
            nn.LeakyReLU(inplace=False, negative_slope=0.5),  # 激活函数/，用于输出
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),  # 批次归一化
            nn.LeakyReLU(inplace=False, negative_slope=0.5),  # 激活函数，用于输出
        )
        
        self.heads = nn.ModuleList()
        
        for _ in range(keypoints_number):
            head = nn.Sequential(
                
                nn.Conv2d(in_channel, in_channel//2, kernel_size=(1, 1), padding=0),
                nn.LeakyReLU(inplace=False, negative_slope=0.5),
                
                nn.Conv2d(in_channel//2, 1, kernel_size=(1, 1), padding=0),
                nn.LeakyReLU(inplace=False, negative_slope=0.5),
            )
            self.heads.append(head)
        
    def forward(self, x) -> list:
        x = self.dsc(x)
        fields = [head(x) for head in self.heads]
        return fields
    