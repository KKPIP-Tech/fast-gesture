import torch
import torch.nn as nn
from torch.nn import functional as F

class UpSample(nn.Module):
    def __init__(self,  in_channels) -> None:
        super(UpSample, self).__init__()
        self.Up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=False),
        )
        
    def forward(self, x, r):

        x = self.Up(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        cat = torch.cat((x, r), 1)
        return cat