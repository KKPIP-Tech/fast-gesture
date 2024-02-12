import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# from fastgesture.layers.mlp import MLP


class KeyPointsDH(nn.Module):
    def __init__(self,head_nums, in_channles=32) -> None:
        super(KeyPointsDH, self).__init__()

        # self.mlp = MLP(in_channles)
        
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
    def forward(self, x) -> list:
        # x = self.mlp(x)
        heatmaps = [head(x) for head in self.heads]

        return heatmaps


class AscriptionDH(nn.Module):
    def __init__(self, in_channles=8) -> None:
        super(AscriptionDH, self).__init__()
        
        self.heads = nn.ModuleList()
        
        for _ in range(3):
            head = nn.Sequential(
                nn.Conv2d(in_channles, in_channles//2, kernel_size=(1, 1), padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channles//2, in_channles//4, kernel_size=(1, 1), padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channles//4, 1, kernel_size=(1, 1), padding=0),
            )
            self.heads.append(head)
        
    def forward(self, x) -> list:
        fields = [head(x) for head in self.heads]

        return fields
    

class BboxDH(nn.Module):
    def __init__(self, in_channels=8, cls_num=5) -> None:
        super(BboxDH, self).__init__()
        
        self.xmatchout = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up3matchx = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        
        self.up2matchx = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.xywhc = nn.ModuleList()
        for _ in range(5):
            head = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Sigmoid()
            )
            self.xywhc.append(head)
        
        self.clsconf = nn.ModuleList()
        for _ in range(cls_num):
            head = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                
            )
            self.clsconf.append(head)
    
    def forward(self, x, up3, up2):
        
        x = self.xmatchout(x)
        
        up3 = self.up3matchx(up3)
        result = x + up3
        
        up2 = F.interpolate(up2, size=(160, 160), mode='bilinear', align_corners=False)
        up2 = self.up2matchx(up2)
        result = result + up2
        
        return_result = []
        return_result.extend([head(result) for head in self.xywhc])
        return_result.extend([head(result) for head in self.clsconf])
        
        return return_result
    

