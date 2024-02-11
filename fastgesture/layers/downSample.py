import torch.nn as nn


class DownSample(nn.Module):
    def __init__(self, in_channels, kernel=3, stride=2, padding=1) -> None:
        super(DownSample, self).__init__()

        self.Down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.Down(x)