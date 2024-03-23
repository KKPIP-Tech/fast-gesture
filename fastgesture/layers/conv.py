import torch.nn as nn


class CommonConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, drop=0.1):
        super(CommonConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)   
        )
        self.use_res_connect = in_channels == out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class ASFDownSample(nn.Module):
    def __init__(self, inchannels, out_channels, kernel=3, stride=1, padding=0, groups=1, drop=0.1) -> None:
        super(ASFDownSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        r = self.conv(x)
        return r

class ASFUpSample(nn.Module):
    def __init__(self, inchannels, out_channels, kernel=3, stride=1, padding=0, groups=1, drop=0.1) -> None:
        super(ASFUpSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=out_channels, out_channels=inchannels, kernel_size=kernel, stride=stride, padding=2, groups=groups),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=inchannels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=2, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
    
    def forward(self, x):
        r = self.conv(x)
        return r

class ASFConv3B3(nn.Module):
    def __init__(self, inchannels, out_channels, kernel=3, stride=1, padding=0, groups=1) -> None:
        super(ASFConv3B3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Tanh(),
        )
    
    def forward(self, x):
        c1 = self.conv1(x)
        # print(f"c1 shape: {c1.shape}")
        c2 = self.conv2(c1)
        # print(f"c2 shape: {c2.shape}")
        c3 = self.conv3(c2)
        # print(f"c3 shape: {c3.shape}")
        
        r = c1 + c2 + c3
        return r

class ASFConv1B1(nn.Module):
    def __init__(self, inchannels, out_channels, kernel=1, stride=1, padding=0, groups=1) -> None:
        super(ASFConv1B1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Tanh(),
        )
    
    def forward(self, x):
        r = self.conv1(x)
        return r
        
        
    