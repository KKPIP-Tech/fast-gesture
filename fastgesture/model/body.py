import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn
from torchsummary import summary

from fastgesture.layers.downSample import DownSample
from fastgesture.layers.upSample import UpSample
from fastgesture.layers.conv import (
    CommonConv, 
    ASFConv3B3, 
    ASFConv1B1, 
    ASFDownSample, 
    ASFUpSample
)
from fastgesture.layers.dsc import DepthwiseSeparableConv
from fastgesture.layers.mlp import MLP
from fastgesture.layers.detectHead import KeyPointsDH, AscriptionDH

class FastGesture(nn.Module):
    def __init__(self, keypoints_num:int=11) -> None:
        super().__init__()
        
        # UNET DownSample ---------------------------------
        self.UNETDownConv1 = CommonConv(1, 64)
        self.UNETDownSample1 = DownSample(64)
        
        self.UNETDownConv2 = CommonConv(64, 64)  # Use Res Connect
        self.UNETDownSample2 = DownSample(64)
        
        self.UNETDownConv3 = CommonConv(64, 128)
        self.UNETDownSample3 = DownSample(128)
        
        self.UNETDownConv4 = CommonConv(128, 128)  # Use Res Connect
        self.UNETDownSample4 = DownSample(128)
        
        self.UNETDownConv5 = CommonConv(128, 128)
        
        # UNET UpSample -----------------------------------
        self.UNETUpSample1 = UpSample(128)
        self.UNETUpConv1 = CommonConv(192, 128)
        
        self.UNETUpSample2 = UpSample(128)
        self.UNETUpConv2 = CommonConv(192, 64)
        
        self.UNETUpSample3 = UpSample(64)
        self.UNETUpConv3 = CommonConv(96, 64)
        
        self.UNETUpSample4 = UpSample(64)
        self.UNETUpConv4 = CommonConv(96, 64)
        
        # MLP ---------------------------------------------
        self.UNETMlp1 = MLP(128)
        self.UNETMlp2 = MLP(128)
        
        # DSC ---------------------------------------------
        self.UNETDSC = DepthwiseSeparableConv(128, 128)
        
        self.ascriptionUNETOutputDSC = DepthwiseSeparableConv(64, 64)
        self.ascriptionUNETDownSampleDSC = DepthwiseSeparableConv(64, 64)
        
        # Ascription Field --------------------------------  
        UNET_output_channel = 64      
        self.unet_to_asf = nn.Sequential(
            nn.Conv2d(UNET_output_channel, UNET_output_channel//2, kernel_size=(1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(UNET_output_channel//2, UNET_output_channel//4, kernel_size=(1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(UNET_output_channel//4, UNET_output_channel//8, kernel_size=(1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(UNET_output_channel//8, 1, kernel_size=(1, 1), padding=0),
            # nn.Sigmoid()
        )
        self.asf_down_conv = ASFDownSample(inchannels=1, out_channels=16)
        self.asf_deep_conv = ASFConv3B3(inchannels=16, out_channels=16, kernel=3, stride=1, padding=1)
        self.asf_light_conv = ASFConv1B1(inchannels=16, out_channels=16, kernel=1)
        self.asf_up_conv = ASFUpSample(inchannels=16, out_channels=8, kernel=3, stride=1, padding=0)
        
        # Detect Head -------------------------------------
        self.UNETKeypointsDH = KeyPointsDH(head_nums=keypoints_num, in_channles=64)
        self.Ascription = AscriptionDH(keypoints_number=keypoints_num)
        
    def forward(self, x):
        
        # x shape: [Batch, 1, 320, 320]
        # UNET Down Sample
        DC1 = self.UNETDownConv1(x)  # [Batch, 8, 320, 320]
        DS1 = self.UNETDownSample1(DC1)  # [Batch, 8, 160, 160]
        
        DC2 = self.UNETDownConv2(DS1)  # [Batch, 8, 160, 160]
        DS2 = self.UNETDownSample2(DC2)  # [Batch, 8, 80, 80]
        
        DC3 = self.UNETDownConv3(DS2)  # [Batch, 16, 80, 80]
        DS3 = self.UNETDownSample3(DC3)  # [Batch, 16, 40, 40]
        
        DC4 = self.UNETDownConv4(DS3)  # [Batch, 16, 40, 40]
        MLPDS4 = self.UNETMlp1(DC4)  # [Batch, 16, 40, 40]
        DS4 = self.UNETDownSample4(MLPDS4)  # [Batch, 16, 20, 20]
        
        DC5 = self.UNETDownConv5(DS4)  # [Batch, 32, 20, 20]
        
        DS5 = self.UNETDSC(DC5)  # [Batch, 32, 20, 20]
        DS5 = self.UNETDSC(DS5)  # [Batch, 32, 20, 20]
        DS5 = self.UNETDSC(DS5)  # [Batch, 32, 20, 20]
        
        MLPDS5 = self.UNETMlp2(DS5)  # [Batch, 32, 20, 20]
        
        # UNET Up Sample
        US1 = self.UNETUpSample1(MLPDS5, DC4)  # [Batch, 32, 40, 40]
        UP1 = self.UNETUpConv1(US1)  # [Batch, 16, 40, 40]
        UP1 = UP1 + MLPDS4  # [Batch, 16, 40, 40]
        
        US2 = self.UNETUpSample2(UP1, DC3)  # [Batch, 24, 80, 80]
        UP2 = self.UNETUpConv2(US2)  # [Batch, 8, 80, 80]
        
        US3 = self.UNETUpSample3(UP2, DC2)  # [Batch, 12, 160, 160]
        UP3 = self.UNETUpConv3(US3)  # [Batch, 8, 160, 160]
        UP3 = UP3 + DC2  # [Batch, 8, 160, 160]
        
        US4 = self.UNETUpSample4(UP3, DC1)  # [Batch, 12, 320, 320]
        UP4 = self.UNETUpConv4(US4)  # [Batch, 8, 320, 320]
        UNET_output = UP4 + DC1  # [Batch, 8, 320, 320]
        
        # print(f"U-NET Output Shape: {UNET_output.shape}")
        
        asf_unet = self.unet_to_asf(UNET_output)
        
        # Get Keypoints Classifications Heatmap
        heatmaps:list = self.UNETKeypointsDH(UNET_output)  # [keypoints_num, Batch, 1, 320, 320]
        
        # Get Ascription Field
        # asf_x_down = self.asf_down_conv(x)
        asf_unet_down = self.asf_down_conv(asf_unet)
        
        asf_deep_out1 = self.asf_deep_conv(asf_unet_down)
        asf_deep_out2 = self.asf_deep_conv(asf_deep_out1)
        asf_deep_out3 = self.asf_deep_conv(asf_deep_out2)
        asf_deep_out4 = self.asf_deep_conv(asf_deep_out3)
        asf_deep_out5 = self.asf_deep_conv(asf_deep_out4)
        asf_deep_out6 = self.asf_deep_conv(asf_deep_out5)
        
        asf_light_out1 = self.asf_light_conv(asf_deep_out6)
        asf_light_out2 = self.asf_light_conv(asf_light_out1)
        
        asf_up = self.asf_up_conv(asf_light_out2)

        ascription_field = self.Ascription(asf_up)
        
        heatmaps.extend(ascription_field)
        
        output = torch.stack(heatmaps, dim=0).squeeze(2)
                        
        return output  #OUTPUT(heatmaps=heatmaps, ascription_field=ascription_field)  #torch.stack([heatmaps, ascription_field],dim=0)
    

if __name__ == "__main__":
    net = FastGesture(keypoints_num=11).to('cuda')
    summary(net, input_size=(1, 160, 160), batch_size=-1, device='cuda')
    
    