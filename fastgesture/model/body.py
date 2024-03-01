import os
import sys
import cv2
import numpy as np

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from fastgesture.layers.conv import CommonConv
from fastgesture.layers.downSample import DownSample
from fastgesture.layers.upSample import UpSample
from fastgesture.layers.dsc import DepthwiseSeparableConv
from fastgesture.layers.mlp import MLP

from fastgesture.layers.detectHead import KeyPointsDH, AscriptionDH, BboxDH


class FastGesture(nn.Module):
    def __init__(self, keypoints_num:int=11, cls_num:int=5) -> None:
        super().__init__()
        
        # UNET DownSample ---------------------------------
        self.UNETDownConv1 = CommonConv(1, 16)
        self.UNETDownSample1 = DownSample(16)
        
        self.UNETDownConv2 = CommonConv(16, 16)  # Use Res Connect
        self.UNETDownSample2 = DownSample(16)
        
        self.UNETDownConv3 = CommonConv(16, 32)
        self.UNETDownSample3 = DownSample(32)
        
        self.UNETDownConv4 = CommonConv(32, 32)  # Use Res Connect
        self.UNETDownSample4 = DownSample(32)
        
        self.UNETDownConv5 = CommonConv(32, 64)
        
        # UNET UpSample -----------------------------------
        self.UNETUpSample1 = UpSample(64)
        self.UNETUpConv1 = CommonConv(64, 32)
        
        self.UNETUpSample2 = UpSample(32)
        self.UNETUpConv2 = CommonConv(48, 16)
        
        self.UNETUpSample3 = UpSample(16)
        self.UNETUpConv3 = CommonConv(24, 16)
        
        self.UNETUpSample4 = UpSample(16)
        self.UNETUpConv4 = CommonConv(24, 16)
        
        # MLP ---------------------------------------------
        self.UNETMlp1 = MLP(32)
        self.UNETMlp2 = MLP(64)
        
        # DSC ---------------------------------------------
        self.UNETDSC = DepthwiseSeparableConv(64, 64)
        
        self.ascriptionUNETOutputDSC = DepthwiseSeparableConv(16, 16)
        self.ascriptionUNETDownSampleDSC = DepthwiseSeparableConv(16, 16)
        
        # Detect Head -------------------------------------
        self.UNETKeypointsDH = KeyPointsDH(head_nums=keypoints_num, in_channles=16)
        self.Ascription = AscriptionDH(in_channles=16)
        self.BBoxDH = BboxDH(in_channels=16, cls_num=cls_num)
        
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
        
        # Get Keypoints Classifications Heatmap
        heatmaps = self.UNETKeypointsDH(UNET_output)  # [keypoints_num, Batch, 1, 320, 320]
        
        # Get Ascription Field
        AF_UOut = self.ascriptionUNETOutputDSC(UNET_output)
        AF_UOut = self.ascriptionUNETOutputDSC(AF_UOut)
        AF_UOut = self.ascriptionUNETOutputDSC(AF_UOut)
        
        AF_UDS1 = self.ascriptionUNETDownSampleDSC(DC1)
        AF_UDS1 = self.ascriptionUNETDownSampleDSC(AF_UDS1)
        AF_UDS1 = self.ascriptionUNETDownSampleDSC(AF_UDS1)
        
        AF_Add = AF_UOut + AF_UDS1  # [Batch, 8, 320, 320]

        ascription_field = self.Ascription(AF_Add)
        
        # Get BBox Result
        # bbox_detect = self.BBoxDH(UNET_output, UP3, UP2) # x,y,w,h,conf,cls,cconf
                
        return heatmaps, ascription_field
    

if __name__ == "__main__":
    net = FastGesture(keypoints_num=11).to('cuda')
    summary(net, input_size=(1, 320, 320), batch_size=-1, device='cuda')
    
    