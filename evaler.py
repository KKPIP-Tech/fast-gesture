import os
import sys
import yaml 
import json
from typing import List, TypedDict
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())

import cv2
import numpy as np
from copy import deepcopy
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fastgesture.model.body import FastGesture
from torch.utils.tensorboard import SummaryWriter

class Evaler:
    def __init__(self, model:FastGesture, 
                 dataloader:DataLoader, 
                 device:str, 
                 tensorboard_writer:SummaryWriter) -> None:
        
        self.model:FastGesture = model
        self.device:str = device
        self.tensorboard_writer:SummaryWriter = tensorboard_writer
        
        # setting tqdm
        self.dataloader = dataloader
        
        # Loss Setting ------------------------------------
        self.criterion_heatmap = nn.MSELoss().to(device=device)
        self.criterion_x_ascription = nn.MSELoss(reduction='mean').to(device=device)
        self.criterion_y_ascription = nn.MSELoss(reduction='mean').to(device=device)
        self.criterion_x_minus = nn.MSELoss(reduction='mean').to(device=device)
        self.criterion_y_minus = nn.MSELoss(reduction='mean').to(device=device)
    
    def _eval(self, epoch:int) -> None:
        
        self.model.eval()
        
        total_loss = 0.0
        avg_loss = 0.0
        
        self.pbar = tqdm(self.dataloader, desc=f"[Eval] -> ")
        
        for index, datapack in enumerate(self.pbar):
            _, tensor_letterbox_img, tensor_kp_cls_labels, tensor_ascription_field, _ = datapack
            
            # translate tensor to device
            tensor_letterbox_img = tensor_letterbox_img.to(self.device)  # [Batch, 1, 320, 320]
            tensor_kp_cls_labels = tensor_kp_cls_labels.permute(1, 0, 2, 3).to(self.device)  # [11, Batch, 320, 320]
            tensor_ascription_field = tensor_ascription_field.permute(1, 0, 2, 3).to(self.device)  # [3, Batch, 320, 320]
            
            forward = self.model(tensor_letterbox_img)
            
            forward_heatmaps = forward.heatmaps
            forward_asf = forward.ascription_field
            
            keypoints_classes_num:int = tensor_kp_cls_labels.shape[0]
            
            # Keypoints Heatmaps Loss ---------------------
            heatmap_loss = 0.0
            for heatmap_class_index in range(keypoints_classes_num):
                heatmap_loss += self.criterion_heatmap(
                    forward_heatmaps[heatmap_class_index],
                    tensor_kp_cls_labels[heatmap_class_index]
                )
            
            # Keypoints Vector X Loss ---------------------
            asf_x_loss = 0.0
            for asf_index in range(0, keypoints_classes_num, 1):
                asf_x_loss += self.criterion_x_ascription(
                    forward_asf[asf_index],
                    tensor_ascription_field[asf_index]
                )
            
            # Keypoints Vector Y Loss ---------------------
            asf_y_loss = 0.0
            for asf_index in range(keypoints_classes_num, keypoints_classes_num*2, 1):
                asf_y_loss += self.criterion_y_ascription(
                    forward_asf[asf_index],
                    tensor_ascription_field[asf_index]
                )
            
            # Keypoints Minus X Loss ----------------------
            x_minus_loss = self.criterion_x_minus(forward_asf[-2], tensor_ascription_field[-2])
            
            # Keypoints Minus Y Loss ----------------------
            y_minus_loss = self.criterion_y_minus(forward_asf[-1], tensor_ascription_field[-1])
            
            loss = heatmap_loss*1.5 + asf_x_loss + asf_y_loss + x_minus_loss + y_minus_loss
            
            total_loss += loss.item()
            
            avg_loss = total_loss / (index + 1)
            
            self.pbar.set_description(
                f"[Epoch {epoch} Eval | loss: {avg_loss:.10f}] -> "
            )
        
        self.tensorboard_writer.add_scalar("val average loss", avg_loss, epoch)
        print()
                