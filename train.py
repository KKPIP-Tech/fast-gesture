
import os
import sys
import argparse
from typing import Union
from copy import deepcopy
from pathlib import Path
import torch.utils
import torch.utils.data

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from thop import profile

from evaler import Evaler
from fastgesture.model.body import FastGesture
from fastgesture.data.datasets import TrainDatasets
from fastgesture.data.val_datasets import ValDatasets
from fastgesture.data.point_average_value import PointsNC, GetPNCS
from fastgesture.utils.checkpoint import ckpt_load, ckpt_save, create_path
from fastgesture.utils.common import (
    select_device, 
    select_optim,
    get_core_num
)

class Train:
    def __init__(self, option, save_path:str, checkpoints=None) -> None:
        
        # get datasets profile ----------------------------
        self._save_path:str = save_path
        print(f"Save Path: {self._save_path}")
        
        self._train_config_file:str = option.data
        self._val_config_file:str = option.val
        print(f"Datasets | Train Set Config File: {self._train_config_file}")
        print(f"Datasets | Val Set Config File: {self._val_config_file}")
        
        self._img_size:Union[int, list] = option.img_size
        print(f"Image Size: {self._img_size}")
        
        self._max_epoch:int = option.epochs
        self._start_epoch:int = 0
        self._view:bool = not option.view  # Demonstrate the training process
        
        # load datasets -----------------------------------
        self._dataloader_workers:int = option.workers if option.workers < get_core_num()[1] else get_core_num()[1]
        self._train_set_batch_size:int = option.batch
        self._val_set_batch_size:int = option.val_batch
        
        self._train_dataloader:DataLoader = self._load_train_datasets(img_size=self._img_size, batch_size=self._train_set_batch_size, 
                                                                      workers=self._dataloader_workers)
        self._val_dataloader:DataLoader = self._load_val_datasets(img_size=self._img_size, batch_size=self._val_set_batch_size, 
                                                                  workers=self._dataloader_workers)
        
        # select device -----------------------------------
        self._device:str = select_device(opt=option)
        print(f"Device: {self._device}")
        
        # init TensorBoard Recoder ------------------------
        self._writer:SummaryWriter = SummaryWriter(log_dir=self._save_path)
    
        # init model --------------------------------------
        self._model:FastGesture = FastGesture(keypoints_num=self._keypoints_classes_num).to(device=self._device)
        self._print_model_info(img_size=self._img_size)
               
        # set loss ----------------------------------------
        self._criterion_heatmap = nn.MSELoss().to(device=self._device)
        self._criterion_x_ascription = nn.MSELoss(reduction='mean').to(device=self._device)
        self._criterion_y_ascription = nn.MSELoss(reduction='mean').to(device=self._device)
        self._criterion_x_minus = nn.MSELoss(reduction='mean').to(device=self._device)
        self._criterion_y_minus = nn.MSELoss(reduction='mean').to(device=self._device)
        
        # set optimizer -----------------------------------
        user_optimizer_setting:str = option.optimizer
        self._optimizer = select_optim(net=self._model, opt=option, user_set_optim=user_optimizer_setting)
        
        self._scaler = GradScaler(enabled=True)
        
        # process resume process --------------------------
        if checkpoints is not None:
            resume_model, resume_optim, resume_scaler, resume_start_epoch = ckpt_load(checkpoints)
            self._model.load_state_dict(resume_model.state_dict())
            self._start_epoch = resume_start_epoch
            self._optimizer.load_state_dict(resume_optim)
            self._scaler.load_state_dict(resume_scaler)
            self._scheduler = optim.lr_scheduler.StepLR(
                self._optimizer, 
                step_size=self._hyper_lr_step_size, gamma=self._hyper_lr_gamma, 
                last_epoch=self._start_epoch-1
            )
        else:
            self._scheduler = optim.lr_scheduler.StepLR(
                self._optimizer, 
                step_size=self._hyper_lr_step_size, gamma=self._hyper_lr_gamma
            )
        
    def train(self) -> None:
        for epoch in range(self._start_epoch, self._max_epoch):
            self._model.train()
            pbar:tqdm = tqdm(self._train_dataloader, desc=f"[Epoch {epoch}] -> ")
            self._current_lr:float = self._scheduler.get_last_lr()[0]
            self._total_loss = 0.0
            avg_loss = 0.0
            
            for index, datapack in enumerate(pbar):
                # load train data from datapack
                (letterbox_image, tensor_letterbox_img, 
                 tensor_kp_cls_labels, tensor_ascription_field, _) = datapack
                if self._view: cv2.imshow("Letterbox Image", letterbox_image[0].cpu().detach().squeeze(0).numpy().astype(np.uint8))
                
                # transfer the data to the target device
                tensor_letterbox_img = tensor_letterbox_img.to(self._device)  # [Batch, 1, img_size, img_size]
                tensor_kp_cls_labels = tensor_kp_cls_labels.permute(1, 0, 2, 3).to(self._device)  # [kp_cls_num, Batch, img_size, img_size]
                tensor_ascription_field = tensor_ascription_field.permute(1, 0, 2, 3).to(self._device)  # [kp_cls_num*2+2, Batch, img_size, img_size]
                
                self._optimizer.zero_grad()
                
                with autocast(enabled=True, dtype=torch.bfloat16):
                    forward = self._model(tensor_letterbox_img)
                    forward_heatmaps, forward_asf_maps = forward[:self._keypoints_classes_num], forward[self._keypoints_classes_num:]
                    
                    # compute loss
                    loss = self._loss_compute(
                        forward_heatmaps=forward_heatmaps,
                        forward_asf_maps=forward_asf_maps,
                        label_heatmaps=tensor_kp_cls_labels,
                        label_asf_maps=tensor_ascription_field,
                    )
                
                self._scaler.scale(loss).backward()
                
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._total_loss += loss.item()
                avg_loss = self._total_loss / (index+1)
                
                if self._device == "cuda":
                    gpu_memory_bytes = torch.cuda.memory_reserved(self._device)
                    gpu_memory_bg = round(gpu_memory_bytes / 1024 / 1024 / 1024, 2)
                    
                    pbar.set_description(
                        f"[Epoch {epoch}, GPU {gpu_memory_bg} G, cur_lr {self._current_lr:.6f}, avg_l {avg_loss:.4f}] -> "
                    )
                else:
                    pbar.set_description(f"[Epoch {epoch}, cur_lr {self._current_lr:.6f}, avg_l {avg_loss:.4f}] -> ")
                
                # write train log into txt
                with open(f"{self._save_path}/log.txt", "a") as f:
                    f.write(f"Epoch: {epoch}, cur_lr: {self._current_lr:.10f}, Batch: {index}, AVG Loss: {avg_loss}, Total Loss: {self._total_loss}\n")

            self._scheduler.step()
            
            # save checkpoints
            ckpt_save(
                model=self._model, optim=self._optimizer, scaler=self._scaler, epoch=epoch, pncs_result=self._train_pncs_result, save_pth=self._save_path, file_name=f"epoch_{str(epoch)}",
            )
            ckpt_save(
                model=self._model, optim=self._optimizer, scaler=self._scaler, epoch=epoch, pncs_result=self._train_pncs_result, save_pth=self._save_path, file_name=f"epoch_{str(epoch)}", last=True
            )
            
            # init Evaler -------------------------------------
            self._evaler:Evaler = Evaler(model=deepcopy(self._model), dataloader=self._val_dataloader, device=self._device, tensorboard_writer=self._writer)
            # verify the training effect of the model
            self._evaler._eval(epoch=epoch)
            
            # use tensorboard to record loss after every epoch
            self._writer.add_scalar("average loss", avg_loss, epoch)
            self._writer.add_scalar("total loss", self._total_loss, epoch)
            self._writer.add_scalar("learning rate", self._current_lr, epoch)
            
        # finish all epoch
        self._writer.close()
                
    def _loss_compute(self, forward_heatmaps, label_heatmaps, forward_asf_maps, label_asf_maps) -> float:
        # compute heatmap loss ----------------------------
        keypoints_loss = 0.0
        for keypoints_index in range(self._keypoints_classes_num):
            if self._view:
                image_to_show = forward_heatmaps.permute(1, 0, 2, 3)[0][keypoints_index].to(torch.float32).cpu().detach().numpy().astype(np.float32)
                cv2.imshow(f"Forward Heatmap No.{keypoints_index}", image_to_show)
            keypoints_loss += self._criterion_heatmap(forward_heatmaps[keypoints_index], label_heatmaps[keypoints_index])
        
        # compute asf vx loss -----------------------------
        asf_x_loss = 0.0
        for asf_index in range(0, self._keypoints_classes_num, 1):
            asf_x_loss += self._criterion_x_ascription(
                forward_asf_maps[asf_index], label_asf_maps[asf_index]
            )
        if self._view:
            image_to_show = label_asf_maps.permute(1, 0, 2, 3)[0][0].to(torch.float32).cpu().detach().numpy().astype(np.float32)
            cv2.imshow(f"asf_label x", image_to_show)

            image_to_show = forward_asf_maps.permute(1, 0, 2, 3)[0][0].to(torch.float32).cpu().detach().numpy().astype(np.float32)  
            cv2.imshow(f"forward asf x", image_to_show)
            
        # compute asf vy loss -----------------------------  
        asf_y_loss = 0.0
        for asf_index in range(self._keypoints_classes_num, self._keypoints_classes_num*2, 1):
            asf_y_loss += self._criterion_y_ascription(
                forward_asf_maps[asf_index], label_asf_maps[asf_index]
            )
        if self._view:
            image_to_show = label_asf_maps.permute(1, 0, 2, 3)[0][self._keypoints_classes_num].to(torch.float32).cpu().detach().numpy().astype(np.float32) 
            cv2.imshow(f"asf_label y", image_to_show)

            image_to_show = forward_asf_maps.permute(1, 0, 2, 3)[0][self._keypoints_classes_num].to(torch.float32).cpu().detach().numpy().astype(np.float32)  
            cv2.imshow(f"forward asf y", image_to_show)
        
        # comput x minus loss -----------------------------
        x_minus_loss = self._criterion_x_minus(forward_asf_maps[-2], label_asf_maps[-2])
        if self._view:
            image_to_show = label_asf_maps.permute(1, 0, 2, 3)[0][-2].to(torch.float32).cpu().detach().numpy().astype(np.float32) 
            cv2.imshow(f"asf_x_minus_label", image_to_show)
            image_to_show = forward_asf_maps.permute(1, 0, 2, 3)[0][-2].to(torch.float32).cpu().detach().numpy().astype(np.float32) 
            cv2.imshow(f"forward asf_x_minus", image_to_show)
        
        # comput y minus loss -----------------------------
        y_minus_loss = self._criterion_y_minus(forward_asf_maps[-1], label_asf_maps[-1])
        if self._view:
            image_to_show = label_asf_maps.permute(1, 0, 2, 3)[0][-1].to(torch.float32).cpu().detach().numpy().astype(np.float32) 
            cv2.imshow(f"asf_y_minus_label", image_to_show)
            image_to_show = forward_asf_maps.permute(1, 0, 2, 3)[0][-1].to(torch.float32).cpu().detach().numpy().astype(np.float32) 
            cv2.imshow(f"forward asf_y_minus", image_to_show)
            cv2.waitKey(1)
        
        # use loss weight from hyper params
        loss = keypoints_loss*self._hyper_loss_weight[0] + asf_x_loss*self._hyper_loss_weight[1] + \
            asf_y_loss*self._hyper_loss_weight[2] + x_minus_loss*self._hyper_loss_weight[3] + y_minus_loss*self._hyper_loss_weight[4]
        # loss = keypoints_loss*1.5 + asf_x_loss*0.2 + asf_y_loss*0.2 + x_minus_loss*0.3 + y_minus_loss*0.3
        return loss

    def _load_train_datasets(self, img_size:Union[int, list], batch_size:int, workers:int) -> DataLoader:

        # get datasets pncs
        pncs_getter:GetPNCS = GetPNCS(config_file=self._train_config_file, img_size=img_size, save_path=self._save_path)
        self._train_pncs_result:PointsNC = pncs_getter.get_pncs()
        
        datasets:torch.utils.data.Dataset = TrainDatasets(config_file=self._train_config_file, img_size=img_size, 
                                     pncs=deepcopy(self._train_pncs_result), limit=None)
        
        # get hyper params
        hyper_params = datasets.get_hyper_params()
        self._hyper_loss_weight:list = hyper_params[0]
        self._hyper_lr_step_size:int = hyper_params[1]
        self._hyper_lr_gamma:float = hyper_params[2]
        
        self._keypoints_classes_num:int = datasets.get_keypoints_class_number()        
        dataloader:DataLoader = DataLoader(dataset=datasets, batch_size=batch_size,
                                           num_workers=workers, shuffle=True)
        return dataloader
    
    def _load_val_datasets(self, img_size:Union[int, list], batch_size:int, workers:int) -> DataLoader:

        # get datasets pncs
        pncs_getter:GetPNCS = GetPNCS(config_file=self._val_config_file, img_size=img_size, save_path=self._save_path)
        pncs_result:PointsNC = pncs_getter.get_pncs()
        
        datasets:torch.utils.data.Dataset = ValDatasets(config_file=self._val_config_file, img_size=img_size, 
                                     pncs=deepcopy(pncs_result), limit=None)
        
        dataloader:DataLoader = DataLoader(dataset=datasets, batch_size=batch_size,
                                           num_workers=workers, shuffle=True)
        return dataloader
    
    def _print_model_info(self, img_size:Union[int, list]) -> None:
        input_size = torch.randn(1, 1, img_size, img_size).to(self._device)
        model_flops, model_params = profile(self._model, inputs=(input_size, ))
        summary(self._model, input_size=(1, img_size, img_size), batch_size=-1, device=self._device)
        print(f"Model FLOPs: {model_flops/1000**3:.2f} G, Params: {model_params/1000**2:.2f} M \n")

def run(option) -> Train:
    if option.resume:
        checkpoints = option.resume if isinstance(option.resume, str) else None
        if checkpoints is None: raise ValueError("Resume Path cannot be empty")
        resume_path:str = str(Path(checkpoints).parent.parent)
        if not os.path.exists(resume_path): raise ValueError("Resume Path Not Exists")
        print(f"The model will continue to be trained using the checkpoint: {option.resume}")
        checkpoints = torch.load(option.resume)
        return Train(option=option, save_path=resume_path, checkpoints=checkpoints)
    else:
        temp_full_path = option.save_path + option.save_name
        save_path = create_path(path=temp_full_path)
        return Train(option=option, save_path=save_path, checkpoints=None)
        
if __name__ == "__main__":
    
    parse = argparse.ArgumentParser()
    parse.add_argument('--data', type=str,default='./data/config.yaml', help='training config path')
    parse.add_argument('--val', type=str,default='./data/config.yaml', help='val set config path')
    parse.add_argument('--epochs', type=int, default=2000, help='max train epoch')
    parse.add_argument('--batch', type=int, default=1, help='batch size')
    parse.add_argument('--val_batch', type=int, default=2, help='batch size')
    parse.add_argument('--img_size', type=int, default=160, help='trian img size')
    parse.add_argument('--device', type=str, default='cuda', help='cuda or cpu or mps')
    parse.add_argument('--save_period', type=int, default=4, help='save per n epoch')
    parse.add_argument('--workers', type=int, default=28, help='thread num to load data')
    parse.add_argument('--shuffle', action='store_false', help='chose to unable shuffle in Dataloader')
    parse.add_argument('--save_path', type=str, default='./run/train/')
    parse.add_argument('--save_name', type=str, default='new_datasets')
    parse.add_argument('--lr', type=float, default=0.001)
    parse.add_argument('--optimizer', type=str, default='Adam', help='only support: [Adam, AdamW, SGD, ASGD]')
    parse.add_argument('--resume', nargs='?', const=True, default=False, help="Choice one path to resume training")
    parse.add_argument('--view', action='store_false', help='chose to able training process')
    parse = parse.parse_args()
    
    trainer:Train = run(option=parse)
    trainer.train()
        
