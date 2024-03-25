import os
import sys
import argparse
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt

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
from torch.utils.tensorboard import SummaryWriter

from fastgesture.model.body import FastGesture
from fastgesture.data.datasets import Datasets
from fastgesture.data.generate import inverse_vxvyd
from fastgesture.model.losses.loss import FocalLoss, CombinedLoss
from fastgesture.utils.checkpoint import ckpt_load, ckpt_save, create_path
from fastgesture.utils.common import (
    select_device, 
    select_optim,
    get_core_num
)


def check_gradient(model):
    """
    检查模型中所有参数的梯度，以识别梯度爆炸或消失。
    如果梯度的绝对值超过某个阈值（这里假定为1e5），则认为是梯度爆炸。
    """
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grad_max = parameter.grad.abs().max()
            if grad_max > 1e5:
                print(f"警告: 梯度爆炸检测在参数 {name} 上，梯度最大值为 {grad_max}")
            elif torch.isnan(grad_max) or torch.isinf(grad_max):
                print(f"警告: 梯度异常（NaN或Inf）在参数 {name} 上，梯度最大值为 {grad_max}")


def draw_arrows_on_image(img):
    # 创建一个与原图像同尺寸的空白图像
    blank_image = np.zeros((320, 320, 3), np.uint8)
    
    # 遍历图像中的每一个像素
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # 提取x向量、y向量和长度dis
            vx = img[y, x, 0]
            vy = img[y, x, 1]
            dis = img[y, x, 2]
            
            if vx == 0 and vy == 0 and dis == 0:
                continue
            
            cx, cy = inverse_vxvyd(point_a=(x/320, y/320), x_vector=vx, y_vector=vy, distance=dis)
            
            cx = int(cx * 320)
            cy = int(cy * 320)
            
            # 在空白图像上绘制箭头
            cv2.arrowedLine(blank_image, (x, y), (cx, cy), (0, 255, 0), 1, tipLength=0.3)
    cv2.imshow('ASC Field', blank_image)
    cv2.waitKey(1)
    

def train(opt, save_path, resume=None):
    
    config_file = opt.data
    print(f"Save Path: {save_path}")
    print(f"Datasets Config File: {config_file}")
    
    device = select_device(opt=opt)
    print(f"Train Device: {device}")
    
    max_epoch = opt.epochs
    start_epoch = 0
    
    # set datasets
    datasets = Datasets(config_file=config_file, img_size=opt.img_size)
    
    cls_num = datasets.get_cls_num()
    keypoints_num = datasets.get_keypoints_num()
    
    dataloader_workers = opt.workers if opt.workers < get_core_num()[1] else get_core_num()[1]
    dataloader = DataLoader(
        dataset=datasets,
        batch_size=opt.batch_size,
        num_workers=dataloader_workers,
        shuffle=True
    )
    
    # set model
    model = FastGesture(keypoints_num=keypoints_num).to(device)
    
    # set loss
    # Loss函数定义
    # criterion_heatmap = FocalLoss(alpha=2, gamma=2)
    criterion_heatmap = nn.MSELoss().to(device=device)
    # criterion_bbox = nn.L1Loss().to(device=device)
    # criterion_confidence = nn.BCEWithLogitsLoss().to(device=device)
    criterion_ascription = nn.SmoothL1Loss(reduction='sum').to(device=device)
    
    # set optimizer
    user_set_optim = opt.optimizer
    optimizer = select_optim(net=model, opt=opt, user_set_optim=user_set_optim)
    
    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=save_path)
    
    if resume is not None:
        resume_model, resume_optim, resume_start_epoch = ckpt_load(resume)
        model.load_state_dict(resume_model, strict=True)
        start_epoch = resume_start_epoch
        optimizer.load_state_dict(resume_optim)
        
    for epoch in range(start_epoch, max_epoch):
        model.train()
        pbar = tqdm(dataloader, desc=f"[Epoch {epoch}] -> ")
        
        total_loss = 0.0
        min_loss = 10000
        max_loss = 10
        
        for index, datapack in enumerate(pbar):
            letterbox_image, tensor_letterbox_img, tensor_kp_cls_labels, tensor_ascription_field, tensor_ascription_mask = datapack
            
            # print(f"Letterbox Image Shape {letterbox_image.shape}")
            # cv2.imshow("Letterbox Image Tensor", cv2.resize(letterbox_image[0].cpu().detach().squeeze(0).numpy().astype(np.uint8), (640, 640)))
            cv2.imshow("Letterbox Image Tensor", letterbox_image[0].cpu().detach().squeeze(0).numpy().astype(np.uint8))

            
            # trans to  target device
            tensor_letterbox_img = tensor_letterbox_img.to(device)  # [Batch, 1, 320, 320]
            tensor_kp_cls_labels = tensor_kp_cls_labels.permute(1, 0, 2, 3).to(device)  # [11, Batch, 320, 320]
            # tensor_bbox_labels = tensor_bbox_labels.permute(1, 0, 2, 3).to(device)  # [5 + names_num, Batch, 160, 160]
            tensor_ascription_field = tensor_ascription_field.permute(1, 0, 2, 3).to(device)  # [3, Batch, 320, 320]
            
            tensor_ascription_mask = tensor_ascription_mask.permute(1, 0, 2, 3).to(device)
            
            optimizer.zero_grad()
            
            f_keypoints_classification, f_ascription = model(tensor_letterbox_img)
            
            f_keypoints_classification = torch.stack(f_keypoints_classification, dim=0).squeeze(2)
            # f_bbox_label = torch.stack(f_bbox_label, dim=0).squeeze(2)
            f_ascription = torch.stack(f_ascription, dim=0).squeeze(2)
            
            
            
            # print(f"f_keypoints_classification {f_keypoints_classification.shape}")
            # print(f"f_bbox_label {f_bbox_label.shape}")
            # print(f"f_ascription {f_ascription.shape}")
            keypoint_regress_loss = 0.0
            for kp_i in range(tensor_kp_cls_labels.shape[0]):
                
                # image_to_show = tensor_kp_cls_labels.permute(1, 0, 2, 3)[0][kp_i].cpu().detach().numpy().astype(np.float32)
                # # print(f"img_to_show {image_to_show.shape}")
                # # image_to_show = image_to_show[0, :, :]
                # # image_to_show = (image_to_show).astype(np.uint8)
                # # print("MAX: ", np.max(image_to_show))
                # cv2.imshow(f"KP Label {kp_i}", image_to_show)
                # cv2.waitKey(1) # 等待按键事件
                image_to_show = f_keypoints_classification.permute(1, 0, 2, 3)[0][kp_i].cpu().detach().numpy().astype(np.float32)
                # print(f"img_to_show {image_to_show.shape}")
                # image_to_show = image_to_show[0, :, :]
                # image_to_show = (image_to_show).astype(np.uint8)
                # print("MAX: ", np.max(image_to_show))
                cv2.imshow(f"Forward {kp_i}", image_to_show)
                # cv2.waitKey(1) # 等待按键事件
                
                keypoint_regress_loss += criterion_heatmap(f_keypoints_classification[kp_i], tensor_kp_cls_labels[kp_i])
            
            # bbox_loss = 0.0
            # for bbox_i in range(5):
            #     # image_to_show = tensor_kp_cls_labels.permute(1, 0, 2, 3)[0][kp_i].cpu().detach().numpy().astype(np.float32)
            #     # # print(f"img_to_show {image_to_show.shape}")
            #     # # image_to_show = image_to_show[0, :, :]
            #     # # image_to_show = (image_to_show).astype(np.uint8)
            #     # # print("MAX: ", np.max(image_to_show))
            #     # cv2.imshow(f"Label {kp_i}", image_to_show)
            #     # cv2.waitKey(1) # 等待按键事件
            #     # image_to_show = f_bbox_label.permute(1, 0, 2, 3)[0][bbox_i].cpu().detach().numpy().astype(np.float32)
            #     # # print(f"img_to_show {image_to_show.shape}")
            #     # # image_to_show = image_to_show[0, :, :]
            #     # # image_to_show = (image_to_show).astype(np.uint8)
            #     # # print("MAX: ", np.max(image_to_show))
            #     # cv2.imshow(f"BBox {bbox_i}", image_to_show)
            #     # cv2.waitKey(1) # 等待按键事件
                
            #     bbox_loss += criterion_bbox(f_bbox_label[bbox_i], tensor_bbox_labels[bbox_i])
            
            # loss += bbox_loss / 5
            
            # classification_loss = 0.0
            # for cls_i in range(tensor_bbox_labels.shape[0]-5, tensor_bbox_labels.shape[0]):
            #     # image_to_show = tensor_bbox_labels.permute(1, 0, 2, 3)[0][cls_i].cpu().detach().numpy().astype(np.float32)
            #     # # print(f"img_to_show {image_to_show.shape}")
            #     # # image_to_show = image_to_show[0, :, :]
            #     # # image_to_show = (image_to_show).astype(np.uint8)
            #     # # print("MAX: ", np.max(image_to_show))
            #     # cv2.imshow(f"Class Label {cls_i}", image_to_show)
            #     # cv2.waitKey(1) # 等待按键事件
            #     # image_to_show = f_bbox_label.permute(1, 0, 2, 3)[0][cls_i].cpu().detach().numpy().astype(np.float32)
            #     # # print(f"img_to_show {image_to_show.shape}")
            #     # # image_to_show = image_to_show[0, :, :]
            #     # # image_to_show = (image_to_show).astype(np.uint8)
            #     # # print("MAX: ", np.max(image_to_show))
            #     # cv2.imshow(f"Class {cls_i}", image_to_show)
            #     # cv2.waitKey(1) # 等待按键事件
            #     classification_loss += criterion_confidence(f_bbox_label[cls_i], tensor_bbox_labels[cls_i])
            
            # loss += classification_loss / (tensor_bbox_labels.shape[0]-5)
            
            asf_loss = 0.0
            for asf_i in range(3):
                asf_loss += criterion_ascription(
                    f_ascription[asf_i]*tensor_ascription_mask, tensor_ascription_field[asf_i]*tensor_ascription_mask
                )
            
            asf_loss = asf_loss/((tensor_ascription_mask.sum() + 1e-6)*3)
            
            image_to_show = tensor_ascription_field.permute(1, 0, 2, 3)[0].cpu().detach().numpy()  # .astype(np.float64)
            # print(f"img_to_show {image_to_show.shape}")
            # image_to_show = image_to_show[0, :, :]
            # image_to_show = (image_to_show).astype(np.uint8)
            # print("MAX: ", np.max(image_to_show))
            # image_to_show = (image_to_show + 1) / 2
            
            # print(np.transpose(image_to_show, (1, 2, 0)))
            # cv2.imshow(f"Label", cv2.resize(np.transpose(image_to_show, (1, 2, 0))*255, (640, 640)))
            cv2.imshow(f"Label", np.transpose(image_to_show, (1, 2, 0))*255)

            # cv2.waitKey(1) # 等待按键事件
            image_to_show = f_ascription.permute(1, 0, 2, 3)[0].cpu().detach().numpy()  # .astype(np.float64)
            # print(f"img_to_show {image_to_show.shape}")
            # image_to_show = image_to_show[0, :, :]
            # image_to_show = (image_to_show).astype(np.uint8)
            # print("MAX: ", np.max(image_to_show))
            # image_to_show = (image_to_show + 1) / 2
            cv2.imshow(f"f_ascription", np.transpose(image_to_show, (1, 2, 0))*255)
            cv2.waitKey(1) # 等待按键事件
            
            # if epoch > 2:
            #     draw_arrows_on_image(np.transpose(image_to_show, (1, 2, 0)))
            
            # loss += asf_loss # / 3
            
            loss = (keypoint_regress_loss / tensor_kp_cls_labels.shape[0])*0.6 + (asf_loss/3)*0.4
            
            # loss_heatmap = criterion_heatmap(f_keypoints_classification, tensor_kp_cls_labels)
            # # loss_bbox = criterion_bbox(f_bbox_label[:, :5, :, :], tensor_bbox_labels[:, :5, :, :])
            # loss_confidence = criterion_confidence(f_bbox_label[:, 5:, :, :], tensor_bbox_labels[:, 5:, :, :])
            # loss_ascription = criterion_ascription(f_ascription, tensor_ascription_field)
            
            # # 总Loss
            # loss = loss_heatmap + loss_bbox + loss_confidence + loss_ascription
            
            # 反向传播和优化
            
            # loss.sum().backward()
            loss.backward()
            # check_gradient(model) 
            optimizer.step()
           
            total_loss += loss.item()
           
            if total_loss < min_loss:
                min_loss = total_loss

            if total_loss > max_loss:
                max_loss = total_loss
            
            avg_loss = total_loss/(index+1)
           
            if device == 'cuda':
            # 获取当前程序占用的GPU显存（以字节为单位）
                gpu_memory_bytes = torch.cuda.memory_reserved(device)
                # print(f"gpu_memory_bytes {gpu_memory_bytes}")
                # 将字节转换为GB
                gpu_memory_GB = round(gpu_memory_bytes / 1024 / 1024 / 1024, 2)
                pbar.set_description(f"Epoch {epoch}, GPU {gpu_memory_GB} G, avg_l {avg_loss:.10f}")
            else:
                pbar.set_description(f"Epoch {epoch}, avg_l {avg_loss:.10f}")
            
            # 将 Loss 写入文本文件
            with open(f"{save_path}/log.txt", "a") as f:
                f.write(f"Epoch: {epoch}, Batch: {index}, AVG Loss: {avg_loss}, Total Loss: {total_loss}\n")

        # 使用 TensorBoard 记录 Loss
        writer.add_scalar("Avg Loss", avg_loss, epoch)
        writer.add_scalar("Total Loss", total_loss, epoch)
            
        ckpt_save(
            model=model, optim=optimizer, epoch=epoch, save_pth=save_path, file_name=f"epoch_{str(epoch)}",
        )
        ckpt_save(
            model=model, optim=optimizer, epoch=epoch, save_pth=save_path, file_name=f"epoch_{str(epoch)}", last=True
        )
    writer.close()
              
def run(opt):
    if opt.resume:
        checkpoints = opt.resume if isinstance(opt.resume, str) else None
        if checkpoints is None:
            raise ValueError("Resume Path cannot be empty")
        resume_path = str(Path(checkpoints).parent.parent)
        if not os.path.exists(resume_path):
            raise ValueError("Resume Path Not Exists")
        print(f"opt.resume {opt.resume}")
        ckpt = torch.load(opt.resume)
        train(opt=opt, save_path=resume_path, resume=ckpt)
    else:
        temp_full_path = opt.save_path + opt.save_name
        save_path = create_path(path=temp_full_path)
        train(opt=opt, save_path=save_path)
        

if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument('--device', type=str, default='cuda', help='cuda or cpu or mps')
    parse.add_argument('--batch_size', type=int, default=1, help='batch size')
    parse.add_argument('--img_size', type=int, default=160, help='trian img size')
    parse.add_argument('--epochs', type=int, default=1000, help='max train epoch')
    parse.add_argument('--data', type=str,default='./data/config.yaml', help='datasets config path')
    parse.add_argument('--save_period', type=int, default=4, help='save per n epoch')
    parse.add_argument('--workers', type=int, default=28, help='thread num to load data')
    parse.add_argument('--shuffle', action='store_false', help='chose to unable shuffle in Dataloader')
    parse.add_argument('--save_path', type=str, default='./run/train/')
    parse.add_argument('--save_name', type=str, default='new_datasets')
    parse.add_argument('--lr', type=float, default=0.001)
    parse.add_argument('--optimizer', type=str, default='Adam', help='only support: [Adam, AdamW, SGD, ASGD]')
    parse.add_argument('--resume', nargs='?', const=True, default=False, help="Choice one path to resume training")
    parse = parse.parse_args()
    
    run(opt=parse)
    
    