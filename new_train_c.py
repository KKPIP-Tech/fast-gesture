import os
import cv2
import numpy as np
import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from model.new_net import FastGesture
from utils.datasets import Datasets


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path

    suffix = 1
    while True:
        new_path = f"{path}{suffix}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
        suffix += 1


def select_optim(net, opt, user_set_optim:str=None):
    if user_set_optim == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    elif user_set_optim == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=opt.lr)
    elif user_set_optim == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=opt.lr)
    elif user_set_optim == "ASGD":
        optimizer = optim.ASGD(net.parameters(), lr=opt.lr)
    else:
        print(f"Your Input Setting Optimizer {user_set_optim} is Not In [Adam, AdamW, SGD, ASGD]")
        print(f"Use Default Optimizer Adam")
        optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    return optimizer


def select_device(opt):
    # set device
    user_set_device = opt.device
    if user_set_device == 'cpu':
        device = user_set_device
    elif user_set_device == 'cuda':
        device = user_set_device if torch.cuda.is_available() else 'cpu'
    elif user_set_device == 'mps':
        device = user_set_device if torch.backends.mps.is_available() else 'cpu'
    else:
        print(f" Your Device Setting: {user_set_device} is not support!")
        device = 'cpu'
    return device


def bbox_loss_fn(predicted_bboxes, true_bboxes, num_max_boxes, mse_loss):
    """
    计算边界框损失
    :param predicted_bboxes: 预测的边界框，形状为 [batch_size, num_max_boxes, 4]
    :param true_bboxes: 实际的边界框，形状为 [batch_size, num_true_boxes, 4]
    :param num_max_boxes: 预测的最大边界框数量
    :return: 边界框损失
    """
    batch_size = predicted_bboxes.size(0)
    

    loss = 0.0
    for i in range(batch_size):
        num_true_boxes = true_bboxes[i].size(0)

        # 为每个实际框找到最佳匹配的预测框
        if num_true_boxes > 0:
            for j in range(num_true_boxes):
                # 假设简单地选择最近的预测框
                distances = torch.norm(predicted_bboxes[i] - true_bboxes[i][j], dim=1)
                best_match = torch.argmin(distances)
                loss += mse_loss(predicted_bboxes[i][best_match], true_bboxes[i][j])

        # 如果预测的框比实际的多，将多余的预测框视为负样本
        if num_max_boxes > num_true_boxes:
            num_neg_samples = num_max_boxes - num_true_boxes
            neg_samples_loss = mse_loss(predicted_bboxes[i][:num_neg_samples], torch.zeros_like(predicted_bboxes[i][:num_neg_samples]))
            loss += torch.sum(neg_samples_loss)

    return loss / batch_size



def train(opt, save_path, resume_pth=None):
    data_json = opt.data
    print(f"save_path: {save_path}")
    print(f"Datasets Config File Root Path: {data_json}")
    log_file = save_path + "/record.log"
    
    device = select_device(opt=opt)
    print(f"Device: {device}")
    max_epoch = opt.epochs
    start_epoch = 0
    
    
    
    # set datasets
    datasets = Datasets(config_file=data_json, img_size=opt.img_size)
    dataloader = DataLoader(
        dataset=datasets,
        batch_size=opt.batch_size,
        num_workers=opt.workers,
        shuffle=True
    )
    
    # init model
    kc = datasets.get_kc()
    nc = datasets.get_nc()
    model = FastGesture(detect_num=(kc + 1), heatmap_channels=1, num_classes=nc).to(device=device)
    
    loss_F = torch.nn.MSELoss().to(device=device)
    class_loss_fn = nn.CrossEntropyLoss().to(device=device)
    box_mse_loss = nn.MSELoss(reduction='none')
    obj_loss_fn = nn.BCEWithLogitsLoss().to(device=device)
    
    # 优化器
    user_set_optim = opt.optimizer
    optimizer = select_optim(net=model, opt=opt, user_set_optim=user_set_optim)
    
    if resume_pth is not None:
        # resume_state_dict = resume_pth['model'].float().state_dict()
        model.load_state_dict(resume_pth['model'], strict=True)
        start_epoch = resume_pth['epoch'] + 1
        optimizer.load_state_dict(resume_pth['optimizer'])
            
    for epoch in range(start_epoch, max_epoch):
        model.train()
        total_loss = 0.0
        min_loss = 10000
        max_loss = 10
        pbar = tqdm(dataloader, desc=f"[Epoch {epoch} ->]")
        for index, datapack in enumerate(pbar):
            images, heatmaps_label, labels, bboxes, objects= datapack
            
            images = images.to(device)
            heatmaps_label = heatmaps_label.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)
            objects = objects.to(device)
            
            # print(f"heatmaps label max {np.max(heatmaps_label[0][0].detach().cpu().numpy().astype(np.float32)*255)}")
            # cv2.imshow("heatmap label", heatmaps_label[0][0].detach().cpu().numpy().astype(np.float32)*255)
            # cv2.waitKey()
            
            f_heatmaps, f_class_scores, f_bboxes, f_obj_scores = model(images)
            # class scores shape: torch.Size([2, 5, 320, 320])
            # bboxes shape: torch.Size([2, 4, 320, 320])
            # obj shape: torch.Size([2, 1, 320, 320])
            
            loss = 0
            
            # 计算损失
            class_loss = class_loss_fn(f_class_scores, labels)
            bbox_loss = bbox_loss_fn(f_bboxes, bboxes, 10, box_mse_loss)
            objects = objects.unsqueeze(1)
            obj_loss = obj_loss_fn(f_obj_scores, objects)
            
            loss = class_loss + bbox_loss + obj_loss
                
            for ni in range(kc+1):  # ni: names index
                # print("Label NI", label[:,ni,...].mean())
                # print(f"Label Shape[{ni}]", label[:,ni,...].shape)
                # print(f"Forward Shape[{ni}]", forward[ni].shape)
                # 选择批次中的第一个图像，并去除批次大小维度
                # image_to_show = f_heatmaps[ni][0].cpu().detach().numpy().astype(np.float32)

                # # # 确保图像是单通道的，尺寸为 (320, 320)
                # image_to_show = image_to_show[0, :, :]

                # # # 转换数据类型并调整像素值范围
                # # # image_to_show = (image_to_show).astype(np.uint8)
                # # # print("MAX: ", np.max(image_to_show))
                # # # 显示图像
                # image_to_show = cv2.resize(image_to_show, (200, 200))
                # cv2.imshow(f"Forward {ni}", image_to_show)
                # cv2.waitKey(1) # 等待按键事件
                # # cv2.waitKey(0)
                loss += loss_F(f_heatmaps[ni], heatmaps_label[:,ni,...].unsqueeze(1))
            
            # loss = loss_F(forward, label)  # 计算损失
            optimizer.zero_grad()  # 因为每次反向传播的时候，变量里面的梯度都要清零
            
            loss.sum().backward()  # 变量得到了grad
            
            optimizer.step()  # 更新参数     
            loss_scalar = loss.sum()      
            total_loss += loss_scalar.item()
            
            # print(f"total loss {total_loss}")

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
                pbar.set_description(f"Epoch {epoch}, GPU {gpu_memory_GB} G, avg_l {avg_loss:.8f}")
            else:
                pbar.set_description(f"Epoch {epoch}, avg_l {avg_loss}")
                
        print(f"[Epoch {epoch}] | -> avg_l: {avg_loss:.4f}, min_l: {min_loss:.4f}, max_l: {max_loss:.4f}")
        
        ckpt = {
            'model': deepcopy(model.state_dict()),
            'optimizer': deepcopy(optimizer.state_dict()),
            'epoch': epoch
        }
        
        torch.save(ckpt, save_path + f'/model_epoch_{epoch}.pt')
        torch.save(ckpt, save_path + f'/last.pt')
        print()

def run(opt):
    # Create
    
    if opt.resume:
        checkpoints = opt.resume if isinstance(opt.resume, str) else None
        if checkpoints is None:
            raise ValueError("Resume Path cannot be empty")
        resume_path = str(Path(checkpoints).parent)
        if not os.path.exists(resume_path):
            raise ValueError("Resume Path Not Exists")
        print(f"opt.resume {opt.resume}")
        ckpt = torch.load(opt.resume)
        train(opt=opt, save_path=resume_path, resume_pth=ckpt)
    else:
        temp_full_path = opt.save_path + opt.save_name
        save_path = create_path(path=temp_full_path)
        train(opt=opt, save_path=save_path)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--device', type=str, default='cuda', help='cuda or cpu or mps')
    parse.add_argument('--batch_size', type=int, default=1, help='batch size')
    parse.add_argument('--img_size', type=int, default=320, help='trian img size')
    parse.add_argument('--epochs', type=int, default=1000, help='max train epoch')
    parse.add_argument('--data', type=str,default='./data', help='datasets config path')
    parse.add_argument('--save_period', type=int, default=4, help='save per n epoch')
    parse.add_argument('--workers', type=int, default=12, help='thread num to load data')
    parse.add_argument('--shuffle', action='store_false', help='chose to unable shuffle in Dataloader')
    parse.add_argument('--save_path', type=str, default='./run/train/')
    parse.add_argument('--save_name', type=str, default='exp')
    parse.add_argument('--lr', type=float, default=0.001)
    parse.add_argument('--optimizer', type=str, default='Adam', help='only support: [Adam, AdamW, SGD, ASGD]')
    # parse.add_argument('--loss', type=str, default='MSELoss', help='[MSELoss]')
    parse.add_argument('--resume', nargs='?', const=True, default=False)
    parse = parse.parse_args()

    run(opt=parse)
    