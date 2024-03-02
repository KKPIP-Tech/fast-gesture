import os
import cv2
import numpy as np
from time import time, sleep
import argparse

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from model.net import HandGestureNet
from utils.datasets_bk import Datasets

torch.cuda.empty_cache()


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path

    suffix = 1
    while True:
        new_path = f"{path}_{suffix}"
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
        suffix += 1


def select_optim(net:HandGestureNet, opt, user_set_optim:str=None):
    if user_set_optim == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    elif user_set_optim == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=opt.lr)
    elif user_set_optim == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=opt.lr)
    elif user_set_optim == "":
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
    

def train(opt, save_path):
    data_json = opt.data
    print(f"save_path: {save_path}")
    print(f"Datasets Config File Root Path: {data_json}")
    log_file = save_path + "/record.log"
    
    device = select_device(opt=opt)
    print(f"Device: {device}")

    # set datasets
    datasets = Datasets(config_file=data_json, img_size=opt.img_size)
    dataloader = DataLoader(
        dataset=datasets,
        batch_size=opt.batch_size,
        num_workers=opt.workers,
        shuffle=True
    )
    
    # 初始化网络
    max_hand_num = datasets.get_max_hand_num()
    kc = datasets.get_kc()
    net = HandGestureNet(max_hand_num=max_hand_num, device=device)
    net.to(device=device)

    # 损失函数
    keypoints_x_loss = nn.MSELoss().to(device=device)
    keypoints_y_loss = nn.MSELoss().to(device=device)
    gestures_loss = nn.L1Loss().to(device=device)
    
    # 优化器
    user_set_optim = opt.optimizer
    optimizer = select_optim(net=net, opt=opt, user_set_optim=user_set_optim)
        
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=1)
    
    for epoch in range(opt.epochs):
        # model.train()
        total_loss = 0.0
        min_loss = 100000
        max_loss = 10
        avg_loss = 0.0
        net.train()  # 设置为训练模式
        pbar = tqdm(dataloader, desc=f"[Epoch {epoch} ->]")
        for index, datapack in enumerate(pbar):
            images, keypoints, class_labels = datapack
            images = images.to(device)
            keypoints_label = keypoints.to(device)
            class_labels = class_labels.to(device)

            optimizer.zero_grad()
            gesture_values, keypoints_outputs = net(images)

            # 修正损失计算
            loss_gesture = 0.0
            loss_keypoint_x = 0.0
            loss_keypoint_y = 0.0

            # class_labels shape: [batch_size, max_hand_num, 1]
            # keypoints_label shape: [batch_size, max_hand_num, kc, 2]
            # print(f"pred_gesture: \n{gesture_values}")  
            # print(f"label_gesture: \n{class_labels}")

            for hi in range(max_hand_num):
                loss_gesture += gestures_loss(gesture_values.squeeze(0)[hi][0], class_labels.squeeze(0)[hi][0])
            
            # # print()
            # print(f"pred_keypoints: \n{keypoints_outputs}") 
            # print(f"label_keypoints: \n{keypoints_label}")
            for hi in range(max_hand_num):
                for li in range(kc):
                    loss_keypoint_x += keypoints_x_loss(keypoints_outputs.squeeze(0)[hi][li][0], keypoints_label.squeeze(0)[hi][li][0])
                    loss_keypoint_y += keypoints_y_loss(keypoints_outputs.squeeze(0)[hi][li][1], keypoints_label.squeeze(0)[hi][li][1])
            
            # loss_k = keypoints_loss(keypoints_outputs, keypoints_label)
            
            loss = loss_gesture + loss_keypoint_x + loss_keypoint_y
            
            # loss_g.backward()
            loss.backward()
            optimizer.step()

            # total_loss += loss_g.item()
            total_loss += loss.item()
            min_loss = min(min_loss, total_loss)
            max_loss = max(max_loss, total_loss)
            avg_loss = total_loss / (index + 1)
            
            pbar.set_description(f"[Epoch {epoch} | avg_loss {avg_loss:.4f}->]")
            
            #         # 检查和打印梯度
            # for name, param in net.named_parameters():
            #     if param.grad is not None:
            #         print(f"层 {name} 的梯度: {param.grad}")
            #     else:
            #         print(f"层 {name} 没有梯度")
        # print(f"loss item: {loss.item()}")
        # if i % 10 == 9:  # 每10个batch打印一次
        scheduler.step(opt.epochs)
        print(f"[Epoch {epoch}] | -> avg_loss: {avg_loss:.4f}, min_loss: {min_loss:.4f}, max_loss: {max_loss:.4f}")
        torch.save(net.state_dict(), save_path + f'/model_epoch_{epoch}.pt')
        print()

    print('Finished Training')

def run(opt):

    # Create
    temp_full_path = opt.save_path + opt.save_name
    save_path = create_path(path=temp_full_path)
    train(opt=opt, save_path=save_path)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--device', type=str, default='mps', help='cuda or cpu or mps')
    parse.add_argument('--batch_size', type=int, default=1, help='batch size')
    parse.add_argument('--img_size', type=int, default=320, help='trian img size')
    parse.add_argument('--epochs', type=int, default=300, help='max train epoch')
    parse.add_argument('--data', type=str,default='./data', help='datasets config path')
    parse.add_argument('--save_period', type=int, default=4, help='save per n epoch')
    parse.add_argument('--workers', type=int, default=6, help='thread num to load data')
    parse.add_argument('--shuffle', action='store_false', help='chose to unable shuffle in Dataloader')
    parse.add_argument('--save_path', type=str, default='./run/train/')
    parse.add_argument('--save_name', type=str, default='exp')
    parse.add_argument('--lr', type=float, default=0.001)
    parse.add_argument('--optimizer', type=str, default='Adam', help='only support: [Adam, AdamW, SGD]')
    # parse.add_argument('--loss', type=str, default='MSELoss', help='[MSELoss]')
    parse.add_argument('--resume', action='store_true')
    parse = parse.parse_args()

    run(opt=parse)