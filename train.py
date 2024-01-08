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
from utils.datasets import Datasets

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

def create_gaussian_kernel(size=15, sigma=5):
    # 生成一个高斯核
    m, n = [(ss - 1.) / 2. for ss in (size, size)]
    y, x = torch.meshgrid(torch.arange(-m, m+1), torch.arange(-n, n+1), indexing='xy')
    h = torch.exp(-(x*x + y*y) / (2*sigma*sigma))
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h / h.sum()
     


def train(opt, save_path):
    data_json = opt.data
    print(f"save_path: {save_path}")
    print(f"Datasets Config File Root Path: {data_json}")
    log_file = save_path + "/record.log"
    
    # set device
    device = opt.device if torch.cuda.is_available() else 'mps'
    print(f"Device: {device}")

    # set datasets
    datasets = Datasets(dataset_conf=data_json, img_size=opt.img_size)
    dataloader = DataLoader(
        dataset=datasets,
        batch_size=opt.batch_size,
        num_workers=opt.workers,
        shuffle=opt.shuffle
    )
    
    # 初始化网络
    max_hand_num = 5
    net = HandGestureNet(max_hand_num=max_hand_num, device=device)
    net.to(device=device)

    # 损失函数
    keypoints_loss = nn.CrossEntropyLoss().to(device=device)
    gestures_loss = nn.CrossEntropyLoss().to(device=device)
    
    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

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
            loss = 0.0

            # class_labels shape: [batch_size, max_hand_num, 1]
            # keypoints_label shape: [batch_size, max_hand_num, kc, 2]
            
            # 手势识别损失
            # class_labels = class_labels.view(-1)  # 将手势标签扁平化
            # print(f"keypoints_label shape: {keypoints_label.shape}")
            # gesture_values = gesture_values.view(-1, gesture_values.size(-1))  # 调整形状以匹配标签
            loss += gestures_loss(gesture_values, class_labels)

            # 关键点检测损失
            # keypoints_label = keypoints_label.view(-1, keypoints_label.shape[-3], keypoints_label.shape[-2], keypoints_label.shape[-1])  # 调整关键点标签的形状
            # keypoints_outputs = keypoints_outputs.view(-1, keypoints_outputs.shape[-2], keypoints_outputs.shape[-1])  # 确保输出形状正确
            loss += keypoints_loss(keypoints_outputs, keypoints_label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            min_loss = min(min_loss, total_loss)
            max_loss = max(max_loss, total_loss)
            avg_loss = total_loss / (index + 1)
            
            pbar.set_description(f"[Epoch {epoch}|avg_loss {avg_loss:.4f}->]")
            
        # print(f"loss item: {loss.item()}")
        # if i % 10 == 9:  # 每10个batch打印一次
        print(f"[{epoch + 1}] | -> avg_loss {avg_loss:.4f}, min_loss {min_loss:.4f}, max_loss {max_loss:.4f}")
        torch.save(net, save_path + f'/model_epoch_{epoch}.pth')

    print('Finished Training')

def run(opt):

    # Create
    temp_full_path = opt.save_path + opt.save_name
    save_path = create_path(path=temp_full_path)
    train(opt=opt, save_path=save_path)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parse.add_argument('--batch_size', type=int, default=8, help='batch size')
    parse.add_argument('--img_size', type=int, default=320, help='trian img size')
    parse.add_argument('--epochs', type=int, default=300, help='max train epoch')
    parse.add_argument('--data', type=str,default='./data', help='datasets config path')
    parse.add_argument('--save_period', type=int, default=4, help='save per n epoch')
    parse.add_argument('--workers', type=int, default=16, help='thread num to load data')
    parse.add_argument('--shuffle', action='store_false', help='chose to unable shuffle in Dataloader')
    parse.add_argument('--save_path', type=str, default='./run/train/')
    parse.add_argument('--save_name', type=str, default='exp')
    parse.add_argument('--lr', type=float, default=0.00001)
    parse.add_argument('--optimizer', type=str, default='Adam', help='only support: [Adam, SGD]')
    parse.add_argument('--loss', type=str, default='MSELoss', help='[MSELoss]')
    parse.add_argument('--resume', action='store_true')
    parse = parse.parse_args()

    run(opt=parse)