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
    print(f"Datasets Config File Root Path: {data_json}")
    log_file = save_path + "/record.log"
    
    # set device
    device = opt.device if torch.cuda.is_available() else 'cpu'
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
    net = HandGestureNet(max_hand_num=max_hand_num)
    net.to(device=device)

    # 损失函数
    keypoints_loss = nn.MSELoss().to(device=device)
    gestures_loss = nn.MSELoss().to(device=device)
    
    # 优化器
    optimizer = optim.SGD(net.parameters(), lr=opt.lr)

    zero_image = np.zeros((opt.img_size, opt.img_size))
    gaussian_kernel = create_gaussian_kernel().to(device).view(1, 1, 15, 15)

    for epoch in range(opt.epochs):
        # model.train()
        total_loss = 0.0
        min_loss = 100000
        max_loss = 10
        net.train()  # 设置为训练模式
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit=" batches")
        for index, datapack in enumerate(pbar):
            
            # 获取输入数据
            images, keypoints, class_labels = datapack
            # 第一是一张原图，尺寸如下 torch.Size([1, 3, 256, 256])；
            # 第二是一组关键点标签，尺寸如下 torch.Size([1, max_hand_num, 21, 256, 256])；
            # 第三是一组类别标签，尺寸如下 torch.Size([1, max_hand_num, 1])；
            
            images = images.to(device)
            keypoint_labels = keypoints.to(device)
            class_labels = class_labels.to(device)
            # print(f"keypoint_labels length: {len(keypoint_labels)}")
            
            # 前向传播
            st = time()
            forward = net(images)
            
            
            loss = 0.0
            
            # print(f"Output Length: {len(forward)}")
            
            for one_batch in zip(forward, keypoint_labels, class_labels):
                
                output_batch, keypoint_label_batch, gesture_label_batch = one_batch
                
                # print(len(keypoint_label_batch))

                pred_gesture_value = [keypoints_value[0] for keypoints_value in output_batch]
                pred_keypoints = [keypoints_value[1] for keypoints_value in output_batch]
            
                # print(f"pred_gesture_value: \n{len(pred_gesture_value)}")
                # print(f"pred_keypoints: \n{len(pred_keypoints)}")
                # print()
                
                for gesture_value_detect, keypoints_detect, heatmaps, gestures in zip(pred_gesture_value, pred_keypoints, keypoint_label_batch, gesture_label_batch):
                    # 创建一个空的张量来存储热图
                    output_heatmaps = torch.zeros((21, opt.img_size, opt.img_size), device=device)

                    for idx, point in enumerate(keypoints_detect):
                        # 在 GPU 上直接生成高斯热图
                        y, x = int(point[1] * opt.img_size), int(point[0] * opt.img_size)
                        if x < opt.img_size and y < opt.img_size:
                            output_heatmaps[idx, y, x] = 1
                            output_heatmaps[idx] = output_heatmaps[idx].unsqueeze(0)
                            output_heatmaps[idx] = F.conv2d(output_heatmaps[idx].unsqueeze(0), gaussian_kernel, padding=7).squeeze(0)
                    tensor_gestures_pred = torch.tensor([gesture_value_detect], dtype=torch.float32, requires_grad=True,device=device)
                    loss += keypoints_loss(output_heatmaps, heatmaps.to(device)) 
                    loss += gestures_loss(tensor_gestures_pred, gestures.to(device))

            # print(f"Net Time: {time() -st}")
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 打印统计信息
            total_loss += loss.item()
            
            if loss < min_loss:
                min_loss = loss
            if loss > max_loss:
                max_loss = loss
            
            avg_loss = total_loss / (index+1)
            
        print(f"loss item: {loss.item()}")
        # if i % 10 == 9:  # 每10个batch打印一次
        print(f"[{epoch + 1}] avg_loss: {avg_loss}")
        torch.save(net, os.path.join("./", f'model_epoch_{epoch}.pth'))


    print('Finished Training')

def run(opt):

    # Create
    temp_full_path = opt.save_path + opt.save_name
    save_path = create_path(path=temp_full_path)
    train(opt=opt, save_path=save_path)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parse.add_argument('--batch_size', type=int, default=4, help='batch size')
    parse.add_argument('--img_size', type=int, default=320, help='trian img size')
    parse.add_argument('--epochs', type=int, default=300, help='max train epoch')
    parse.add_argument('--data', type=str,default='./data', help='datasets config path')
    parse.add_argument('--save_period', type=int, default=4, help='save per n epoch')
    parse.add_argument('--workers', type=int, default=6, help='thread num to load data')
    parse.add_argument('--shuffle', action='store_false', help='chose to unable shuffle in Dataloader')
    parse.add_argument('--save_path', type=str, default='./run/train/')
    parse.add_argument('--save_name', type=str, default='exp')
    parse.add_argument('--lr', type=float, default=0.0001)
    parse.add_argument('--optimizer', type=str, default='Adam', help='only support: [Adam, SGD]')
    parse.add_argument('--loss', type=str, default='MSELoss', help='[MSELoss]')
    parse.add_argument('--resume', action='store_true')
    parse = parse.parse_args()

    run(opt=parse)