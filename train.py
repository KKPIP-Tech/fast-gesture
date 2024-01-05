import os
import cv2
from time import time, sleep
import argparse

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from model.net import HandGestureNetwork
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
    
    # 参数设置
    max_hand_num = 10
    num_epochs = 10  # 训练轮数
    learning_rate = 0.000001  # 学习率

    # 初始化网络
    net = HandGestureNetwork(max_hand_num)
    
    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()  # 可以根据需要选择适当的损失函数
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(opt.epochs):
        # model.train()
        running_loss = 0.0
        net.train()  # 设置为训练模式
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit=" batches")
        for index, datapack in enumerate(pbar):
            
            # 获取输入数据
            images, keypoints, class_labels = datapack
            # 第一是一张原图，尺寸如下 torch.Size([1, 3, 256, 256])；
            # 第二是一组关键点标签，尺寸如下 torch.Size([1, max_hand_num, 21, 256, 256])；
            # 第三是一组类别标签，尺寸如下 torch.Size([1, max_hand_num, 1])；
            
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播 + 反向传播 + 优化
            gesture_outputs, keypoint_outputs = net(images, keypoints, class_labels)
            
            # 计算损失
            loss = 0
            # 在计算损失时
            for j in range(max_hand_num):
                # 转换 class_labels 为 one-hot 编码以匹配 gesture_outputs 的形状
                one_hot_labels = F.one_hot(class_labels[:, j].to(torch.int64), num_classes=20)
                
                # 确保 one_hot_labels 和 gesture_outputs 的形状相同
                one_hot_labels = one_hot_labels.to(torch.float32).view(gesture_outputs[j].shape)
                
                loss += criterion(gesture_outputs[j], one_hot_labels)
                loss += criterion(keypoint_outputs[j], keypoints[:, j, :, :, :])

            loss.backward()
            optimizer.step()

            # 打印统计信息
            running_loss += loss.item()
            print(f"loss item: {loss.item()}")
            # if i % 10 == 9:  # 每10个batch打印一次
            print(f"[{epoch + 1}] loss: {running_loss / 10:.3f}")
 

    print('Finished Training')

def run(opt):

    # Create
    temp_full_path = opt.save_path + opt.save_name
    save_path = create_path(path=temp_full_path)
    train(opt=opt, save_path=save_path)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
    parse.add_argument('--batch_size', type=int, default=1, help='batch size')
    parse.add_argument('--img_size', type=int, default=256, help='trian img size')
    parse.add_argument('--epochs', type=int, default=300, help='max train epoch')
    parse.add_argument('--data', type=str,default='./data', help='datasets config path')
    parse.add_argument('--save_period', type=int, default=4, help='save per n epoch')
    parse.add_argument('--workers', type=int, default=16, help='thread num to load data')
    parse.add_argument('--shuffle', action='store_false', help='chose to unable shuffle in Dataloader')
    parse.add_argument('--save_path', type=str, default='./run/train/')
    parse.add_argument('--save_name', type=str, default='exp')
    parse.add_argument('--lr', type=float, default=0.0001)
    parse.add_argument('--optimizer', type=str, default='Adam', help='only support: [Adam, SGD]')
    parse.add_argument('--loss', type=str, default='MSELoss', help='[MSELoss]')
    parse.add_argument('--resume', action='store_true')
    parse = parse.parse_args()

    run(opt=parse)