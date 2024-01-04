import os
import cv2
from time import time, sleep
import argparse

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.net import GestureNet
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
    print(f"data json: {data_json}")
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
    # 初始化模型
    # num_heatmaps = len(datasets[0][1])  # Heatmap 的数量
    # num_hand_types = 2  # 手的类别数量（0 或 1）
    # num_gesture_types = opt.num_gesture_types  # 手势类别的数量，需要在 opt 中指定
    model = GestureNet(21, 5).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(opt.epochs):
        model.train()
        total_loss = 0
        for index, (images, heatmaps, gesture_types) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", unit=" batches")):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            gesture_types = gesture_types.to(device)

            outputs = model(images, heatmaps, gesture_types)

            # 假设损失计算基于手势类型预测
            loss = criterion(outputs, gesture_types[:, :, :5])  # 根据实际情况调整
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{opt.epochs}], Loss: {total_loss/len(dataloader)}")

def run(opt):

    # Create
    temp_full_path = opt.save_path + opt.save_name
    save_path = create_path(path=temp_full_path)
    train(opt=opt, save_path=save_path)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parse.add_argument('--batch_size', type=int, default=1, help='batch size')
    parse.add_argument('--img_size', type=int, default=256, help='trian img size')
    parse.add_argument('--epochs', type=int, default=300, help='max train epoch')
    parse.add_argument('--data', type=str,default='./data', help='datasets config path')
    parse.add_argument('--save_period', type=int, default=1, help='save per n epoch')
    parse.add_argument('--workers', type=int, default=1, help='thread num to load data')
    parse.add_argument('--shuffle', action='store_false', help='chose to unable shuffle in Dataloader')
    parse.add_argument('--save_path', type=str, default='./run/train/')
    parse.add_argument('--save_name', type=str, default='exp')
    parse.add_argument('--lr', type=float, default=0.0001)
    parse.add_argument('--optimizer', type=str, default='Adam', help='only support: [Adam, SGD]')
    parse.add_argument('--loss', type=str, default='MSELoss', help='[MSELoss]')
    parse.add_argument('--resume', action='store_true')
    parse = parse.parse_args()

    run(opt=parse)