import os
import cv2
from time import time, sleep
import argparse

import torch
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
    
     # 加载模型
    model = GestureNet(n_channels=3, n_classes=21, img_size=opt.img_size, gesture_types=5).to(device)
    
    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # 训练循环
    for epoch in range(opt.epochs):
        model.train()
        total_loss = 0
        for index, datapack in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} ->", bar_format="{l_bar}{bar}{r_bar}", colour="green", unit=" batches")):
            image, landmarks, gesture_type, _ = datapack
            image = image.to(device)
            gesture_type = gesture_type.to(device)

            optimizer.zero_grad()
            outputs = model(image, landmarks)
            loss = criterion(outputs, gesture_type)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 打印每个周期的损失
        print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")
        
        # 保存每个周期的模型
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch}.pth"))

    print("Training completed.")


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