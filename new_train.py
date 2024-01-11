import os
import cv2
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from model.new_net import U_Net
from utils.datasets import Datasets


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


def select_optim(net:U_Net, opt, user_set_optim:str=None):
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
    
    # init model
    # kc = datasets.get_kc()
    model = U_Net(detect_num=1).to(device=device)
    
    loss_F = torch.nn.MSELoss()
    loss_F.to(device=device)
    
    # 优化器
    user_set_optim = opt.optimizer
    optimizer = select_optim(net=model, opt=opt, user_set_optim=user_set_optim)
    
    
    for epoch in range(opt.epochs):
        model.train()
        total_loss = 0.0
        min_loss = 10000
        max_loss = 10
        pbar = tqdm(dataloader, desc=f"[Epoch {epoch} ->]")
        for index, datapack in enumerate(pbar):
            images, heatmaps_label = datapack
            
            images = images.to(device)
            heatmaps_label = heatmaps_label.to(device)
            
            # print(f"heatmaps label max {np.max(heatmaps_label[0][0].detach().cpu().numpy().astype(np.float32)*255)}")
            # cv2.imshow("heatmap label", heatmaps_label[0][0].detach().cpu().numpy().astype(np.float32)*255)
            # cv2.waitKey()
            
            forward = model(images)
            loss = 0
            
            for ni in range(1):  # ni: names index
                # print("Label NI", label[:,ni,...].mean())
                # print(f"Label Shape[{ni}]", label[:,ni,...].shape)
                # print(f"Forward Shape[{ni}]", forward[ni].shape)
                # 选择批次中的第一个图像，并去除批次大小维度
                image_to_show = forward[ni][0].cpu().detach().numpy().astype(np.float32)

                # 确保图像是单通道的，尺寸为 (320, 320)
                image_to_show = image_to_show[0, :, :]

                # 转换数据类型并调整像素值范围
                # image_to_show = (image_to_show).astype(np.uint8)

                # 显示图像
                cv2.imshow("Forward", image_to_show)
                cv2.waitKey(1) # 等待按键事件
                # cv2.waitKey(0)
                loss += loss_F(forward[ni], heatmaps_label[:,ni,...].unsqueeze(1))
            
            # loss = loss_F(forward, label)  # 计算损失
            optimizer.zero_grad()  # 因为每次反向传播的时候，变量里面的梯度都要清零
            
            loss.backward()  # 变量得到了grad
            
            optimizer.step()  # 更新参数          
            total_loss += loss.item()
            

            if loss < min_loss:
                min_loss = loss

            if loss > max_loss:
                max_loss = loss
            
            avg_loss = total_loss/(index+1)
            
            if device == 'cuda':
            # 获取当前程序占用的GPU显存（以字节为单位）
                gpu_memory_bytes = torch.cuda.max_memory_reserved(device)
                # 将字节转换为GB
                gpu_memory_GB = round(gpu_memory_bytes / 1024 / 1024 / 1024, 2)
                pbar.set_description(f"Epoch {epoch}, GPU_Mem {gpu_memory_GB} GiB, avg_loss {avg_loss}")
            else:
                pbar.set_description(f"Epoch {epoch}, avg_loss {avg_loss}")
                
        print(f"[Epoch {epoch}] | -> avg_loss: {avg_loss:.4f}, min_loss: {min_loss:.4f}, max_loss: {max_loss:.4f}")
        torch.save(model.state_dict(), save_path + f'/model_epoch_{epoch}.pt')
        print()

def run(opt):
    # Create
    temp_full_path = opt.save_path + opt.save_name
    save_path = create_path(path=temp_full_path)
    train(opt=opt, save_path=save_path)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--device', type=str, default='cuda', help='cuda or cpu or mps')
    parse.add_argument('--batch_size', type=int, default=1, help='batch size')
    parse.add_argument('--img_size', type=int, default=640, help='trian img size')
    parse.add_argument('--epochs', type=int, default=1000, help='max train epoch')
    parse.add_argument('--data', type=str,default='./data', help='datasets config path')
    parse.add_argument('--save_period', type=int, default=4, help='save per n epoch')
    parse.add_argument('--workers', type=int, default=6, help='thread num to load data')
    parse.add_argument('--shuffle', action='store_false', help='chose to unable shuffle in Dataloader')
    parse.add_argument('--save_path', type=str, default='./run/train/')
    parse.add_argument('--save_name', type=str, default='exp')
    parse.add_argument('--lr', type=float, default=0.001)
    parse.add_argument('--optimizer', type=str, default='Adam', help='only support: [Adam, AdamW, SGD, ASGD]')
    # parse.add_argument('--loss', type=str, default='MSELoss', help='[MSELoss]')
    parse.add_argument('--resume', action='store_true')
    parse = parse.parse_args()

    run(opt=parse)