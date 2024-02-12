import os
import sys
import argparse
from copy import deepcopy
from pathlib import Path

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

from fastgesture.model.body import FastGesture
from fastgesture.data.datasets import Datasets
from fastgesture.utils.checkpoint import ckpt_load, ckpt_save, create_path
from fastgesture.utils.common import (
    select_device, 
    select_optim,
    get_core_num
)


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
    dataloader_workers = opt.workers if opt.workers < get_core_num()[1] else get_core_num()[1]
    dataloader = DataLoader(
        dataset=datasets,
        batch_size=opt.batch_size,
        num_workers=dataloader_workers,
        shuffle=True
    )
    
    # set model
    model = FastGesture(keypoints_num=11, cls_num=5)
    
    # set optimizer
    user_set_optim = opt.optimizer
    optimizer = select_optim(net=model, opt=opt, user_set_optim=user_set_optim)
    
    if resume is not None:
        model, optim, start_epoch = ckpt_load(resume)
        model.load_state_dict(model, strict=True)
        start_epoch = start_epoch
        optimizer.load_state_dict(optim)
        
    for epoch in range(start_epoch, max_epoch):
        
        pbar = tqdm(dataloader, desc=f"[Epoch {epoch}] -> ")
        for index, datapack in enumerate(pbar):
            pass


def run(opt):
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
    parse.add_argument('--data', type=str,default='./data/config.yaml', help='datasets config path')
    parse.add_argument('--save_period', type=int, default=4, help='save per n epoch')
    parse.add_argument('--workers', type=int, default=1, help='thread num to load data')
    parse.add_argument('--shuffle', action='store_false', help='chose to unable shuffle in Dataloader')
    parse.add_argument('--save_path', type=str, default='./run/train/')
    parse.add_argument('--save_name', type=str, default='exp')
    parse.add_argument('--lr', type=float, default=0.001)
    parse.add_argument('--optimizer', type=str, default='Adam', help='only support: [Adam, AdamW, SGD, ASGD]')
    parse.add_argument('--resume', nargs='?', const=True, default=False, help="Choice one path to resume training")
    parse = parse.parse_args()
    
    run(opt=parse)
    
    