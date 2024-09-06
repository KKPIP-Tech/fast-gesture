import psutil
import torch
from torch import optim


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


def get_core_num():
    """
    获取 CPU 信息
    """
    # 获取逻辑 CPU 数量
    logical_cpus = psutil.cpu_count(logical=True)
    
    # 获取物理 CPU 数量
    physical_cpus = psutil.cpu_count(logical=False)
    
    return logical_cpus, physical_cpus