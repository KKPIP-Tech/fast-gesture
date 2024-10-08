import os
from copy import deepcopy
import torch


def ckpt_save(model, optim, scaler, epoch, pncs_result, save_pth, file_name, best=False, last=False):
    ckpt = {
        "model": model,
        "optimizer": optim.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "pncs_result":deepcopy(pncs_result)
    }
    
    save_pth = save_pth + "/weights"
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
        return save_pth
    
    torch.save(ckpt, save_pth + "/" + file_name + ".pt")
    if best:
        torch.save(ckpt, save_pth + "/best.pt")
    if last:
        torch.save(ckpt, save_pth + "/last.pt")
        
        
def ckpt_load(model_path, export=False):
    
    model = model_path['model']
    start_epoch = model_path['epoch'] + 1
    optim = model_path['optimizer']
    scaler = model_path['scaler']
    pncs_result = model_path['pncs_result']

    if not export: return model, optim, scaler, start_epoch
    return model, pncs_result


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
    
    
    