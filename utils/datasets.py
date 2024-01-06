import os
import yaml
import sys
import cv2
import json
import random
import numpy as np
from time import time, sleep
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__()) 

import torch 
import torch.utils.data
from torch import _nnpack_available
from torchvision import datasets, transforms, models

from utils.augment import letterbox

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


class Datasets(torch.utils.data.Dataset):
    def __init__(self, dataset_conf, img_size, device) -> None:
        
        self.device = device
        
        yaml_file = dataset_conf + '/config.yaml'
        print(f"Config File: {yaml_file}")
        
        if isinstance(img_size, int):
            self.height, self.width = img_size, img_size
        elif isinstance(img_size, list):
            self.height, self.width = img_size
        else:
            raise ValueError("img_size is not int or list")
        
        with open(yaml_file) as file:
            config = yaml.safe_load(file)
        
        datasets_root = config['root']
        datasets_resolution = config['resolution']
        names = config['names']
        self.names = names
        self.nc = int(config['nc'])
        self.kc = int(config['kc'])
        
        self.kernel_size = config['kernel_size']
        self.sigma_x = config['sigma_x']
        self.sigma_y = config['sigma_y']
        
        self.max_hand_num = config['max_hand_num']
        # self.max_hand_num += 1
        
        print(f"names: {names}")
        
        images_path = []
        for name in names:
            search_images_path = datasets_root + '/' + name + '/images/'
            search_labels_path = datasets_root + '/' + name + '/labels/'
            for datapack in os.walk(search_images_path):
                for filename in datapack[2]:
                    image_path = search_images_path + filename
                    label_path = search_labels_path + filename.replace(".jpg", ".json")
                    images_path.append([image_path, label_path, names.index(name)])
        self.images_path = images_path
        
    def __getitem__(self, index):
        st = time()
        while True:
            image_path, label_path, name_index = self.images_path[index]
            
            original_image = cv2.imread(image_path)
            image_height, image_width = original_image.shape[:2]
            letterbox_image = letterbox(original_image, [self.height, self.width], (114, 114, 114))[0]

            with open(label_path) as label_content:
                json_data = json.load(label_content)
            if len(json_data) == 0:
                index = (index + random.randint(-len(self.images_path), len(self.images_path))) % len(self.images_path)
                continue
            
            zero_image = torch.zeros((image_height, image_width), device=self.device)
            object_labels = [[torch.zeros((self.height, self.width), device=self.device) for _ in range(self.kc)] for _ in range(self.max_hand_num)]
            type_labels = [[torch.tensor(19, device=self.device)] for _ in range(self.max_hand_num)]
            
            hand_cnt = 0
            for one_hand_data in json_data:
                heatmaps = [zero_image.clone() for _ in range(self.kc)]
                key_points = one_hand_data['points']
                
                for keypoint in key_points:
                    heatmaps_index = int(keypoint['id'])
                    x = float(keypoint['x'])
                    x = x if 0 <= x <= 1 else 1
                    y = float(keypoint['y'])
                    y = y if 0 <= y <= 1 else 1
                    heatmaps[heatmaps_index][int(y*image_height-1), int(x*image_width-1)] = 255
                
                for heatmaps_index in range(len(heatmaps)):
                    heatmap = heatmaps[heatmaps_index]
                    heatmap = cv2.GaussianBlur(heatmap.cpu().numpy(), self.kernel_size, self.sigma_x, self.sigma_y)
                    heatmap = torch.from_numpy(heatmap).to(self.device)
                    heatmap_amax = torch.amax(heatmap)
                    if heatmap_amax != 0:
                        heatmap /= heatmap_amax / 255
                    heatmap /= 255.0
                    letterbox_heatmap = letterbox(heatmap.cpu().numpy(), [self.height, self.width], 0, is_mask=True)[0]
                    heatmaps[heatmaps_index] = torch.from_numpy(letterbox_heatmap).to(self.device)
                
                if hand_cnt < self.max_hand_num:
                    object_labels[hand_cnt] = heatmaps
                    type_labels[hand_cnt] = [torch.tensor(name_index, device=self.device)]
                    hand_cnt += 1
                    if hand_cnt == self.max_hand_num:
                        break

            letterbox_image = transforms.ToTensor()(letterbox_image).to(self.device)
            object_labels = torch.stack([torch.stack(hand) for hand in object_labels])
            type_labels = torch.tensor(type_labels, dtype=torch.float32).to(self.device)
            return letterbox_image, object_labels, type_labels
        
    def __len__(self):
        return len(self.images_path)


