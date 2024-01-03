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

class Datasets(torch.utils.data.Dataset):
    def __init__(self, dataset_conf, img_size) -> None:
        
        yaml_file = dataset_conf + '/config.yaml'
        print(dataset_conf)
        sleep(2)
        
        if isinstance(img_size, int):
            self.height, self.weight = img_size, img_size
        elif isinstance(img_size, list):
            self.height, self.weight = img_size
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
        while True:
            # 应用数据增强到图像
            image_path, label_path, name_index = self.images_path[index]
            
            # image process --------
            original_image = cv2.imread(image_path)
            img_height, img_width = original_image.shape[:2]
            letterbox_image, scale_ratio, left_padding, top_padding = letterbox(
                image=original_image,
                target_shape=[self.height, self.weight],
                fill_color=(114, 114, 114)
            )
            
            zero_image = np.zeros((img_height, img_width))
            
            # cv2.imshow("letterbox_image", letterbox_image)
            
            # label process --------
            landmarks = [zero_image.copy() for _ in range(self.kc)]
            gesture_type = []
            with open(label_path) as label_content:
                json_data = json.load(label_content)
            if len(json_data) == 0:
                continue
            
            for one_object in json_data:
                x_cache = []
                y_cache = []
                z_cache = []
                # Left: 0; Right: 1
                hand_label = 0 if one_object['hand_label'] == "Left" else 1
                points = one_object['points']
                for single_point in points:
                    try:
                        id = int(single_point['id'])
                        x = float(single_point['x'])
                        y = float(single_point['y'])
                        z = float(single_point['z'])
                        heatmap = landmarks[id]
                        heatmap[int(y*img_height-1)][int(x*img_width-1)] = 255
                        x_cache.append(x)
                        y_cache.append(y)
                        z_cache.append(z)
                        gesture_type.append(np.array([hand_label, id, x, y, name_index]))
                    except:
                        continue
            break
        
        for cnt in range(len(landmarks)):

            heatmap = landmarks[cnt]
            heatmap = cv2.GaussianBlur(heatmap, 
                                       self.kernel_size, 
                                       self.sigma_x, 
                                       self.sigma_y)
            heatmap_amax = np.amax(heatmap)
            if heatmap_amax != 0:
                heatmap /= heatmap_amax / 255
            heatmap /= 255.0
            heatmap = letterbox(
                image=heatmap,
                target_shape=[self.height, self.weight],
                fill_color=0,
                is_mask=True
            )[0]

            landmarks[cnt] = np.array(heatmap)
        
        letterbox_image = transforms.ToTensor()(letterbox_image)
        landmarks = torch.tensor(np.array(landmarks), dtype=torch.float32)
        gesture_type = torch.tensor(np.array(gesture_type), dtype=torch.float32)
            
        return letterbox_image, landmarks, gesture_type
        
    def __len__(self):
        return len(self.images_path)


