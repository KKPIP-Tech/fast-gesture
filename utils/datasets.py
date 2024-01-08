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
            cnt = 0
            for datapack in os.walk(search_images_path):
                for filename in datapack[2]:
                    image_path = search_images_path + filename
                    label_path = search_labels_path + filename.replace(".jpg", ".json")
                    images_path.append([image_path, label_path, names.index(name)])
                    # cnt += 1
                    # if cnt == 20:
                    #     break
        self.images_path = images_path
        
    def __getitem__(self, index):
        st = time()
        while True:
            # 图像的路径，图像及其标签的路径，类别索引
            image_path, label_path, name_index = self.images_path[index]
            
            # 图像处理 --------------
            original_image = cv2.imread(image_path)
            image_height, image_width = original_image.shape[:2]
            letterbox_image, scale_ratio, left_padding, top_padding = letterbox(
                image=original_image,
                target_shape=[self.height, self.width],
                fill_color=(114, 114, 114)
            )
            
            # cv2.imshow("letterbox image", letterbox_image)
            
            # Labels 处理 ------------
            with open(label_path) as label_content:
                json_data = json.load(label_content)
            if len(json_data) == 0  :
                # 如果 json label 当中没有任何数据
                while True:
                    random_int = random.randint(
                        (index - len(self.images_path)),
                        (len(self.images_path) - index)
                    )
                    temp_index = index + random_int
                    if 0 < temp_index < len(self.images_path):
                        index += random_int
                        break
                continue
            
            # 原始的遮罩
            zero_image = np.zeros((image_height, image_width))
            
            # 按照手的最大数量填充 object_labels 和 type_labels 以确保在多 Workers 和多 Batch Size 时不会出现尺寸错误的问题
            # object_labels = [[np.zeros((self.height, self.width)) for _ in range(self.kc)] for _ in range(self.max_hand_num)]  # 每个手的关键点及手势类别数据为一个元素
            
            object_labels = [[np.asarray([0.0, 0.0]).copy() for _ in range(self.kc)] for _ in range(self.max_hand_num)]
            type_labels = [[np.asarray(19)] for _ in range(self.max_hand_num)]  # 其索引值对应 object_labels 当中的元素位置，用于存储对应手的手势类别
            
            hand_cnt = 0
            for one_hand_data in json_data:
                # 处理单个手的数据
                heatmaps = [zero_image.copy() for _ in range(self.kc)]  # 跟据关键点的总数生成对应数量的 heatmap
                
                keypoint_label = [np.asarray([0.0, 0.0]).copy() for _ in range(self.kc)]
                
                # # Left: 0; Right: 1
                # hand_label = 0 if one_hand_data['hand_label'] == "Left" else 1 
                
                key_points = one_hand_data['points']
                
                for keypoint in key_points:
                    # 处理单个点的数据
                    heatmaps_index = int(keypoint['id'])
                    x = float(keypoint['x'])
                    x = x if 0 <= x <= 1 else 1
                    y = float(keypoint['y'])
                    y = y if 0 <= y <= 1 else 1
                    float_x = x*self.width-1+left_padding
                    float_x = float_x if float_x < (self.width - 1) else self.width - 1
                    float_y = y*self.height-1+top_padding
                    float_y = float_y if float_y < (self.height - 1) else self.height - 1
                    
                    
                    
                    keypoint_label[heatmaps_index] = np.asarray([
                        float((float_x/self.width*2)-1),
                        float((float_x/self.height*2)-1)
                    ])
                    

                
                if hand_cnt < self.max_hand_num:
                    object_labels[hand_cnt] = keypoint_label
                    type_labels[hand_cnt] = [np.asarray(name_index)]
                    hand_cnt += 1
                    if hand_cnt == self.max_hand_num:
                        break
                else:
                    break
            break         
        
        # print(f"object_labels length in datasets: {len(object_labels)}")
        
        letterbox_image = transforms.ToTensor()(letterbox_image)
        object_labels = torch.tensor(np.array(object_labels), dtype=torch.float32)
        type_labels = torch.tensor(np.array(type_labels), dtype=torch.float32)
        # print(f"Dataset Upload Time: {time() - st}")
        return letterbox_image, object_labels, type_labels
        
    def __len__(self):
        return len(self.images_path)


