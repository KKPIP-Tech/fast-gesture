import os
import yaml
import sys
import cv2
import json
from time import time
import numpy as np
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())

import torch
import torch.utils.data
from torchvision import transforms

from utils.augment import letterbox


class Datasets(torch.utils.data.Dataset):
    def __init__(self, config_file, img_size):
        
        config_file = config_file + "/config.yaml"
        print(f"Datasets Config File: {config_file}")
        
        if isinstance(img_size, int):
            self.height, self.width = img_size, img_size
        elif isinstance(img_size, list):
            self.height, self.width = img_size
        else:
            raise ValueError("img_size is not int or list")

        # Load Datasets Config ---------
        with open(config_file) as file:
            config = yaml.safe_load(file)
            
        self.datasets_path = config['root']
        self.namse = config['names']
        self.nc = int(config['nc'])
        self.kc = int(config['kc'])
        
        self.max_hand_num = config['max_hand_num']
        self.datapack = self.load_data(
            names=self.namse, 
            datasets_path=self.datasets_path,
            limit=None)
    
    def __getitem__(self, index):
        start_time = time()
        # get image path, lebel path, name index
        img_path, leb_path, ni = self.datapack[index]
        
        # image process ---------------------
        original_img = cv2.imread(img_path)
        letterbox_img, scale_ratio, left_padding, top_padding = letterbox(
            image=original_img,
            target_shape=[self.height, self.width],
            fill_color=(114, 114, 114)
        )
        
        # labels process --------------------
        with open(leb_path) as label_file:
            leb_json = json.load(label_file)
        
        # Keypoints Labels Shape [max_hand_num, kc, 2]
        empty_keypoints_label = [np.asarray([0.0, 0.0]).copy() for _ in range(self.kc)]
        keypoints_labels = [
            empty_keypoints_label.copy() for _ in range(self.max_hand_num)
        ]
        # Gestures Labels Shape [max_hand_num, 1]
        gestures_labels = [
            [np.asarray(6)] for _ in range(self.max_hand_num)
        ]
        
        # load process data -----------------
        hand_index:int = 0
        for single_hand_data in leb_json:
            # Single Hand Keypoints Label Temp
            empty_label_copy = empty_keypoints_label.copy()
            points_json = single_hand_data['points']
            for keypoint in points_json:
                key_index = int(keypoint['id'])
                x = float(keypoint['x'])
                x = x if 0 <= x <= 1 else 1
                y = float(keypoint['y'])
                y = y if 0 <= y <= 1 else 1
                
                float_x = x*self.width-1+left_padding
                float_x = float_x if float_x < (self.width - 1) else self.width - 1
                float_y = y*self.height-1+top_padding
                float_y = float_y if float_y < (self.height - 1) else self.height - 1
                
                # normalization [0, 1]
                empty_label_copy[key_index] = np.asarray(
                    [
                        float(float_x/self.width),
                        float(float_x/self.height)
                    ]
                )
            
            keypoints_labels[hand_index] = empty_label_copy
            gestures_labels[hand_index] = [np.asarray(ni)]
            
            hand_index += 1
            is_index_over_limit: bool = hand_index == self.max_hand_num
            if is_index_over_limit:
                break
        
        # convert data to tensor ------------
        tensor_letterbox_img = transforms.ToTensor()(letterbox_img)
        tensor_keypoints_lab = torch.tensor(np.asarray(keypoints_labels), dtype=torch.float32)
        tensor_gestures_leb = torch.tensor(np.asarray(gestures_labels), dtype=torch.float32)
        
        return tensor_letterbox_img, tensor_keypoints_lab, tensor_gestures_leb
     
    def __len__(self):
        return len(self.datapack)
    
    def get_max_hand_num (self) -> int:
        return self.max_hand_num
    
    @staticmethod
    def load_data(names, datasets_path, limit:int = None):
        # Load All Images Path, Labels Path and name index
        datapack: list = []  # [[img, leb, name_index]...]
        search_path: list = []
        
        for name in names:
            # get all search dir by name index
            target_image_path = datasets_path + '/' + name + '/images/'
            target_label_path = datasets_path + '/' + name + '/labels/'
            search_path.append([target_image_path, target_label_path])
        
        for target_image_path, target_label_path in search_path:
            index_count:int = 0
            for path_pack in os.walk(target_image_path):
                for filename in path_pack[2]:
                    img = target_image_path + filename
                    label_name = filename.replace(".jpg", ".json")
                    leb = target_label_path + label_name
                    name_index = names.index(name)
                    datapack.append(
                        [img, leb, name_index]
                    )
                    index_count += 1
                    if limit is None:
                        continue
                    if index_count < limit:
                        continue
                    break
        
        return datapack