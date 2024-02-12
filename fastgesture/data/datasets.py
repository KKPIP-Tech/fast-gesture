import os
import sys
import yaml 
import json
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())

import cv2
import numpy as np
from copy import deepcopy
import random
from tqdm import tqdm

import torch
import torch.utils.data
from torchvision import transforms
from torchvision.transforms import functional as F

from fastgesture.data.augment import letterbox, shearing
from fastgesture.data.generate import get_vxvyd


class Datasets(torch.utils.data.Dataset):
    def __init__(self, config_file, img_size) -> None:
        
        print(f"Datasets Config File: {config_file}")
        
        if isinstance(img_size, int):
            self.height, self.width = img_size, img_size
        elif isinstance(img_size, list):
            self.height, self.width = img_size[1], img_size[0] 
        else:
            raise ValueError("img_size is not int or list")

        # Load Datasets Config ---------
        with open(config_file) as file:
            config = yaml.safe_load(file)
            
        self.datasets_path = config['root']
        self.names = config['names']
        self.nc = int(config['nc'])
        self.kc = int(config['kc'])
        self.target_points_id = config['target_points_id']
        print(f"Your Target Points ID of Mediapipe Is: {self.target_points_id}")
        
        self.max_hand_num = config['max_hand_num']
        self.datapack = self.load_data(
            names=self.names, 
            datasets_path=self.datasets_path,
            limit=None)
        
    def __getitem__(self, index):
        # get image path, lebel path, name index
        img_path, leb_path, ni = self.datapack[index]
        
        # image process -----------------------------------
        orin_image = cv2.imread(img_path)
        grey_image = cv2.cvtColor(orin_image, cv2.COLOR_BGR2GRAY)
        image_height, image_width = orin_image.shape[:2]
        del orin_image
        letterbox_image, scale_ratio, left_padding, top_padding = letterbox(
            image=grey_image,
            target_shape=(self.width, self.height),
            fill_color=114,
            single_channel=True
        )
        del grey_image
         
        cv2.imshow("letterbox_image", letterbox_image)
        cv2.waitKey(1)
        
        # label process -----------------------------------
        with open(leb_path) as label_file:
            label_json = json.load(label_file)
        
        preprocess_points_info = []
        for single_hand_data in label_json:
            points_json = single_hand_data['points']
            gesture = single_hand_data['gesture']
            hand_label = single_hand_data['hand_label']
            
            new_points_info = []
            x_cache = []
            y_cache = []
            for keypoint in points_json:
                target_id = self.get_keypoints_index(id=int(keypoint['id']))
                if target_id is None:
                    continue
                point = {
                    "id": target_id,
                    "x": float(keypoint['x']),
                    "y": float(keypoint['y']),
                    "z": float(keypoint['z'])
                }
                new_points_info.append(point)
                x_cache.append(float(keypoint['x']))
                y_cache.append(float(keypoint['y']))
            
            # get center point
            xmax = max(x_cache)
            xmin = min(x_cache)
            bbox_width = xmax - xmin
            center_x = xmin + ((xmax - xmin)/2)
            
            ymax = max(y_cache)
            ymin = min(y_cache)
            bbox_height = ymax - ymin
            center_y = ymin + ((ymax - ymin)/2)
            
            control_point = (center_x, center_y)
            
            single_hand_data = {
                'hand_label': hand_label,
                'gesture': gesture,
                'points': new_points_info,
                'control_point': control_point,
                'bbox': [center_x, center_y, bbox_width, bbox_height]
            }
            preprocess_points_info.append(single_hand_data)
        
        # get keypoints classification heatmaps
        keypoint_classfication_label = self.generate_keypoints_heatmap(
            points_info=preprocess_points_info,
            img_height=image_height,
            img_width=image_width,
            scale_ratio=scale_ratio,
            left_padding=left_padding,
            top_padding=top_padding
        )
        # cv2.imshow("classfication img", keypoint_classfication_label[0])
        # cv2.waitKey(1)
        
        # get bbox label and gesture label
        bbox_label = self.generate_bbox(
            points_info=preprocess_points_info,
            img_height=image_height,
            img_width=image_width,
            scale_ratio=scale_ratio,
            left_padding=left_padding,
            top_padding=top_padding
        )
        # cv2.imshow("cls img", cv2.resize(bbox_label[-1], (320, 320)))
        # cv2.waitKey()
        
        

        return letterbox_image, keypoint_classfication_label, bbox_label
    
    def __len__(self):
        return len(self.datapack)
    
    def generate_keypoints_heatmap(self, points_info, img_height, img_width, scale_ratio, left_padding, top_padding)->list:
        
        empty_heatmap = np.zeros((self.height, self.width))
        keypoint_classfication_label = [
            deepcopy(empty_heatmap) for _ in range(len(self.target_points_id))
        ]
        
        for one_hand in points_info: 
            points = one_hand['points']
            for point in points:
                id = point['id']
                x = int(int(img_width*scale_ratio)*point['x']) + left_padding - 1
                x = x if x < self.width else self.width - 1
                y = int(int(img_height*scale_ratio)*point['y']) + top_padding - 1
                y = y if y < self.height else self.height -1 
                temp_heatmap = keypoint_classfication_label[id]
                # print(f"temp_heatmap x:{x}, y{y}")
                # print()
                temp_heatmap[y][x] = 1
                keypoint_classfication_label[id] = temp_heatmap

        return keypoint_classfication_label
    
    def generate_bbox(self, points_info, img_height, img_width, scale_ratio, left_padding, top_padding)->list:
        
        empty_label = np.zeros((160, 160))
        bbox_label = [
            deepcopy(empty_label) for _ in range(5)
        ]
        
        cls_label = [
            deepcopy(empty_label) for _ in range(len(self.names))
        ]
        
        scales = 2
        
        for one_hand in points_info:
            cx, cy, w, h = one_hand['bbox']
            gesture = one_hand['gesture']
            
            x = int((int(int(img_width*scale_ratio)*cx) + int(left_padding) - 1)/scales)
            x = x if x < self.width else self.width - 1
            y = int((int(int(img_height*scale_ratio)*cy) + int(top_padding) - 1)/scales)
            y = y if y < self.height else self.height -1 
                        
            bbox_label[0][y][x] = cx
            bbox_label[1][y][x] = cy
            bbox_label[2][y][x] = w
            bbox_label[3][y][x] = h

            w = int(int(img_width*scale_ratio)*w) + int(top_padding) - 1
            w = w if w < self.width else self.width -1 
            h = int(int(img_height*scale_ratio)*h) + int(top_padding) - 1
            h = h if h < self.height else self.height -1 
            
            cls_label[gesture][
                int(y-(h/4/scales)):int(y+(h/4/scales)),
                int(x-(w/4/scales)):int(x+(w/4/scales)) 
            ] = 1
        
        bbox_label.extend(cls_label)
        
        return bbox_label
            
    @staticmethod
    def load_data(names, datasets_path, limit:int = None):
        # Load All Images Path, Labels Path and name index
        datapack: list = []  # [[img, leb, name_index]...]
        search_path: list = []
        
        for name in names:
            # get all search dir by name index
            target_image_path = datasets_path + '/' + name + '/images/'
            target_label_path = datasets_path + '/' + name + '/labels/'
            search_path.append([target_image_path, target_label_path, name])
        
        for target_image_path, target_label_path, name in search_path:
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
    
    @staticmethod
    def get_keypoints_index(id):
        keypoints_id = [0, 4, 8, 12, 16, 20]
        if id not in keypoints_id:
            return None
        return keypoints_id.index(id)

    