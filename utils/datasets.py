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
            self.height, self.width = img_size[1], img_size[0] 
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
        # get image path, lebel path, name index
        img_path, leb_path, ni = self.datapack[index]
        
        # image process ---------------------
        original_img = cv2.imread(img_path)
        original_height, original_width = original_img.shape[:2]
        
        # canny, drawContours 
        grey_img = cv2.cvtColor(original_img.copy(), cv2.COLOR_BGR2GRAY)
        canny_img = cv2.Canny(grey_img, 0, 80, 50)
        contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(original_img,contours,-1,(0,0,255),2) 
        resize_img = cv2.resize(original_img, (self.width, self.height), cv2.INTER_AREA)
        
        # cv2.imshow("Canny_image", resize_img)
        # cv2.waitKey()
        
        # labels process --------------------
        with open(leb_path) as label_file:
            leb_json = json.load(label_file)
        
        # Keypoints Label Shape [kc, self.height, self.width]
        empty_heatmap_original_size = np.zeros((original_height, original_width))
        heatmaps_label = [
            empty_heatmap_original_size.copy() for _ in range(self.kc + 1)
        ] 
        
        # Gesture and boxx label [hand_num, 5] id, cx, cy, w, h
        empty_gesture_bbox = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        gesture_bbox_label = [
            empty_gesture_bbox.copy() for _ in range(self.max_hand_num)
        ]
        
        '''
        kc + 1 是因为需要在 heatmaps_label 最后添加一张绘制了所有的点的 heatmap。
        heatmaps_label 除最后一张之外，从索引 0 到索引 20 分别对应了关键点的类别。
        除最后一张之外，每一张 heatmap 对应一类关键点。
        '''
        
        # load process data
        # draw all keypoints in the last heatmap of heatmaps_label
        hand_count = 0
        for single_hand_data in leb_json:
            points_json = single_hand_data['points']
            x_cache = []
            y_cache = []
            for keypoint in points_json:
                key_index = int(keypoint['id'])
                x = float(keypoint['x'])
                x = x if 0 <= x <= 1 else 1
                y = float(keypoint['y'])
                y = y if 0 <= y <= 1 else 1
                target_allkp_heatmap = heatmaps_label[-1]  # 该 heatmap 有所有的关键点
                target_index_heatmap = heatmaps_label[key_index]  # 该 heatmap 只有该类关键点
                
                x_cache.append(x)
                y_cache.append(y)
                
                int_x = int(original_width * x - 1)
                int_x = int_x if int_x <= (original_width - 1) else original_width - 1
                int_y = int(original_height * y - 1)
                int_y = int_y if int_y <= (original_height - 1) else original_height -1 
                
                target_allkp_heatmap[int_y][int_x] = 255  # height, width
                target_index_heatmap[int_y][int_x] = 255  # height, width

                heatmaps_label[-1] = target_allkp_heatmap
                heatmaps_label[key_index] = target_index_heatmap
            
            # get cx, cy, w, h
            max_x = max(x_cache)
            min_x = min(x_cache)
            max_y = max(y_cache)
            min_y = min(y_cache)
            
            w = max_x - min_x
            h = max_y - min_y
            cx = w/2 + min_x
            cy = h/2 + min_y
            single_hand_bbox_label = [ni, cx, cy, w, h]
            gesture_bbox_label[hand_count] = np.array(single_hand_bbox_label)
            hand_count += 0
            if hand_count < self.max_hand_num:
                continue
            break
                   
        # # GaussianBlur and resize
        for heatmap_index in range(len(heatmaps_label)):
            heatmap = heatmaps_label[heatmap_index]
            gaussian_kernel = (25, 25)
            heatmap = cv2.GaussianBlur(heatmap, gaussian_kernel, 1, 0)
            heatmap_amax = np.max(heatmap)
            if heatmap_amax != 0:
                heatmap /= heatmap_amax / 255
            heatmap_amax = np.max(heatmap)
            # heatmap /= 255
            resize_heatmap = cv2.resize(heatmap, (self.width, self.height), cv2.INTER_AREA)
            # cv2.imshow("resize heatmap", resize_heatmap)
            # cv2.waitKey()
            # print(resize_heatmap.shape)
            heatmaps_label[heatmap_index] = resize_heatmap
        
        # # convert data to tensor -------------
        tensor_img = transforms.ToTensor()(resize_img)
        tensor_heatmap_label = torch.tensor(np.array(heatmaps_label), dtype=torch.float32)
        tensor_bbox_label = torch.tensor(np.array(gesture_bbox_label), dtype=torch.float32)

        return tensor_img, tensor_heatmap_label, tensor_bbox_label

    def __len__(self):
        return len(self.datapack)
    
    def get_max_hand_num(self) -> int:
        return self.max_hand_num
    
    def get_kc(self) -> int:
        return self.kc
    
    @staticmethod
    def load_data(names, datasets_path, limit:int = None):
        # Load All Images Path, Labels Path and name index
        datapack: list = []  # [[img, leb, name_index]...]
        search_path: list = []
        
        names_length = len(names) + 1
        
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
                    name_index = names.index(name) / names_length
                    
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
    