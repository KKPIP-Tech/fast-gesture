import os
import cv2
import sys
import yaml
import json
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())

import torch
import torch.utils.data
from torchvision import transforms
from torchvision.transforms import functional as F

from utils.augment import preprocess


class Datasets(torch.utils.data.Dataset):
    def __init__(self, config:str, img_size:int or list) -> None:
        
        config_file = config + "/config.yaml"
        
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
        
        self.target_points_id = config['target_points_id']
        print(f"Your Target Points ID of Mediapipe Is: {self.target_points_id}")
        
        self.datapack = self.load_data(
            names=self.namse, 
            datasets_path=self.datasets_path,
            limit=None)
    
    def __getitem__(self, index):
        
        # get image path and label path
        image_path, label_path = self.datapack[index]
        orin_image = cv2.imread(image_path)
        
        # image_preprocess ------------------------------------------------------
        letterbox_image, scale_ratio, left_padding, top_padding = preprocess(
            image=orin_image,
            is_mask=False,
            target_shape=(self.height, self.width)
        )
        
        # label process ---------------------------------------------------------
        with open(label_path) as label_file:
            label_json = json.load(label_file)
        
        for single_hand in label_json:
            gesture_type:int = int(single_hand["gesture"])
            hand_type:int = int(single_hand["hand_label"])
            points_coord:list = single_hand["points"]
            
            #
            target_points = self.points_fillter(target_id=self.target_points_id, points=points_coord)
        
            
        
        
        return 1
    
    def __len__(self):
        return len(self.datapack)
    
    def get_kc(self) -> int:
        return self.kc
    
    def get_nc(self) -> int:
        return self.nc
    
    @staticmethod
    def load_data(names, datasets_path, limit:int = None):
        # Load All Images Path, Labels Path and name index
        datapack: list = []  # [[img, leb]...]
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
                    
                    datapack.append(
                        [img, leb]
                    )
                    index_count += 1
                    if limit is None:
                        continue
                    if index_count < limit:
                        continue
                    break
        
        return datapack
    
    @staticmethod
    def points_fillter(target_id:list, points:list)->list:
        
        target_points = []
        for item in points:
            id = item["id"]
            x = item["x"]
            y = item["y"]
            z = item["z"]
            if id not in target_id:
                continue
            
            target_id.append({
                "id":id,
                "x": x,
                "y": y,
                "z": z
            })
        return target_points
            
