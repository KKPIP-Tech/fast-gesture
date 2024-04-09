import os
import sys
import yaml
import json
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from random import shuffle
from typing import List, TypedDict, Union

sys.path.append(Path(__file__).parent.parent.absolute().__str__())

import cv2
from PIL import Image
from cv2 import Mat
import numpy as np

from fastgesture.data.generate import get_vxvyd
# from generate import get_vxvyd


class Points(TypedDict):
    point_id: int
    x: float
    y: float
    z: float
    
class PrepocessLabel(TypedDict):
    hand_label: int
    gesture: int
    points: Points
    control_point:tuple
    bbox: list

class NormalizationCoefficient(TypedDict):
    point_id: int
    x_coefficient: float
    y_coefficient: float
    
class PointsNC(TypedDict):
    points_number: int
    ncs: List[NormalizationCoefficient]
    
class GetPNCS:
    def __init__(self, config_file:str, img_size:Union[int, list], save_path:str=None, value_thres:List[float]=[0.8, 0.8]) -> None:
        
        print(rf"The dataset profile has been successfully loaded from \"{config_file}\"")
        
        if isinstance(img_size, int):
            self.height, self.width = img_size, img_size
        elif isinstance(img_size, list):
            self.height, self.width = img_size[1], img_size[0] 
        else:
            raise ValueError("img_size is not int or list")
        
        self.x_value_thres:float = value_thres[0]
        self.y_value_thres:float = value_thres[1]
        
        with open(config_file) as file:
            config = yaml.safe_load(file)

        self.datasets_path:str = config['root']
        print(rf"The datasets path is \"{self.datasets_path}\"")
        self.datapack:list = self.load_data(
            datasets_path=self.datasets_path,
            limit=10000
        )
        shuffle(self.datapack)
        self.target_points_id:list = config['target_points_id']
        self.save_path:str = save_path
        
        self.image_hw_info:list = []
        self.points_counter:int = 0
        
        template: NormalizationCoefficient = {
            "point_id": None,
            "x_coefficient": 0,
            "y_coefficient": 0
        }
        self.ncs:List[NormalizationCoefficient] = [deepcopy(template) for _ in range(len(self.target_points_id))]
    
    def get_pncs(self) -> PointsNC:
        
        exists_datasets_info = self.datasets_path + "/datasets_info.json"
        if os.path.exists(exists_datasets_info):
            with open(exists_datasets_info) as label_file:
                label_json = json.load(label_file)
            return label_json
        
        x_coe_list = [deepcopy([]) for _ in range(len(self.target_points_id))]
        y_coe_list = [deepcopy([]) for _ in range(len(self.target_points_id))]
        
        for img_path, leb_path in tqdm(self.datapack, desc="Loading Data", unit=" items"):
            # image = cv2.imread(img_path)
            
            with Image.open(img_path) as img_pil:
                img_array = np.array(img_pil)
            image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_height, img_width = image.shape[:2]
            self.image_hw_info.append([img_height, img_width])
            
            scale_ratio = min(self.height / img_height, self.width / img_width)
            scale_ratio = min(scale_ratio, 1.0)
            
            # print(f"scale ratio {scale_ratio}")
            
            del img_array, image, img_height, img_width
            
            with open(leb_path) as label_file:
                label_json = json.load(label_file)
            
            for single_hand_data in label_json:
                points_json = single_hand_data['points']
                
                x_cache = []
                y_cache = []
                for keypoint in points_json:
                    self.points_counter += 1
                    x = int(keypoint['x'])*scale_ratio
                    y = int(keypoint['y'])*scale_ratio
                    x_cache.append(x)
                    y_cache.append(y)
                xmax = max(x_cache)
                xmin = min(x_cache)
                center_x = xmin + ((xmax - xmin)/2)
                
                mid_x = sum([abs(x) for x in x_cache])/len(x_cache)
                
                ymax = max(y_cache)
                ymin = min(y_cache)
                center_y = ymin + ((ymax - ymin)/2)
                
                mid_y = sum([abs(x) for x in y_cache])/len(y_cache)
                
                control_point = (center_x, center_y)
                
                for keypoint in points_json:
                    point_id = int(keypoint['id'])
                    x = int(keypoint['x'])*scale_ratio
                    y = int(keypoint['y'])*scale_ratio
                    point_a:tuple = (x, y)
                    
                    vx, vy, dis = get_vxvyd(point_a=point_a, control_point=control_point)
                    
                    x_coe_list[point_id].append(abs(vx))
                    y_coe_list[point_id].append(abs(vy))
                    
        for index, (vx_list, vy_list) in tqdm(enumerate(zip(x_coe_list, y_coe_list)), desc="Calculating COE", unit=" classes"):
            
            # print(index)
            self.ncs[index]["point_id"] = index
            
            vx_list = sorted([abs(num) for num in tqdm(vx_list, desc="Processing VX List", unit=" points")], reverse=False)
            vy_list = sorted([abs(num) for num in tqdm(vy_list, desc="Processing VY List", unit=" points")], reverse=False)
            x_coe = vx_list[int(len(vx_list)*self.x_value_thres)]
            y_coe = vy_list[int(len(vy_list)*self.y_value_thres)]
            
            # x_coe = self.find_peak_density_values(vx_list)
            # y_coe = self.find_peak_density_values(vy_list)
            
            self.ncs[index]["x_coefficient"] = x_coe
            self.ncs[index]["y_coefficient"] = y_coe
            
        result:PointsNC = {
            "points_number": self.points_counter,
            "ncs": self.ncs
        }          
        
        for index, line in enumerate(self.ncs):
            print(f"{index} Point NC: {line}")
        
        with open(self.datasets_path + "/" + "datasets_info.json", "w") as new_json:
            json.dump([result], new_json, indent=4)
        
        return result
                
    @staticmethod
    def load_data(datasets_path, limit:int = -1):
        # Load All Images Path, Labels Path and name index
        datapack: list = []  # [[img, leb, name_index]...]
        search_path: list = []
        
        # for name in names:
            # get all search dir by name index
        target_image_path = datasets_path + '/images/'
        target_label_path = datasets_path + '/labels/'
        search_path.append([target_image_path, target_label_path])
        
        for target_image_path, target_label_path in search_path:
            
            for path_pack in os.walk(target_image_path):
                for filename in tqdm(path_pack[2], desc="Loading Files"):
                    img = target_image_path + filename
                    label_name = filename.replace(".jpg", ".json")
                    leb = target_label_path + label_name
                    # name_index = names.index(name)
                    
                    datapack.append(
                        [img, leb]
                    )
                    # index_count += 1
        shuffle(datapack)
        return datapack[:limit]
    
if __name__ == "__main__":
    a = GetPNCS(
        config_file="./data/config.yaml",    
        img_size=160,
        save_path=123
    )
    
    print(a.get_pncs())