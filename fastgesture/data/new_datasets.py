import os
import sys
import yaml 
import json
from typing import List, TypedDict, Union
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

from fastgesture.data.augment import (
    letterbox, 
    rotate_image, 
    rotate_point, 
    resize_image, 
    resize_point,
    translate,
    exposure
)
from fastgesture.data.generate import get_vxvyd, inverse_vxvyd
from fastgesture.data.point_average_value import PointsNC, NormalizationCoefficient

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
    
class TrainDatasets(torch.utils.data.Dataset):
    def __init__(self, config_file:str, img_size:Union[int, list], pncs:PointsNC, limit:int=None) -> None:
        
        # load datasets config ----------------------------
        with open(config_file) as file:
            config = yaml.safe_load(file)

        self._train_datasets_path:str = config['train_set']
        print(f"Train Datasets Path: {self._train_datasets_path}")
        
        self._keypoints_cls_num:int = config['keypoints_classes_num']
        self._keypoint_radius:int = config['keypoint_radius']
                
        # load augmentation params ------------------------
        self._exposure_limit:list[int] = config['exposure']
        self._degree_limit:int = config['degree']
        self._sp_noise_limit:float = config['sp_noise']
        self._scale_limit:list[float] = config['scale']
        self._translate_limit:float = config['translate']
        
        # set image size and pncs -------------------------
        if isinstance(img_size, int):
            self._height, self._width = img_size, img_size
        elif isinstance(img_size, list):
            self._height, self._width = img_size[1], img_size[0] 
        else:
            raise ValueError("img_size is not int or list")
        
        self._pncs:PointsNC = pncs[0]
        
        # load image:label from target path ---------------
        self._datapack = self._load_data(datasets_path=self._train_datasets_path, limit=limit)
    
    def __len__(self) -> int:
        return len(self._datapack)
    
    def get_keypoints_class_number(self) -> int:
        return self._keypoints_cls_num
    
    def __getitem__(self, index:int):    
        
        image_path, label_path = self._datapack[index]
        
        # generating data augmentation parameters ---------
        _degree:int = random.randint(-self._degree_limit, self._degree_limit)
        _exposure:int = random.randint(self._exposure_limit[0], self._exposure_limit[1])
        _sp_noise:float = random.uniform(0, self._sp_noise_limit)
        _scale:float = random.uniform(self._scale_limit[0], self._scale_limit[1])
        _translate:int = random.randint(0, int(self._height*self._translate_limit))
        
        # image preprocess --------------------------------
        _original_image = cv2.imread(image_path)
        
        # exposure
        _original_image = exposure(image=_original_image, exposure=_exposure)
        
        # sp_noise
        _original_image = self._sp_noise(image=_original_image, proportion=_sp_noise)
        
        # rotate
        _original_image, _rotate_matrix = rotate_image(image=_original_image, angle=_degree)
        
        # resize
        (_h, _w) = _original_image.shape[:2]
        _original_center = (_w // 2, _h // 2)
        _original_image = resize_image(image=_original_image, scale_factor=_scale, center=_original_center, fill_color=114)
        _original_height, _original_width = _original_image.shape[:2]
        _grey_image = cv2.cvtColor(_original_image, cv2.COLOR_BGR2GRAY)
        del _original_image
        
        # letterbox
        _letterbox_image, _scale_ratio, _left_padding, _top_padding = letterbox(
            image=_grey_image, target_shape=(self._width, self._height),
            fill_color=114, single_channel=True
        )
        del _grey_image
        
        # offset
        _letterbox_image, self._trans_off_x, self._trans_off_y = translate(
            image=_letterbox_image, max_offset=_translate, fill=114
        )
        self._draw_copy = cv2.cvtColor(deepcopy(_letterbox_image), cv2.COLOR_GRAY2BGR)
        
        # label process -----------------------------------
        with open(label_path) as label_file:
            _label_json = json.load(label_file)
        
        _preprocess_points_info:List[PrepocessLabel] = []
        for single_hand_data in _label_json:
            _points_json = single_hand_data['points']
            _gesture = single_hand_data['gesture']
            _hand_label = single_hand_data['hand_type']

            _new_points_info = []
            _x_cache = []
            _y_cache = []
            
            for keypoint in _points_json:
                # original points data
                x = int(keypoint['x'])
                y = int(keypoint['y'])
                
                # rotate
                x, y = rotate_point((x, y), _rotate_matrix)
                
                # center scale
                x, y = resize_point((x, y), scale_factor=_scale, image_shape=(_original_height, _original_width))
                
                # letterbox
                x = (x*_scale_ratio) + _left_padding + self._trans_off_x 
                y = (y*_scale_ratio) + _top_padding + self._trans_off_y
                
                point = {
                    "point_id": int(keypoint['id']),
                    "x": x,
                    "y": y,
                }
                _new_points_info.append(point)
                _x_cache.append(x)
                _y_cache.append(y)
            
            # get control point
            xmax, xmin = max(_x_cache), min(_x_cache)
            bbox_width = xmax - xmin
            center_x = xmin + ((xmax - xmin)/2)
            
            ymax, ymin = max(_y_cache), min(_y_cache)
            bbox_height = ymax - ymin
            center_y = ymin + ((ymax - ymin)/2)
            
            control_point = (center_x, center_y)
            
            cv2.circle(self._draw_copy, (int(center_x), int(center_y)), 3, (0, 255, 0), 1)

            single_hand_data:PrepocessLabel = {
                'hand_label': _hand_label,
                'gesture': _gesture,
                'points': _new_points_info,
                'control_point': control_point,
                'bbox': [center_x, center_y, bbox_width, bbox_height]
            }
            _preprocess_points_info.append(single_hand_data)
        
        # get keypoints classification heatmaps -----------
        keypoint_classfication_label:list = self._generate_keypoints_heatmap(points_info=_preprocess_points_info)
        ascription_field, ascription_mask = self._get_ascription(points_info=_preprocess_points_info)
        
        # convert data to tensor
        tensor_letterbox_img = transforms.ToTensor()(_letterbox_image)
        tensor_kp_cls_labels = torch.tensor(np.array(keypoint_classfication_label), dtype=torch.float32)
        tensor_ascription_field = torch.tensor(np.array(ascription_field), dtype=torch.float32)
        tensor_ascription_mask = transforms.ToTensor()(ascription_mask)
        
        return deepcopy(_letterbox_image), tensor_letterbox_img, tensor_kp_cls_labels, tensor_ascription_field, tensor_ascription_mask
            
    def _generate_keypoints_heatmap(self, points_info:List[PrepocessLabel]) -> list:
        
        empty_heatmap = np.zeros((self._height, self._width))
        keypoints_classification_label:list = [deepcopy(empty_heatmap) for _ in range(self._keypoints_cls_num)]
        
        for one_hand in points_info:
            points:List[Points] = one_hand["points"]
            for point in  points:
                point_id:int = point['point_id']
                x:int = int(point['x'])
                if x >= self._width: continue
                y:int = int(point['y'])
                if y >= self._height: continue
                
                temp_heatmap = keypoints_classification_label[point_id]
                cv2.circle(temp_heatmap, (x, y), self._keypoint_radius, 1, -1)
                keypoints_classification_label[point_id] = temp_heatmap
        
        return keypoints_classification_label
    
    def _get_ascription(self, points_info:List[PrepocessLabel]) -> list:
        
        empty_field_map = np.zeros((self._height, self._width))
        ascription_field:list = [deepcopy(empty_field_map) for _ in range(self._keypoints_cls_num*2+2)]
        ascription_mask = deepcopy(empty_field_map)
        
        ncs:List[NormalizationCoefficient] = self._pncs["ncs"]
        for one_hand in points_info:
           empty_zero_map = deepcopy(empty_field_map) 
           
           control_point = one_hand["control_point"]
           
           control_point_x = int(control_point[0])
           if control_point_x >= self._width: continue
           control_point_y = int(control_point[1])
           if control_point_y >= self._height: continue
           control_point:tuple = (control_point_x, control_point_y)
           
           points:List[Points] = one_hand["points"]
           for point in points:
                point_id:int = point["point_id"]
                x:int = int(point['x'])
                if x >= self._width: continue
                y:int = int(point['y'])
                if y >= self._height: continue
                point_a = (x, y)
                
                # get target point coe
                x_coe:float = ncs[point_id]["x_coefficient"]
                y_coe:float = ncs[point_id]["y_coefficient"]

                points_on_zero = deepcopy(empty_zero_map)
                
                # calculate the pixel range of the connecting area between two points
                cv2.circle(points_on_zero, point_a, self._keypoint_radius, (255, 255, 255), -1)
                cv2.circle(points_on_zero, control_point, self._keypoint_radius, (255, 255, 255), -1)
                
                dx = control_point[0] - point_a[0]
                dy = control_point[1] - point_a[1]
                angle = np.arctan2(dy, dx)
                
                # 计算长方形的四个顶点
                offset_x = self._keypoint_radius * np.sin(angle)
                offset_y = self._keypoint_radius * np.cos(angle)
                p1 = (int(point_a[0] - offset_x), int(point_a[1] + offset_y))
                p2 = (int(point_a[0] + offset_x), int(point_a[1] - offset_y))
                p3 = (int(control_point[0] + offset_x), int(control_point[1] - offset_y))
                p4 = (int(control_point[0] - offset_x), int(control_point[1] + offset_y))

                # 绘制长方形
                pts = np.array([p1, p2, p3, p4], np.int32)
                cv2.fillPoly(points_on_zero, [pts], (255, 255, 255))
                
                y_coords, x_coords = np.where(points_on_zero > 1)
                
                vx, vy, dis = get_vxvyd(point_a=point_a, control_point=control_point)
                # print(f"vx, vy: {vx, x_coe, vy, y_coe}")
                vx = vx/x_coe
                vy = vy/y_coe
                
                x_minus:bool = True if vx < 0 else False
                y_minus:bool = True if vy < 0 else False
                
                eval_vx, eval_vy = vx*x_coe, vy*y_coe
                end_x, end_y = inverse_vxvyd((x, y), eval_vx, eval_vy)
                cv2.line(self._draw_copy, (x, y), (int(end_x), int(end_y)), (255, 0, 0), 1, 1)
                
                for blur_x, blur_y in zip(x_coords, y_coords):
                    ascription_field[point_id][blur_y][blur_x] = -vx if x_minus else vx
                    ascription_field[point_id + self._keypoints_cls_num][blur_y][blur_x] = -vy if y_minus else vy

                    if x_minus: 
                        ascription_field[-2][blur_y][blur_x] = 1
                    if y_minus: 
                        ascription_field[-1][blur_y][blur_x] = 1
                    
                    ascription_mask[blur_y][blur_x] = 1
        ascription_field = np.array(ascription_field, dtype=np.float64)
        
        return ascription_field, ascription_mask
        
    @staticmethod
    def _load_data(datasets_path:str, limit:int=None) -> list:
        
        datapack:list = []
        search_path:list = []
        
        target_image_path = datasets_path + '/images/'
        target_label_path = datasets_path + '/labels/'
        search_path.append([target_image_path, target_label_path])
        
        for target_image_path, target_label_path in search_path:
            for path_pack in os.walk(target_image_path):
                for filename in path_pack[2]:
                    image = target_image_path + filename
                    label_name = filename.replace(".jpg", ".json")
                    label = target_label_path + label_name

                    datapack.append([image, label])

        random.shuffle(datapack)
        if limit is None: return datapack
        return datapack[:limit]
    
    @staticmethod
    def _sp_noise(image: np.ndarray, proportion:float) -> np.ndarray:
        height, width = image.shape[:2]  # 获取高度宽度像素值
        num = int(height * width * proportion)  # 准备加入的噪声点数量

        # 生成随机位置
        w_random = np.random.randint(0, width, num)
        h_random = np.random.randint(0, height, num)

        # 生成随机噪声类型（黑或白）
        noise_type = np.random.randint(0, 2, num)

        # 对于噪声类型为0的，生成较暗的值
        mask = noise_type == 0
        image[h_random[mask], w_random[mask]] = np.random.randint(50, 200, (mask.sum(), 3))

        # 对于噪声类型为1的，生成较亮的值
        mask = noise_type == 1
        image[h_random[mask], w_random[mask]] = np.random.randint(200, 254, (mask.sum(), 3))

        return image