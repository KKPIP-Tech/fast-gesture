import os
import sys
from pathlib import Path
from typing import List, TypedDict, Tuple

sys.path.append(Path(__file__).parent.parent.absolute().__str__())

import cv2
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
from torchvision import transforms

from fastgesture.model.body import FastGesture
from fastgesture.data.augment import letterbox
from fastgesture.data.generate import inverse_vxvyd


class KeypointsCenter(TypedDict):
    keypoint_id: int
    center_x: int
    center_y: int
    conf: float
    start_coord: Tuple[int]
    end_coord: Tuple[int]


class KeypointsType(TypedDict):
    type: int
    points: List[KeypointsCenter]

                                                  
class FGInfer:
    def __init__(self, device:chr='cuda', img_size:tuple=(320, 320),
                 weights:str=None, conf:float=0.6,
                 keypoints_num:int=11, cls_num:int=5) -> None:
        
        if device is None:
            raise ValueError("device choice cannot be empty!")
        self.device = self.select_device(user_set_device=device)
        
        self.img_size = img_size
        self.conf_threshold = conf
        self.weight:str = weights
        self.keypoints_num = keypoints_num
        self.cls_num = cls_num
        
        self.model_init()  # init model
    
    def model_init(self) -> None:          
        with torch.no_grad():
            self.model = FastGesture(
                keypoints_num=self.keypoints_num,
            #   cls_num=self.cls_num
            ).to(self.device)
            
            # self.model = torch.jit.load("./run/train/20240411/weights/jit.pt").cuda()
            
            checkpoints = torch.load(self.weight)
            pre_trained = checkpoints['model']
            self.pncs = checkpoints["pncs_result"][0]["ncs"]
            self.model = pre_trained  #.load_state_dict(pre_trained, strict=True)
            # self.model.eval()
    
    def infer(self, image:np.ndarray=None) -> Tuple[np.ndarray, List[KeypointsType]]:
    
        # image pre-process ---------------------------------------------------
        if image is None:
            raise ValueError("Input Image Cannot Be Empty")
        letterbox_image, self.scale_ratio, self.left_padding, self.top_padding = letterbox(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            target_shape=(self.img_size[0], self.img_size[1]),
            fill_color=114,
            single_channel=True
        )
        del image
        # tensor image shape [1, 1, 320, 320]
        tensor_image = transforms.ToTensor()(deepcopy(letterbox_image)).to(self.device).unsqueeze(0)
        
        # infer ---------------------------------------------------------------
        forward = self.model(tensor_image)
        del tensor_image
        f_keypoints_classification, f_ascription = forward.heatmaps, forward.ascription_field
        
        
        # get keypoints from heatmaps -----------------------------------------
        self.keypoints:List[KeypointsType] = []
        for keypoints_type in range(self.keypoints_num):
            heatmap = f_keypoints_classification.permute(1, 0, 2, 3)[0][keypoints_type].cpu().detach().numpy()#.astype(np.float32)
            keypoint:KeypointsType = self.extract_heatmap_center(heatmap=heatmap, keypoints_type=keypoints_type)
            self.keypoints.append(keypoint)
        
        ascription_maps:list = []
        for asf_index in range(self.keypoints_num*2+2):  # 3 for vx, vy, dis
            ascription_field_map = f_ascription.permute(1, 0, 2, 3)[0][asf_index].cpu().detach().numpy()#.astype(np.float32)
            ascription_maps.append(ascription_field_map)
        
        self.get_asf_direction(ascription_maps=ascription_maps)
        
        return letterbox_image, self.keypoints
        
    def convert(self) -> Tuple[np.ndarray, List[KeypointsType]]:
        
        pass
        
    @staticmethod
    def select_device(user_set_device):
        # set device
        if user_set_device == 'cpu':
            device = user_set_device
        elif user_set_device == 'cuda':
            device = user_set_device if torch.cuda.is_available() else 'cpu'
        elif user_set_device == 'mps':
            device = user_set_device if torch.backends.mps.is_available() else 'cpu'
        else:
            print(f" Your Device Setting: {user_set_device} is not support!")
            device = 'cpu'
        return device
    
    def extract_heatmap_center(self, heatmap, keypoints_type) -> KeypointsType:
        binary_heatmap = np.where(heatmap > self.conf_threshold, 1, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
        
        centers:list = []
        for keypoints_id, contour in enumerate(contours):
            M = cv2.moments(contour)
            if M["m00"] == 0: continue# 避免除以零 
            centerX = int(M["m10"] / M["m00"])
            centerY = int(M["m01"] / M["m00"])
            
            # 计算质心的置信度
            confidence = heatmap[centerY, centerX]
            keypoints_info:KeypointsCenter = {
                "keypoint_id": keypoints_id,
                "center_x": centerX,
                "center_y": centerY,
                "conf": confidence,
                "start_coord": None,
                "end_coord": None,
            }
            centers.append(keypoints_info)
        
        result:KeypointsType = {
            "type":  keypoints_type,
            "points": centers
        }
        return result
    
    def get_asf_direction(self, ascription_maps:list) -> None:
        
        for type_index, single_type_keypoints in enumerate(self.keypoints):
            keypoints_type = single_type_keypoints["type"]
            keypoints_points = single_type_keypoints["points"]
            
            for point_index, point in enumerate(keypoints_points):
                keypoint_id = point["keypoint_id"]
                center_x = point["center_x"]
                center_y = point["center_y"]
                conf = point["conf"]
                
                vx = ascription_maps[keypoints_type][center_y][center_x]
                vy = ascription_maps[keypoints_type + self.keypoints_num][center_y][center_x]
                
                # vx = self.find_nearest_gaussian(ascription_maps[keypoints_type], (center_x, center_y))
                # vy = self.find_nearest_gaussian(ascription_maps[keypoints_type+self.keypoints_num], (center_x, center_y))
                
                x_coe = self.pncs[keypoints_type]["x_coefficient"]
                y_coe = self.pncs[keypoints_type]["y_coefficient"]
                
                x_minus:bool = True if ascription_maps[-2][center_y][center_x] > 0.5 else False
                y_minus:bool = True if ascription_maps[-1][center_y][center_x] > 0.5 else False
                
                print(f"max value in number {keypoints_type} asf map is: {np.max(ascription_maps[keypoints_type])}")
                # vx = vx*(self.img_size[0]*6)
                # vy = vy*(self.img_size[1]*6)
                
                vx = x_coe*vx
                if x_minus:
                    vx = vx*-1
                vy = y_coe*vy
                if y_minus:
                    vy = vy*-1
                
                # print(f"xydis {vx, vy}")
                end_x, end_y = inverse_vxvyd((center_x, center_y), vx, vy)
                new_points_info:KeypointsCenter = {
                    "keypoint_id": keypoint_id,
                    "center_x": center_x,
                    "center_y": center_y,
                    "conf": conf,
                    "start_coord": (int(center_x), int(center_y)),
                    "end_coord": (int(end_x), int(end_y))
                }
                keypoints_points[point_index] = new_points_info
                
            new_type_info:KeypointsType = {
                "type": keypoints_type,
                "points": keypoints_points
            }
            self.keypoints[type_index] = new_type_info
    
    @staticmethod
    def find_nearest_gaussian(heatmap, point):
        y_indices, x_indices = np.indices(heatmap.shape)
        distances = np.sqrt((x_indices - point[0])**2 + (y_indices - point[1])**2)
        min_distance_index = np.argmin(distances)
        nearest_value = heatmap.flatten()[min_distance_index]
        return nearest_value
    
if __name__ == "__main__":
    import time
    weight:str = "/home/kd/Documents/Codes/fast-gesture/run/train/20240412/weights/last.pt"
    
    fg_model = FGInfer(device='cuda', img_size=(160, 160), weights=weight, conf=0.3, keypoints_num=11)
    
    capture = cv2.VideoCapture(0)
    
    img = "/home/kd/WD_1_Data4T/Datasets/AmountData3/images/OpenGesture_89388240229.jpg"
    
    
    while True:
        st = time.time()
        frame = cv2.imread(img)       
        ret, frame = capture.read()
        cv2.imshow(f"Frame", frame)
        letterbox_image, keypoints = fg_model.infer(image=frame)
        letterbox_image = cv2.cvtColor(letterbox_image, cv2.COLOR_GRAY2BGR)
        for keypoint in keypoints:
            keypoints_type = keypoint["type"]
            points = keypoint["points"]
            # print(f"type: {keypoints_type}|points: {points}")
            for point in points:
                center_x = point["center_x"]
                center_y = point["center_y"]
                conf = point["conf"]
                start_coord = point["start_coord"]
                end_coord = point["end_coord"]
                cv2.circle(letterbox_image, (center_x, center_y),
                    1, (0, 0, 255), -1, 1
                )
                cv2.putText(letterbox_image, f"{str(keypoints_type)}|{conf:.2f}",
                    (center_x, center_y), 1, 1, (0, 0, 255), 1, 1
                )
                if start_coord is None or end_coord is None:
                    continue
                cv2.line(letterbox_image, start_coord, end_coord, (0, 255, 0), 1, 1)
                
        cv2.imshow("Draw", cv2.resize(letterbox_image, (640, 640)))
        cv2.waitKey(1)
        
        print(f"FPS: {1/(time.time() - st)}")
            
    
    
