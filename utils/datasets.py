import os
import yaml
import sys
import cv2
import json
from PIL import Image
from time import time
import numpy as np
import random
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())

import torch
import torch.utils.data
from torchvision import transforms
from torchvision.transforms import functional as F

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
        
        # set data transforms
        seed_value = random.randint(1, 10000000)
        img_transforms = self.create_transforms(fill=114)
        
        # image process ---------------------
        original_img = cv2.imread(img_path)
        original_height, original_width = original_img.shape[:2]
        
        # canny, drawContours
        grey_img = cv2.cvtColor(original_img.copy(), cv2.COLOR_BGR2GRAY)
        # canny_img = cv2.Canny(grey_img, 0, 100, 80)
        # contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(original_img,contours,-1,(0,0,255),2) 
        resize_img = cv2.resize(grey_img, (self.width, self.height), cv2.INTER_AREA) 
        pil_img = Image.fromarray(resize_img)
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
        resize_img = img_transforms(pil_img)
        
        # cv2.imshow("Canny_image", resize_img)
        # cv2.waitKey(1)
        
        # labels process --------------------
        with open(leb_path) as label_file:
            leb_json = json.load(label_file)
        
        # Keypoints Label Shape [kc, self.height, self.width]
        empty_heatmap_original_size = np.zeros((original_height, original_width))
        heatmaps_label = [
            empty_heatmap_original_size.copy() for _ in range(self.kc + 1)
        ] 
        
        # labels
        # labels.append(np.array([ni, self.height, self.width]))
        empty_label = np.zeros((original_height, original_width))
        labels = [empty_label.copy() for _ in range(self.nc)]
        # bboxes xy xy
        bboxes = []
        #objects
        objects = np.zeros((self.height, self.width), dtype=np.float32)
        
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
                labels_temp = labels[ni]
                # key_index = self.get_keypoints_index(id=id)
                # if key_index is None:
                #     continue
                
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
                labels_temp[int_y][int_x] = 1

                heatmaps_label[-1] = target_allkp_heatmap
                heatmaps_label[key_index] = target_index_heatmap
                labels[ni] = labels_temp
            
            # get cx, cy, w, h
            max_x = int(max(x_cache))
            min_x = int(min(x_cache))
            max_y = int(max(y_cache))
            min_y = int(min(y_cache))
            
            w = max_x - min_x
            h = max_y - min_y
            cx = w/2 + min_x
            cy = h/2 + min_y
            single_hand_bbox_label = np.array([cx, cy, h, w])
            bboxes.append(single_hand_bbox_label)
            objects[min_y:max_y, min_x:max_x] = 1.0
            hand_count += 0
            if hand_count < self.max_hand_num:
                continue
            break
        
        # # GaussianBlur and resize
        for heatmap_index in range(len(heatmaps_label)):
            heatmap = heatmaps_label[heatmap_index]
            gaussian_kernel = (35, 35)
            heatmap = cv2.GaussianBlur(heatmap, gaussian_kernel, 1, 0)
            heatmap_amax = np.max(heatmap)
            if heatmap_amax != 0:
                heatmap /= heatmap_amax / 255
            heatmap_amax = np.max(heatmap)
            # print(f"heatmap amax {heatmap_amax}")
            # heatmap /= 255
            resize_heatmap = cv2.resize(heatmap, (self.width, self.height), cv2.INTER_AREA)
            # cv2.imshow("resize heatmap", resize_heatmap)
            # cv2.waitKey()
            # print(resize_heatmap.shape)
            heatmaps_label[heatmap_index] = resize_heatmap
        
        for label_index in range(len(labels)):
            label_temp = labels[label_index]
            label_temp = cv2.resize(label_temp, (self.width, self.height), cv2.INTER_LINEAR)
            labels[label_index] = label_temp
        
        # get transformed heatmaps
        transformed_heatmaps = []
        heatmap_transforms = self.create_transforms(fill=0)
        for heatmap in heatmaps_label:
            heatmap[heatmap > 0.1] = 255
            heatmap = F.to_pil_image((heatmap).astype(np.uint8))
            random.seed(seed_value)
            np.random.seed(seed_value)
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)
            transformed_heatmap = heatmap_transforms(heatmap)
            transformed_heatmaps.append(transformed_heatmap)
        # 转换处理后的热图为张量
        
        tensor_heatmap_label = torch.stack(transformed_heatmaps)
        
        # # convert data to tensor -------------
        # tensor_img = transforms.ToTensor()(resize_img)
        # tensor_heatmap_label = torch.tensor(np.array(heatmaps_label), dtype=torch.float32)
        tensor_labels = torch.tensor(np.array(labels), dtype=torch.float32)
        tensor_bboxes = torch.tensor(np.array(bboxes), dtype=torch.float32)
        tensor_objects = torch.tensor(objects, dtype=torch.float32)

        # print(f"tensor_heatmap_label[-1] max {torch.max(tensor_heatmap_label[-1])}")
        # cv2.imshow("Transformed Heatmap", tensor_heatmap_label[-1].cpu().detach().squeeze(0).numpy().astype(np.float64))
        # cv2.imshow("Transformed Image", resize_img.cpu().detach().squeeze(0).numpy().astype(np.float64))
        # cv2.waitKey(1)
        
        
        return resize_img, tensor_heatmap_label, tensor_labels, tensor_bboxes, tensor_objects

    def __len__(self):
        return len(self.datapack)
    
    def get_max_hand_num(self) -> int:
        return self.max_hand_num
    
    def get_kc(self) -> int:
        return self.kc
    
    def get_nc(self) -> int:
        return self.nc
    
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
    
    def create_transforms(self, fill=0):
        
        transform_list = []

        # Random horizontal flipping
        transform_list.append(transforms.RandomHorizontalFlip())

        # Random rotation
        transform_list.append(transforms.RandomRotation(30, fill=fill))

        # Random scaling
        
        scale_transform = transforms.RandomAffine(0, translate=(0.2, 0.6), scale=(0.6, 1.2), shear=0, fill=fill)
        transform_list.append(scale_transform)

        # Random cropping
        # crop_transform = transforms.RandomResizedCrop(size=(self.height, self.width), scale=(0.1, 0.6))
        # transform_list.append(crop_transform)


        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Combine all transforms
        return transforms.Compose(transform_list)
    
    @staticmethod
    def set_seed(seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

