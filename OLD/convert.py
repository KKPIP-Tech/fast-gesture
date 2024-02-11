import os
import sys
import yaml
import tqdm
import json
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.absolute().__str__())

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


if __name__ == "__main__":
    
    config_file = "/home/kd/Documents/Codes/fast-gesture/data/config.yaml"
    # Load Datasets Config ---------
    with open(config_file) as file:
        config = yaml.safe_load(file)
    
    datasets_path = config['root']
    namse = config['names']
    nc = int(config['nc'])
    kc = int(config['kc'])
    
    datapack = load_data(
        names=namse, 
        datasets_path=datasets_path,
        limit=None)
    
    NAMES = ["one", "two", "five", "hold", "pickup"]
    
    for img_path, leb_path, ni in tqdm.tqdm(datapack):
        
        all_gesture_coord = []
        # 打开并读取 YAML 文件
        with open(leb_path, 'r') as file:
            data =json.load(file)  # 使用 safe_load 读取 YAML 文件内容
        # print(data)
        for item in data:
            # print(f"Item: {item}")
            gesture = ni
            hand_label = item['hand_label']
            points = item['points']
            points_list = []
            for si in points:
                data = {
                    "id": si["id"],
                    "x": si["x"],
                    "y": si["y"],
                    "z": si["z"]
                }
                points_list.append(data)
            
            if hand_label == "Left":
                hand_label = 0
            else:
                hand_label = 1
            
            single_hand_result = {
                "gesture": gesture,
                "hand_label": hand_label,
                "points": points
            }
            all_gesture_coord.append(single_hand_result)
            
        # 将修改后的数据保存回 YAML 文件
        with open(leb_path, 'w') as file:
            json.dump(all_gesture_coord, file)
            
        del file
        del data
        del all_gesture_coord
        



