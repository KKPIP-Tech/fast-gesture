# Fast Gesture

## Quick Start
<details open>
<summary>Install</summary>

**recommend install cuda 11.8 with torch 2.2.1 use follow command**: 
```
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```

if you want to use default env setting:

```
pip install -r requirements.txt
```

</details>

<details open>
<summary>Scratch on custom data</summary>

**Label Structure**
```json
[
  {
    "gesture": 2,
    "hand_type": 0,
    "points": [
      { "id": 0, "x": 686, "y": 470 },
      { "id": 1, "x": 632, "y": 408 },
      { "id": 2, "x": 612, "y": 341 },
      { "id": 3, "x": 690, "y": 398 },
      { "id": 4, "x": 681, "y": 295 },
      { "id": 5, "x": 722, "y": 398 },
      { "id": 6, "x": 716, "y": 283 },
      { "id": 7, "x": 740, "y": 399 },
      { "id": 8, "x": 724, "y": 297 },
      { "id": 9, "x": 749, "y": 400 },
      { "id": 10, "x": 734, "y": 320 }
    ],
    "bbox": [592, 255, 769, 498]
  },
  {
    "gesture": 0,
    "hand_type": 0,
    "points": [
      { "id": 0, "x": 932, "y": 402 },
      { "id": 1, "x": 850, "y": 332 },
      { "id": 2, "x": 862, "y": 277 },
      { "id": 3, "x": 844, "y": 292 },
      { "id": 4, "x": 812, "y": 188 },
      { "id": 5, "x": 875, "y": 278 },
      { "id": 6, "x": 875, "y": 266 },
      { "id": 7, "x": 908, "y": 276 },
      { "id": 8, "x": 905, "y": 271 },
      { "id": 9, "x": 939, "y": 282 },
      { "id": 10, "x": 932, "y": 279 }
    ],
    "bbox": [793, 156, 958, 434]
  }
]
```
In the example above, we can know that there have two hands in single picture, "gesture" and "hand_type" params have no any sense in the current version. There is also no guarantee that these two fields will be modified in the future.

Field "points" saved data of different key point class.

Field "bbox" saved data of bounding box of the target object.

*Make sure your dataset structure as follows*

**datasets structure**

```
|-- fg_datasets
|   |-- train
|   |   |-- images
|   |   |-- labels
|   |   |-- datasets_info.json
|   |
|   |-- val
|   |   |-- images
|   |   |-- labels
```
"datasets_info.json" will automatically generated when you use it for the first time.

If you change value of `img-size`, you must delet it. Otherwise, the training result is incorrect.

Single GPU
```
python train.py --batch 32 --workers 28 --save_name 20240430 --lr 0.001 
```
</details>

## Deployment
<p align="center">
<img src="./assets/Input_Output.png" height="300px"/>
</p>

Input Shape: **[batch, channel, img_size, img_size]**

Output Shape: **[maps, batch, img_size, img_size]**, 
- Key Points Heatmap: [:keypoint_cls] 
- ASF Map: [keypoint_cls:]
- - ASF X Map in ASF Map: [0:keypoint_cls]
- - ASF Y Map in ASF Map: [keypoint_cls:keypoint_cls*2]
- - ASF X Minus in ASF Map: [-2]
- - ASF Y Minux in ASF Map: [-1]

Pytorch
```
ckpt = torch.load({weight})
model = model_path['model']
pncs = model_path['pncs_result']

model.eval()
```

ONNX
```
python export_onnx.py --weights {/path/weight} --save_path {/path} --save_name {onnx file name}
```

## FAQ

If you have any question, welcome to join my QQ Group.

913211989 ( 小猫不要摸鱼 )

进群令牌：fGithub