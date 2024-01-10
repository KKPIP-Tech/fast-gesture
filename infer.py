import torch
from torchvision import transforms
import cv2
import numpy as np
from time import time
from model.net import HandGestureNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
net = HandGestureNet(max_hand_num=5, device=device)
# 加载模型
# net = HandGestureNet(max_hand_num=您的最大手部数量)  # 创建模型实例
net.load_state_dict(torch.load("run/train/exp/model_epoch_3.pt"))  # 加载模型文件
net.to(device)
net.eval()

names = ["one", "two", "five", "hold", "pickup"]

capture = cv2.VideoCapture(0)

while True:
# 读取图像并进行预处理
    st = time()
    ret, image = capture.read()
    if not ret:
        continue
    # image = cv2.imread("101.jpg")
    image = cv2.resize(image, (320, 320))  # 假设模型输入为256x256
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        outputs = net(image_tensor)

    # print(len(outputs[0]))
    class_labels, keypoints = outputs
    
    # 处理输出并在图像上绘制结果
    for output in zip(class_labels[0], keypoints[0]):
        # print(output)
        gesture_value, keypoints = output
        gesture_label = gesture_value.item()  # 假设有一个类别标签列表
        print(f"Gesture ID: {gesture_label}")
        if gesture_label > len(names):
            continue
        gesture_label = str(gesture_label)  # names[gesture_label]
        keypoints = keypoints.cpu().detach().numpy() 
        
        print(f"keypoints on cpu: \n{keypoints}")
        
        # 绘制关键点
        for x, y in keypoints:
            # x = (x+1)/2
            # y = (y+1)/2
            x, y = int(x * 320), int(y * 320)  # 将坐标转换回原图尺寸
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

        # 计算手势框坐标并绘制
        x_min, y_min = np.min(keypoints, axis=0)
        x_max, y_max = np.max(keypoints, axis=0)
        x_min, y_min, x_max, y_max = [int(val * 256) for val in [x_min, y_min, x_max, y_max]]
        x_min = int((x_min+1)/2)
        y_min = int((y_min+1)/2)
        x_max = int((x_max+1)/2)
        y_max = int((y_max+1)/2)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # 在框上方打印手势类别
        cv2.putText(image, gesture_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 显示图像
    cv2.imshow("Output", image)
    cv2.waitKey(1)
    print(f"FPS: ", (1000/(time() - st)))
# cv2.destroyAllWindows()
