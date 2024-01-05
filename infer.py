import torch
import torchvision.transforms as transforms
import cv2

# 加载模型
model = HandGestureNetwork(max_hand_num=5)
model.load_state_dict(torch.load('path_to_saved_model.pth'))
model.eval()

# 准备输入数据
def prepare_input(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 推理
def infer(model, image_path):
    input_tensor = prepare_input(image_path)
    with torch.no_grad():
        gesture_outputs, keypoint_outputs = model(input_tensor, None, None)

    # 解析输出
    gestures = [torch.argmax(go, dim=1).item() for go in gesture_outputs]
    keypoints = [ko.numpy() for ko in keypoint_outputs]

    return gestures, keypoints

# 示例使用
image_path = 'path_to_test_image.jpg'
gestures, keypoints = infer(model, image_path)

# 显示结果
print("Detected Gestures:", gestures)
print("Detected Keypoints:", keypoints)
