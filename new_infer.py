import torch
import torch.nn.functional as F
import cv2
import time
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from model.new_net import FastGesture, MLPUNET
import torch.quantization

def non_max_suppression(boxes, scores, threshold=0.5):
    """
    非极大值抑制 (NMS) 实现
    :param boxes: bounding boxes, 形状 [num_boxes, 4]
    :param scores: 每个框的置信度, 形状 [num_boxes]
    :param threshold: 重叠阈值
    :return: 保留下来的框的索引
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        idxs = (overlap <= threshold).nonzero().squeeze()
        order = order[idxs + 1]

    return torch.LongTensor(keep)

# Get BBox, label, conf
def post_process_v2(heatmaps, class_scores, bboxes, obj_scores, conf_threshold=0.5, nms_threshold=0.4):
    batch_size = class_scores.size(0)
    num_classes = class_scores.size(1)
    
    obj_scores = torch.sigmoid(obj_scores)
    heatmaps = torch.sigmoid(heatmaps[-1])

    final_results = []

    # 假设 bboxes 的形状是 [batch_size, N, 4]，其中 N 是边界框的数量
    for i in range(batch_size):
        per_image_results = []
        for n in range(bboxes.size(1)):  # 遍历每个边界框
            box = bboxes[i, n]  # 获取单个边界框
            for c in range(num_classes):
                # 提取对应于边界框区域的类别得分
                # 这需要一些方法来确定哪些类别得分属于这个边界框
                # 下面是一个简化的示例，您需要根据您的实际情况调整
                class_score = class_scores[i, c].mean()  # 假设简单取均值
                
                if class_score > conf_threshold:
                    x1, y1, x2, y2 = box
                    total_score = class_score.item()  # 置信度
                    per_image_results.append([c, x1.item(), y1.item(), x2.item(), y2.item(), total_score])

        final_results.append(per_image_results)

    return final_results





# 使用函数提取信息
# results = post_process_v2(heatmaps, class_scores, bboxes, obj_scores)


# 加载模型
def load_model(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    return model


# 图像预处理
def preprocess_image(image):
    # image = cv2.imread(image_path)
    # resize_img = cv2.resize(image, (320, 320))
    
    # # canny, drawContours
    # grey_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    # canny_img = cv2.Canny(grey_img, 0, 100, 80)
    # contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image,contours,-1,(0,0,255),2) 
    resize_img = cv2.resize(image, (320, 320), cv2.INTER_AREA) 
    resize_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("resize img", resize_img)
    
    tensor_img = transforms.ToTensor()(resize_img)
    return tensor_img.unsqueeze(0)  # 增加batch维度


# 推理并绘制边界框
def infer_and_draw_boxes(image, model, device):
    
    
    image = image.to(device)
    
    with torch.no_grad():
        # heatmaps, class_scores, bboxes, obj_scores = model(image)
        heatmaps = model(image)
    
    image_to_show = heatmaps[-1][0].cpu().detach().numpy().astype(np.float32)

    # # 确保图像是单通道的，尺寸为 (320, 320)
    image_to_show = image_to_show[0, :, :]

    # # 转换数据类型并调整像素值范围
    # # image_to_show = (image_to_show).astype(np.uint8)
    # # print("MAX: ", np.max(image_to_show))
    # # 显示图像
    image_to_show = cv2.resize(image_to_show, (200, 200))
    cv2.imshow(f"Forward {-1}", image_to_show)
    cv2.waitKey(1) # 等待按键事件
    
    # print(f"obj_scores: {obj_scores}")

    # results = post_process_v2(heatmaps, class_scores, bboxes, obj_scores)

    # 将图像转换为PIL图像，用于绘制边界框
    # orig_image = Image.open(image_path)
    # draw = ImageDraw.Draw(orig_image)
    # font = ImageFont.load_default()

    # for box in results[0]:  # 假设批次大小为1
    #     class_id, x1, y1, x2, y2, confidence = box
    #     draw.rectangle(((x1, y1), (x2, y2)), outline='red', width=2)
    #     draw.text((x1, y1), f'{class_id} {confidence:.2f}', fill='blue', font=font)

    # orig_image.show()  # 显示图像
    # orig_image.save('output.jpg')  # 保存图像


# 主程序
if __name__ == '__main__':
    device = "cuda"#torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    # model = FastGesture(detect_num=22, heatmap_channels=1, num_classes=5)  # 初始化模型，假设FastGesture是您的模型类
    model = MLPUNET(22)
    model = model.to(device)

    model = load_model('/home/kd/Documents/Codes/fast-gesture/run/train/exp62/last.pt', model)  # 加载模型
    model.eval()  # 设置模型为评估模式
    
    image_path = '101.jpg'  # 测试图像路径
    
    capture = cv2.VideoCapture(0)
    
    while True:
        ret, image = capture.read()
        if not ret:
            continue
        st = time.time()
        image = preprocess_image(image)
        infer_and_draw_boxes(image, model, device)
        print(1/(time.time()-st))
