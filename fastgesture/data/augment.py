import cv2
import random
import numpy as np
from scipy.ndimage import affine_transform


def letterbox(image, target_shape=(640, 640), fill_color=(72, 72, 72), single_channel=False):
    shape = image.shape[:2]
    if isinstance(target_shape, int):
        target_shape = [target_shape, target_shape]
    elif isinstance(target_shape, list):
        target_shape = [target_shape[0], target_shape[1]]
    if target_shape[0] != target_shape[1]:
        raise ValueError("Letterbox target shape width and height not same")
    
    scale_ratio = min(target_shape[0] / shape[0], target_shape[1] / shape[1])
    scale_ratio = min(scale_ratio, 1.0)
    new_unpad = int(round(shape[1] * scale_ratio)), int(round(shape[0] * scale_ratio))
    dw, dh = target_shape[1] - new_unpad[0], target_shape[0] - new_unpad[1]  # wh padding
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_AREA)
    top_padding, bottom_padding = int(round(dh - 0.1)), int(round(dh + 0.1))
    left_padding, right_padding = int(round(dw - 0.1)), int(round(dw + 0.1))
    if single_channel:
        # 如果是遮罩图像
        image = cv2.copyMakeBorder(
            image, 
            top_padding, 
            bottom_padding, 
            left_padding, 
            right_padding, 
            cv2.BORDER_CONSTANT, value=fill_color)  # add border
        
        return image, scale_ratio, left_padding, top_padding
    else:
        # 如果不是遮罩图像
        image = cv2.copyMakeBorder(
            image, 
            top_padding, 
            bottom_padding, 
            left_padding, 
            right_padding, 
            cv2.BORDER_CONSTANT, value=fill_color)  # add border
        
        return image, scale_ratio, left_padding, top_padding


def shearing_img(image:np.ndarray, shearing_factor:float, axis:int=0) -> np.ndarray:
    """
    对一张灰度图进行Shearing操作。
    
    参数:
    - image: 输入的灰度图像，numpy数组格式。
    - shearing_factor: Shearing系数，决定了拉伸或压缩的程度。
    - axis: Shearing操作的轴，0为X轴，1为Y轴。
    
    返回:
    - sheared_image: 经过Shearing变换后的图像。
    """
    # 根据Shearing方向构造变换矩阵
    if axis == 0:  # X轴Shearing
        transform_matrix = np.array([[1, shearing_factor],
                                      [0, 1]])
    else:  # Y轴Shearing
        transform_matrix = np.array([[1, 0],
                                      [shearing_factor, 1]])
    
    # 应用仿射变换
    sheared_image = affine_transform(
        image, transform_matrix, offset=0, order=1, mode='constant', cval=0.0
    )
    
    return sheared_image


def get_rotated_dimensions(width, height, angle):
    """
    计算旋转后的图像尺寸
    :param width: 原图像宽度
    :param height: 原图像高度
    :param angle: 旋转角度
    :return: 旋转后的图像尺寸 (新宽度, 新高度)
    """
    radians = np.deg2rad(angle)
    new_width = int(abs(width * np.cos(radians)) + abs(height * np.sin(radians)))
    new_height = int(abs(height * np.cos(radians)) + abs(width * np.sin(radians)))
    return new_width, new_height

def rotate_image(image, angle):
    """
    旋转图像并确保所有内容都在新尺寸内
    :param image: 要旋转的图像
    :param angle: 旋转角度
    :return: 旋转后的图像
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    new_width, new_height = get_rotated_dimensions(w, h, angle)
    
    # 计算新的旋转中心和旋转矩阵
    new_center = (new_width // 2, new_height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_center[0] - center[0])
    M[1, 2] += (new_center[1] - center[1])

    # 执行旋转
    rotated_image = cv2.warpAffine(image, M, (new_width, new_height), borderValue=(114, 114, 114))
    return rotated_image, M

def rotate_point(point, M):
    """
    根据图像的旋转矩阵旋转关键点
    :param point: 要旋转的点(x, y)
    :param M: 图像旋转矩阵
    :return: 旋转后的点
    """
    v = np.array([[point[0]], [point[1]], [1]])
    calculated = np.dot(M, v)
    return int(calculated[0][0]), int(calculated[1][0])

def resize_image(image, scale_factor, center, fill_color=(114, 114, 114)):
    h, w = image.shape[:2]
    
    # 计算缩放后的图像尺寸
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    # 缩放图像
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 创建一个空白图像用于填充
    result = np.full((h, w, 3), fill_color, dtype=np.uint8)
    
    # 计算缩放图像在空白图像中的位置
    start_x = center[0] - new_w // 2
    start_y = center[1] - new_h // 2
    
    # 确保坐标不会出界
    end_x = min(start_x + new_w, w)
    end_y = min(start_y + new_h, h)
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    
    # 将缩放后的图像复制到空白图像上
    result[start_y:end_y, start_x:end_x] = resized_image[max(0, -start_y):min(new_h, h - start_y), max(0, -start_x):min(new_w, w - start_x)]
    
    return result

def resize_point(point, scale_factor, image_shape):

    # 计算关键点相对于缩放中心的相对位置
    original_h, original_w = image_shape
    center_x, center_y = original_w / 2, original_h / 2
    x, y = point

    adjusted_x = center_x - (center_x - x)*scale_factor
    adjusted_y = center_y - (center_y - y)*scale_factor
    
    return int(adjusted_x), int(adjusted_y)

def translate(image, max_offset, fill=114):
    # 生成随机平移量
    offset_x = random.randint(-max_offset, max_offset)
    offset_y = random.randint(-max_offset, max_offset)
    
    # 创建新的图像数组，初始化为114
    translated_image = np.full_like(image, fill)
    
    # 计算平移后的图像区域
    if offset_x > 0:
        x_start_new, x_end_new = offset_x, image.shape[1]
        x_start_old, x_end_old = 0, image.shape[1] - offset_x
    else:
        x_start_new, x_end_new = 0, image.shape[1] + offset_x
        x_start_old, x_end_old = -offset_x, image.shape[1]
    
    if offset_y > 0:
        y_start_new, y_end_new = offset_y, image.shape[0]
        y_start_old, y_end_old = 0, image.shape[0] - offset_y
    else:
        y_start_new, y_end_new = 0, image.shape[0] + offset_y
        y_start_old, y_end_old = -offset_y, image.shape[0]
    
    # 复制图像到新的位置
    translated_image[y_start_new:y_end_new, x_start_new:x_end_new] = image[y_start_old:y_end_old, x_start_old:x_end_old]
    
    return translated_image, offset_x, offset_y

def exposure(image, exposure):
    B, G, R = cv2.split(image)
    B = B.astype(float)
    G = G.astype(float)
    R = R.astype(float)
    
    # 调整曝光度
    B += exposure
    G += exposure
    R += exposure
    
    # 保证调整后的像素值仍然在合法范围内 [0, 255]
    B = np.clip(B, 0, 255)
    G = np.clip(G, 0, 255)
    R = np.clip(R, 0, 255)
    
    # 转换回 uint8
    B = B.astype(np.uint8)
    G = G.astype(np.uint8)
    R = R.astype(np.uint8)
    
    image = cv2.merge((B, G, R))
    
    return image    

if __name__ == "__main__":
    image = cv2.cvtColor(cv2.imread("/home/kd/Documents/Codes/fast-gesture/101.jpg"), cv2.COLOR_BGR2GRAY)
    new_img = shearing_img(image=image, shearing_factor=0.5, axis=0)
    # 示例使用
    cv2.imshow("img shearing", new_img)
    cv2.waitKey(5000)
    pass