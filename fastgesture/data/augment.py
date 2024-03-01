import cv2
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


if __name__ == "__main__":
    image = cv2.cvtColor(cv2.imread("/home/kd/Documents/Codes/fast-gesture/101.jpg"), cv2.COLOR_BGR2GRAY)
    new_img = shearing_img(image=image, shearing_factor=0.5, axis=0)
    # 示例使用
    cv2.imshow("img shearing", new_img)
    cv2.waitKey(5000)
    pass