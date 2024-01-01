import cv2
import numpy as np


def letterbox(image, target_shape=(640, 640), fill_color=(72, 72, 72), is_mask=False):
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
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top_padding, bottom_padding = int(round(dh - 0.1)), int(round(dh + 0.1))
    left_padding, right_padding = int(round(dw - 0.1)), int(round(dw + 0.1))
    if is_mask:
        image = cv2.copyMakeBorder(
            image, 
            top_padding, 
            bottom_padding, 
            left_padding, 
            right_padding, 
            cv2.BORDER_CONSTANT, value=0)  # add border
        
        return image, scale_ratio, left_padding, top_padding
    else:
        image = cv2.copyMakeBorder(
            image, 
            top_padding, 
            bottom_padding, 
            left_padding, 
            right_padding, 
            cv2.BORDER_CONSTANT, value=fill_color)  # add border
        
        return image, scale_ratio, left_padding, top_padding

