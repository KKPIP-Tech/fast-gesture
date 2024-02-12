import math
import numpy as np
from decimal import Decimal, getcontext


def get_vxvyd(control_point:tuple, point_a:tuple) -> (float):
    """
    计算从点 A 到目标点的 x 向量、y 向量以及两点之间的距离。
    
    参数:
    - point_a: 点 A 的坐标，格式为 (x, y)，其中 x 和 y 是归一化的值。
    - control_point: 目标点的坐标，格式为 (x, y)，其中 x 和 y 是归一化的值。
    
    返回值:
    - x_vector: 从点 A 到目标点的 x 向量。
    - y_vector: 从点 A 到目标点的 y 向量。
    - distance: 点 A 与目标点之间的距离。
    """
    # 计算 x 向量和 y 向量
    x_vector = control_point[0] - point_a[0]
    y_vector = control_point[1] - point_a[1]
    
    # 计算距离
    distance = math.sqrt(x_vector**2 + y_vector**2)
    
    return x_vector, y_vector, distance



# 示例点 A 和目标点的坐标
point_a = (0.5, 0.5)  # 点 A 的坐标
target_point = (0.8, 0.2)  # 目标点的坐标

# 计算并打印结果
x_vector, y_vector, distance = get_vxvyd(point_a, target_point)
print(f"从点 A 到目标点的 x 向量: {x_vector}")
print(f"从点 A 到目标点的 y 向量: {y_vector}")
print(f"点 A 与目标点之间的距离: {distance}")