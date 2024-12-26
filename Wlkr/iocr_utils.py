import math
from enum import Enum

import numpy as np


def calc_theta_abs(p1, p2):
    angle = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
    angle = abs(angle)
    angle = min(angle, abs(angle - 180))
    return float(angle)


def calc_distance(point1, point2):
    # 这两个 Mat 有点多余？
    m1 = np.array([[point1[0], point1[1]]], dtype=np.float32)
    m2 = np.array([[point2[0], point2[1]]], dtype=np.float32)
    line = m1 - m2

    x = line[0, 0]
    y = line[0, 1]
    z = x ** 2 + y ** 2
    return np.sqrt(z)


def point_in_region(src_region, dst_point):
    A = src_region[0]
    B = src_region[1]
    C = src_region[2]
    D = src_region[3]

    a = (B[0] - A[0]) * (dst_point[1] - A[1]) - (B[1] - A[1]) * (dst_point[0] - A[0])
    b = (C[0] - B[0]) * (dst_point[1] - B[1]) - (C[1] - B[1]) * (dst_point[0] - B[0])
    c = (D[0] - C[0]) * (dst_point[1] - C[1]) - (D[1] - C[1]) * (dst_point[0] - C[0])
    d = (A[0] - D[0]) * (dst_point[1] - D[1]) - (A[1] - D[1]) * (dst_point[0] - D[0])

    if (a >= 0 and b >= 0 and c >= 0 and d >= 0) or (a <= 0 and b <= 0 and c <= 0 and d <= 0):
        return True
    return False


def calc_warp_point(point, M):
    """

    :param point: 透视变换（或仿射变换）前的点
    :param M: 3x3的齐次矩阵
    :return: 变换后的点
    """
    vector = np.array([point[0], point[1], 1], dtype=np.float64).reshape((3, 1))
    warp_mat = np.dot(M, vector)
    x = warp_mat[0, 0]
    y = warp_mat[1, 0]
    r = warp_mat[2, 0]
    p_x = x / r if r != 0 else x
    p_y = y / r if r != 0 else y
    return [p_x, p_y]


class PointType(Enum):
    LeftTop = 1
    RightTop = 2
    RightBottom = 3
    LeftBottom = 4
    Center = 5
    WrapPoint = 6


def sort_region_by(region_list, region_key_name, pt: PointType, threshold=1.5,
                   compare_method=None):
    """
    按照左上角的坐标排序文本区域
    :param region_list:
    :param region_key_name:
    :param pt:
    :param threshold: 角度(像素)阈值，用于判断是否换行
    :param compare_method: 比较角度(theta)还是像素(px)
    :return: 排序后的文本区域矩阵，一个二维列表
    """
    if not region_list or len(region_list) == 0:
        return region_list
    # y轴排序
    y_sort = sorted(region_list, key=lambda x: get_point(x, region_key_name, pt)[0])
    y_sort = sorted(y_sort, key=lambda x: get_point(x, region_key_name, pt)[1])
    #print(y_sort)
    # 分行
    matrix = []
    line = [y_sort[0]]
    for i in range(1, len(y_sort)):
        if compare_method == "px":
            flag = compare_by_px(y_sort, i, line, region_key_name, pt, threshold)
        else:
            flag = compare_by_theta(y_sort, i, line, region_key_name, pt, threshold)
        if flag:
            matrix.append(line)
            line = [y_sort[i]]
        else:
            line.append(y_sort[i])
    matrix.append(line)
    # x轴排序
    for x_sort in matrix:
        x_sort.sort(key=lambda x: get_point(x, region_key_name, pt)[0])
    # logging.info(matrix)
    return matrix


def compare_by_px(y_sort, i, line, region_key_name, pt: PointType, threshold=15):
    p1 = get_point(y_sort[i], region_key_name, pt)
    # 此参照物更适合像素作为阈值，因为倾斜像素阈值会被放大
    p2 = get_point(y_sort[i - 1], region_key_name, pt)
    dif_len = abs(p1[1] - p2[1])
    # logging.info("#" * 10)
    # logging.info(p1)
    # logging.info(p2)
    # logging.info(dif_len)
    # logging.info("#" * 10)
    return dif_len > threshold


def compare_by_theta(y_sort, i, line, region_key_name, pt: PointType, threshold=15):
    p1 = get_point(y_sort[i], region_key_name, pt)
    # 角度按理不会被放大，故适合取行首
    p2 = get_point(line[0], region_key_name, pt)
    theta = calc_theta_abs(p1, p2)
    # logging.info("#" * 10)
    # logging.info(p1)
    # logging.info(p2)
    # logging.info(theta)
    # logging.info("#" * 10)
    return theta > threshold


def get_point(x, region_key_name, pt):
    """
    由于数据结构不统一
    pt为左右上下返回x[region_key_name][n]
    pt为其他则返回x[region_key_name]
    """
    if pt == PointType.LeftTop:
        return x[region_key_name][0]
    elif pt == PointType.RightTop:
        return x[region_key_name][1]
    elif pt == PointType.RightBottom:
        return x[region_key_name][2]
    elif pt == PointType.LeftBottom:
        return x[region_key_name][3]
    elif pt == PointType.Center:
        return [
            (x[region_key_name][0][0] +
             x[region_key_name][1][0] +
             x[region_key_name][2][0] +
             x[region_key_name][3][0]) / 4,
            (x[region_key_name][0][1] +
             x[region_key_name][1][1] +
             x[region_key_name][2][1] +
             x[region_key_name][3][1]) / 4
        ]
    else:
        return x[region_key_name]
