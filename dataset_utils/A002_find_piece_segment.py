import json

import cv2
import numpy as np

from dataset_utils.A001_find_go_board_grid import save_middle_mat, reset_dir


# def find_piece_segment():
#     # 读取图像
#     image = cv2.imread('../assets/material/B002.png')
#     save_middle_mat(image, "ori")
#     # 灰度化
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     save_middle_mat(gray, "gray")
#     # 使用高斯滤波平滑图像
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     save_middle_mat(blur, "blur")
#     # 使用Canny边缘检测
#     edges = cv2.Canny(blur, 50, 150)
#     save_middle_mat(edges, "edges")
#     # 寻找轮廓
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     main_contour = max(contours, key=cv2.contourArea)
#     # 画出主轮廓（红色）
#     cv2.drawContours(image, [main_contour], -1, (0, 0, 255), 2)
#     # epsilon参数控制近似程度
#     epsilon = 0.02 * cv2.arcLength(main_contour, True)
#     approx = cv2.approxPolyDP(main_contour, epsilon, True)
#     # 将轮廓点转换为16个平均分布的点
#     num_points = 8
#     points = np.linspace(0, len(approx) - 1, num_points, dtype=int)
#     sampled_points = approx[points]
#     print(sampled_points)
#
#     # 将16个点连接起来
#     sampled_points = sampled_points.reshape((-1, 1, 2))
#     cv2.polylines(image, [sampled_points], isClosed=True, color=(0, 255, 0), thickness=2)
#     save_middle_mat(image, "polylines")
#     # 显示图像
#     cv2.imshow('Image with Polyline', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#

def find_piece_segment():
    # 创建一张空白的60x60的图像（你可以替换为你的实际图像）
    image = cv2.imread('../assets/material/W005.png')

    # 中心点
    center = (30, 30)

    # 画一个直径为60像素的圆
    radius = 30
    cv2.circle(image, center, radius, (0, 255, 0), 2)

    # 取圆上均匀分布的16个点
    num_points = 16
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points_x = center[0] + radius * np.cos(theta)
    points_y = center[1] + radius * np.sin(theta)
    points = np.array(list(zip(points_x, points_y)), dtype=np.int32).reshape((-1, 1, 2))

    # 在图像上画出这16个点（红色）
    for point in points:
        cv2.circle(image, tuple(point[0]), 2, (0, 0, 255), -1)


    cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)

    arr_reshaped = points.reshape((16,2))
    arr_list = arr_reshaped.tolist()

    # 将 Python 列表保存为 JSON 文件
    with open('../assets/material/piece_segmentation.json', 'w') as json_file:
        json.dump(arr_list, json_file)

    # 显示图像
    cv2.imshow('Circle with Points', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_empty_segment():
    # 创建一张空白的60x60的图像（你可以替换为你的实际图像）
    image = cv2.imread('../assets/material/W005.png')

    # 中心点
    center = (30, 30)

    # 画一个直径为60像素的圆
    radius = 15
    cv2.circle(image, center, radius, (0, 255, 0), 2)

    # 取圆上均匀分布的16个点
    num_points = 16
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points_x = center[0] + radius * np.cos(theta)
    points_y = center[1] + radius * np.sin(theta)
    points = np.array(list(zip(points_x, points_y)), dtype=np.int32).reshape((-1, 1, 2))

    # 在图像上画出这16个点（红色）
    for point in points:
        cv2.circle(image, tuple(point[0]), 2, (0, 0, 255), -1)


    cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)

    arr_reshaped = points.reshape((16,2))
    arr_list = arr_reshaped.tolist()

    # 将 Python 列表保存为 JSON 文件
    with open('../assets/material/empty_segmentation.json', 'w') as json_file:
        json.dump(arr_list, json_file)

    # 显示图像
    cv2.imshow('Circle with Points', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    reset_dir()
    find_piece_segment()
    find_empty_segment()
