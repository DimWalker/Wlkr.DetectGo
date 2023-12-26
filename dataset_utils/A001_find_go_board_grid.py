import json
import math
import os

import cv2
import numpy as np

from Wlkr.iocr_utils import calc_distance

mid_dir = "../output/mid_files"
# 设置保存计数器
cnt_save = 0


def reset_dir():
    global cnt_save
    cnt_save = 0

    if not os.path.exists(mid_dir):
        os.makedirs(mid_dir)
    else:
        files = os.listdir(mid_dir)
        for file in files:
            file_path = os.path.join(mid_dir, file)
            os.remove(file_path)


def save_middle_mat(img, name):
    if not os.path.exists(mid_dir):
        os.makedirs(mid_dir)
    global cnt_save
    cnt_save += 1
    img_path = os.path.join(mid_dir, f"{cnt_save:03d}_{name}.jpg")
    cv2.imwrite(img_path, img)


def find_go_board_squares(image_path, bin_threshold=160, diff_threshold=5
                          , crop_px=None, brightness=None):
    print(image_path)
    # 读取图像
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    min_x, max_x, min_y, max_y = 0, width, 0, height
    if crop_px and isinstance(crop_px, int):
        min_x, max_x, min_y, max_y = crop_px, width - crop_px, crop_px, height - crop_px
    elif crop_px and isinstance(crop_px, list):
        min_x, max_x, min_y, max_y = crop_px[0], width - crop_px[2], crop_px[1], height - crop_px[3]
    # 裁剪图像
    cp_img = img[min_y:max_y, min_x: max_x]

    if brightness and isinstance(brightness, int):
        # 图像亮度
        cp_img = cv2.add(cp_img, np.ones_like(cp_img) * brightness, dtype=cv2.CV_8U)
        save_middle_mat(cp_img, "brightness")

    # 灰度
    binary_inv = cv2.cvtColor(cp_img, cv2.COLOR_BGR2GRAY)
    save_middle_mat(binary_inv, "gray")

    # 二值化
    if bin_threshold > 0:
        _, binary_inv = cv2.threshold(binary_inv, bin_threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        _, binary_inv = cv2.threshold(binary_inv, -bin_threshold, 255, cv2.THRESH_BINARY)
    save_middle_mat(binary_inv, "bin_inv")

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # # 开运算，去除小点
    # binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
    # save_middle_mat(binary_inv, "open")

    ver = binary_inv.copy()
    # 膨胀竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    ver = cv2.dilate(ver, kernel, iterations=1)
    save_middle_mat(ver, "v_dilate")
    # 腐蚀横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    ver = cv2.erode(ver, kernel, iterations=3)
    save_middle_mat(ver, "v_erode")
    # 腐蚀线的头尾会被削掉，补偿回来
    # 另外像素为1的点不会被当做轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
    ver = cv2.dilate(ver, kernel, iterations=3)
    save_middle_mat(ver, "v_dilate")

    hor = binary_inv.copy()
    # 膨胀横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    hor = cv2.dilate(hor, kernel, iterations=1)
    save_middle_mat(hor, "h_dilate")
    # 腐蚀竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    hor = cv2.erode(hor, kernel, iterations=3)
    save_middle_mat(hor, "h_erode")
    # 腐蚀线的头尾会被削掉，补偿回来
    # 另外像素为1的点不会被当做轮廓
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    hor = cv2.dilate(hor, kernel, iterations=3)
    save_middle_mat(hor, "h_dilate")

    # 交点
    intersection_image = cv2.bitwise_and(ver, hor)
    save_middle_mat(intersection_image, "intersection")
    # 膨胀交点，有个棋盘的星位不是点而是4直角
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    intersection_image = cv2.dilate(intersection_image, kernel, iterations=3)
    save_middle_mat(intersection_image, "i_dilate")

    # 找到轮廓
    contours, _ = cv2.findContours(intersection_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 计算每个轮廓的中心点
    center_points = []
    for contour in contours:
        # 这计算方式中心点不准确，会偏移
        # M = cv2.moments(contour)
        # if M["m00"] != 0:
        #     cx = int(M["m10"] / M["m00"])
        #     cy = int(M["m01"] / M["m00"])
        #     center_points.append([cx, cy])

        # 计算最小包围矩形
        # rect = cv2.minAreaRect(contour)
        # 获取矩形中心点坐标
        # x, y = rect[0]

        # 计算最大矩形，左上角
        x, y, w, h = cv2.boundingRect(contour)
        center_points.append([x, y])
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness=1)

    # # 填充轮廓为黑色
    # cv2.drawContours(img, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
    # cv2.drawContours(intersection_image, contours, -1, (0, 0, 0), thickness=cv2.FILLED)
    # # 红点观察，这坑爹玩意为什么裁剪后不准？
    # for p in center_points:
    #     img[p[0]][p[1]] = [0, 0, 255]
    #     intersection_image[p[0]][p[1]] = 0
    # save_middle_mat(img, "red_img")
    # save_middle_mat(intersection_image, "red_itr")

    center_points = sorted(center_points, key=lambda x: x[0])
    center_points = sorted(center_points, key=lambda x: x[1])
    points_count = len(center_points)
    # 输出中心点坐标
    print("center_points:", center_points)
    print("points_count: " + str(points_count))

    if points_count != 361:
        raise Exception("交点数目错误，19线棋盘应为361点: " + image_path)

    matrix = []
    row = [center_points[0]]
    for idx, point in enumerate(center_points):
        if idx == 0:
            continue
        diff = abs(center_points[idx - 1][1] - point[1])
        if diff <= diff_threshold:
            row.append(point)
        else:
            matrix.append(row)
            row = [point]
    matrix.append(row)

    # x轴排序
    for x_sort in matrix:
        x_sort.sort(key=lambda x: x[0])

    # 计算正方形横线的平均长度，竖线偷懒略
    line_ttl = 0
    for row in matrix:
        line_ttl += calc_distance(row[0], row[-1])
        # 候选：int、round、math.ceil
    avg_line_len = int(line_ttl / len(matrix) / len(matrix[0]))
    print("avg_line_len: " + str(avg_line_len))

    # 裁剪补正
    if crop_px and isinstance(crop_px, int):
        for r, row in enumerate(matrix):
            for c, cell in enumerate(row):
                matrix[r][c] = (cell[0] + crop_px, cell[1] + crop_px)
    elif crop_px and isinstance(crop_px, list):
        for r, row in enumerate(matrix):
            for c, cell in enumerate(row):
                matrix[r][c] = (cell[0] + crop_px[0], cell[1] + crop_px[1])

    for row in matrix:
        print('1,' * len(row))
    # 保存棋盘数据
    json_path = image_path + ".json"
    json_obj = {
        "board_region": [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]],
        "matrix": matrix,
        "points_count": points_count,
        "avg_line_len": avg_line_len
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f)


def try_to_find():
    pass
    img_path = r"../assets/material/O001.png"
    reset_dir()
    find_go_board_squares(img_path)

    img_path = r"../assets/material/O002.png"
    reset_dir()
    find_go_board_squares(img_path, crop_px=5)

    img_path = r"../assets/material/O003.png"
    reset_dir()
    find_go_board_squares(img_path, bin_threshold=96, crop_px=5)

    # 这张图黑白是反的，还需要调整亮度
    img_path = r"../assets/material/O004.png"
    reset_dir()
    find_go_board_squares(img_path, bin_threshold=-128, crop_px=40, brightness=-50)

    # img_path = r"../assets/material/O005.png"
    # reset_dir()
    # find_go_board_squares(img_path, bin_threshold=96)

    img_path = r"../assets/material/O005.png"
    reset_dir()
    find_go_board_squares(img_path, crop_px=60)

    img_path = r"../assets/material/O006.png"
    reset_dir()
    find_go_board_squares(img_path)

    img_path = r"../assets/material/O007.png"
    reset_dir()
    find_go_board_squares(img_path,bin_threshold=96)

    img_path = r"../assets/material/O008.png"
    reset_dir()
    find_go_board_squares(img_path , bin_threshold=96, crop_px=30)

    img_path = r"../assets/material/O009.png"
    reset_dir()
    find_go_board_squares(img_path, crop_px=30)

    img_path = r"../assets/material/O010.png"
    reset_dir()
    find_go_board_squares(img_path, crop_px=50)

    img_path = r"../assets/material/O011.png"
    reset_dir()
    find_go_board_squares(img_path, bin_threshold=96, crop_px=[60, 60, 70, 70])

    img_path = r"../assets/material/O012.png"
    reset_dir()
    find_go_board_squares(img_path, bin_threshold=96, crop_px=[70, 50, 70, 70])

    img_path = r"../assets/material/O013.png"
    reset_dir()
    find_go_board_squares(img_path ,bin_threshold=-128, crop_px=40, brightness=-50)

    img_path = r"../assets/material/O014.png"
    reset_dir()
    find_go_board_squares(img_path,bin_threshold=180, crop_px=60)

    img_path = r"../assets/material/O015.png"
    reset_dir()
    find_go_board_squares(img_path)

    img_path = r"../assets/material/O016.png"
    reset_dir()
    find_go_board_squares(img_path)



    # 这张得想新算法     XXXXO013.png


if __name__ == "__main__":
    try_to_find()
