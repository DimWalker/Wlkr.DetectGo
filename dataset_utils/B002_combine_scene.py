# 读取 JPG 图像
import json
import random
import cv2
import numpy as np


def combine_scene_image(jpg_path, png_path, train_size=640):
    """

    :param jpg_path:
    :param png_path:
    :param train_size: Yolo训练的输出尺寸正方形
    :return:
    """
    jpg_img = cv2.imread(jpg_path)
    png_img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    jpg_h, jpg_w, _ = jpg_img.shape
    png_h, png_w, _ = png_img.shape

    # 上一步transform_diagram最大图片是500像素，需要保证大于png
    min_jpg_len = min(jpg_h, jpg_w)
    max_png_len = max(png_h, png_w)
    if min_jpg_len <= max_png_len:
        new_len = max_png_len + 100
        if jpg_w < jpg_h:
            jpg_h = int(new_len / jpg_w * jpg_h)
            jpg_w = new_len
        else:
            jpg_w = int(new_len / jpg_h * jpg_w)
            jpg_h = new_len
        jpg_img = cv2.resize(jpg_img, (jpg_w, jpg_h))

    # 随机偏移
    offset_x = random.randint(0, jpg_w - png_w)
    offset_y = random.randint(0, jpg_h - png_h)
    # 获取 PNG 图像的 Alpha 通道
    alpha_channel = png_img[:, :, 3]
    # 将 PNG 图像合成到 JPG 图像中
    for c in range(0, 3):
        jpg_img[offset_y:png_h + offset_y, offset_x:png_w + offset_x, c] = \
            jpg_img[offset_y:png_h + offset_y, offset_x:png_w + offset_x, c] * \
            (1 - alpha_channel / 255) + png_img[:, :, c] * (alpha_channel / 255)

    if jpg_h > jpg_w:
        zoom = train_size / jpg_h
    else:
        zoom = train_size / jpg_w

    if zoom != 1:
        offset_x = int(offset_x * zoom)
        offset_y = int(offset_y * zoom)
        new_width = int(jpg_w * zoom)
        new_height = int(jpg_h * zoom)
        jpg_img = cv2.resize(jpg_img, (new_width, new_height))

    return jpg_img, offset_x, offset_y, zoom


def offset_json_obj(json_path, offset_x, offset_y, zoom):
    with open(json_path, "r", encoding="utf-8") as f:
        json_obj = json.load(f)
    for p, pt in enumerate(json_obj["board_region"]):
        json_obj["board_region"][p] = [int(pt[0] * zoom + offset_x), int(pt[1] * zoom + offset_y)]
    for r, row in enumerate(json_obj["matrix"]):
        for c, cell in enumerate(row):
            json_obj["matrix"][r][c] = [int(cell[0] * zoom + offset_x), int(cell[1] * zoom + offset_y)]
    for r, row in enumerate(json_obj["pieces_seg"]):
        for c, cell in enumerate(row):
            for p, pt in enumerate(cell):
                json_obj["pieces_seg"][r][c][p] = [int(pt[0] * zoom + offset_x), int(pt[1] * zoom + offset_y)]
    for p, pt in enumerate(json_obj["dst_pts"]):
        json_obj["dst_pts"][p] = [int(pt[0] * zoom + offset_x), int(pt[1] * zoom + offset_y)]
    for r, row in enumerate(json_obj["regions"]):
        for c, cell in enumerate(row):
            for p, pt in enumerate(cell):
                json_obj["regions"][r][c][p] = [int(pt[0] * zoom + offset_x), int(pt[1] * zoom + offset_y)]
    for c, cell in enumerate(json_obj["corners"]):
        for p, pt in enumerate(cell):
            json_obj["corners"][c][p] = [int(pt[0] * zoom + offset_x), int(pt[1] * zoom + offset_y)]
    for c, cell in enumerate(json_obj["row_regions"]):
        for p, pt in enumerate(cell):
            json_obj["row_regions"][c][p] = [int(pt[0] * zoom + offset_x), int(pt[1] * zoom + offset_y)]
    for c, cell in enumerate(json_obj["col_regions"]):
        for p, pt in enumerate(cell):
            json_obj["col_regions"][c][p] = [int(pt[0] * zoom + offset_x), int(pt[1] * zoom + offset_y)]
    return json_obj


def draw_region(image, pts):
    if len(pts) == 1:
        # 顶点坐标需要reshape成OpenCV所需的格式
        points = np.array(pts[0], np.int32).reshape((-1, 2))
        # 画四边形
        cv2.polylines(image, [points], isClosed=True,
                      color=(
                          random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255),
                      thickness=2)
    elif len(pts) == 4 and isinstance(pts[0], list) and len(pts[0]) == 2:
        points = np.array(pts, np.int32).reshape((4, 2))
        # 画四边形
        cv2.polylines(image, [points], isClosed=True,
                      color=(
                          random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255),
                      thickness=2)
    else:
        # 定义矩形的左上角和右下角坐标
        x1, y1 = pts[0], pts[1]
        x2, y2 = pts[0] + pts[2], pts[1] + pts[3]
        # 定义矩形的颜色（BGR 格式）
        color = (0, 255, 0)  # 这里使用绿色
        # 绘制矩形
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
