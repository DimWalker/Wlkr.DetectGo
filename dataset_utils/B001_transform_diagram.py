# 必要知识：透视变换、齐次矩阵
import json
import os
import random

import cv2
import numpy as np

import cv2
import numpy as np

from Wlkr.Common.FileUtils import GetFileNameSplit

tmp_disable_factor = 37
factor_cnt = 0


def random_perspective_transform(image, factor=100):
    global factor_cnt
    factor_cnt += 1
    h, w = image.shape[:2]

    # 创建新的图片
    # 这个只是猜测有待确认，应该不会查过 原始像素 + factor * 2
    n_h, n_w = h + factor * 2, w + factor * 2
    new_image = np.ones((n_h, n_w, 4), dtype=np.uint8, ) * 0
    new_image[factor:n_h - factor, factor:n_w - factor] = image

    # 定义原始图像上的四个点
    src_pts = np.float32(
        [[0 + factor, 0 + factor], [w + factor, 0 + factor],
         [w + factor, h + factor], [0 + factor, h + factor]])

    tmp_fac = factor
    if factor_cnt % tmp_disable_factor == 0:
        tmp_fac = 0
    # 定义目标图像上的四个点，通过随机扰动原始点
    dst_pts = src_pts + np.random.uniform(-tmp_fac, tmp_fac, size=src_pts.shape).astype(np.float32)
    # dst_pts = np.float32([[0 + 100, 0 + 100], [w + 100, 0 + 100], [w + 100, h + 100], [0 + 100, h + 100]])
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # 应用透视变换
    warped_image = cv2.warpPerspective(new_image, M, (n_h, n_w), borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(255, 255, 255, 0))

    # 数据集图片是683 * 512或 512 * 683
    # 控制在500像素以内
    zoom = random.randint(3, 10) / 20
    # 计算新的宽度和高度
    new_width = int(n_w * zoom)
    new_height = int(n_h * zoom)

    # 使用cv2.resize函数缩小图像
    resized_image = cv2.resize(warped_image, (new_width, new_height))
    dst_pts = dst_pts * zoom
    return resized_image, M, factor, zoom, dst_pts


def refresh_matrix(json_obj, M, factor, zoom, dst_pts):
    json_obj["dst_pts"] = dst_pts.astype(np.int32).tolist()
    regions = []
    for r, row in enumerate(json_obj["matrix"]):
        line = []
        for c, pnt in enumerate(row):
            x, y = pnt
            x += factor
            y += factor
            # 中心点不满足需求
            json_obj["matrix"][r][c] = [int(f * zoom) for f in calc_warp_point(M, x, y)]

            # 棋盘四个角补正
            # 变更为向内扩展1格，增加特征
            len = int(json_obj["avg_line_len"] / 2)
            if (r == 0 and c == 0):
                lt = [int(f * zoom) for f in calc_warp_point(M, x - len, y - len)]
                rt = [int(f * zoom) for f in calc_warp_point(M, x + len * 4, y - len)]
                rb = [int(f * zoom) for f in calc_warp_point(M, x + len * 4, y + len * 4)]
                lb = [int(f * zoom) for f in calc_warp_point(M, x - len, y + len * 4)]
            elif (r == 0 and c == 18):
                lt = [int(f * zoom) for f in calc_warp_point(M, x - len * 4, y - len)]
                rt = [int(f * zoom) for f in calc_warp_point(M, x + len, y - len)]
                rb = [int(f * zoom) for f in calc_warp_point(M, x + len, y + len * 4)]
                lb = [int(f * zoom) for f in calc_warp_point(M, x - len * 4, y + len * 4)]
            elif (r == 18 and c == 0):
                lt = [int(f * zoom) for f in calc_warp_point(M, x - len, y - len * 4)]
                rt = [int(f * zoom) for f in calc_warp_point(M, x + len * 4, y - len * 4)]
                rb = [int(f * zoom) for f in calc_warp_point(M, x + len * 4, y + len)]
                lb = [int(f * zoom) for f in calc_warp_point(M, x - len, y + len)]
            elif (r == 18 and c == 18):
                lt = [int(f * zoom) for f in calc_warp_point(M, x - len * 4, y - len * 4)]
                rt = [int(f * zoom) for f in calc_warp_point(M, x + len, y - len * 4)]
                rb = [int(f * zoom) for f in calc_warp_point(M, x + len, y + len)]
                lb = [int(f * zoom) for f in calc_warp_point(M, x - len * 4, y + len)]
            else:
                # 棋子region
                lt = [int(f * zoom) for f in calc_warp_point(M, x - len, y - len)]
                rt = [int(f * zoom) for f in calc_warp_point(M, x + len, y - len)]
                rb = [int(f * zoom) for f in calc_warp_point(M, x + len, y + len)]
                lb = [int(f * zoom) for f in calc_warp_point(M, x - len, y + len)]
            line.append([lt, rt, rb, lb])
            # 4个角区域
        regions.append(line)
    json_obj["regions"] = regions
    json_obj["avg_line_len"] = int(json_obj["avg_line_len"] * zoom)
    return json_obj


def calc_warp_point(M, ori_x, ori_y):
    matrix = M.copy()
    vector = np.array([ori_x, ori_y, 1], dtype=np.float64).reshape((3, 1))
    warp_mat = np.dot(matrix, vector)
    x, y, r = warp_mat[0, 0], warp_mat[1, 0], warp_mat[2, 0]
    w_x = x / r if r != 0 else x
    w_y = y / r if r != 0 else y
    return [int(w_x), int(w_y)]


def try_to_warp():
    # 读取围棋棋盘图片
    board_image = cv2.imread("../output/diagram_img/O001_20231214140252621478.png",
                             cv2.IMREAD_UNCHANGED)

    # 生成透视变换后的图像
    transformed_image, M, factor, zoom, dst_pts = random_perspective_transform(board_image, factor=100)

    # 显示原始图像和变换后的图像
    # cv2.imshow("Original Board", board_image)
    # cv2.imshow("Transformed Board", transformed_image)
    cv2.imwrite("../output/warp.png", transformed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    with open("../assets/material/O001.png.json", "r", encoding="utf-8") as r:
        json_obj = json.load(r)
        json_obj = refresh_matrix(json_obj, M, factor, zoom, dst_pts)
    with open("../output/warp.png.json", "w", encoding="utf-8") as w:
        json.dump(json_obj, w)

    for row in json_obj["regions"]:
        for region in row:
            draw_region(transformed_image, region)
    cv2.imwrite("../output/draw_region.png", transformed_image)


def draw_region(image, region):
    # 定义四个点的坐标
    points = np.array(region, np.int32)
    # 顶点坐标需要reshape成OpenCV所需的格式
    points = points.reshape((-1, 1, 2))
    # 画四边形
    cv2.polylines(image, [points], isClosed=True,
                  color=(
                      random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255),
                  thickness=1)


def do_warp():
    output_dir = "../output/diagram_warp"
    mtrl_dir = "../assets/material"
    diagram_dir = "../output/diagram_img"
    label_path = "../output/diagram_img/label.txt"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # diagrams = os.listdir(diagram_dir)
    for line in lines:
        dia_path = line.rstrip().split('\t')[-1]
        print("warping " + dia_path)
        bn, pre, ext = GetFileNameSplit(dia_path)
        if not dia_path.endswith(".png"):
            continue
        board_image = cv2.imread(dia_path, cv2.IMREAD_UNCHANGED)
        transformed_image, M, factor, zoom, dst_pts = random_perspective_transform(board_image, factor=100)
        save_path = os.path.join(output_dir, bn)
        cv2.imwrite(save_path, transformed_image)
        tmpl_name = pre.split('_')[0]
        tmpl_path = os.path.join(mtrl_dir, tmpl_name + ".png.json")
        with open(tmpl_path, "r", encoding="utf-8") as r:
            json_obj = json.load(r)
            json_obj = refresh_matrix(json_obj, M, factor, zoom, dst_pts)
        with open(save_path + ".json", "w", encoding="utf-8") as w:
            json.dump(json_obj, w)


if __name__ == "__main__":
    # try_to_warp()
    # arr = np.array([1, 1])
    # print(arr * 2)
    do_warp()
