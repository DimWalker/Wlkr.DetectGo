# 必要知识：透视变换、齐次矩阵
import json
import os
import random
import threading
import cv2
import numpy as np

from Wlkr.Common.FileUtils import GetFileNameSplit
from dataset_utils.B000_combine_board import load_diagram
from dataset_utils.B002_combine_scene import draw_region

tmp_disable_factor = 37
factor_cnt = 0
lock_factor_cnt = threading.Lock()


def get_next_factor_cnt():
    global factor_cnt
    with lock_factor_cnt:
        factor_cnt += 1
        sub_id = factor_cnt
    return sub_id


def random_perspective_transform(image, factor=100):
    sub_factor_cnt = get_next_factor_cnt()
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
    if sub_factor_cnt % tmp_disable_factor == 0:
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
    zoom = random.randint(4, 10) / 20
    # 计算新的宽度和高度
    new_width = int(n_w * zoom)
    new_height = int(n_h * zoom)

    # 使用cv2.resize函数缩小图像
    resized_image = cv2.resize(warped_image, (new_width, new_height))
    dst_pts = dst_pts * zoom
    return resized_image, M, factor, zoom, dst_pts


def refresh_matrix(json_obj, M, factor, zoom, dst_pts, dia_tmpl_path):
    # 由于后面代码会改变json_obj的值，故必须放在前面
    json_obj = calc_piece_segmentation(json_obj, M, factor, zoom, dia_tmpl_path)

    json_obj["dst_pts"] = dst_pts.astype(np.int32).tolist()
    regions = []
    cornets = []
    # 这段会改变json_obj的原来的的值
    for p, pt in enumerate(json_obj["board_region"]):
        x, y = pt
        x += factor
        y += factor
        json_obj["board_region"][p] = [int(f * zoom) for f in calc_warp_point(M, x, y)]
    for r, row in enumerate(json_obj["matrix"]):
        line = []
        for c, pt in enumerate(row):
            x, y = pt
            x += factor
            y += factor
            json_obj["matrix"][r][c] = [int(f * zoom) for f in calc_warp_point(M, x, y)]

            # 棋盘四个角补正
            # 变更为向内扩展1格，增加特征
            length = int(json_obj["avg_line_len"] / 2)
            # 角
            if r == 0 and c == 0:
                lt = [int(f * zoom) for f in calc_warp_point(M, x - length, y - length)]
                rt = [int(f * zoom) for f in calc_warp_point(M, x + length * 4, y - length)]
                rb = [int(f * zoom) for f in calc_warp_point(M, x + length * 4, y + length * 4)]
                lb = [int(f * zoom) for f in calc_warp_point(M, x - length, y + length * 4)]
                cornets.append([lt, rt, rb, lb])
            elif r == 0 and c == 18:
                lt = [int(f * zoom) for f in calc_warp_point(M, x - length * 4, y - length)]
                rt = [int(f * zoom) for f in calc_warp_point(M, x + length, y - length)]
                rb = [int(f * zoom) for f in calc_warp_point(M, x + length, y + length * 4)]
                lb = [int(f * zoom) for f in calc_warp_point(M, x - length * 4, y + length * 4)]
                cornets.append([lt, rt, rb, lb])
            elif r == 18 and c == 0:
                lt = [int(f * zoom) for f in calc_warp_point(M, x - length, y - length * 4)]
                rt = [int(f * zoom) for f in calc_warp_point(M, x + length * 4, y - length * 4)]
                rb = [int(f * zoom) for f in calc_warp_point(M, x + length * 4, y + length)]
                lb = [int(f * zoom) for f in calc_warp_point(M, x - length, y + length)]
                cornets.append([lt, rt, rb, lb])
            elif r == 18 and c == 18:
                lt = [int(f * zoom) for f in calc_warp_point(M, x - length * 4, y - length * 4)]
                rt = [int(f * zoom) for f in calc_warp_point(M, x + length, y - length * 4)]
                rb = [int(f * zoom) for f in calc_warp_point(M, x + length, y + length)]
                lb = [int(f * zoom) for f in calc_warp_point(M, x - length * 4, y + length)]
                cornets.append([lt, rt, rb, lb])
            # 棋子region
            lt = [int(f * zoom) for f in calc_warp_point(M, x - length, y - length)]
            rt = [int(f * zoom) for f in calc_warp_point(M, x + length, y - length)]
            rb = [int(f * zoom) for f in calc_warp_point(M, x + length, y + length)]
            lb = [int(f * zoom) for f in calc_warp_point(M, x - length, y + length)]
            line.append([lt, rt, rb, lb])
            # 4个角区域
        regions.append(line)
    json_obj["regions"] = regions
    json_obj["corners"] = cornets
    json_obj["avg_line_len"] = int(json_obj["avg_line_len"] * zoom)
    calc_row_col(json_obj)
    return json_obj


def calc_row_col(json_obj):
    # 行
    rows = []
    for r, row in enumerate(json_obj["regions"]):
        row_region = [[row[0][0][0], row[0][0][1]],
                      [row[18][1][0], row[18][1][1]],
                      [row[18][2][0], row[18][2][1]],
                      [row[0][3][0], row[0][3][1]]]
        rows.append(row_region)
    cols = []
    for c in range(19):
        col_region = [
            [json_obj["regions"][0][c][0][0], json_obj["regions"][0][c][0][1]],
            [json_obj["regions"][0][c][1][0], json_obj["regions"][0][c][1][1]],
            [json_obj["regions"][18][c][2][0], json_obj["regions"][18][c][2][1]],
            [json_obj["regions"][18][c][3][0], json_obj["regions"][18][c][3][1]],
        ]
        cols.append(col_region)
    json_obj["row_regions"] = rows
    json_obj["col_regions"] = cols


def calc_piece_segmentation(json_obj, M, factor, zoom, dia_tmpl_path):
    diagram = load_diagram(dia_tmpl_path)
    with open("../assets/material/piece_segmentation.json", "r", encoding="utf-8") as f:
        p_seg = json.load(f)
    with open("../assets/material/empty_segmentation.json", "r", encoding="utf-8") as f:
        e_seg = json.load(f)
    # 棋子原图像素60px
    zoom_len = json_obj["avg_line_len"] / 60
    for s, seq in enumerate(p_seg):
        seq_x, seq_y = seq
        p_seg[s] = [seq_x * zoom_len, seq_y * zoom_len]
    for s, seq in enumerate(e_seg):
        seq_x, seq_y = seq
        e_seg[s] = [seq_x * zoom_len, seq_y * zoom_len]

    length = int(json_obj["avg_line_len"] / 2)
    pieces_seg = []

    for r, row in enumerate(json_obj["matrix"]):
        line = []
        for c, pnt in enumerate(row):
            x, y = pnt
            # 棋子region左上角
            x += factor - length
            y += factor - length
            warp_seq = []
            # 需要棋谱
            if diagram[r][c] == 0:
                for s, seq in enumerate(e_seg):
                    seq_x, seq_y = seq
                    warp_seq.append(
                        [int(f * zoom) for f in calc_warp_point(M, seq_x + x, seq_y + y)]
                    )
            else:
                for s, seq in enumerate(p_seg):
                    seq_x, seq_y = seq
                    warp_seq.append(
                        [int(f * zoom) for f in calc_warp_point(M, seq_x + x, seq_y + y)]
                    )
            line.append(warp_seq)
        pieces_seg.append(line)
    json_obj["pieces_seg"] = pieces_seg
    json_obj["diagram"] = diagram.tolist()
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
    board_image = cv2.imread("../output/diagram_img/O001_20231227100023624787.png",
                             cv2.IMREAD_UNCHANGED)
    dia_tmpl_path = "../output/diagram/aa_my_label/003.txt"
    # 生成透视变换后的图像
    transformed_image, M, factor, zoom, dst_pts = random_perspective_transform(board_image, factor=100)
    # 显示原始图像和变换后的图像
    cv2.imwrite("../output/warp.png", transformed_image)
    with open("../assets/material/O001.png.json", "r", encoding="utf-8") as r:
        json_obj = json.load(r)
        json_obj = refresh_matrix(json_obj, M, factor, zoom, dst_pts, dia_tmpl_path)
    with open("../output/warp.png.json", "w", encoding="utf-8") as w:
        json.dump(json_obj, w)

    # for row in json_obj["regions"]:
    #     for region in row:
    #         draw_region(transformed_image, region)
    # for row in json_obj["pieces_seg"]:
    #     for region in row:
    #         draw_region(transformed_image, region)
    draw_region(transformed_image, json_obj["board_region"])

    cv2.imwrite("../output/draw_region.png", transformed_image)


def draw_all_warp():
    output_dir = "../output/warp_draw"
    warp_dir = "../output/diagram_warp"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    lines = os.listdir(warp_dir)

    for line in lines:
        if not line.endswith(".png"):
            continue
        warp_path = os.path.join(warp_dir, line)
        bn, _, _ = GetFileNameSplit(warp_path)
        json_path = warp_path + ".json"
        with open(json_path, "r", encoding="utf-8") as f:
            json_obj = json.load(f)

        img = cv2.imread(warp_path)
        draw_region(img, json_obj["board_region"])
        cv2.imwrite(os.path.join(output_dir, bn), img)


# def draw_region(image, region):
#     # 定义四个点的坐标
#     points = np.array(region, np.int32)
#     # 顶点坐标需要reshape成OpenCV所需的格式
#     points = points.reshape((-1, 1, 2))
#     # 画四边形
#     cv2.polylines(image, [points], isClosed=True,
#                   color=(
#                       random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255),
#                   thickness=1)


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
    cnt = 0
    for line in lines:
        dia_tmpl_path, dia_path = line.rstrip().split('\t')

        bn, pre, ext = GetFileNameSplit(dia_path)
        if not dia_path.endswith(".png"):
            continue
        print("warping " + dia_path)
        board_image = cv2.imread(dia_path, cv2.IMREAD_UNCHANGED)
        transformed_image, M, factor, zoom, dst_pts = random_perspective_transform(board_image, factor=100)
        save_path = os.path.join(output_dir, bn)
        cv2.imwrite(save_path, transformed_image)
        tmpl_name = pre.split('_')[0]
        tmpl_path = os.path.join(mtrl_dir, tmpl_name + ".png.json")
        with open(tmpl_path, "r", encoding="utf-8") as r:
            json_obj = json.load(r)
            json_obj = refresh_matrix(json_obj, M, factor, zoom, dst_pts, dia_tmpl_path)
        with open(save_path + ".json", "w", encoding="utf-8") as w:
            json.dump(json_obj, w)

        cnt += 1
        if cnt > cnt_limit:
            break


cnt_limit = 10000000
if __name__ == "__main__":
    # 构建端正的行训练集
    tmp_disable_factor = 1
    # try_to_warp()
    # arr = np.array([1, 1])
    # print(arr * 2)
    do_warp()
    # draw_all_warp()
