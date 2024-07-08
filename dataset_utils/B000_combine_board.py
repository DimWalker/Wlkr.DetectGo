"""
合并棋盘与棋子，并输出label.txt
"""

import json
import os
import random
import zipfile
from datetime import datetime
from io import StringIO

import cv2
import numpy as np
from PIL import Image

from Wlkr.Common.FileUtils import GetFileNameSplit
from dataset_utils.B002_combine_scene import draw_region

mtrl_dir = r"../assets/material"


def load_diagram(diagram_path):
    with open(diagram_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = StringIO(content.replace("[", "").replace("]", ""))
    diagram = np.genfromtxt(content, delimiter=' ', dtype=int, encoding="utf-8")
    return diagram


def init_material_list():
    arr_o = []
    arr_b = []
    arr_w = []
    # 检查文件夹是否存在
    if os.path.exists(mtrl_dir):
        # 获取文件夹中的所有项
        items = os.listdir(mtrl_dir)
        # 遍历文件夹的第一层内容
        for item in items:
            item_path = os.path.join(mtrl_dir, item)
            # 检查是否为文件
            if os.path.isfile(item_path) and item.endswith(".png"):
                if item.startswith("O"):
                    arr_o.append(os.path.join(mtrl_dir, item))
                elif item.startswith("B"):
                    arr_b.append(os.path.join(mtrl_dir, item))
                elif item.startswith("W"):
                    arr_w.append(os.path.join(mtrl_dir, item))
    else:
        raise Exception(f"The folder {mtrl_dir} does not exist.")
    return arr_o, arr_b, arr_w


def add_piece(board_img, piece_img, norm_loc, board_info):
    # resized_img = piece_img.resize((board_info["avg_line_len"], board_info["avg_line_len"]))
    x, y = board_info["matrix"][norm_loc[0]][norm_loc[1]]
    offset = int(board_info["avg_line_len"] / 2)
    position = [x - offset, y - offset]
    board_img.paste(piece_img, position, piece_img)


def combine_board_image(o_path, b_path, w_path, diagram_path=None, save_path=None):
    # 打开三张图片
    board_img = Image.open(o_path).convert("RGBA")
    black_img = Image.open(b_path).convert("RGBA")
    white_img = Image.open(w_path).convert("RGBA")
    with open(o_path + ".json", "r", encoding="utf-8") as f:
        board_info = json.load(f)

    black_img = black_img.resize((board_info["avg_line_len"], board_info["avg_line_len"]))
    white_img = white_img.resize((board_info["avg_line_len"], board_info["avg_line_len"]))

    if diagram_path:
        print("handling :" + diagram_path)
        diagram = load_diagram(diagram_path)
        for r, row in enumerate(diagram):
            for c, cell in enumerate(row):
                if cell == 1:
                    add_piece(board_img, black_img, [r, c], board_info)
                elif cell == 2:
                    add_piece(board_img, white_img, [r, c], board_info)
    else:
        cnt = 0
        for r in range(19):
            for c in range(19):
                if cnt % 2 == 0:
                    add_piece(board_img, black_img, [r, c], board_info)
                else:
                    add_piece(board_img, white_img, [r, c], board_info)
                cnt += 1

    if save_path:
        board_img.save(save_path, "PNG")
    else:
        # 获取当前日期和时间
        current_datetime = datetime.now()
        # 将日期和时间格式化为指定格式
        formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S%f")
        board_img.save(f"../output/combine_board_{formatted_datetime}.png", "PNG")


def extract_diagram():
    zip_dir = "../assets/diagram"
    output_dir = "../output/diagram"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    diagram_list = []
    for filename in os.listdir(zip_dir):
        src_path = os.path.join(zip_dir, filename)
        _, pre, _ = GetFileNameSplit(filename)
        dst_path = os.path.join(output_dir, pre)
        if not os.path.exists(dst_path):
            with zipfile.ZipFile(src_path, 'r') as zip_ref:
                print(" unzip " + src_path + " to " + output_dir)
                # 解压 ZIP 文件到指定目标文件夹
                # 目前是zip文件内有同名文件夹
                zip_ref.extractall(output_dir)
                # 获取解压后的所有文件名列表
                file_list = zip_ref.namelist()
        diagram_list += [os.path.join(dst_path, x) for x in os.listdir(dst_path)]
    return diagram_list


def try_to_combine():
    o, b, w = init_material_list()
    for i in range(4):
        for _ in range(10):
            r0 = random.randint(0, len(o) - 1)
            r1 = random.randint(0, len(b) - 1)
            r2 = random.randint(0, len(w) - 1)
            combine_board_image(o[i], b[r1], w[r2])


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


def draw_board_region():
    output_dir = "../output/board_draw"
    mtrl_path = "../assets/material"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dia_dir = "../output/diagram_img"
    files = os.listdir(dia_dir)

    ok_list = []
    for file in files:
        if not file.endswith(".png"):
            continue

        tmpl_name = file[:4] + ".png.json"
        if tmpl_name in ok_list:
            continue
        with open(os.path.join(mtrl_path, tmpl_name), "r", encoding="utf-8") as f:
            json_obj = json.load(f)
        img = cv2.imread(os.path.join(dia_dir, file))
        draw_region(img, json_obj["board_region"])
        cv2.imwrite(os.path.join(output_dir, file), img)
        ok_list.append(tmpl_name)


def do_combine():
    o, b, w = init_material_list()
    diagram_list = extract_diagram()
    output_dir = "../output/diagram_img"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label = []
    cnt = 0
    # threads = []
    for d in diagram_list:
        r0 = random.randint(0, len(o) - 1)
        r1 = random.randint(0, len(b) - 1)
        r2 = random.randint(0, len(w) - 1)
        # 获取当前日期和时间
        current_datetime = datetime.now()
        # 将日期和时间格式化为指定格式
        formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S%f")
        _, pre, _ = GetFileNameSplit(o[r0])
        save_path = os.path.join(output_dir, f"{pre}_{formatted_datetime}.png")
        combine_board_image(o[r0], b[r1], w[r2], diagram_path=d, save_path=save_path)
        label.append(d + "\t" + save_path + "\n")
        cnt += 1
        if cnt == cnt_limit:
            break
    with open(os.path.join(output_dir, "label.txt"), "w", encoding="utf-8") as l:
        l.writelines(label)


cnt_limit = 50000
if __name__ == "__main__":
    # try_to_combine()
    do_combine()
    # draw_board_region()
