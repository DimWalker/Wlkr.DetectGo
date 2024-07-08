"""
整合前面的所有步骤，合并为一个制作流程
方便重开
"""


import os
import shutil

from dataset_utils.B000_combine_board import do_combine
from dataset_utils.B001_transform_diagram import do_warp
from dataset_utils.B002_combine_scene_board import do_coco_dataset, all_to_sub
from dataset_utils.B003_coco_to_yolo_fmt import coco_to_yolo_fmt
from dataset_utils.Z999_common_utils import remove_module_dir


def go_board_dataset_all(categories_list_all):
    dataset_name = "go_board_dataset_all"
    dataset_type = "eval"
    remove_module_dir(os.path.join("../output", dataset_name, dataset_type))
    do_coco_dataset(dataset_name, dataset_type, categories_list_all, "all")
    dataset_name = "go_board_dataset_all"
    dataset_type = "train"
    remove_module_dir(os.path.join("../output", dataset_name, dataset_type))
    do_coco_dataset(dataset_name, dataset_type, categories_list_all, "all")


def go_board_dataset_sub():
    """
    生成多个子集
    :return:
    """
    # 棋盘，角
    categories_list_b_c = [
        {
            "id": 1,
            "name": "board",
            "supercategory": "board"
        },
        {
            "id": 2,
            "name": "corner",
            "supercategory": "board"
        },
    ]
    dataset_name = "go_board_dataset_all"
    dataset_type = "eval"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_b_c, "b_c")
    dataset_name = "go_board_dataset_all"
    dataset_type = "train"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_b_c, "b_c")

    # 行
    categories_list_r = [
        {
            "id": 1,
            "name": "row",
            "supercategory": "board"
        },
    ]
    dataset_name = "go_board_dataset_all"
    dataset_type = "eval"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_r, "r")
    dataset_name = "go_board_dataset_all"
    dataset_type = "train"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_r, "r")

    # 列
    categories_list_c = [
        {
            "id": 1,
            "name": "col",
            "supercategory": "board"
        },
    ]
    dataset_name = "go_board_dataset_all"
    dataset_type = "eval"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_c, "c")
    dataset_name = "go_board_dataset_all"
    dataset_type = "train"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_c, "c")

    # 黑白棋子
    categories_list_b_w = [
        {
            "id": 1,
            "name": "black",
            "supercategory": "piece"
        },
        {
            "id": 2,
            "name": "white",
            "supercategory": "piece"
        },
    ]
    dataset_name = "go_board_dataset_all"
    dataset_type = "eval"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_b_w, "b_w")
    dataset_name = "go_board_dataset_all"
    dataset_type = "train"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_b_w, "b_w")

    # 空棋子
    categories_list_n = [
        {
            "id": 1,
            "name": "empty",
            "supercategory": "piece"
        },
    ]
    dataset_name = "go_board_dataset_all"
    dataset_type = "eval"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_n, "n")
    dataset_name = "go_board_dataset_all"
    dataset_type = "train"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_n, "n")


if __name__ == "__main__":
    # 基础棋谱
    remove_module_dir("../output/diagram_img")
    do_combine()

    # 数据增强：透视变换
    remove_module_dir("../output/diagram_warp")
    do_warp()

    # 棋谱到场景 all
    categories_list_all = [
        {
            "id": 1,
            "name": "board",
            "supercategory": "board"
        },
        {
            "id": 2,
            "name": "corner",
            "supercategory": "board"
        },
        {
            "id": 3,
            "name": "row",
            "supercategory": "board"
        },
        {
            "id": 4,
            "name": "col",
            "supercategory": "board"
        },
        {
            "id": 5,
            "name": "black",
            "supercategory": "piece"
        },
        {
            "id": 6,
            "name": "white",
            "supercategory": "piece"
        },
        {
            "id": 7,
            "name": "empty",
            "supercategory": "piece"
        },
    ]
    go_board_dataset_all(categories_list_all)
    go_board_dataset_sub()

    # coco转yolo格式
    # 相对路径的处理不够好，写死了，故相关路径的格式不能变
    coco_to_yolo_fmt("../output/go_board_dataset_all/train"
                     , "../output/go_board_dataset_all/train/all"
                     , "coco_data_all.json", "coco_data_all.txt")
    coco_to_yolo_fmt("../output/go_board_dataset_all/eval"
                     , "../output/go_board_dataset_all/eval/all"
                     , "coco_data_all.json", "coco_data_all.txt")
