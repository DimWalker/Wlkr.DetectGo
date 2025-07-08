"""
整合前面的所有步骤，合并为一个制作流程
方便重开
"""
import logging
import os
import shutil
import sys

# 跟目录，兼容linux
code_root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, code_root_path)

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
    categories_list_o_c = [
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
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_o_c, "o_c")
    dataset_name = "go_board_dataset_all"
    dataset_type = "train"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_o_c, "o_c")

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

    # 黑白空
    categories_list_b_w_n = [
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
        {
            "id": 3,
            "name": "empty",
            "supercategory": "piece"
        },
    ]
    dataset_name = "go_board_dataset_all"
    dataset_type = "eval"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_b_w_n, "b_w_n")
    dataset_name = "go_board_dataset_all"
    dataset_type = "train"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_b_w_n, "b_w_n")

    # 黑白空
    categories_list_ocbwn = [
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
            "name": "black",
            "supercategory": "piece"
        },
        {
            "id": 4,
            "name": "white",
            "supercategory": "piece"
        },
        {
            "id": 5,
            "name": "empty",
            "supercategory": "piece"
        },
    ]
    dataset_name = "go_board_dataset_all"
    dataset_type = "eval"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_ocbwn, "ocbwn")
    dataset_name = "go_board_dataset_all"
    dataset_type = "train"
    all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_ocbwn, "ocbwn")


def coco_to_yolo(label_type):
    # 相对路径的处理不够好，写死了，故相关路径的格式不能变
    ds_train_path = "../output/go_board_dataset_all/train"
    ds_eval_path = "../output/go_board_dataset_all/eval"

    output_train_path = f"../output/go_board_dataset_all/train/{label_type}"
    output_eval_path = f"../output/go_board_dataset_all/eval/{label_type}"
    json_file_path = f"coco_data_{label_type}.json"
    txt_file_path = f"coco_data_{label_type}.txt"
    coco_to_yolo_fmt(ds_train_path
                     , output_train_path
                     , json_file_path, txt_file_path)
    coco_to_yolo_fmt(ds_eval_path
                     , output_eval_path
                     , json_file_path, txt_file_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

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
    coco_to_yolo("all")  # 所有
    coco_to_yolo("o_c")  # 棋盘、四个角
