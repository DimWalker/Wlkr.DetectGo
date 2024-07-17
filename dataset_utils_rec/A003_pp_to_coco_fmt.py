import json
import logging
import os
from datetime import datetime

import cv2

from Wlkr.Common.FileUtils import GetFileNameSplit
from dataset_utils.B002_combine_scene_board import get_next_image_info_id, cale_ppt_from_region, find_category_id, \
    get_next_annotation_info_id, info, licenses_list
from dataset_utils.B003_coco_to_yolo_fmt import coco_to_yolo_fmt


def pplabel_2_coco(categories_list):
    """
    细菌感染感冒几天了，copy代码懒得优化了
    :param pre_dir_name:
    :return:
    """
    output_dir = "../output/warp_back_straight/"
    with open(os.path.join(output_dir, "Label.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    img_list = []
    ann_list = []
    for line in lines:
        file_name, json_obj = line.rstrip().split("\t")
        print("handing " + file_name)
        bn, pre, ext = GetFileNameSplit(file_name)
        json_obj = json.loads(json_obj)
        # 获取当前日期和时间
        current_datetime = datetime.now()
        # 将日期和时间格式化为指定格式
        formatted_datetime = current_datetime.strftime("%Y-%m-%d")
        jpg_img = cv2.imread(os.path.join(output_dir, bn))
        h, w, _ = jpg_img.shape

        sub_image_info_id = get_next_image_info_id()
        # 定义图像信息
        image_info = {
            "id": sub_image_info_id,
            "width": w,  # 替换为实际图像宽度
            "height": h,  # 替换为实际图像高度
            "file_name": bn,  # 替换为实际图像文件名
            "license": None,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": formatted_datetime
        }
        img_list.append(image_info)

        for rgn in json_obj:
            cate_name = rgn["transcription"]
            if cate_name == "1":
                cate_name = "black"
            elif cate_name == "2":
                cate_name = "white"
            else:
                cate_name = "empty"
            seg, area, roi = cale_ppt_from_region(rgn["points"])
            # todo:
            cat_id = find_category_id(cate_name, categories_list)
            # 定义注释信息
            annotation_info = {
                "id": get_next_annotation_info_id(),
                "image_id": sub_image_info_id,
                "category_id": cat_id,  # 替换为实际类别ID
                "segmentation": [seg],
                "area": area,  # 替换为实际区域面积
                "bbox": roi,  # 替换为实际边界框信息 [x, y, width, height]
                "iscrowd": 0
            }
            ann_list.append(annotation_info)
    coco_data = {
        "info": info,
        "licenses": licenses_list,
        "images": img_list,
        "annotations": ann_list,
        "categories": categories_list
    }
    with open(os.path.join(output_dir, 'coco_data.json'), 'w', encoding="utf-8") as json_file:
        json.dump(coco_data, json_file)


def coco_to_yolo(label_type="bwn"):
    # 相对路径的处理不够好，写死了，故相关路径的格式不能变
    ds_path = "../output/warp_back_straight/"

    output_path = f"../output/warp_back_straight/{label_type}"

    json_file_path = f"coco_data.json"
    txt_file_path = f"coco_data_{label_type}.txt"
    coco_to_yolo_fmt(ds_path
                     , output_path
                     , json_file_path, txt_file_path)


if __name__ == "__main__":
    pass
    # categories_list_b_w_n = [
    #     {
    #         "id": 1,
    #         "name": "black",
    #         "supercategory": "piece"
    #     },
    #     {
    #         "id": 2,
    #         "name": "white",
    #         "supercategory": "piece"
    #     },
    #     {
    #         "id": 3,
    #         "name": "empty",
    #         "supercategory": "piece"
    #     },
    # ]
    # pplabel_2_coco(categories_list_b_w_n)

    coco_to_yolo()
