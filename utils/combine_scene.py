# 读取 JPG 图像
import datetime
import json
import os.path
import random

import cv2

# jpg_image = cv2.imread('../assets/unclassified/images/05june05_static_street_boston__p1010736.jpg')  # 请替换为实际 JPG 图片路径
# # 读取 PNG 图像（包含 Alpha 通道）
# png_image = cv2.imread('../output/warp.png', cv2.IMREAD_UNCHANGED)  # 请替换为实际 PNG 图片路径
#
# # 检查图像是否成功加载
# if jpg_image is not None and png_image is not None:
#     # 获取 PNG 图像的 Alpha 通道
#     alpha_channel = png_image[:, :, 3]
#
#     # 将 PNG 图像合成到 JPG 图像中
#     for c in range(0, 3):
#         jpg_image[:500, :500, c] = jpg_image[:500, :500, c] * (1 - alpha_channel / 255) + \
#                                    png_image[:, :, c] * (alpha_channel / 255)
#
#     # 保存合成后的图像
#     cv2.imwrite('output_image.jpg', jpg_image)
#
#     # 显示原始 JPG 图像和合成后的图像
#     cv2.imshow('Original JPG Image', jpg_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Failed to load the images.")
import numpy as np

from Wlkr.Common.FileUtils import GetFileNameSplit


def combine_scene_image(jpg_path, png_path):
    jpg_img = cv2.imread(jpg_path)
    png_img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    jpg_h, jpg_w, _ = jpg_img.shape
    png_h, png_w, _ = png_img.shape
    # 随机偏移
    offset_x = random.randint(0, jpg_w - png_w)
    offset_y = random.randint(0, jpg_h - png_h)
    # 获取 PNG 图像的 Alpha 通道
    alpha_channel = png_img[:, :, 3]
    # 将 PNG 图像合成到 JPG 图像中
    for c in range(0, 3):
        jpg_img[:png_h, :png_w, c] = jpg_img[:png_h, :png_w, c] * (1 - alpha_channel / 255) + \
                                     png_img[:, :, c] * (alpha_channel / 255)
    return jpg_img, offset_x, offset_y


def offset_json_obj(json_path, offset_x, offset_y):
    with open(json_path, "r", encoding="utf-8") as f:
        json_obj = json.load(f)
    for r, row in enumerate(json_obj["matrix"]):
        for c, cell in enumerate(row):
            json_obj["matrix"][r][c] = [cell[0] + offset_x, cell[1] + offset_y]
    for c, cell in enumerate(json_obj["dst_pnts"]):
        json_obj["dst_pnts"][c] = [cell[0] + offset_x, cell[1] + offset_y]
    for r, row in enumerate(json_obj["regions"]):
        for c, cell in enumerate(row):
            for rg, region in enumerate(cell):
                json_obj["regions"][r][c][rg] = [region[0] + offset_x, region[1] + offset_y]
    return json_obj


################## Coco Dataset
# 定义类别信息

info = {
    "description": "COCO Dataset",
    "url": "",
    "version": "1.0",
    "year": 2023,
    "contributor": "Walker Chan",
    "date_created": "2023-12-14"
}
licenses_list = []
category_info_list = [
    {
        "id": 1,
        "name": "black",
        "supercategory": "black"
    },
    {
        "id": 2,
        "name": "white",
        "supercategory": "white"
    },
    {
        "id": 3,
        "name": "empty",
        "supercategory": "empty"
    },
    {
        "id": 4,
        "name": "board",
        "supercategory": "board"
    },
    # {
    #     "id": 5,
    #     "name": "corner",
    #     "supercategory": "corner"
    # }
]
image_info_id = 0
annotation_info_id = 0


def coco_image_info(json_obj, jpg_img, dia_path, tmpl_path):
    global image_info_id
    global annotation_info_id

    bn, pre, ext = GetFileNameSplit(dia_path)

    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 将日期和时间格式化为指定格式
    formatted_datetime = current_datetime.strftime("%Y-%m-%d")
    h, w, _ = jpg_img.shape

    # 定义图像信息
    image_info_id += 1
    image_info = {
        "id": image_info_id,
        "width": h,  # 替换为实际图像宽度
        "height": w,  # 替换为实际图像高度
        "file_name": bn,  # 替换为实际图像文件名
        "license": None,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": formatted_datetime
    }

    ann_list = []

    # 棋盘
    annotation_info_id += 1
    seg, area, roi = cale_ppt_from_region(json_obj["dst_pts"])
    annotation_info = {
        "id": annotation_info_id,
        "image_id": image_info_id,
        "category_id": 4,
        "segmentation": seg,
        "area": area,  # 替换为实际区域面积
        "bbox": roi,  # 替换为实际边界框信息 [x, y, width, height]
        "iscrowd": 0
    }
    ann_list.append(annotation_info)
    diagram = np.genfromtxt(tmpl_path, delimiter=' ', dtype=np.int32, encoding="utf-8")

    for r, row in enumerate(json_obj["regions"]):
        for c, cell in enumerate(row):
            annotation_info_id += 1
            seg, area, roi = cale_ppt_from_region(cell)
            cat_id = 1 if diagram[r][c] == 1 else 2 if diagram[r][c] == 2 else 3
            # 定义注释信息
            annotation_info = {
                "id": annotation_info_id,
                "image_id": image_info_id,
                "category_id": cat_id,  # 替换为实际类别ID
                "segmentation": seg,
                "area": area,  # 替换为实际区域面积
                "bbox": roi,  # 替换为实际边界框信息 [x, y, width, height]
                "iscrowd": 0
            }
            ann_list.append(annotation_info)
    return image_info, ann_list


def cale_ppt_from_region(region):
    # 计算包围四个点形成的四边形的最大矩形
    roi = cv2.boundingRect(region)
    seg = [x for point in region for x in point]
    # 定义四个点坐标
    points = np.array(region, dtype=np.int32)
    # 使用 reshape 将四个点组织为一个轮廓
    contour = points.reshape((-1, 1, 2))
    # 计算四边形的面积
    area = cv2.contourArea(contour)
    return seg, area, roi


def coco_dataset():
    output_dir = "../output/dataset"
    scene_dir = "../output/scene_images"
    diagram_warp_dir = "../output/diagram_warp"
    label_path = "../output/diagram_img/label.txt"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 获取文件夹中的所有项
    scene_list = os.listdir(scene_dir)
    diagram_list = []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        dia_path, warp_path = line.rstrip().split('\t')
        warp_path = warp_path.replace("/diagram_img/", "/diagram_warp/")
        diagram_list.append([dia_path, warp_path])

    for scene in scene_list:
        scene_path = os.path.join(scene_dir, scene)
        dia_path, warp_path = diagram_list[random.randint(0, len(diagram_list) - 1)]
        jpg_img, offset_x, offset_y = combine_scene_image(scene_path, warp_path)

        json_obj = offset_json_obj(warp_path + ".json", offset_x, offset_x)
