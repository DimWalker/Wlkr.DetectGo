import json
import os.path
import random
from datetime import datetime
import cv2
import numpy as np

################## Coco Dataset
# 定义类别信息
from utils.B002_combine_scene import combine_scene_image, offset_json_obj

info = {
    "description": "COCO Dataset",
    "url": "",
    "version": "1.0",
    "year": 2023,
    "contributor": "Walker Chan",
    "date_created": "2023-12-14"
}
licenses_list = []
categories_list = [
    {
        "id": 1,
        "name": "board",
        "supercategory": "board"
    },
    {
        "id": 2,
        "name": "corner",
        "supercategory": "board"
    }
    # {
    #     "id": ,
    #     "name": "row",
    #     "supercategory": "board"
    # }
    # {
    #     "id": ,
    #     "name": "col",
    #     "supercategory": "board"
    # }

]
image_info_id = 0
annotation_info_id = 0


def find_category_id(name):
    res = [x for x in categories_list if x["name"] == name]
    if len(res) == 1:
        return res[0]["id"]
    raise Exception("category_id not found.")


def coco_image_info(json_obj, jpg_img, scene):
    global image_info_id
    global annotation_info_id
    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 将日期和时间格式化为指定格式
    formatted_datetime = current_datetime.strftime("%Y-%m-%d")
    h, w, _ = jpg_img.shape

    # 定义图像信息
    image_info_id += 1
    image_info = {
        "id": image_info_id,
        "width": w,  # 替换为实际图像宽度
        "height": h,  # 替换为实际图像高度
        "file_name": scene,  # 替换为实际图像文件名
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
        "category_id": find_category_id("board"),
        "segmentation": [seg],
        "area": area,  # 替换为实际区域面积
        "bbox": roi,  # 替换为实际边界框信息 [x, y, width, height]
        "iscrowd": 0
    }
    ann_list.append(annotation_info)
    return image_info, ann_list


def cale_ppt_from_region(region):
    rgns = np.array(region, dtype=np.int32)
    # 计算包围四个点形成的四边形的最大矩形
    roi = cv2.boundingRect(rgns)
    seg = [x for point in region for x in point]
    # 定义四个点坐标
    points = np.array(region, dtype=np.int32)
    # 使用 reshape 将四个点组织为一个轮廓
    contour = points.reshape((-1, 1, 2))
    # 计算四边形的面积
    area = cv2.contourArea(contour)
    return seg, area, roi


def do_coco_dataset():
    output_dir = "../output/dataset"
    # 手动下载并解压到 scene_dir
    # https://aistudio.baidu.com/datasetdetail/93975
    scene_dir = "../output/scene_images_2"
    # diagram_warp_dir = "../output/diagram_warp"
    label_path = "../output/diagram_img/label.txt"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 获取文件夹中的所有项
    scene_list = os.listdir(scene_dir)
    diagram_list = []
    coco_data = {
        "info": info,
        "licenses": licenses_list,
        "images": None,
        "annotations": None,
        "categories": categories_list
    }
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        dia_path, warp_path = line.rstrip().split('\t')
        warp_path = warp_path.replace("/diagram_img", "/diagram_warp")
        diagram_list.append([dia_path, warp_path])

    images = []
    annotations = []

    cnt = 0
    for scene in scene_list:
        if not scene.endswith(".jpg"):
            continue

        print("scene: " + scene)
        scene_path = os.path.join(scene_dir, scene)
        dia_path, warp_path = diagram_list[random.randint(0, len(diagram_list) - 1)]
        jpg_img, offset_x, offset_y, zoom = combine_scene_image(scene_path, warp_path)
        json_obj = offset_json_obj(warp_path + ".json", offset_x, offset_y, zoom)

        img_info, ann_list = coco_image_info(json_obj, jpg_img, scene)
        cv2.imwrite(os.path.join(output_dir, scene), jpg_img)
        images.append(img_info)
        annotations += ann_list
        cnt += 1
        if cnt == cnt_limit:
            break
    coco_data["images"] = images
    coco_data["annotations"] = annotations
    with open(os.path.join(output_dir, 'coco_data.json'), 'w') as json_file:
        json.dump(coco_data, json_file, indent=2)


cnt_limit = 99999999
if __name__ == "__main__":
    #do_coco_dataset()
    pass
