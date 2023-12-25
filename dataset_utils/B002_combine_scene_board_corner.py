# 读取 JPG 图像
import json
import os.path
import random
from datetime import datetime
import cv2
import numpy as np

################## Coco Dataset
# 定义类别信息
from dataset_utils.B002_combine_scene import combine_scene_image, offset_json_obj, draw_region

info = {
    "description": "COCO Dataset",
    "url": "",
    "version": "1.0",
    "year": 2023,
    "contributor": "Walker Chan",
    "date_created": datetime.now().strftime("%Y-%m-%d")
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
    }

]
image_info_id = 0
annotation_info_id = 0


def find_category_id(name):
    res = [x for x in categories_list if x["name"] == name]
    if len(res) == 1:
        return res[0]["id"]
    raise Exception("category_id not found.")


def coco_image_info(json_obj, jpg_img, dia_path, scene):
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
    diagram = np.genfromtxt(dia_path, delimiter=' ', dtype=np.int32, encoding="utf-8")

    # 角
    for r, row in enumerate(json_obj["regions"]):
        for c, cell in enumerate(row):
            if (r == 0 and c == 0) or (r == 0 and c == 18) or (r == 18 and c == 0) or (r == 18 and c == 18):
                annotation_info_id += 1
                seg, area, roi = cale_ppt_from_region(cell)
                cat_id = find_category_id("corner")
                # 定义注释信息
                annotation_info = {
                    "id": annotation_info_id,
                    "image_id": image_info_id,
                    "category_id": cat_id,  # 替换为实际类别ID
                    "segmentation": [seg],
                    "area": area,  # 替换为实际区域面积
                    "bbox": roi,  # 替换为实际边界框信息 [x, y, width, height]
                    "iscrowd": 0
                }
                ann_list.append(annotation_info)

    # 行
    for r, row in enumerate(json_obj["regions"]):
        row_region = [[row[0][0][0], row[0][0][1]],
                      [row[18][1][0], row[18][1][1]],
                      [row[18][2][0], row[18][2][1]],
                      [row[0][3][0], row[0][3][1]]]
        annotation_info_id += 1
        seg, area, roi = cale_ppt_from_region(row_region)
        cat_id = find_category_id("row")
        # 定义注释信息
        annotation_info = {
            "id": annotation_info_id,
            "image_id": image_info_id,
            "category_id": cat_id,  # 替换为实际类别ID
            "segmentation": [seg],
            "area": area,  # 替换为实际区域面积
            "bbox": roi,  # 替换为实际边界框信息 [x, y, width, height]
            "iscrowd": 0
        }
        ann_list.append(annotation_info)

    for c in range(19):
        col_region = [
            [json_obj["regions"][0][c][0][0], json_obj["regions"][0][c][0][1]],
            [json_obj["regions"][0][c][1][0], json_obj["regions"][0][c][1][1]],
            [json_obj["regions"][18][c][2][0], json_obj["regions"][18][c][2][1]],
            [json_obj["regions"][18][c][3][0], json_obj["regions"][18][c][3][1]],
        ]
        annotation_info_id += 1
        seg, area, roi = cale_ppt_from_region(col_region)
        cat_id = find_category_id("col")
        # 定义注释信息
        annotation_info = {
            "id": annotation_info_id,
            "image_id": image_info_id,
            "category_id": cat_id,  # 替换为实际类别ID
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
    output_dir = "../output/" + dataset_name + "/" + dataset_type
    # 手动下载并解压到 scene_dir
    # https://aistudio.baidu.com/datasetdetail/93975
    scene_dir = "../output/scene_images_" + dataset_type
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

        img_info, ann_list = coco_image_info(json_obj, jpg_img, dia_path, scene)
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


def try_to():
    output_dir = "../output/scene_draw/" + dataset_type
    dataset_dir = "../output/" + dataset_name + "/" + dataset_type
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(dataset_dir, "coco_data.json")) as f:
        coco_data = json.load(f)
    for image in coco_data["images"]:
        img_path = os.path.join(dataset_dir, image["file_name"])
        img = cv2.imread(img_path)
        for ann in coco_data["annotations"]:
            if image["id"] == ann["image_id"]:
                draw_region(img, ann["segmentation"])
                draw_region(img, ann["bbox"])
        cv2.imwrite(os.path.join(output_dir, image["file_name"]), img)


dataset_name = "go_board_dataset_v3"
dataset_type = "train"
cnt_limit = 99999999
if __name__ == "__main__":
    do_coco_dataset()
    try_to()
