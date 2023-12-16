# 读取 JPG 图像
import json
import os.path
import random
from datetime import datetime
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
    for r, row in enumerate(json_obj["matrix"]):
        for c, cell in enumerate(row):
            json_obj["matrix"][r][c] = [cell[0] * zoom + offset_x, cell[1] * zoom + offset_y]
    for p, point in enumerate(json_obj["dst_pts"]):
        json_obj["dst_pts"][p] = [point[0] * zoom + offset_x, point[1] * zoom + offset_y]
    for r, row in enumerate(json_obj["regions"]):
        for c, cell in enumerate(row):
            for rg, region in enumerate(cell):
                json_obj["regions"][r][c][rg] = [region[0] * zoom + offset_x, region[1] * zoom + offset_y]
    return json_obj


def draw_region(image, pts):
    if len(pts) == 1:
        # 顶点坐标需要reshape成OpenCV所需的格式
        points = np.array(pts[0], np.int32).reshape((4, 2))
        # 画四边形
        cv2.polylines(image, [points], isClosed=True,
                      color=(
                          random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 255),
                      thickness=1)
    else:
        # 定义矩形的左上角和右下角坐标
        x1, y1 = pts[0], pts[1]
        x2, y2 = pts[0] + pts[2], pts[1] + pts[3]
        # 定义矩形的颜色（BGR 格式）
        color = (0, 255, 0)  # 这里使用绿色
        # 绘制矩形
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=1)


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
categories_list = [
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
        "supercategory": "board"
    },
    {
        "id": 4,
        "name": "board",
        "supercategory": "board"
    },
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
    # {
    #     "id": ,
    #     "name": "corner",
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

    for r, row in enumerate(json_obj["regions"]):
        for c, cell in enumerate(row):
            annotation_info_id += 1
            seg, area, roi = cale_ppt_from_region(cell)
            cat_id = find_category_id("black" if diagram[r][c] == 1 else "white" if diagram[r][c] == 2 else "empty")
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
    output_dir = "../output/dataset"
    # 手动下载并解压到 scene_dir
    # https://aistudio.baidu.com/datasetdetail/93975
    scene_dir = "../output/scene_images"
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

        img_info, ann_list = coco_image_info(json_obj, jpg_img, dia_path, warp_path, scene)
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
    do_coco_dataset()
