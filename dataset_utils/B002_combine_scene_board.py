"""
场景合成，制作数据集
"""



import json
import os.path
import random
import threading
from datetime import datetime
import cv2
import numpy as np

################## Coco Dataset
# 定义类别信息
from Wlkr.Common.FileUtils import GetFileNameSplit
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


# categories_list = [
#     {
#         "id": 1,
#         "name": "board",
#         "supercategory": "board"
#     },
#     {
#         "id": 2,
#         "name": "corner",
#         "supercategory": "board"
#     },
#     {
#         "id": 3,
#         "name": "row",
#         "supercategory": "board"
#     },
#     {
#         "id": 4,
#         "name": "col",
#         "supercategory": "board"
#     },
#     {
#         "id": 5,
#         "name": "black",
#         "supercategory": "piece"
#     },
#     {
#         "id": 6,
#         "name": "white",
#         "supercategory": "piece"
#     },
#     {
#         "id": 7,
#         "name": "empty",
#         "supercategory": "piece"
#     },
# ]


def find_category_id(name, categories_list, raise_ex=True):
    res = [x for x in categories_list if x["name"] == name]
    if len(res) == 1:
        return res[0]["id"]
    if raise_ex:
        raise Exception("category_id not found.")
    else:
        return None


image_info_id = 0
annotation_info_id = 0
lock_image_info_id = threading.Lock()
lock_annotation_info_id = threading.Lock()


def get_next_image_info_id():
    global image_info_id
    with lock_image_info_id:
        image_info_id += 1
        sub_id = image_info_id
    return sub_id


def get_next_annotation_info_id():
    global annotation_info_id
    with lock_annotation_info_id:
        annotation_info_id += 1
        sub_id = annotation_info_id
    return sub_id


def coco_image_info(json_obj, jpg_img, dia_path, scene, categories_list):
    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 将日期和时间格式化为指定格式
    formatted_datetime = current_datetime.strftime("%Y-%m-%d")
    h, w, _ = jpg_img.shape

    sub_image_info_id = get_next_image_info_id()
    # 定义图像信息
    image_info = {
        "id": sub_image_info_id,
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
    if find_category_id("board", categories_list, False) is not None:
        seg, area, roi = cale_ppt_from_region(json_obj["board_region"])
        annotation_info = {
            "id": get_next_annotation_info_id(),
            "image_id": sub_image_info_id,
            "category_id": find_category_id("board", categories_list),
            "segmentation": [seg],
            "area": area,  # 替换为实际区域面积
            "bbox": roi,  # 替换为实际边界框信息 [x, y, width, height]
            "iscrowd": 0
        }
        ann_list.append(annotation_info)

    # 角
    if find_category_id("corner", categories_list, False) is not None:
        for corner in json_obj["corners"]:
            seg, area, roi = cale_ppt_from_region(corner)
            cat_id = find_category_id("corner", categories_list)
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
    # 行
    if find_category_id("row", categories_list, False) is not None:
        for row_region in json_obj["row_regions"]:
            seg, area, roi = cale_ppt_from_region(row_region)
            cat_id = find_category_id("row", categories_list)
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
    # 列
    if find_category_id("col", categories_list, False) is not None:
        for col_region in json_obj["col_regions"]:
            seg, area, roi = cale_ppt_from_region(col_region)
            cat_id = find_category_id("col", categories_list)
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
    # 棋子，圆形
    if find_category_id("black", categories_list, False) is not None:
        diagram = np.genfromtxt(dia_path, delimiter=' ', dtype=np.int32, encoding="utf-8")
        for r, row in enumerate(json_obj["pieces_seg"]):
            for c, cell in enumerate(row):
                seg, area, roi = cale_ppt_from_region(cell)
                cat_id = find_category_id(
                    "black" if diagram[r][c] == 1 else ("white" if diagram[r][c] == 2 else "empty"),
                    categories_list)
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


def do_coco_dataset(dataset_name, dataset_type, categories_list, categories_name):
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
    scene_label = []
    for scene in scene_list:
        if not scene.endswith(".jpg"):
            continue
        logging.info("scene: " + scene)
        scene_path = os.path.join(scene_dir, scene)
        dia_path, warp_path = diagram_list[random.randint(0, len(diagram_list) - 1)]
        jpg_img, offset_x, offset_y, zoom = combine_scene_image(scene_path, warp_path)
        json_obj = offset_json_obj(warp_path + ".json", offset_x, offset_y, zoom)
        img_info, ann_list = coco_image_info(json_obj, jpg_img, dia_path, scene, categories_list)
        cv2.imwrite(os.path.join(output_dir, scene), jpg_img)
        images.append(img_info)
        annotations += ann_list
        scene_label.append(f"{scene_path}\t{warp_path}\n")
        cnt += 1
        if cnt == cnt_limit:
            break
    coco_data["images"] = images
    coco_data["annotations"] = annotations
    with open(os.path.join(output_dir, f'coco_data_{categories_name}.json'), 'w', encoding="utf-8") as json_file:
        json.dump(coco_data, json_file)  # , indent=2
    with open(os.path.join(output_dir, "coco_label.txt"), 'w', encoding="utf-8") as sc_lb:
        sc_lb.writelines(scene_label)


def all_to_sub(dataset_name, dataset_type, categories_list_all, categories_list_sub: dict, categories_name_sub):
    """
    前置条件 coco_data_all.json
    :param dataset_name:
    :param dataset_type:
    :param categories_list_all:
    :param categories_list_sub:
    :param categories_name_sub:
    :return:
    """
    output_dir = "../output/" + dataset_name + "/" + dataset_type
    logging.info(output_dir)
    logging.info(categories_name_sub)
    with open(os.path.join(output_dir, 'coco_data_all.json'), 'r', encoding="utf-8") as json_file:
        json_obj = json.loads(json_file.read())
    id_list = []
    for a in categories_list_all:
        for s in categories_list_sub:
            if a["name"] == s["name"]:
                id_list.append(a["id"])
                s["ori_id"] = a["id"]

    json_obj["categories"] = categories_list_sub
    json_obj["annotations"] = [x for x in json_obj["annotations"]
                               if x["category_id"] in id_list]

    id_cnt = 0
    for x in json_obj["annotations"]:
        id_cnt += 1
        x["id"] = id_cnt
        x["category_id"] = [y["id"] for y in categories_list_sub
                            if y["ori_id"] == x["category_id"]][0]

    for x in categories_list_sub:
        if "ori_id" in x:
            del x["ori_id"]
    json_obj["categories"] = categories_list_sub
    with open(os.path.join(output_dir, f'coco_data_{categories_name_sub}.json'), 'w', encoding="utf-8") as json_file:
        json.dump(json_obj, json_file)


def try_to(dataset_name, dataset_type):
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
                # draw_region(img, ann["bbox"])
        cv2.imwrite(os.path.join(output_dir, image["file_name"]), img)


def pplabel_2_coco(pre_dir_name, categories_list):
    output_dir = "../output/diagram_det_rec_dataset/" + pre_dir_name
    with open(os.path.join(output_dir, "Label.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()

    img_list = []
    ann_list = []
    for line in lines:
        file_name, json_obj = line.rstrip().split("\t")
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
            seg, area, roi = cale_ppt_from_region(rgn["points"])
            cat_id = find_category_id("row")
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
        json.dump(coco_data, json_file, indent=2)


# 这个归档了，当它完美没bug了
# dataset_name = "go_board_dataset_v3"


cnt_limit = 99999999
# lock_obj = threading.Lock()
if __name__ == "__main__":
    # dataset_name = "diagram_det_rec_dataset"
    # dataset_name = "go_board_dataset_all"
    # dataset_type = "eval"

    # dataset_type = "eval"
    # do_coco_dataset()
    # dataset_type = "train"
    # do_coco_dataset()
    # try_to()

    pplabel_2_coco("ppocrlabel_dataset_eval")
    pplabel_2_coco("ppocrlabel_dataset")
