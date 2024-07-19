import json
import os.path
import shutil
from copy import deepcopy

import cv2
import numpy as np
from Wlkr.Common.FileUtils import GetFileNameSplit
from Wlkr.iocr_utils import calc_distance
from dataset_utils.B001_transform_diagram import calc_warp_point
from dataset_utils.Z999_common_utils import remove_module_dir
from yolo_utils.yolov8_utils import YOLOv8


def load_coco_data(raw_dir):
    with open(os.path.join(raw_dir, "coco_data_all.json"), "r", encoding="utf-8") as f:
        coco_data = json.load(f)
    return coco_data


def load_label(raw_dir):
    with open(os.path.join(raw_dir, "coco_label.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    sc_wp_list = []
    for line in lines:
        sc_wp_list.append(line.rstrip().split('\t'))
    return sc_wp_list


def find_category_id(coco_data, name, raise_ex=True):
    res = [x for x in coco_data["categories"] if x["name"] == name]
    if len(res) == 1:
        return res[0]["id"]
    if raise_ex:
        raise Exception("category_id not found.")
    else:
        return None


def find_row_segmentation(coco_data, sc_wp_list, img_name, cate_id_list: list, M):
    img_info = [x for x in coco_data["images"] if x["file_name"] == img_name]
    if len(img_info) == 0:
        return None
    warp_img_path = [x for x in sc_wp_list if img_name in x[0]]
    if len(warp_img_path) == 0:
        return None

    img_info = img_info[0]
    warp_img_path = warp_img_path[0][1]
    ann_list = [x for x in coco_data["annotations"] if x["category_id"] in cate_id_list
                and x["image_id"] == img_info["id"]]
    with open(warp_img_path + ".json", "r", encoding="utf-8") as f:
        json_obj = json.load(f)
    # PPOCRLabel的数据结构，p_dir实际为可选
    # p_dir/img_name [{"transcription": "", "points": [[],[],[],[]], "difficult": false, "key_cls": "None"}]
    trs = []

    diagram_shape = deepcopy(json_obj["diagram"])
    # todo:貌似无法兼容行与棋子的代码
    l = len(ann_list)
    if l > 19:
        diagram_shape = [item for sublist in diagram_shape for item in sublist]
    for a, ann in enumerate(ann_list):

        # 棋子是16点，其他是4点
        # 棋子是二维数组，其他一维
        pnts = []
        for pidx in range(int(len(ann["segmentation"][0]) / 2)):
            pnts.append(
                calc_warp_point(M,
                                ann["segmentation"][0][pidx * 2],
                                ann["segmentation"][0][pidx * 2 + 1])
            )

            # lt = calc_warp_point(M, ann["segmentation"][0][0], ann["segmentation"][0][1])
            # rt = calc_warp_point(M, ann["segmentation"][0][2], ann["segmentation"][0][3])
            # rb = calc_warp_point(M, ann["segmentation"][0][4], ann["segmentation"][0][5])
            # lb = calc_warp_point(M, ann["segmentation"][0][6], ann["segmentation"][0][7])
        if l > 19:
            transcription = str(diagram_shape[a])
            # 计算最小x和最大x
            min_x = min(point[0] for point in pnts)
            max_x = max(point[0] for point in pnts)
            # 计算最小y和最大y
            min_y = min(point[1] for point in pnts)
            max_y = max(point[1] for point in pnts)
            pnts = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
        else:
            transcription = ''.join(map(str, diagram_shape[a]))

        text_region = {"transcription": transcription,
                       "points": pnts,  # [lt, rt, rb, lb],
                       "difficult": False,
                       "key_cls": "None"}
        trs.append(text_region)
    return json.dumps(trs, ensure_ascii=False)


def find_row_segmentation_straight(img_path, M):
    with open(img_path + ".json", "r", encoding="utf-8") as f:
        json_obj = json.load(f)
    trs = []

    diagram_shape = deepcopy(json_obj["diagram"])
    pieces_seg_shape = deepcopy(json_obj["pieces_seg"])
    diagram_shape = [item for sublist in diagram_shape for item in sublist]
    pieces_seg_shape = [item for sublist in pieces_seg_shape for item in sublist]

    for a, ann in enumerate(diagram_shape):
        # 棋子是16点
        pnts = []
        for pidx in range(len(pieces_seg_shape[a])):
            pnts.append(
                calc_warp_point(M,
                                pieces_seg_shape[a][pidx][0],
                                pieces_seg_shape[a][pidx][1])
            )

        transcription = str(diagram_shape[a])
        # 计算最小x和最大x
        min_x = min(point[0] for point in pnts)
        max_x = max(point[0] for point in pnts)
        # 计算最小y和最大y
        min_y = min(point[1] for point in pnts)
        max_y = max(point[1] for point in pnts)
        pnts = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

        text_region = {"transcription": transcription,
                       "points": pnts,
                       "difficult": False,
                       "key_cls": "None"}
        trs.append(text_region)

    # 棋盘
    pnts = []
    for pidx in range(len(json_obj["dst_pts"])):
        pnts.append(
            calc_warp_point(M,
                            json_obj["dst_pts"][pidx][0],
                            json_obj["dst_pts"][pidx][1])
        )
    transcription = "board"
    # 计算最小x和最大x
    min_x = min(point[0] for point in pnts)
    max_x = max(point[0] for point in pnts)
    # 计算最小y和最大y
    min_y = min(point[1] for point in pnts)
    max_y = max(point[1] for point in pnts)
    pnts = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

    text_region = {"transcription": transcription,
                   "points": pnts,
                   "difficult": False,
                   "key_cls": "None"}
    trs.append(text_region)
    # 4个角
    for a, ann in enumerate(json_obj["corners"]):
        pnts = []
        for pidx in range(len(ann)):
            pnts.append(
                calc_warp_point(M,
                                ann[pidx][0],
                                ann[pidx][1])
            )

        transcription = "corner"
        # 计算最小x和最大x
        min_x = min(point[0] for point in pnts)
        max_x = max(point[0] for point in pnts)
        # 计算最小y和最大y
        min_y = min(point[1] for point in pnts)
        max_y = max(point[1] for point in pnts)
        pnts = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

        text_region = {"transcription": transcription,
                       "points": pnts,
                       "difficult": False,
                       "key_cls": "None"}
        trs.append(text_region)
    return json.dumps(trs, ensure_ascii=False)


def warp_back(output_dir, dataset_name, pre_dir_name, raw_dir, cate_name_list: list):
    # todo: fileState.txt需要绝对路径，目前hardcode写死路径
    abs_dir = os.path.dirname(os.path.dirname(__file__))
    abs_dir = abs_dir.replace("/", "\\")
    abs_dir = os.path.join(abs_dir, "output", dataset_name, pre_dir_name)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    coco_data = load_coco_data(raw_dir)
    sc_wp_list = load_label(raw_dir)
    input_list = os.listdir(raw_dir)
    cate_id_list = []
    for x in cate_name_list:
        cate_id_list.append(find_category_id(coco_data, x))
    res_list = []
    stat_list = []
    for img_path in input_list:
        if not img_path.endswith(".jpg"):
            continue
        bn, _, _ = GetFileNameSplit(img_path)
        img_path = os.path.join(raw_dir, img_path)
        M, save_name = board_warp_back(img_path, output_dir, False)
        if M is None:
            continue
        line = find_row_segmentation(coco_data, sc_wp_list, bn, cate_id_list, M)
        if line is not None:
            res_list.append(f"{pre_dir_name}/{save_name}\t{line}\n")
            stat_list.append(f"{abs_dir}\\{save_name}\t1\n")
        else:
            print("line is None.")
    with open(os.path.join(output_dir, "Label.txt"), "w", encoding="utf-8") as f:
        f.writelines(res_list)
    with open(os.path.join(output_dir, "fileState.txt"), "w", encoding="utf-8") as f:
        f.writelines(stat_list)


def warp_back_straight(output_dir, raw_dir, cate_name_list: list):
    """
    从json记录中
    :param cnt_limit:
    :param skip_cnt:
    :return:
    """

    abs_dir = os.path.dirname(os.path.dirname(__file__))
    abs_dir = abs_dir.replace("/", "\\")
    abs_dir = os.path.join(abs_dir, "output", "warp_back_straight")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_list = os.listdir(raw_dir)
    res_list = []
    stat_list = []

    for img_path in input_list:
        if not img_path.endswith(".png"):
            continue

        bn, _, _ = GetFileNameSplit(img_path)
        img_path = os.path.join(raw_dir, img_path)

        M, save_name = board_warp_back_straight(img_path, output_dir, False)
        if M is None:
            continue

        line = find_row_segmentation_straight(img_path, M)
        if line is not None:
            res_list.append(f"warp_back_straight/{save_name}\t{line}\n")
            stat_list.append(f"{abs_dir}\\{save_name}\t1\n")
        else:
            print("line is None.")
    with open(os.path.join(output_dir, "Label.txt"), "w", encoding="utf-8") as f:
        f.writelines(res_list)
    with open(os.path.join(output_dir, "fileState.txt"), "w", encoding="utf-8") as f:
        f.writelines(stat_list)


def board_warp_back(image_path, output_dir, skip_save=None):
    print("warp_back " + image_path)
    bn, pre, ext = GetFileNameSplit(image_path)
    result = model(image_path)

    # v8格式
    json_obj = []
    for idx in range(len(result[0].boxes)):
        boxes = result[0].boxes
        json_obj.append(
            {
                "cls": int(boxes.cls[idx]),
                "conf": float(boxes.conf[idx]),
                "name": result[0].names[int(boxes.cls[idx])],
                "xmin": float(boxes.xyxy[idx][0]),
                "xmax": float(boxes.xyxy[idx][2]),
                "ymin": float(boxes.xyxy[idx][1]),
                "ymax": float(boxes.xyxy[idx][3])
            }
        )

    # v5格式
    # json_obj = result[0].pandas().xyxy[0].to_json(orient='records')
    # json_obj = json.loads(json_obj)

    corners = []
    for cls in json_obj:
        if cls["name"] == "corner":
            corners.append(cls)
    img = cv2.imread(image_path)
    if len(corners) == 4:
        # 原图截图
        # color = (0, 255, 0)  # BGR颜色，这里是绿色
        # thickness = 2
        # min_x, min_y, max_x, max_y = -1, -1, -1, -1
        # for corner in corners:
        #     if corner["xmin"] < min_x or min_x == -1:
        #         min_x = corner["xmin"]
        #     if corner["ymin"] < min_y or min_y == -1:
        #         min_y = corner["ymin"]
        #     if corner["xmax"] > max_x or max_x == -1:
        #         max_x = corner["xmax"]
        #     if corner["ymax"] > max_y or max_y == -1:
        #         max_y = corner["ymax"]
        #     cv2.rectangle(img, (int(corner["xmin"]), int(corner["ymin"])),
        #                   (int(corner["xmax"]), int(corner["ymax"])), color, thickness)
        # min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
        # crop_img = img[min_y:max_y, min_x:max_x]
        # cv2.imwrite(os.path.join(output_dir, pre + "_min_max" + ext), crop_img)

        # 计算rec任务，棋子高32像素，32*19=608
        # 24*19=456，为什么这个像素warp back后的东西很奇怪，如没有图像
        wb_len = 608
        of_len = 60
        src_pts = [[of_len, of_len], [wb_len + of_len, of_len],
                   [wb_len + of_len, wb_len + of_len], [of_len, wb_len + of_len]]
        dst_pts = [
            calc_anchor_point(src_pts[0], corners),
            calc_anchor_point(src_pts[1], corners),
            calc_anchor_point(src_pts[2], corners),
            calc_anchor_point(src_pts[3], corners)
        ]
        # draw_region(img, dst_pts)
        # if not os.path.exists(os.path.join(output_dir, "..", "mid_img")):
        #     os.makedirs(os.path.join(output_dir, "..", "mid_img"))
        # cv2.imwrite(os.path.join(output_dir, "..", "mid_img", pre + "_wb" + ext), img)

        M = cv2.getPerspectiveTransform(np.float32(dst_pts), np.float32(src_pts))
        save_name = pre + "_wb" + ext
        if not skip_save:
            new_image = img.copy()
            warped_image = cv2.warpPerspective(new_image, M, (wb_len + of_len * 2, wb_len + of_len * 2),
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=(255, 255, 255, 0))
            # 保存
            cv2.imwrite(os.path.join(output_dir, pre + "_wb" + ext), warped_image)
        return M, save_name
    else:
        print("corners len: " + str(len(corners)))
        return None, None


def board_warp_back_straight(image_path, output_dir, skip_save=None):
    print("board_warp_back_straight " + image_path)
    with open(image_path + ".json", "r", encoding="utf-8") as f:
        json_obj = json.load(f)
    wb_len = 608
    of_len = 60
    src_pts = [[of_len, of_len], [wb_len + of_len, of_len],
               [wb_len + of_len, wb_len + of_len], [of_len, wb_len + of_len]]
    dst_pts = json_obj["dst_pts"]
    M = cv2.getPerspectiveTransform(np.float32(dst_pts), np.float32(src_pts))

    bn, pre, ext = GetFileNameSplit(image_path)
    img = cv2.imread(image_path)
    save_name = pre + "_wb" + ext
    if not skip_save:
        new_image = img.copy()
        warped_image = cv2.warpPerspective(new_image, M, (wb_len + of_len * 2, wb_len + of_len * 2),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=(255, 255, 255, 0))
        # 保存
        cv2.imwrite(os.path.join(output_dir, pre + "_wb" + ext), warped_image)
    return M, save_name


def calc_anchor_point(src_pt, corners):
    dst_pt = None
    min_len = None
    for corner in corners:
        center = [(corner["xmin"] + corner["xmax"]) / 2, (corner["ymin"] + corner["ymax"]) / 2]
        l = calc_distance(src_pt, center)
        if dst_pt is None:
            dst_pt = center
            min_len = l
        else:
            if l < min_len:
                dst_pt = center
                min_len = l

        # if dst_pt is None:
        #     dst_pt = [corner["xmin"], corner["ymin"]]
        #     min_len = calc_distance(src_pt, dst_pt)
        # else:
        #     l = calc_distance(src_pt, [corner["xmin"], corner["ymin"]])
        #     if l < min_len:
        #         dst_pt = [corner["xmin"], corner["ymin"]]
        #         min_len = l
        # l = calc_distance(src_pt, [corner["xmax"], corner["ymin"]])
        # if l < min_len:
        #     dst_pt = [corner["xmax"], corner["ymin"]]
        #     min_len = l
        # l = calc_distance(src_pt, [corner["xmax"], corner["ymax"]])
        # if l < min_len:
        #     dst_pt = [corner["xmax"], corner["ymax"]]
        #     min_len = l
        # l = calc_distance(src_pt, [corner["xmin"], corner["ymax"]])
        # if l < min_len:
        #     dst_pt = [corner["xmin"], corner["ymax"]]
        #     min_len = l
    return dst_pt


# cnt_limit = 2000000

if __name__ == "__main__":
    pass
    # model = YOLOv8("../yolov8n_o_c/runs/detect/train4/weights/best.pt")

    # warp back后，空棋子的偏移较大（没在中心的十字上）
    # 改用其他方式生成棋子的训练集
    cate_name_lsit = ["board", "corner", "black", "white", "empty"]
    # raw_dir = "../output/go_board_dataset_all/eval"
    # dataset_name = "diagram_pplabel_dataset_bwn"
    # pre_dir_name = "eval"
    # output_dir = f"../output/{dataset_name}/{pre_dir_name}"
    # warp_back(output_dir, dataset_name, pre_dir_name, raw_dir, cate_name_lsit)
    #
    # raw_dir = "../output/go_board_dataset_all/train"
    # dataset_name = "diagram_pplabel_dataset_bwn"
    # pre_dir_name = "train"
    # output_dir = f"../output/{dataset_name}/{pre_dir_name}"
    # warp_back(output_dir, dataset_name, pre_dir_name, raw_dir, cate_name_lsit)

    # 端正的数据集

    raw_dir = "../output/diagram_warp"
    output_dir = f"../output/warp_back_straight"
    remove_module_dir(output_dir)
    warp_back_straight(output_dir, raw_dir, cate_name_lsit)
