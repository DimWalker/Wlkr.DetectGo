import json
import os

import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR

from Wlkr.Common.FileUtils import GetFileNameSplit
from dataset_utils_rec.A000_warp_back import calc_anchor_point


def board_warp_back(image_path, skip_save=None):
    """
    todo: 角度少于阈值时，不warp_back
    todo: corner in board
    todo: 角少于4时怎么处理
    :return:
    """

    logging.info("warp_back " + image_path)
    bn, pre, ext = GetFileNameSplit(image_path)
    result = model_bc(image_path)

    json_obj = result.pandas().xyxy[0].to_json(orient='records')
    json_obj = json.loads(json_obj)

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
        logging.info("corners < 4")
        return None, None


def board_split_row(image_path, threshole= 0.75,skip_save=None):
    logging.info("split row " + image_path)
    bn, pre, ext = GetFileNameSplit(image_path)
    result = model_r(image_path)
    json_obj = result.pandas().xyxy[0].to_json(orient='records')
    json_obj = json.loads(json_obj)

    # 加载了两次图片，感觉可以优化
    img = cv2.imread(image_path)
    row_cnt = 0

    json_obj = sorted(json_obj, key=lambda x: x["xmin"])
    json_obj = sorted(json_obj, key=lambda x: x["ymin"])
    rows = []
    for cls in json_obj:
        if cls["confidence"] < threshole:
            logging.info(f'{cls["confidence"]} < {threshole}')
            continue
        min_x, max_x, min_y, max_y = int(cls["xmin"]), int(cls["xmax"]), \
                                     int(cls["ymin"]), int(cls["ymax"])
        row = img[min_y:max_y, min_x:max_x]
        row_path = os.path.join(output_dir, pre + "_" + str(row_cnt)) + ext
        cv2.imwrite(row_path, row)
        rows.append((cls, row_path))
        row_cnt += 1
    return rows


def row_rec(rows):
    res_list = []
    for cls, img_path in rows:
        res = ocr.ocr(img_path, det=False, cls=False)
        # logging.info(res)
        res_list.append(res)
    return res_list


output_dir = "../output/detect_go"
input_dir = "../output/go_board_dataset_v3/eval"
if __name__ == "__main__":

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    weights_path = r'../runs/train/board_corner/weights/best.pt'
    model_bc = torch.hub.load(r'../../yolov5', 'custom', path=weights_path, source='local')

    weights_path = r'../runs/train/row/weights/best.pt'
    model_r = torch.hub.load(r'../../yolov5', 'custom', path=weights_path, source='local')

    ocr = PaddleOCR(use_pdserving=False,
                    use_angle_cls=False,
                    det=False,
                    cls=False,
                    use_gpu=True,
                    lang='ch',
                    show_log=True,
                    enable_mkldnn=False,
                    rec_model_dir="../runs/train/inference/diagram_rec/",
                    rec_char_dict_path="../runs/train/inference/diagram_rec/piece_keys.txt"
                    )
    img_list = os.listdir(input_dir)
    res_list = []
    for img_path in img_list:
        # img_path = r"../output/go_board_dataset_v3/eval/0a2aece808faa1108ac606b65bdcf75bc5ccca65.jpg"
        img_path = os.path.join(input_dir, img_path)
        M, wp_name = board_warp_back(img_path)
        if wp_name is None:
            continue
        wp_name = os.path.join(output_dir, wp_name)
        rows = board_split_row(wp_name,0.67)
        rec_res = row_rec(rows)
        res_list.append({
            "img_path": img_path,
            "rec_res": rec_res
        })
    with open(os.path.join(output_dir, "Rec_Res.txt"), "w", encoding="utf-8") as f:
        json.dump(res_list)
