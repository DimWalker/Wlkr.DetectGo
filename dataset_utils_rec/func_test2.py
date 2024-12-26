import json
import os

import cv2
import numpy as np
import torch

from Wlkr.Common.FileUtils import GetFileNameSplit
from Wlkr.iocr_utils import calc_distance


def calc_anchor_point(src_pt, corners):
    dst_pt = None
    min_len = None
    for corner in corners:
        if dst_pt is None:
            dst_pt = [corner["xmin"], corner["ymin"]]
            min_len = calc_distance(src_pt, dst_pt)

        l = calc_distance(src_pt, [corner["xmin"], corner["ymin"]])
        if l < min_len:
            dst_pt = [corner["xmin"], corner["ymin"]]
            min_len = l
        l = calc_distance(src_pt, [corner["xmax"], corner["ymin"]])
        if l < min_len:
            dst_pt = [corner["xmax"], corner["ymin"]]
            min_len = l
        l = calc_distance(src_pt, [corner["xmax"], corner["ymax"]])
        if l < min_len:
            dst_pt = [corner["xmax"], corner["ymax"]]
            min_len = l
        l = calc_distance(src_pt, [corner["xmin"], corner["ymax"]])
        if l < min_len:
            dst_pt = [corner["xmin"], corner["ymax"]]
            min_len = l
    return dst_pt


def board_warp_back(image_path, output_dir):
    bn, pre, ext = GetFileNameSplit(image_path)
    result = model(image_path)
    logging.info(result)
    json_obj = result.pandas().xyxy[0].to_json(orient='records')
    json_obj = json.loads(json_obj)

    corners = []
    for cls in json_obj:
        if cls["name"] == "corner":
            corners.append(cls)
    img = cv2.imread(image_path)
    if len(corners) == 4:
        color = (0, 255, 0)  # BGR颜色，这里是绿色
        thickness = 2

        min_x, min_y, max_x, max_y = -1, -1, -1, -1
        for corner in corners:
            if corner["xmin"] < min_x or min_x == -1:
                min_x = corner["xmin"]
            if corner["ymin"] < min_y or min_y == -1:
                min_y = corner["ymin"]
            if corner["xmax"] > max_x or max_x == -1:
                max_x = corner["xmax"]
            if corner["ymax"] > max_y or max_y == -1:
                max_y = corner["ymax"]
            cv2.rectangle(img, (int(corner["xmin"]), int(corner["ymin"])),
                          (int(corner["xmax"]), int(corner["ymax"])), color, thickness)

        min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)

        crop_img = img[min_y:max_y, min_x:max_x]
        # cv2.imshow('corner_minmax_crop', crop_img)
        cv2.imwrite(os.path.join(output_dir, pre + "_min_max" + ext), crop_img)

        # 计算rec任务，棋子高32像素，32*19=608
        # 24*19=456，为什么这个像素warp back后的东西很奇怪？
        wb_len=608
        src_pts = [[0, 0], [wb_len, 0], [wb_len, wb_len], [0, wb_len]]
        dst_pts = [
            calc_anchor_point(src_pts[0], corners),
            calc_anchor_point(src_pts[1], corners),
            calc_anchor_point(src_pts[2], corners),
            calc_anchor_point(src_pts[3], corners)
        ]
        M = cv2.getPerspectiveTransform(np.float32(dst_pts), np.float32(src_pts))
        new_image = img.copy()
        warped_image = cv2.warpPerspective(new_image, M, (wb_len, wb_len), borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=(255, 255, 255, 0))
        # cv2.imshow('corner_warp_crop', warped_image)
        cv2.imwrite(os.path.join(output_dir, pre + "_warp_back" + ext), warped_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        logging.info("corners < 4")


def show_all():
    output_dir = "../output/warp_back"
    input_dir = "../output/go_board_dataset_v3/eval"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_list = os.listdir(input_dir)
    for img_path in input_list:
        if not img_path.endswith(".jpg"):
            continue
        board_warp_back(os.path.join(input_dir, img_path), output_dir)


if __name__ == "__main__":
    # OK
    #image_path = r'..\output\go_board_dataset_v3\eval\0bd64ecc6579b78313d2e90226550051985674d7.jpg'
    # 直线可以，像素太差
    #image_path = r'..\output\go_board_dataset_v3\eval\0b3dc72c2003eda0e4fa0b9994a583bcacea95db.jpg'
    # OK
    #image_path = r'..\output\go_board_dataset_v3\eval\0ccfe16896bed86140adb065c119090498eaebcb.jpg'
    weights_path = r'..\runs\train\exp\weights\best.pt'
    model = torch.hub.load(r'../../yolov5', 'custom', path=weights_path, source='local')
    #board_warp_back(image_path)

    show_all()
