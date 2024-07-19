import json
import logging
import os
from datetime import datetime

import cv2
import numpy as np
from numpy import ndarray

from Wlkr.Common.FileUtils import GetFileNameSplit
from Wlkr.iocr_utils import point_in_region, calc_warp_point, sort_region_by, PointType
from dataset_utils_rec.A001_warp_back import calc_anchor_point
from yolo_utils.yolov8_utils import YOLOv8, find_images_in_folder


class Go_Detector():
    qi_mapping = {
        "black": "1",
        "white": "2",
        "empty": "0"
    }

    def __init__(self):
        self.model_o_c = YOLOv8("../yolov8n_o_c/runs/detect/train4/weights/best.pt")
        self.model_bwn = YOLOv8("../yolov8l_bwn/runs/detect/train4/weights/best.pt")
        self.output_dir = "../output/Go_Detector"
        self.qi_max_det = 400
        self.qi_conf = 0.75
        self.skip_save = False
        # self.threshold_qi = {
        #     "black": 0.8,
        #     "white": 0.8,
        #     "empty": 0.6,
        # }

    def execute(self, image_path):
        board_objects = self.detect_o_c(image_path)
        board_objects = self.calc_board_warp_back(board_objects)
        board_objects, warped_img_list = self.crop_board_warp_back(board_objects, image_path)
        board_objects = self.detect_bwn(board_objects, warped_img_list, image_path)
        board_objects = self.qi_regions_to_diagram(board_objects)
        self.print_diagram(board_objects)
        self.dump_result(board_objects, image_path)
        return board_objects

    def detect_o_c(self, image_path):
        logging.info("detect_o_c " + image_path)

        result = self.model_o_c(image_path)
        # v8格式转v5格式
        json_obj = []
        boxes = result[0].boxes
        for idx in range(len(boxes)):
            json_obj.append({
                "cls": int(boxes.cls[idx]),
                "conf": float(boxes.conf[idx]),
                "name": result[0].names[int(boxes.cls[idx])],
                "xmin": float(boxes.xyxy[idx][0]),
                "xmax": float(boxes.xyxy[idx][2]),
                "ymin": float(boxes.xyxy[idx][1]),
                "ymax": float(boxes.xyxy[idx][3])
            })

        board_objects = [x for x in json_obj if x["name"] == "board"]
        corner_objects = [x for x in json_obj if x["name"] == "corner"]
        for board in board_objects:
            board["corners"] = []
            board_region = [[board["xmin"], board["ymin"]],
                            [board["xmax"], board["ymin"]],
                            [board["xmax"], board["ymax"]],
                            [board["xmin"], board["ymax"]]]
            for corner in corner_objects:
                if point_in_region(board_region,
                                   [(corner["xmin"] + corner["xmax"]) / 2,
                                    (corner["ymin"] + corner["ymax"]) / 2]):
                    board["corners"].append(corner)
        return board_objects

    def calc_board_warp_back(self, board_objects):
        for board in board_objects:
            corners = board["corners"]
            if len(corners) == 4:
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
                M = cv2.getPerspectiveTransform(np.float32(dst_pts), np.float32(src_pts))
                board["M"] = M
                lt = calc_warp_point([board["xmin"], board["ymin"]], M)
                rt = calc_warp_point([board["xmax"], board["ymin"]], M)
                rb = calc_warp_point([board["xmax"], board["ymax"]], M)
                lb = calc_warp_point([board["xmin"], board["ymax"]], M)

                # 变换后，理论上是摆正的棋盘的点，(60,60)为了配合角的中心点
                board["src_pts"] = src_pts
                # 变换前，四个角的region中心点
                board["dst_pts"] = dst_pts
                # 变换后的正方形边长
                board["src_len"] = wb_len + of_len * 2
                # 变换后的棋盘region，有问题，这里不是正方形
                # 原本的意图是想利用它来确认棋子是否在棋盘的
                # 貌似没法利用，只能重新再识别一次棋盘？
                board["warped_region"] = [lt, rt, rb, lb]
        return board_objects

    def crop_board_warp_back(self, board_objects, image_path):
        img = cv2.imread(image_path)
        warped_img_list = []
        for board in board_objects:
            if "M" in board:
                M = board["M"]
                # 需要复制吗？
                # new_image = img.copy()
                warped_image = cv2.warpPerspective(img, M, (board["src_len"], board["src_len"]),
                                                   borderMode=cv2.BORDER_CONSTANT,
                                                   borderValue=(255, 255, 255, 0))
                warped_img_list.append(warped_image)
            else:
                warped_img_list.append(None)
        return board_objects, warped_img_list

    def detect_bwn(self, board_objects, warped_img_list, image_path):
        # todo : 再找一次棋盘
        bn, pre, ext = GetFileNameSplit(image_path)
        for idx_b, board in enumerate(board_objects):
            if warped_img_list[idx_b] is not None:
                result = self.model_bwn(warped_img_list[idx_b], max_det=400)
                json_obj = []
                boxes = result[0].boxes
                for idx in range(len(boxes)):
                    qi_obj = {
                        "cls": int(boxes.cls[idx]),
                        "conf": float(boxes.conf[idx]),
                        "name": result[0].names[int(boxes.cls[idx])],
                        "xmin": float(boxes.xyxy[idx][0]),
                        "xmax": float(boxes.xyxy[idx][2]),
                        "ymin": float(boxes.xyxy[idx][1]),
                        "ymax": float(boxes.xyxy[idx][3])
                    }
                    qi_obj["region"] = [[qi_obj["xmin"], qi_obj["ymin"]],
                                        [qi_obj["xmax"], qi_obj["ymin"]],
                                        [qi_obj["xmax"], qi_obj["ymax"]],
                                        [qi_obj["xmin"], qi_obj["ymax"]]]
                    json_obj.append(qi_obj)

                qi_regions = sort_region_by(json_obj, "region", PointType.Center, 3)
                board["qi_regions"] = qi_regions

                if not self.skip_save:
                    # now = datetime.now()
                    # formatted_time = now.strftime('%Y%m%d%H%M%S%f')
                    cv2.imwrite(f"{self.output_dir}/{pre}_{idx_b}_warp.{ext}", warped_img_list[idx_b])
                    result[0].save(filename=f"{self.output_dir}/{pre}_{idx_b}__det.{ext}"
                                   , conf=True, labels=True, show_name=None)  # save to disk

        return board_objects

    def qi_regions_to_diagram(self, board_objects):

        for idx, board in enumerate(board_objects):
            if "qi_regions" not in board:
                continue
            matrix = []
            for idx_r, row in enumerate(board["qi_regions"]):
                matrix.append([])
                for idx_c, col in enumerate(board["qi_regions"][idx_r]):
                    matrix[idx_r].append(Go_Detector.qi_mapping[board["qi_regions"][idx_r][idx_c]["name"]])
            board["diagram"] = matrix
        return board_objects

    def print_diagram(self, board_objects):
        for idx, board in enumerate(board_objects):
            if "diagram" in board:
                d = ""
                for idx_r, row in enumerate(board["diagram"]):
                    d += ",".join(row) + "\n"
                print(f"diagram {idx}:\n" + d)

    def dump_result(self, board_objects, image_path):
        bn, pre, ext = GetFileNameSplit(image_path)

        for idx, board in enumerate(board_objects):
            if "M" not in board:
                continue
            if isinstance(board["M"], ndarray):
                board["M"] = board["M"].tolist()
            json_path = os.path.join(self.output_dir, f"{pre}_{idx}.json")
            with open(json_path, mode="w", encoding="utf-8") as f:
                json.dump(board_objects, f)

    # def board_warp_back(self, image_path, skip_save=None):
    #     """
    #     todo: 角度少于阈值时，不warp_back
    #     todo: corner in board
    #     todo: 角少于4时怎么处理
    #     :return:
    #     """
    #
    #     logging.info("board_warp_back " + image_path)
    #     bn, pre, ext = GetFileNameSplit(image_path)
    #     result = self.model_o_c(image_path)
    #
    #     # v8格式转v5格式
    #     json_obj = []
    #     for idx in range(len(result[0].boxes)):
    #         boxes = result[0].boxes
    #         json_obj.append(
    #             {
    #                 "cls": int(boxes.cls[idx]),
    #                 "conf": float(boxes.conf[idx]),
    #                 "name": result[0].names[int(boxes.cls[idx])],
    #                 "xmin": float(boxes.xyxy[idx][0]),
    #                 "xmax": float(boxes.xyxy[idx][2]),
    #                 "ymin": float(boxes.xyxy[idx][1]),
    #                 "ymax": float(boxes.xyxy[idx][3])
    #             }
    #         )
    #
    #     corners = []
    #
    #     for cls in json_obj:
    #         if cls["name"] == "corner":
    #             corners.append(cls)
    #     img = cv2.imread(image_path)
    #     if len(corners) == 4:
    #         wb_len = 608
    #         of_len = 60
    #         src_pts = [[of_len, of_len], [wb_len + of_len, of_len],
    #                    [wb_len + of_len, wb_len + of_len], [of_len, wb_len + of_len]]
    #         # 四个角的region中心点，计算距离从而得出角的4个点
    #         dst_pts = [
    #             calc_anchor_point(src_pts[0], corners),
    #             calc_anchor_point(src_pts[1], corners),
    #             calc_anchor_point(src_pts[2], corners),
    #             calc_anchor_point(src_pts[3], corners)
    #         ]
    #         M = cv2.getPerspectiveTransform(np.float32(dst_pts), np.float32(src_pts))
    #
    #         new_image = img.copy()
    #         warped_image = cv2.warpPerspective(new_image, M, (wb_len + of_len * 2, wb_len + of_len * 2),
    #                                            borderMode=cv2.BORDER_CONSTANT,
    #                                            borderValue=(255, 255, 255, 0))
    #         n_h, n_w = wb_len + of_len * 2, wb_len + of_len * 2
    #         # o_h, o_w, _ = warped_image.shape
    #         # h, w = min(o_h, n_h), min(o_w, n_w)
    #         new_warped_img = np.zeros((n_h, n_w, 3), dtype=np.uint8)
    #         new_warped_img[:, :] = warped_image
    #
    #         save_name = None
    #         if not skip_save:
    #             # 保存
    #             save_name = pre + "_wb" + ext
    #             cv2.imwrite(os.path.join(self.output_dir, pre + "_wb" + ext), warped_image)
    #         return M, new_warped_img, save_name
    #     else:
    #         print("corners != 4")
    #         return None, None, None
    #
    # def qi_find(self, image_path, skip_save=None):
    #     results = self.model_bwn(image_path, max_det=400, conf=0.8)
    #     if not skip_save:
    #         if not os.path.exists(self.output_dir):
    #             os.makedirs(self.output_dir)
    #         for result in results:
    #             # 格式化时间输出，其中：
    #             # yyyy 表示四位年份
    #             # MM   表示两位月份
    #             # dd   表示两位日期
    #             # HH   表示两位小时（24小时制）
    #             # mm   表示两位分钟
    #             # ss   表示两位秒
    #             # fffffff 表示七位微秒
    #             now = datetime.now()
    #             formatted_time = now.strftime('%Y%m%d%H%M%S%f')
    #             result.save(filename=f"{self.output_dir}/{formatted_time}.jpg")  # save to disk
    #     return results


if __name__ == "__main__":
    # img_path = r"F:\Project_Private\Wlkr.DetectGo\output\go_board_dataset_all\eval\static_street_cambridge_outdoor_july_2005__img_2722.jpg"
    # img_path = r"F:\Project_Private\Wlkr.DetectGo\output\go_board_dataset_all\eval\static_street_outdoor_palma_mallorca_spain__IMG_0413.jpg"
    image_path_list = find_images_in_folder("F:\Project_Private\Wlkr.DetectGo\output\go_board_dataset_all\eval")
    gd = Go_Detector()
    for image_path in image_path_list:
        board_objects = gd.execute(image_path)

    pass
