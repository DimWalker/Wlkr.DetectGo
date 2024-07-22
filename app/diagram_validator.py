import json
import os

import numpy as np

from Wlkr.Common.FileUtils import GetFileNameSplit


def val_diagram():
    det_res_path = "../output/Go_Detector"
    eval_path = "../output/go_board_dataset_all/eval"
    coco_label = eval_path + "/coco_label.txt"
    diagram_path = "../output/diagram_warp"

    with open(coco_label, mode="r", encoding="utf-8") as f:
        lines = f.readlines()

    ttl_cnt = 0
    scc_cnt = 0
    err_cnt = 0
    for file_name in os.listdir(det_res_path):
        if not file_name.endswith(".json"):
            continue

        ttl_cnt += 1
        bn, pre, ext = GetFileNameSplit(file_name)
        img_name = pre[0: pre.rfind("_")]
        # coco_label找棋谱mapping关系
        for line in lines:
            if img_name in line:
                diagram_path = line.split("\t")[1].rstrip() + ".json"
                with open(os.path.join(det_res_path, file_name), mode="r", encoding="utf-8") as f:
                    j1 = json.load(f)
                with open(diagram_path, mode="r", encoding="utf-8") as f:
                    j2 = json.load(f)
                d1 = j1["diagram"]
                d1 = [[int(cell) for cell in row] for row in d1]
                try:
                    d1 = np.array(d1)
                    d2 = np.array(j2["diagram"])
                    if np.array_equal(d1, d2):
                        scc_cnt += 1
                    else:
                        err_cnt += 1
                except Exception as ex:
                    err_cnt += 1

    print(f"ttl_cnt: {ttl_cnt}, scc_cnt: {scc_cnt}, err_cnt: {err_cnt}")


if __name__ == "__main__":
    val_diagram()
