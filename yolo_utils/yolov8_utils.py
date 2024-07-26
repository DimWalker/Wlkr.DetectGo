import os
import shutil

from ultralytics import YOLO


class YOLOv8:

    def __init__(self, model_path):
        self.model = YOLO(model_path)  # pretrained YOLOv8n model

    def __call__(self, img_path, **kwargs):
        """
        等同于self.predict
        :param img_path: 图片路径，图片路径数组等
        :return:
        """
        return self.predict(img_path, **kwargs)

    def predict(self, img_path, **kwargs):
        """

        :param img_path: 图片路径，图片路径数组等
        :param max_det: 默认300个目标
        :param conf: 可信分值
        :return:
        """
        # Run batched inference on a list of images
        # results = self.model(img_path)  # return a list of Results objects
        results = self.model.predict(img_path, **kwargs)
        return results

    def result_to_regions(self, result):
        """
        将Yolov8结果，转换为自定义的json格式
        :param result: 单个结果，如batch_size是1时，即results[0]
        :return:
        """
        json_obj = []
        boxes = result.boxes
        for idx in range(len(boxes)):
            o = {
                "cls": int(boxes.cls[idx]),
                "conf": float(boxes.conf[idx]),
                "name": result.names[int(boxes.cls[idx])],
                "xmin": float(boxes.xyxy[idx][0]),
                "xmax": float(boxes.xyxy[idx][2]),
                "ymin": float(boxes.xyxy[idx][1]),
                "ymax": float(boxes.xyxy[idx][3])
            }
            o["region"] = [[o["xmin"], o["ymin"]],
                           [o["xmax"], o["ymin"]],
                           [o["xmax"], o["ymax"]],
                           [o["xmin"], o["ymax"]]]
            o["center"] = [(o["xmin"] + o["xmax"]) / 2, (o["ymin"] + o["ymax"]) / 2]
            json_obj.append(o)
        return json_obj


def o_c_test():
    output_dir = "results_o_c"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    model = YOLOv8("../yolov8n_o_c/runs/detect/train4/weights/best.pt")
    results = model.predict("../output/go_board_dataset_all/eval/")
    # Process results list
    cnt = 0
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        # result.show()  # display to screen
        result.save(filename=f"{output_dir}/{cnt}.jpg")  # save to disk
        cnt += 1


def find_images_in_folder(folder_path):
    """
    在指定的文件夹中查找所有图片文件，不包含子文件夹中的文件。
    :param folder_path: 要搜索的文件夹路径
    :return: 图片文件的完整路径列表
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = []
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # 检查文件是否是图片
        if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in image_extensions:
            image_files.append(file_path)
    return image_files


def bwn_test():
    output_dir = "results_bwn"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    model = YOLOv8("../yolov8l_bwn/runs/detect/train4/weights/best.pt")
    img_list = find_images_in_folder("../output/warp_back_straight/")
    img_list = img_list[:50]
    results = []
    for img_path in img_list:
        results += model.predict(img_path)
    # Process results list
    cnt = 0
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        # result.show()  # display to screen
        result.save(filename=f"{output_dir}/{cnt}.jpg")  # save to disk
        cnt += 1


if __name__ == "__main__":
    # o_c_test()
    bwn_test()
