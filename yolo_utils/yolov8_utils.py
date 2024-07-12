import os
import shutil

from ultralytics import YOLO


class YOLOv8:

    def __init__(self, model_path):
        self.model = YOLO(model_path)  # pretrained YOLOv8n model

    def __call__(self, img_path):
        """
        等同于self.predict
        :param img_path: 图片路径，图片路径数组等
        :return:
        """
        return self.predict(img_path)

    def predict(self, img_path):
        """

        :param img_path: 图片路径，图片路径数组等
        :return:
        """
        # Run batched inference on a list of images
        results = self.model(img_path)  # return a list of Results objects
        return results


if __name__ == "__main__":
    if os.path.exists("results"):
        shutil.rmtree("results")
    os.makedirs("results")

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
        result.save(filename=f"results/{cnt}.jpg")  # save to disk
        cnt += 1
