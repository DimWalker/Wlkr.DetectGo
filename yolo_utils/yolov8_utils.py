import os

from ultralytics import YOLO


class YOLOv8:

    def __init__(self, model_path):
        self.model = YOLO(model_path)  # pretrained YOLOv8n model

    def predict(self, img_path):
        # Run batched inference on a list of images
        results = self.model(img_path)  # return a list of Results objects
        return results


if __name__ == "__main__":
    model = YOLOv8("../yolov8n_o_c/runs/detect/train4/weights/best.pt")
    results = model.predict("../output/go_board_dataset_all/eval/")
    # Process results list

    os.makedirs("results")
    cnt = 0
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        #result.show()  # display to screen
        result.save(filename=f"results/{cnt}.jpg")  # save to disk
        cnt += 1
