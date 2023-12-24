import torch

#model = torch.hub.load('.', 'custom', 'runs/train/exp4/weights/best.pt', source='local')

model = torch.hub.load('ultralytics/yolov5', 'custom', r'F:\Project_Private\Wlkr.DetectGo\runs\train\exp4\weights\best.pt')
# Images
img = r"F:\Project_Private\Wlkr.DetectGo\output\go_board_dataset_v3\eval\0a2aece808faa1108ac606b65bdcf75bc5ccca65.jpg"  # or file, Path, PIL, OpenCV, numpy, list
# Inference
results = model(img)
# Results
results.print()
