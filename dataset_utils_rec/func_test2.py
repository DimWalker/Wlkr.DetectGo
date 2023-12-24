import json

import torch

# 指定你的权重文件路径
weights_path = r'..\runs\train\exp4\weights\best.pt'
# 指定图像文件路径
image_path = r'..\output\go_board_dataset_v3\eval\0a2aece808faa1108ac606b65bdcf75bc5ccca65.jpg'

#
# # 加载模型
# model = torch.load(weights_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
#
# # 切换模型为评估模式
# model.eval()
#
# # 打印模型结构
# print(model)

model = torch.hub.load(r'D:\WorkArea\Private_Project\yolov5', 'custom', path=weights_path, source='local')
# print(model)
result = model(image_path)
print(result)
json_obj = result.pandas().xyxy[0].to_json(orient='records')
json_obj = json.loads(json_obj)

corners = []
for cls in json_obj:
    if cls["name"] == "corner":
        corners.append(cls)
if len(corners) == 4:
    min_x, min_y, max_x, max_y = -1, -1, -1, -1
    for corner in corners:
        if corner["xmin"] < min_x or min_x == -1:
            min_x = corner["xmin"]
        if corner["ymin"] < min_y or min_y == -1:
            min_y = corner["ymin"]
        if corner["xmax"] > max_x or max_x == -1:
            max_x = corner["xmax"]
        if corner["ymax"] < max_y or max_y == -1:
            max_y = corner["ymax"]
    print((min_x, min_y, max_x, max_y))
else:
    print("corners < 4")
