import os
import cv2
import numpy as np
from PIL import Image


def resize_images(src_path, dst_path, target_size, crop=None):
    try:
        # Open the image file
        with Image.open(src_path) as img:
            w, h = img.size
            if crop:
                if w > h:
                    px = (w - h) / 2
                    img = img.crop((px, 0, w - px, h))
                elif w < h:
                    px = (h - w) / 2
                    img = img.crop((0, px, w, h - px))
            else:
                if w != h:
                    max_len = max(w, h)
                    new_image = Image.new('RGBA', (max_len, max_len), (0, 0, 0, 0))
                    if w > h:
                        px = (w - h) / 2
                        new_image.paste(img, (0, px))
                    else:
                        px = (h - w) / 2
                        new_image.paste(img, (px, 0))
                    img = new_image
            # Resize the image
            resized_img = img.resize(target_size)
            # Save the resized image
            resized_img.save(dst_path)
    except Exception as e:
        print(f"Error processing {src_path}: {str(e)}")


def resize_dir():
    src_dir = r"../assets/unclassified/kou"
    dst_dir = r"../assets/material"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for filename in os.listdir(src_dir):
        if not filename.endswith(".png"):
            continue
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        if filename.startswith("B") or filename.startswith("W"):
            resize_images(src_path, dst_path, (60, 60))
        else:
            resize_images(src_path, dst_path, (800, 800), True)


def change_color():
    # 读取图像
    image = cv2.imread('../assets/unclassified/kou/O018.png', cv2.IMREAD_UNCHANGED)  # 替换为你的图像路径
    # 定义颜色范围
    lower_color = np.array([220, 220, 220], dtype=np.uint8)
    upper_color = np.array([255, 255, 255], dtype=np.uint8)

    # 创建掩码，标记在颜色范围内的像素
    mask = cv2.inRange(image[:, :, :3], lower_color, upper_color)
    # 将颜色范围内的像素改为透明
    image[mask > 0, 3] = 0


    cv2.imwrite('../assets/unclassified/kou/O018.png', image)


if __name__ == "__main__":
    resize_dir()
    #change_color()
