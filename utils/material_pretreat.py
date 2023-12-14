import os
import cv2
import numpy as np
from PIL import Image


def resize_images(src_path, dst_path, target_size=(60, 60), crop=None):
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
            # Resize the image
            resized_img = img.resize(target_size)
            # Save the resized image
            resized_img.save(dst_path)
    except Exception as e:
        print(f"Error processing {src_path}: {str(e)}")


def resize_dir():
    src_dir = r"../assets/未分类素材/扣"
    dst_dir = r"../assets/material"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        if filename.startswith("B") or filename.startswith("W"):
            resize_images(src_path, dst_path)
        else:
            resize_images(src_path, dst_path, (800, 800), True)


if __name__ == "__main__":
    resize_dir()
