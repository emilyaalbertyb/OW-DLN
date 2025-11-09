#读取两张图片，对比差异，提取出差异部分
# 1.读取两张图片
# 2.将两张图片转换为灰度图
# 3.计算两张图片的差异
# 4.对差异进行二值化处理
# 5.保存差异图片
# 6.提取差异部分
# 7.在原图框选出差异部分

import cv2
import numpy as np
import os

def extract_defect(left_image_path, right_image_path, save_path):
    # 读取两张图片
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)
    # 将两张图片转换为灰度图
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    # 计算两张图片的差异
    diff = cv2.absdiff(left_gray, right_gray)
    # 对差异进行二值化处理
    _, binary = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
    # 保存差异图片
    cv2.imwrite(os.path.join(save_path, 'diff2.jpg'), binary)
    # 提取差异部分
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # 在原图框选出差异部分
        cv2.rectangle(left_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # 保存带有差异部分的图片
    cv2.imwrite(os.path.join(save_path, 'defect2.jpg'), left_image)

if __name__ == '__main__':
    left_image_path = 'img/0aea0556667f66d81013262223.jpg'
    right_image_path = 'repaired_img/0aea0556667f66d81013262223.jpg'
    save_path = 'extract_defect'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    extract_defect(left_image_path, right_image_path, save_path)