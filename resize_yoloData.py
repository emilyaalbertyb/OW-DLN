#将2446*1000全部resize到512*512，然后将对应的标注yolo格式的txt中的坐标也按照比例缩放，

import os

import cv2
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

from PIL import Image
import os

from PIL import Image
import os


def draw_boxes_on_image(image_path, label_path):
    """
    在图片上绘制 YOLO 格式的标注框
    """
    # 读取图片
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 方便 Matplotlib 显示
    # 获取图像的宽度和高度
    height, width, _ = image.shape
    #读取标注文件后画出框
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 处理每个标注
    for line in lines:
        # 拆分每一行
        parts = line.strip().split()

        # 解析标注信息
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # 将相对坐标转换为像素坐标
        x_center_pixel = int(x_center * width)
        y_center_pixel = int(y_center * height)
        box_width_pixel = int(box_width * width)
        box_height_pixel = int(box_height * height)

        # 计算左上角和右下角的坐标
        x1 = int(x_center_pixel - box_width_pixel / 2)
        y1 = int(y_center_pixel - box_height_pixel / 2)
        x2 = int(x_center_pixel + box_width_pixel / 2)
        y2 = int(y_center_pixel + box_height_pixel / 2)

        # 绘制矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框，线宽为2

    return image

def compare_images(image1_path, label1_path, image2_path, label2_path):
    """
    比较两张图片，绘制其标注框并显示在对比图中
    """
    # 在两张图片上绘制标注框
    image1 = draw_boxes_on_image(image1_path, label1_path)
    image2 = draw_boxes_on_image(image2_path, label2_path)

    # 显示对比图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title("Image 1 with Annotations")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title("Image 2 with Annotations")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# #只需要进行resize图片，标注文件不需要改，因为是比例
# # 根据标签文件画出框
# image1_path = "dataset/images/train/0a8f67b9c53152c20940589623.jpg"
# label1_path = "dataset/labels/train/0a8f67b9c53152c20940589623.txt"
# image2_path = "yolo_resize/dataset/images/train/0a8f67b9c53152c20940589623.jpg"
# label2_path = "dataset/labels/train/0a8f67b9c53152c20940589623.txt"
#
# # 调用函数显示对比图
# compare_images(image1_path, label1_path, image2_path, label2_path)


# resize图片和标注文件
# 设置路径和尺寸
# input_image_dir = "dataset/images/train"
# input_label_dir = "dataset/labels/train"
# output_image_dir = "yolo_resize/dataset/images/train"
# output_label_dir = "yolo_resize/dataset/labels/train"
# original_size = (2446, 1000)
# new_size = (512, 512)
#
# # 处理数据集
# process_dataset(input_image_dir, input_label_dir, output_image_dir, output_label_dir, original_size, new_size)
#
# print("处理完成！")


#resize文件夹中的图片

# def resize_image(image_path, new_size, output_image_dir):
#     """
#     将图片调整为指定尺寸
#     """
#     # 打开图片
#     image = Image.open(image_path)
#     # 调整尺寸
#     resized_image = image.resize(new_size)
#     # 保存图片
#     resized_image.save(output_image_dir + '/' + os.path.basename(image_path))
#
# # 设置路径和尺寸
# input_image_dir = "yolo_results/bus.jpg"
# output_image_dir = "yolo_resize/dataset/images"
# new_size = (512, 512)
#
# resize_image(input_image_dir, new_size, output_image_dir)

import os
from PIL import Image

# 设置文件夹路径
folder_path = 'defect_dataset/defect_dataset'  # 替换为你的图片文件夹路径
output_folder = 'defect_dataset/resize_defect_dataset'  # 替换为输出图片的文件夹路径

# 创建输出文件夹（如果不存在的话）
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中的所有图片文件
for filename in os.listdir(folder_path):
    # 检查文件是否是图片（可以根据文件扩展名判断）
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # 拼接文件路径
        img_path = os.path.join(folder_path, filename)

        # 打开图片
        with Image.open(img_path) as img:
            # 调整图片大小到 512x512
            img_resized = img.resize((512, 512))

            # 保存调整后的图片到输出文件夹
            output_path = os.path.join(output_folder, filename)
            img_resized.save(output_path)
            print(f"Saved resized image: {output_path}")
