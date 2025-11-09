import cv2
import os
import random


def generate_random_masks(image_path, output_folder):
    # 读取原始图片
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"无法读取图片: {image_path}")
        return

    # 获取图片尺寸
    height, width = original_img.shape[:2]

    # 创建全黑的掩码图像
    mask = np.zeros((height, width), dtype=np.uint8)

    # 随机生成1-4个矩形
    num_rectangles = random.randint(1, 4)

    for _ in range(num_rectangles):
        # 随机生成矩形大小（小于200x200）
        rect_width = random.randint(10, 200)
        rect_height = random.randint(10, 200)

        # 随机生成矩形位置
        x = random.randint(0, width - rect_width - 1)
        y = random.randint(0, height - rect_height - 1)

        # 在掩码上绘制白色矩形
        cv2.rectangle(mask, (x, y), (x + rect_width, y + rect_height), 255, -1)

    # 获取原始图片文件名（不带扩展名）
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 保存掩码图像
    output_path = os.path.join(output_folder, f"{base_name}_mask.png")
    cv2.imwrite(output_path, mask)
    print(f"已生成掩码: {output_path}")


# 主程序
if __name__ == "__main__":
    import numpy as np

    # 设置输入文件夹和输出文件夹
    input_folder = "data/inpainting_fabric/normal/train/"  # 替换为你的图片文件夹路径
    output_folder = "data/inpainting_fabric/image_output_folder/"  # 掩码输出文件夹

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有图片文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            generate_random_masks(image_path, output_folder)

    print("所有掩码生成完成！")