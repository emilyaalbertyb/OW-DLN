import os
import random
import csv


def generate_csv(image_folder, output_csv, validation_ratio=0.1):
    # 获取文件夹中的所有图片文件
    all_images = [f for f in os.listdir(image_folder) if f.endswith('.png') and 'mask' not in f]

    # 生成图像路径和掩码路径的对应关系
    data = []
    for image in all_images:
        mask = image.replace('.png', '_mask.png')
        image_path = os.path.join(image_folder, image)
        mask_path = os.path.join(image_folder, mask)
        data.append([image_path, mask_path])

    # 按照比例随机划分为训练集和验证集
    random.shuffle(data)
    num_validation = int(len(data) * validation_ratio)
    validation_data = data[:num_validation]
    train_data = data[num_validation:]

    # 写入CSV文件
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'mask_path', 'partition'])

        # 写入训练集
        for image_path, mask_path in train_data:
            writer.writerow([image_path, mask_path, 'train'])

        # 写入验证集
        for image_path, mask_path in validation_data:
            writer.writerow([image_path, mask_path, 'validation'])


# 调用函数，指定文件夹路径和输出CSV文件路径
image_folder = 'D:/ye/Stable-Diffusion-Inpaint-main/data/inpainting_fabric/custom_inpainting/'
output_csv = 'D:/ye/Stable-Diffusion-Inpaint-main/data/inpainting_fabric/fabric.csv'
generate_csv(image_folder, output_csv)
