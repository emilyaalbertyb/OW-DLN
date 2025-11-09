import os
import json
from PIL import Image, ImageDraw


def generate_and_resize_masks(annotation_file, image_folder, output_folder, size=(512, 512)):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 加载标注文件
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # 按图片名分组标注信息
    grouped_annotations = {}
    for annotation in annotations:
        image_name = annotation['name']
        bbox = annotation['bbox']
        if image_name not in grouped_annotations:
            grouped_annotations[image_name] = []
        grouped_annotations[image_name].append(bbox)

    # 遍历图片文件夹，生成掩码
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        # 检查是否在标注中
        if image_name not in grouped_annotations:
            continue

        # 打开图像
        with Image.open(image_path) as img:
            original_size = img.size
            mask = Image.new('L', original_size, 0)  # 创建白色背景的掩码
            draw = ImageDraw.Draw(mask)

            # 绘制标注的bbox为黑色区域
            for bbox in grouped_annotations[image_name]:
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], fill=255)

            # Resize 图片和掩码到目标尺寸
            img_resized = img.resize(size, Image.ANTIALIAS)
            mask_resized = mask.resize(size, Image.NEAREST)

            # 保存图片和掩码
            base_name = os.path.splitext(image_name)[0]  # 去掉图片后缀
            img_resized.save(os.path.join(output_folder, f"{base_name}.png"))
            mask_resized.save(os.path.join(output_folder, f"{base_name}_mask.png"))
            print(f"Saved resized image and mask: {base_name}")


# 使用示例
annotation_file = "dataset/anno_train.json"  # 标注文件路径
image_folder = "dataset/defect_dataset/defect_dataset"  # 图片文件夹路径
output_folder = "output_masks"  # 掩码输出文件夹路径

generate_and_resize_masks(annotation_file, image_folder, output_folder, size=(512, 512))
