"""
unet/dataset/
    labels/
        train/
            *.txt  # YOLO格式的标注文件
        val/
            *.txt
    train/
        defect_images/
            *.jpg
        inpainted_images/
            *.jpg
    val/
        defect_images/
            *.jpg
        inpainted_images/
            *.jpg
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

def create_mask_from_yolo(label_path, image_size, save_path):
    """
    从YOLO格式的标注文件创建掩码
    Args:
        label_path: 标注文件路径
        image_size: 原始图片尺寸 (width, height)
        save_path: 掩码保存路径
    """
    # 创建空白掩码
    mask = np.zeros(image_size[::-1], dtype=np.uint8)  # height, width
    
    # 如果标注文件不存在，返回空掩码
    if not os.path.exists(label_path):
        cv2.imwrite(save_path, mask)
        return
    
    # 读取标注文件
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # 处理每个标注框
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            # YOLO格式：中心点坐标和宽高（归一化到0-1）
            x_center = float(parts[1]) * image_size[0]
            y_center = float(parts[2]) * image_size[1]
            width = float(parts[3]) * image_size[0]
            height = float(parts[4]) * image_size[1]
            
            # 计算框的左上角和右下角坐标
            x1 = int(x_center - width/2)
            y1 = int(y_center - height/2)
            x2 = int(x_center + width/2)
            y2 = int(y_center + height/2)
            
            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_size[0], x2)
            y2 = min(image_size[1], y2)
            
            # 在掩码上绘制矩形
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    # 保存掩码
    cv2.imwrite(save_path, mask)

def process_dataset(dataset_root, image_size=(256, 256)):
    """
    处理整个数据集
    Args:
        dataset_root: 数据集根目录
        image_size: 输出掩码的尺寸
    """
    # 创建保存目录
    for split in ['train', 'val']:
        mask_dir = os.path.join(dataset_root, split, 'masks')
        os.makedirs(mask_dir, exist_ok=True)
        
        # 获取标注文件列表
        label_dir = os.path.join(dataset_root, 'labels', split)
        if not os.path.exists(label_dir):
            continue
            
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        
        print(f"Processing {split} set...")
        for label_file in tqdm(label_files):
            # 构建文件路径
            label_path = os.path.join(label_dir, label_file)
            mask_name = label_file.replace('.txt', '_mask.png')
            mask_path = os.path.join(mask_dir, mask_name)
            
            # 创建掩码
            create_mask_from_yolo(label_path, image_size, mask_path)

def main():
    # 设置数据集路径
    dataset_root = 'unet/dataset'
    
    # 获取原始图片尺寸
    # 假设所有图片尺寸相同，读取第一张图片获取尺寸
    sample_image_path = None
    for split in ['train', 'val']:
        defect_dir = os.path.join(dataset_root, split, 'defect_images')
        if os.path.exists(defect_dir):
            images = [f for f in os.listdir(defect_dir) if f.endswith(('.jpg', '.png'))]
            if images:
                sample_image_path = os.path.join(defect_dir, images[0])
                break
    
    if sample_image_path is None:
        print("Error: No images found in the dataset")
        return
    
    # 读取样本图片获取尺寸
    with Image.open(sample_image_path) as img:
        image_size = img.size  # (width, height)
    
    print(f"Creating masks with size: {image_size}")
    process_dataset(dataset_root, image_size)
    print("Mask creation completed!")

if __name__ == "__main__":
    main() 