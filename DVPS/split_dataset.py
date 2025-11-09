"""
将数据集自动分割为训练集和验证集

数据集结构：
unet/dataset/
    defect_images/
        *.jpg
    inpainted_images/
        *.jpg
    labels/
        *.txt

分割后的结构：
unet/dataset/
    train/
        defect_images/
            *.jpg
        inpainted_images/
            *.jpg
        masks/
            *.png
    val/
        defect_images/
            *.jpg
        inpainted_images/
            *.jpg
        masks/
            *.png
    labels/
        train/
            *.txt
        val/
            *.txt
"""

import os
import shutil
import random
from tqdm import tqdm

def split_dataset(dataset_root, val_ratio=0.2, seed=42):
    """
    将数据集分割为训练集和验证集
    
    Args:
        dataset_root: 数据集根目录
        val_ratio: 验证集比例
        seed: 随机种子
    """
    random.seed(seed)
    
    # 源文件夹
    defect_dir = os.path.join(dataset_root, 'defect_images')
    inpaint_dir = os.path.join(dataset_root, 'inpainted_images')
    label_dir = os.path.join(dataset_root, 'labels')
    mask_dir = os.path.join(dataset_root, 'masks')
    
    # 获取所有图片文件名（不包含扩展名）
    image_names = [os.path.splitext(f)[0] for f in os.listdir(defect_dir) 
                  if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # 随机打乱
    random.shuffle(image_names)
    
    # 计算验证集大小
    val_size = int(len(image_names) * val_ratio)
    
    # 分割数据集
    val_names = image_names[:val_size]
    train_names = image_names[val_size:]
    
    print(f"Total images: {len(image_names)}")
    print(f"Training set: {len(train_names)}")
    print(f"Validation set: {len(val_names)}")
    
    # 创建目录结构
    for split in ['train', 'val']:
        for subdir in ['defect_images', 'inpainted_images', 'masks']:
            os.makedirs(os.path.join(dataset_root, split, subdir), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, 'labels', split), exist_ok=True)
    
    # 移动文件
    def move_files(file_names, split):
        for name in tqdm(file_names, desc=f'Moving {split} files'):
            # 移动缺陷图片
            src_defect = os.path.join(defect_dir, f"{name}.jpg")
            dst_defect = os.path.join(dataset_root, split, 'defect_images', f"{name}.jpg")
            if os.path.exists(src_defect):
                shutil.copy2(src_defect, dst_defect)
            
            # 移动修复图片
            src_inpaint = os.path.join(inpaint_dir, f"{name}.jpg")
            dst_inpaint = os.path.join(dataset_root, split, 'inpainted_images', f"{name}.jpg")
            if os.path.exists(src_inpaint):
                shutil.copy2(src_inpaint, dst_inpaint)
            
            # 移动mask文件
            src_mask = os.path.join(mask_dir, f"{name}_mask.png")  # mask文件通常以_mask.png结尾
            dst_mask = os.path.join(dataset_root, split, 'masks', f"{name}_mask.png")
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
            
            # 移动标签文件
            src_label = os.path.join(label_dir, f"{name}.txt")
            if os.path.exists(src_label):
                dst_label = os.path.join(dataset_root, 'labels', split, f"{name}.txt")
                shutil.copy2(src_label, dst_label)
    
    # 移动训练集和验证集文件
    move_files(train_names, 'train')
    move_files(val_names, 'val')
    
    print("Dataset split completed!")

def main():
    dataset_root = 'D:/YQM/ultralytics-main/unet/dataset'
    split_dataset(dataset_root, val_ratio=0.1)

if __name__ == '__main__':
    main()