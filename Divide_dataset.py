#划分数据集和测试集，然后把数据集和测试集的图片和txt标注文件都放到对应的文件夹中
import os
import random
import shutil

def split_dataset(image_folder, txt_folder, train_ratio=0.8):
    # 获取文件夹中的图片名称
    image_names = os.listdir(image_folder)
    image_names = [os.path.splitext(name)[0] for name in image_names]
    # 获取文件夹中的txt文件名称
    txt_names = os.listdir(txt_folder)
    txt_names = [os.path.splitext(name)[0] for name in txt_names]
    # 获取图片名称的交集
    common_names = set(image_names) & set(txt_names)
    common_names = list(common_names)
    # 打乱图片名称的顺序
    random.shuffle(common_names)
    # 划分训练集和测试集
    train_size = int(len(common_names) * train_ratio)
    train_names = common_names[:train_size]
    test_names = common_names[train_size:]
    # 创建训练集和测试集文件夹
    train_image_folder = os.path.join('dataset_re', 'images', 'train')
    train_txt_folder = os.path.join('dataset_re', 'labels', 'train')
    test_image_folder = os.path.join('dataset_re', 'images', 'val')
    test_txt_folder = os.path.join('dataset_re', 'labels', 'val')
    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(train_txt_folder, exist_ok=True)
    os.makedirs(test_image_folder, exist_ok=True)
    os.makedirs(test_txt_folder, exist_ok=True)
    # 移动图片和txt文件到对应的文件夹
    for name in train_names:
        shutil.move(os.path.join(image_folder, name+'.jpg'), train_image_folder)
        shutil.move(os.path.join(txt_folder, name+'.txt'), train_txt_folder)
    for name in test_names:
        shutil.move(os.path.join(image_folder, name+'.jpg'), test_image_folder)
        shutil.move(os.path.join(txt_folder, name+'.txt'), test_txt_folder)

if __name__ == '__main__':
    image_folder = 'train_re_output'  # 图片文件夹路径
    txt_folder = 'yolo_annotations'  # txt文件夹路径
    split_dataset(image_folder, txt_folder)