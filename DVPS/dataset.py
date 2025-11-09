import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class DefectDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): 数据目录，应包含以下子目录：
                - defect_images: 包含有疵点的原始图片
                - inpainted_images: 修复后的图片
                - masks: 疵点的二值掩码标签
        """
        self.data_dir = data_dir
        self.defect_dir = os.path.join(data_dir, 'defect_images')
        self.inpainted_dir = os.path.join(data_dir, 'inpainted_images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        
        self.image_files = sorted(os.listdir(self.defect_dir))
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
            
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取图片文件名
        img_name = self.image_files[idx]
        
        # 读取有疵点的图片
        defect_path = os.path.join(self.defect_dir, img_name)
        defect_img = Image.open(defect_path).convert('RGB')
        
        # 读取修复后的图片
        inpainted_path = os.path.join(self.inpainted_dir, img_name)
        inpainted_img = Image.open(inpainted_path).convert('RGB')
        
        # 读取掩码标签
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '_mask.png'))
        mask = Image.open(mask_path).convert('L')
        
        # 应用变换
        defect_img = self.transform(defect_img)
        inpainted_img = self.transform(inpainted_img)
        mask = self.mask_transform(mask)
        
        # 将输入图片拼接在一起
        input_tensor = torch.cat([defect_img, inpainted_img], dim=0)
        
        return input_tensor, mask 