import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from unet import UNetModel
from tqdm import tqdm
import numpy as np
import cv2

def load_image(image_path):
    """加载并预处理图片"""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def mask_to_yolo_format(mask, min_area=15):
    """
    将掩码转换为YOLO格式的标注
    Args:
        mask: 预测的掩码 (H, W) 范围在0-1之间
        min_area: 最小连通区域面积，小于此面积的区域将被忽略
    Returns:
        list of [class_id, x_center, y_center, width, height]
    """
    # 将掩码转换为二值图像
    binary = (mask.numpy() > 0.5).astype(np.uint8)
    
    # 找到连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    boxes = []
    # 跳过背景（第一个组件）
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
            
        # 获取边界框坐标
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # 转换为YOLO格式 (归一化的中心点坐标和宽高)
        x_center = (x + w/2) / mask.shape[1]
        y_center = (y + h/2) / mask.shape[0]
        width = w / mask.shape[1]
        height = h / mask.shape[0]
        
        # class_id 设为 0
        boxes.append([0, x_center, y_center, width, height])
    
    return boxes

def predict_mask(model, defect_img_path, inpainted_img_path, device='cuda'):
    """预测掩码"""
    # 加载图片
    defect_img = load_image(defect_img_path)
    inpainted_img = load_image(inpainted_img_path)
    
    # 合并输入
    x = torch.cat([defect_img, inpainted_img], dim=0)
    x = x.unsqueeze(0)
    x = x.to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)
    
    return pred.cpu().squeeze()

def process_image_folders(model, defect_dir, inpainted_dir, save_dir, device='cuda'):
    """处理整个文件夹的图片"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, 'visualizations')
    mask_dir = os.path.join(save_dir, 'masks')
    label_dir = os.path.join(save_dir, 'labels')  # YOLO格式标签的保存目录
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    # 获取所有图片文件名
    defect_images = sorted(os.listdir(defect_dir))
    
    # 使用tqdm显示进度
    for img_name in tqdm(defect_images, desc="Processing images"):
        # 构建完整路径
        defect_path = os.path.join(defect_dir, img_name)
        inpainted_path = os.path.join(inpainted_dir, img_name)
        
        # 检查对应的修复图片是否存在
        if not os.path.exists(inpainted_path):
            print(f"Warning: No matching inpainted image for {img_name}")
            continue
            
        try:
            # 预测掩码
            pred_mask = predict_mask(model, defect_path, inpainted_path, device)
            
            # 保存掩码
            mask_save_path = os.path.join(mask_dir, f"{os.path.splitext(img_name)[0]}_mask.png")
            plt.imsave(mask_save_path, pred_mask, cmap='gray')
            
            # 转换为YOLO格式并保存标签
            boxes = mask_to_yolo_format(pred_mask)
            label_save_path = os.path.join(label_dir, f"{os.path.splitext(img_name)[0]}.txt")
            with open(label_save_path, 'w') as f:
                for box in boxes:
                    f.write(' '.join(map(str, box)) + '\n')
            
            # 保存可视化结果（包括边界框）
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # 显示原始缺陷图片
            defect_img = Image.open(defect_path).convert('RGB')
            defect_img = defect_img.resize((256, 256))
            axes[0].imshow(defect_img)
            axes[0].set_title('Defect Image')
            axes[0].axis('off')
            
            # 显示修复后的图片
            inpainted_img = Image.open(inpainted_path).convert('RGB')
            inpainted_img = inpainted_img.resize((256, 256))
            axes[1].imshow(inpainted_img)
            axes[1].set_title('Inpainted Image')
            axes[1].axis('off')
            
            # 显示预测的掩码
            axes[2].imshow(pred_mask, cmap='gray')
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')
            
            # 显示带边界框的图片
            axes[3].imshow(defect_img)
            for box in boxes:
                _, x_center, y_center, width, height = box
                # 转换回像素坐标
                x = int((x_center - width/2) * 256)
                y = int((y_center - height/2) * 256)
                w = int(width * 256)
                h = int(height * 256)
                rect = plt.Rectangle((x, y), w, h, fill=False, color='red')
                axes[3].add_patch(rect)
            axes[3].set_title('Bounding Boxes')
            axes[3].axis('off')
            
            plt.tight_layout()
            vis_save_path = os.path.join(vis_dir, f"{os.path.splitext(img_name)[0]}_vis.png")
            plt.savefig(vis_save_path)
            plt.close()
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = UNetModel(
        in_channels=6,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=8,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
    ).to(device)
    
    # 加载训练好的权重
    checkpoint_path = 'D:/YQM/ultralytics-main/unet/checkpoints/checkpoint_epoch_69.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置输入和输出目录
    defect_dir = 'D:/YQM/ultralytics-main/split_task/task1/images'
    inpainted_dir = 'D:/YQM/ultralytics-main/split_task/task1/inpainted_images'
    save_dir = 'D:/YQM/ultralytics-main/split_task/task1/unet_predict'
    
    # 处理所有图片
    process_image_folders(model, defect_dir, inpainted_dir, save_dir, device)

if __name__ == '__main__':
    main() 