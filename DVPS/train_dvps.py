"""python train_dvps.py \
    --batch_size 16 \
    --lr 1e-4 \
    --epochs 100 \
    --device cuda \
    --save_dir checkpoints \
    --vis_interval 1
"""

"""data/
    train/
        defect_images/
            image1.jpg
            image2.jpg
            ...
        inpainted_images/
            image1.jpg
            image2.jpg
            ...
        masks/
            image1_mask.png
            image2_mask.png
            ...
    val/
        defect_images/
            ...
        inpainted_images/
            ...
        masks/
            ...
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from torchvision import transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

from fp16_util import (
    convert_module_to_f16,
    convert_module_to_f32,
    make_master_params,
    model_grads_to_master_grads,
    master_params_to_model_params,
    zero_grad
)
from dataset import DefectDataset
from unet import UNetModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="unet/checkpoints")
    parser.add_argument("--vis_interval", type=int, default=1, 
                      help="每隔多少个epoch保存一次可视化结果")
    return parser.parse_args()

def dice_loss(pred, target):
    smooth = 1.0
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = torch.device(args.device)
        self.model.to(self.device)
        
        # 添加 current_epoch 属性
        self.current_epoch = 0
        
        # 设置优化器
        self.optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        
        # FP16 设置
        self.use_fp16 = args.fp16
        if self.use_fp16:
            self.model.convert_to_fp16()
            self.master_params = make_master_params(self.model.parameters())
        
        # 添加可视化保存目录
        self.vis_dir = os.path.join(args.save_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def visualize_predictions(self, epoch, inputs, targets, outputs, batch_idx=0):
        """
        保存预测结果的可视化图片
        Args:
            epoch: 当前轮次
            inputs: 输入张量 [B, 6, H, W]
            targets: 目标掩码 [B, 1, H, W]
            outputs: 模型输出 [B, 1, H, W]
            batch_idx: 要可视化的batch中的图片索引
        """
        # 将输出转换为概率图
        pred_mask = torch.sigmoid(outputs[batch_idx]).detach().cpu()
        
        # 分离原始图片和修复图片
        defect_img = inputs[batch_idx, :3]
        inpainted_img = inputs[batch_idx, 3:]
        target_mask = targets[batch_idx]
        
        # 创建图片网格
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # 显示原始有疵点图片
        axes[0, 0].imshow(defect_img.permute(1, 2, 0).cpu())
        axes[0, 0].set_title('Defect Image')
        axes[0, 0].axis('off')
        
        # 显示修复后的图片
        axes[0, 1].imshow(inpainted_img.permute(1, 2, 0).cpu())
        axes[0, 1].set_title('Inpainted Image')
        axes[0, 1].axis('off')
        
        # 显示真实掩码
        axes[1, 0].imshow(target_mask.squeeze().cpu(), cmap='gray')
        axes[1, 0].set_title('Ground Truth Mask')
        axes[1, 0].axis('off')
        
        # 显示预测掩码
        axes[1, 1].imshow(pred_mask.squeeze().cpu(), cmap='gray')
        axes[1, 1].set_title('Predicted Mask')
        axes[1, 1].axis('off')
        
        # 保存图片
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, f'epoch_{epoch}.png')
        plt.savefig(save_path)
        plt.close()
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.compute_loss(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.compute_loss(outputs, targets)
                
                # 反向传播
                if self.use_fp16:
                    model_grads_to_master_grads(self.model.parameters(), self.master_params)
                    self.optimizer.zero_grad()
                    loss.backward()
                    master_params_to_model_params(self.model.parameters(), self.master_params)
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
                
                # 在每个epoch的最后一个batch保存可视化结果
                if batch_idx == len(dataloader) - 1:
                    self.visualize_predictions(epoch, inputs, targets, outputs)
        
        return total_loss / len(dataloader)
    
    def compute_loss(self, outputs, targets):
        """
        组合BCE损失和Dice损失
        """
        bce_loss = nn.BCEWithLogitsLoss()(outputs, targets)
        dice = dice_loss(outputs, targets)
        return bce_loss + dice
    
    def save_checkpoint(self, epoch, loss):
        """
        保存模型检查点
        """
        import os
        os.makedirs(self.args.save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        path = os.path.join(self.args.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
    
    def train(self, train_dataloader, val_dataloader=None):
        """
        完整的训练流程
        """
        for epoch in range(self.args.epochs):
            self.current_epoch = epoch  # 更新当前epoch
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_dataloader, epoch)
            
            # 验证
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # 保存检查点
            if (epoch + 1) % 10 == 0:  # 每10个epoch保存一次
                self.save_checkpoint(epoch, train_loss)
    
    def validate(self, dataloader):
        """
        验证函数
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.compute_loss(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.compute_loss(outputs, targets)
                
                total_loss += loss.item()
                
                # 在验证集上也保存一些可视化结果
                if batch_idx == 0:  # 只保存第一个batch的结果
                    self.visualize_predictions(
                        f"val_epoch_{self.current_epoch}", 
                        inputs, 
                        targets, 
                        outputs
                    )
        
        return total_loss / len(dataloader)

def main():
    args = parse_args()
    
    # 创建数据集
    train_dataset = DefectDataset(
        data_dir='unet/dataset/train',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    val_dataset = DefectDataset(
        data_dir='unet/dataset/val',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 初始化UNet模型
    model = UNetModel(
        in_channels=6,  # 3(RGB) * 2(原图和修复图)
        model_channels=64,  # 基础通道数
        out_channels=1,  # 输出单通道掩码
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
    )
    
    trainer = Trainer(model, args)
    trainer.train(train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()