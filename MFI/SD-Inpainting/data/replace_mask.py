#将data\inpainting_fabric\custom_inpainting中的mask随机替换成data\inpainting_fabric\mask中的mask，但是要保持原来文件夹中mask的名称不变
import os
import random
import shutil

# 设置路径
custom_mask_dir = r'data/inpainting_fabric/custom_inpainting'
source_mask_dir = r'data/inpainting_fabric/mask'

# 获取源mask列表
source_masks = [os.path.join(source_mask_dir, f) for f in os.listdir(source_mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 遍历custom_inpainting目录下的所有mask文件
for filename in os.listdir(custom_mask_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        dst_path = os.path.join(custom_mask_dir, filename)
        src_path = random.choice(source_masks)  # 随机选择一个源mask
        shutil.copy(src_path, dst_path)  # 替换内容但保持文件名不变
        print(f"Replaced {dst_path} with {src_path}")
