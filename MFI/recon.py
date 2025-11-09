import os
from os.path import exists

import torch
from torchvision import transforms, utils
from PIL import Image

def load_images_from_folder(folder_path, transform):
    """加载文件夹中的所有图片，并应用transform"""
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB')
            images.append(transform(img))
    return images

def test_and_save_reconstruction(model, test_image_dir, output_dir, device):
    """测试并保存重建图像"""
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 根据模型要求调整大小
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # 加载测试图片路径
    image_paths = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if
                   os.path.isfile(os.path.join(test_image_dir, f))]

    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for img_path in image_paths:
            # 加载并处理单张图片
            img = Image.open(img_path).convert('RGB')
            input_image = transform(img).unsqueeze(0).to(device)  # 增加 batch 维度

            # 获取重建结果
            reconstructed_image, _, _ = model(input_image)

            # 保存重建图像
            reconstructed_image_path = os.path.join(output_dir, os.path.basename(img_path))
            utils.save_image(
                reconstructed_image,
                reconstructed_image_path,
                normalize=True,
                value_range=(-1, 1),
            )

def main():
    model_path = "checkpoint_MD_repair/all_img2/vqvae_300.pt"
    test_image_dir = "D:/YQM/ultralytics-main/defect_dataset/resize_defect_dataset_yuanshi"
    output_dir = "inpainting_output_yuanshi"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    from vqvae import VQVAE
    model = VQVAE()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # 测试和保存重建图像
    test_and_save_reconstruction(model, test_image_dir, output_dir, device)

if __name__ == "__main__":
    main()
