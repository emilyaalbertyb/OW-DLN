import torch
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from PIL import Image
import os
import glob

# 加载模型
def load_model(model_path, device):
    from vqvae import VQVAE
    model = VQVAE()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# 重建图像并保存对比结果
def test_and_save_reconstruction(model, test_image_dir, output_dir, device):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载文件夹中所有图像路径
    test_image_paths = glob.glob(os.path.join(test_image_dir, "*"))
    if not test_image_paths:
        print(f"No images found in {test_image_dir}")
        return

    # 定义预处理
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 假设输入图像大小是 512x512
        transforms.ToTensor(),
    ])

    for i, image_path in enumerate(test_image_paths):
        # 加载和预处理图像
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {image_path}: {e}")
            continue

        input_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 512, 512]

        # 获取重建结果
        with torch.no_grad():
            reconstructed_img = model(input_tensor)

        # 保存对比图像
        comparison = torch.cat([input_tensor.cpu(), reconstructed_img[0].cpu()], dim=0)  # 拼接原图和重建图
        save_path = os.path.join(output_dir, f"comparison_{i + 1}.png")
        save_image(make_grid(comparison, nrow=2, normalize=True), save_path)
        print(f"Saved comparison image to {save_path}")

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 替换为你的模型路径和测试图像文件夹路径
    model_path = "D:/vq-vae-2-pytorch-master/checkpoint_D/vqvae_400.pt"
    test_image_dir = "D:/vq-vae-2-pytorch-master/test_image"  # 文件夹路径
    output_dir = "re_output"

    # 加载模型
    model = load_model(model_path, device)

    # 测试并保存重建结果
    test_and_save_reconstruction(model, test_image_dir, output_dir, device)

if __name__ == "__main__":
    main()
