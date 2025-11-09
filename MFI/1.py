# import cv2
# import numpy as np
#
# def remove_background(image_path):
#     # 读取照片
#     image = cv2.imread(image_path)
#
#     # 创建与照片大小相同的掩码
#     mask = np.zeros(image.shape[:2], np.uint8)
#
#     # 创建GrabCut算法所需的矩形边界
#     rect = (1, 1, image.shape[1] - 1, image.shape[0] - 1)
#
#     # 使用GrabCut算法去除背景
#     cv2.grabCut(image, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)
#
#     # 创建新的掩码，将可能是前景的像素设置为前景
#     mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#
#     # 将原始图像与新掩码相乘，去除背景
#     image = image * mask2[:, :, np.newaxis]
#
#     # 显示去除背景后的图像
#     cv2.imshow("Image without background", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# # 调用函数去除照片背景
# remove_background("1.jpg")
import os

import torch
from PIL import Image

import os
import random
from PIL import Image, ImageDraw

# 定义掩码的大小
mask_size = (64, 64)

# 定义原图片文件夹路径
input_folder_path = r'E:\orginal\512_512\512'

# 定义保存修改后图片的文件夹路径
output_folder_path = r'E:\orginal\512_512\mask_img'

# 创建保存修改后图片的文件夹
os.makedirs(output_folder_path, exist_ok=True)

# 遍历原图片文件夹中的每张图片
for filename in os.listdir(input_folder_path):
    if filename.endswith('.jpeg'):
        # 打开原图片
        img_path = os.path.join(input_folder_path, filename)
        img = Image.open(img_path)

        # 创建一个新的掩码图像
        mask = Image.new('L', img.size, 0)

        # 随机生成掩码的位置
        x = random.randint(0, img.size[0] - mask_size[0])
        y = random.randint(0, img.size[1] - mask_size[1])

        # 在掩码图像上绘制一个白色的矩形
        draw = ImageDraw.Draw(mask)
        draw.rectangle([(x, y), (x + mask_size[0], y + mask_size[1])], fill=255)

        # 将掩码图像覆盖在原图片上
        img.paste(mask, (0, 0), mask)

        # 保存修改后的图片到另一个文件夹
        output_img_path = os.path.join(output_folder_path, filename)
        img.save(output_img_path)




# import argparse
# import sys
# import os
#
# import torch
# from PIL import Image
# from torch import nn, optim
# from torch.utils.data import DataLoader
#
# from torchvision import datasets, transforms, utils
#
# from tqdm import tqdm
#
# from vqvae import VQVAE
# from scheduler import CycleScheduler
# import distributed as dist
#
#
# def test(loader, model, device):
#     if dist.is_primary():
#         loader = tqdm(loader)
#     model.eval()
#
#     criterion = nn.MSELoss()
#     latent_loss_weight = 0.25
#     sample_size = 128
#     mse_sum = 0
#     mse_n = 0
#
#     with torch.no_grad():
#         for i, (img, label) in enumerate(loader):
#             img = img.to(device)
#             sample = img[:sample_size]
#
#             # 图像补全 - 对缺失区域的像素设置为白色
#             # 假设缺失区域用二值掩模表示
#             inpainted = torch.where(torch.isnan(sample), torch.ones_like(sample), sample)
#
#             out, latent_loss = model(inpainted)
#             recon_loss = criterion(out, img)
#             latent_loss = latent_loss.mean()
#             loss = recon_loss + latent_loss_weight * latent_loss
#             part_mse_sum = loss.item() * img.shape[0]
#             part_mse_n = img.shape[0]
#             comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
#             comm = dist.all_gather(comm)
#
#             for part in comm:
#                 mse_sum += part["mse_sum"]
#                 mse_n += part["mse_n"]
#
#     avg_mse = mse_sum / mse_n
#     print(f"Average MSE: {avg_mse:.5f}")
#     print(img.shape[0])
#     # # 保存重建图像
#     # utils.save_image(out, f"recon/reconstructed_{i}.png")
#
#     # 创建网格以显示原始图像和重建图像
#     grid = utils.make_grid(torch.cat([img, out], dim=0), nrow=img.shape[0])
#
#     # 保存网格图像
#     utils.save_image(grid, f"cropfix/original_reconstructed_{i}.png",
#                      normalize=True,
#                      range=(-1, 1),
#                      )
#
#
# def main(args):
#     device = "cuda"
#
#     transform = transforms.Compose(
#         [
#             transforms.Resize(args.size),
#             transforms.CenterCrop(args.size),
#             transforms.ToTensor(),
#         ]
#  )
#
#     dataset = datasets.ImageFolder(args.path, transform=transform, target_transform=mask_transform)
#     sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
#     loader = DataLoader(
#         dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
#     )
#     model = VQVAE().to(device)
#
#     if args.distributed:
#         model = nn.parallel.DistributedDataParallel(
#             model,
#             device_ids=[dist.get_local_rank()],
#             output_device=dist.get_local_rank(),
#         )
#
#     model.load_state_dict(torch.load(args.checkpoint))
#
#     test(loader, model, device)
#
#
# # 添加一个新函数来将掩模转换为张量
# def mask_transform(mask_path):
#     mask = Image.open(mask_path).convert("L")
#     mask = transforms.ToTensor()(mask)
#     return mask
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--n_gpu", type=int, default=1)
#     parser.add_argument("--distributed", action="store_true")
#     port = (
#         2 ** 15
#         + 2 **14
#         + hash(os.getuid if sys.platform != "win32" else 1) % 2 ** 14
#     )
#     parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
#
#     parser.add_argument("--size", type=int, default=256)
#     parser.add_argument("checkpoint", type=str)
#     parser.add_argument("path", type=str)
#
#     args = parser.parse_args()
#
#     print(args)
#
#     dist.launch(main, args.n_gpu, 1, 0, args.dist_url,args=(args,))
#
# import argparse
# import sys
# import os
#
# import torch
# from PIL import Image
# from torch import nn, optim
# from torch.utils.data import DataLoader
#
# from torchvision import datasets, transforms, utils
#
# from tqdm import tqdm
#
# from vqvae import VQVAE
# from scheduler import CycleScheduler
# import distributed as dist
# import random
# from torchvision.transforms.functional import to_pil_image
#
#
#
# def test(loader, model, device):
#     if dist.is_primary():
#         loader = tqdm(loader)
#     model.eval()
#
#     criterion = nn.MSELoss()
#     latent_loss_weight=0.25
#     sample_size = 128
#     mse_sum = 0
#     mse_n = 0
#
#     with torch.no_grad():
#         for i, (img, label) in enumerate(loader):
#             img = img.to(device)
#             mask = torch.zeros_like(img)
#             mask[:, :, random.randint(0, img.shape[2] - 64):random.randint(64, img.shape[2]),
#             random.randint(0, img.shape[3] - 64):random.randint(64, img.shape[3])] = .0
#             img_masked = img * (1 - mask)
#             print(img.shape)
#             sample = img[:sample_size]
#
#             out, latent_loss = model(img_masked)
#             recon_loss = criterion(out, img)
#             inpainting_loss = criterion(out * mask, img * mask)
#             latent_loss = latent_loss.mean()
#             loss = recon_loss + latent_loss_weight * latent_loss +inpainting_loss
#             part_mse_sum = loss.item() * img.shape[0]
#             part_mse_n = img.shape[0]
#             comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
#             comm = dist.all_gather(comm)
#
#             for part in comm:
#                 mse_sum += part["mse_sum"]
#                 mse_n += part["mse_n"]
#
#     avg_mse = mse_sum / mse_n
#     print(f"Average MSE: {avg_mse:.5f}")
#     print(img.shape[0])
#     # # Save reconstructed images
#     # utils.save_image(out, f"recon/reconstructed_{i}.png")
#
#     # 保存修复后的图像
#     utils.save_image(out * (1 - mask) + img_masked, f"recon/reconstructed_{i}.png")
#
#     # 创建包含原始图像、掩码图像和修复图像的网格
#     grid = utils.make_grid(torch.cat([img, img_masked, out * (1 - mask) + img_masked], dim = 0), nrow = img.shape[0])
#
#     # 保存网格图像
#     utils.save_image(grid, f"small256/original_masked_reconstructed_{i}.png", normalize=True, range=(-1, 1))
#
#
# def main(args):
#     device = "cuda"
#
#     transform = transforms.Compose(
#
#         [
#             transforms.Resize(args.size),
#             transforms.CenterCrop(args.size),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#         ]
#     )
#
#     dataset = datasets.ImageFolder(args.path, transform=transform)
#     sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
#     loader = DataLoader(
#         dataset,batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
#     )
#     model = VQVAE().to(device)
#
#     if args.distributed:
#         model = nn.parallel.DistributedDataParallel(
#             model,
#             device_ids=[dist.get_local_rank()],
#             output_device=dist.get_local_rank(),
#         )
#
#     model.load_state_dict(torch.load(args.checkpoint))
#
#     test(loader, model, device)
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--n_gpu", type=int, default=1)
#     parser.add_argument("--distributed", action="store_true")
#     port = (
#         2 ** 15
#         + 2 ** 14
#         + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
#     )
#     parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
#
#     parser.add_argument("--size", type=int, default=256)
#     parser.add_argument("checkpoint", type=str)
#     parser.add_argument("path", type=str)
#
#     args = parser.parse_args()
#
#     print(args)
#
#     dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
