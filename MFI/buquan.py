# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.transforms as transforms
# from PIL import Image
#
# # 定义VQ-VAE模型
# class VQVAE(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim):
#         super(VQVAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, embedding_dim, kernel_size=4, stride=2, padding=1),
#             nn.ReLU()
#         )
#         self.vq = VQ(num_embeddings, embedding_dim)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # 编码
#         z = self.encoder(x)
#         # VQ层
#         codes, indices = self.vq(z)
#         # 解码
#         x_hat = self.decoder(codes)
#         return x_hat, indices
#
# # 添加随机掩码
# def add_mask(image, mask_size):
#     width, height = image.size
#     mask = Image.new('RGB', (mask_size, mask_size))
#     mask_pixels = mask.load()
#     for i in range(mask_size):
#         for j in range(mask_size):
#             mask_pixels[i, j] = (0, 0, 0)  # 黑色掩码
#     x = torch.randint(0, width - mask_size, (1,))
#     y = torch.randint(0, height - mask_size, (1,))
#     image.paste(mask, (x, y))
#     return image, mask
#
# # 图像补全
# def complete_image(image, model, mask_size, x, y):
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),
#         transforms.ToTensor()
#     ])
#     image = transform(image).unsqueeze(0)
#     mask = torch.zeros(1, 3, 64, 64)
#     mask[:, :, y:y+mask_size, x:x+mask_size] = 1.0
#     input_image = image * (1.0 - mask)
#     output_image, _ = model(input_image)
#     completed_image = input_image + output_image * mask
#     return completed_image.squeeze(0)
#
# # 加载训练好的VQ-VAE模型
# model = VQVAE(num_embeddings=512, embedding_dim=64)
# model.load_state_dict(torch.load('vqvae_model.pth'))
# model.eval()
#
# # 加载输入图像
# input_image = Image.open('input_image.jpg')
#
# # 添加随机掩码
# mask_size = 64
# masked_image, x, y = add_mask(input_image, mask_size)
#
# # 图像补全
# completed_image = complete_image(masked_image, model, mask_size, x, y)
#
# # 显示结果
# input_image.show()
# masked_image.show()
# completed_image = transforms.ToPILImage()(completed_image)
# completed_image.show()



import argparse
import sys
import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist


def add_random_mask(img):
    img_cpu = img.cpu()
    img_np = img_cpu.numpy()
    img_pil =Image.fromarray(np.uint8(img_np * 255))
    mask_size = 64
    w, h = img_pil.size
    top = random.randint(0, h - mask_size)
    left = random.randint(0, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    mask = Image.new('RGB', (w, h), color=(0, 0, 0))
    draw = ImageDraw.Draw(mask)
    draw.rectangle((left, top, right, bottom), fill=(255, 255, 255))

    masked_img = Image.composite(img, mask, mask)
    return masked_img


def test(loader, model, device):
    if dist.is_primary():
        loader = tqdm(loader)
    model.eval()

    criterion = nn.MSELoss()
    latent_loss_weight = 0.25
    sample_size = 128
    mse_sum = 0
    mse_n = 0

    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            img = img.to(device)

            masked_img = add_random_mask(img)
            sample = masked_img[:sample_size]

            out, latent_loss = model(sample)
            recon_loss = criterion(out, img)
            latent_loss = latent_loss.mean()
            loss = recon_loss + latent_loss_weight * latent_loss
            part_mse_sum = loss.item() * img.shape[0]
            part_mse_n = img.shape[0]
            comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
            comm = dist.all_gather(comm)

            for part in comm:
                mse_sum += part["mse_sum"]
                mse_n += part["mse_n"]

    avg_mse = mse_sum / mse_n
    print(f"Average MSE: {avg_mse:.5f}")
    print(img.shape[0])

    # Create a grid to display original, masked, and images
    grid = utils.make_grid(torch.cat([img, masked_img, out], dim=0), nrow=img[0])

    # Save the grid
    utils.save_image(grid, "original_masked_reconstructed_{i}.png",
                     normalize=True,
                     range=(-1, 1),
                     )


def main(args):
    device = "cuda"

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder(args.path, transform=transform)
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
    )
    model = VQVAE().to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    model.load_state_dict(torch.load(args.checkpoint))

    test(loader, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--distributed", action="store_true")
    port = (
            2 ** 15
            + 2 ** 14
            + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))

