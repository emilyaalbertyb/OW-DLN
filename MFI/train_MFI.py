import argparse
import json
import sys
import os
import argparse
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from vqvae import VQVAE
# from vqvqe_mask import VQVAE
from scheduler import CycleScheduler
import distributed as dist
import torch.nn.functional as F


def load_annotations(json_path):
    """
    加载标注文件，解析为一个字典。

    Args:
        json_path: JSON 文件路径。

    Returns:
        annotations: 一个字典，键为图像名称，值为疵点 bbox 列表。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = {}
    for item in data:
        img_name = item["name"]
        bbox = item["bbox"]
        if img_name not in annotations:
            annotations[img_name] = []
        annotations[img_name].append(bbox)

    return annotations

def get_mask_from_annotations(img_paths, annotations, img_size, orig_width, orig_height, device):
    """
    根据标注信息生成掩码（先按原始尺寸生成，再整体缩放）。

    Args:
        img_paths: 当前 batch 中每张图片的路径列表。
        annotations: 字典格式的标注信息，每个条目是一个列表，包含疵点区域的 bbox。
        img_size: 模型输入图像的尺寸 (H, W)。
        orig_width: 原始图像宽度。
        orig_height: 原始图像高度。

    Returns:
        masks: 一个形状为 (batch_size, 1, H, W) 的张量，表示每张图片的掩码。
    """
    batch_size = len(img_paths)
    # 初始化掩码，使用原始图像尺寸
    masks = torch.ones((batch_size, 1, orig_height, orig_width), device=device)

    for i, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path)
        if img_name in annotations:
            for bbox in annotations[img_name]:
                x_min, y_min, x_max, y_max = bbox
                # 将坐标转换为整数
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                # 在原始尺寸上绘制掩码
                masks[i, :, y_min:y_max, x_min:x_max] = 0

    # 将掩码整体缩放到目标尺寸
    masks = F.interpolate(masks, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return masks


import torch
import torch.nn.functional as F


# def repair_image(input, mask, patch_size=20):
#     """
#     利用掩码边缘的正常区域修复掩码区域。
#     :param input: 输入图像 (B, C, H, W)
#     :param mask: 掩码 (B, 1, H, W)，1表示正常区域，0表示需要修复的区域
#     :param patch_size: 搜索边缘区域的窗口大小
#     :return: 修复后的图像 (B, C, H, W)
#     """
#     B, C, H, W = input.size()
#
#     # 将修复后的图像初始化为输入图像
#     repaired_img = input.clone()
#
#     # 定义一个卷积核，用于检测边缘
#     kernel = torch.ones((1, 1, 3, 3), device=input.device)
#
#     # 遍历每张图片进行修复
#     for b in range(B):
#         # 计算缺陷区域的边缘：边缘为缺陷区域(0)周围的正常区域(1)
#         boundary = F.conv2d(mask[b:b + 1], kernel, padding=1) * (1 - mask[b:b + 1])
#         boundary_indices = torch.nonzero(boundary.squeeze() > 0, as_tuple=False)
#
#         if boundary_indices.size(0) == 0:
#             # 如果没有边缘，跳过修复
#             continue
#
#         # 获取需要修复的区域
#         defect_indices = torch.nonzero(mask[b, 0] == 0, as_tuple=False)
#
#         for y, x in defect_indices:
#             # 从边缘区域中随机选择一个点
#             idx = torch.randint(0, boundary_indices.size(0), (1,))
#             edge_y, edge_x = boundary_indices[idx].squeeze()
#
#             # 以选中的边缘点为中心，获取 patch_size 区域内的正常像素
#             top = max(0, edge_y - patch_size // 2)
#             left = max(0, edge_x - patch_size // 2)
#             bottom = min(H, top + patch_size)
#             right = min(W, left + patch_size)
#
#             # 从边缘的 patch 中随机选取一个像素用于填充
#             patch = input[b, :, top:bottom, left:right]
#             mask_patch = mask[b, :, top:bottom, left:right]
#             valid_pixels = torch.nonzero(mask_patch.squeeze() == 1, as_tuple=False)
#
#             if valid_pixels.size(0) > 0:
#                 # 随机选择一个有效像素
#                 patch_idx = torch.randint(0, valid_pixels.size(0), (1,))
#                 patch_y, patch_x = valid_pixels[patch_idx].squeeze()
#                 repaired_img[b, :, y, x] = patch[:, patch_y, patch_x]
#
#     return repaired_img


#
# def repair_image(input, mask, patch_size=40):
#     """
#     利用正常区域的随机像素块修复掩码区域。
#     :param input: 输入图像 (B, C, H, W)
#     :param mask: 掩码 (B, 1, H, W)，1表示正常区域，0表示需要修复的区域
#     :param patch_size: 用于填充的随机像素块大小
#     :return: 修复后的图像 (B, C, H, W)
#     """
#     B, C, H, W = input.size()
#
#     # 修复后的图像初始化为输入图像
#     repaired_img = input.clone()
#
#     # 遍历每张图片进行修复
#     for b in range(B):
#         # 获取当前图片的正常区域像素位置 (1 表示正常区域)
#         normal_indices = torch.nonzero(mask[b, 0] == 1, as_tuple=False)
#
#         if normal_indices.size(0) == 0:
#             # 如果没有正常区域，跳过修复
#             continue
#
#         # 随机选择一个中心点
#         idx = torch.randint(0, normal_indices.size(0), (1,))
#         center_y, center_x = normal_indices[idx].squeeze()
#
#         # 确定随机块的边界
#         top = max(0, center_y - patch_size // 2)
#         left = max(0, center_x - patch_size // 2)
#         bottom = min(H, top + patch_size)
#         right = min(W, left + patch_size)
#
#         # 从正常区域提取像素块
#         patch = input[b, :, top:bottom, left:right]
#
#         # 获取需要修复的区域 (0 表示需要修复的区域)
#         defect_indices = torch.nonzero(mask[b, 0] == 0, as_tuple=False)
#
#         # 填充缺陷区域
#         for y, x in defect_indices:
#             # 计算从 patch 中选取的像素
#             repaired_img[b, :, y, x] = patch[:, (y - top) % patch.size(1), (x - left) % patch.size(2)]
#
#     return repaired_img

def repair_image(input, mask, kernel_size=9):
    """
    修复图像，将掩码逻辑反转，掩码为1的区域保持不变，掩码为0的区域进行修复。
    :param input: 输入图像 (B, C, H, W)
    :param mask: 掩码 (B, C, H, W)，1表示正常区域，0表示需要修复的区域
    :param kernel_size: 卷积核大小，决定修复区域的平滑程度
    :return: 修复后的图像 (B, C, H, W)
    """
    B, C, H, W = input.size()

    # 创建多通道均值卷积核
    repair_kernel = torch.ones(C, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
    repair_kernel = repair_kernel.to(input.device)

    # 对多通道输入进行卷积
    repaired_img = F.conv2d(input, repair_kernel, groups=C, padding=kernel_size // 2)

    # 替换反转掩码区域（掩码为0的位置进行修复）
    reversed_mask = 1 - mask  # 掩码反转：0变为1，1变为0
    repaired_img = input * mask + repaired_img * reversed_mask
    return repaired_img

# def repair_image(img, masks):
#     """
#     修复疵点的图像，反转掩码区域，即掩码为0的位置需要修复。
#     :param img: 输入图像 (B, C, H, W)
#     :param masks: 掩码图像 (B, 1, H, W) ，值为 0 或 1，1 表示正常区域，0 表示需要修复的区域
#     :return: 修复后的图像 (B, C, H, W)
#     """
#     # 将掩码区域反转
#     reversed_masks = 1 - masks  # 原为1的区域变为0，原为0的区域变为1
#
#     repaired_images = img.clone()  # 初始化修复图像为原始图像
#     for i in range(img.size(0)):  # 遍历每张图像
#         mask = reversed_masks[i]  # 单张反转后的掩码 (1, H, W)
#
#         # 膨胀掩码，扩展修复区域
#         dilated_mask = F.conv2d(mask.unsqueeze(0), torch.ones(1, 1, 3, 3).to(mask.device), padding=1) > 0
#         dilated_mask = dilated_mask.float()  # 膨胀后的掩码区域
#
#         for c in range(img.size(1)):  # 对每个通道单独处理
#             channel = img[i, c]  # 单通道图像
#             repaired_channel = channel.clone()
#
#             # 计算非掩码区域（正常区域）的均值
#             mean_value = channel[~mask.squeeze(0).bool()].mean()
#
#             # 用均值填补掩码（反转后为1）区域
#             repaired_channel[mask.squeeze(0).bool()] = mean_value
#             repaired_images[i, c] = repaired_channel
#
#     return repaired_images

def preprocess_repaired_images(loader, annotations, img_size, orig_width, orig_height, device):
    """
    预处理修复后的图像并缓存结果。
    :param loader: 数据加载器
    :param annotations: 图像标注信息
    :param img_size: 输入图像尺寸
    :param orig_width: 原始图像宽度
    :param orig_height: 原始图像高度
    :param device: 设备（CPU/GPU）
    :return: 缓存的修复图像字典 {路径: 修复图像张量}
    """
    repaired_cache = {}
    for img, _, paths in loader:
        img = img.to(device)
        masks = get_mask_from_annotations(paths, annotations, img_size, orig_width, orig_height, device)
        masks = masks.to(device).squeeze(0)

        # 生成修复图像
        repaired_imgs = repair_image(img, masks)

        # 缓存修复后的图像
        for path, repaired_img in zip(paths, repaired_imgs):
            repaired_cache[path] = repaired_img.detach()  # 使用 `detach` 避免显存占用

    return repaired_cache

def get_repaired_images(loader):

    repaired_cache = {}
    for img, _, paths in loader:
        img = img
        # 生成修复图像
        repaired_imgs = img.cpu()

        # 缓存修复后的图像
        for path, repaired_img in zip(paths, repaired_imgs):
            repaired_cache[path] = repaired_img.detach()  # 使用 `detach` 避免显存占用

    return repaired_cache
def train(epoch, loader, re_loader, model, optimizer, scheduler, device, annotations, img_size, orig_width, orig_height):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()
    discriminator_criterion = nn.BCELoss()

    latent_loss_weight = 0.25
    discriminator_loss_weight = 0.1
    sample_size = 25
    alpha = 0.50
    # alpha = 0
    mse_sum = 0
    mse_n = 0
    discriminator_loss_sum = 0
    discriminator_loss_n = 0
    # re_iter = iter(re_loader)
    repaired_cache = get_repaired_images(re_loader)
    for i, (img, label, paths) in enumerate(loader):

        # repaired_img, repair_label, repair_paths = next(re_iter)

        # repaired_img = repaired_img.to(device)

        # 从re_loader中获取修复图像,根据路径取，保证顺序一致
        # repaired_cache = get_repaired_images(re_loader, device)

        re_paths = [path.replace('dataset/defect_dataset\\defect_dataset\\', 'dataset/defect_repair_img\\defect_repair_img\\') for path in paths]
        # 将path中的路径
        repaired_img = torch.stack([repaired_cache[path] for path in re_paths]).to(device)

        model.zero_grad()

        img = img.to(device)


        # repaired_img = torch.stack([repaired_cache[path] for path in paths]).to(device)

        # 根据路径和标注信息生成掩码
        masks = get_mask_from_annotations(paths, annotations, img_size, orig_width, orig_height, device)
        masks = masks.to(device)
        masks = masks.squeeze(0)

        # # 使用修复图像
        # repaired_img = repair_image(img, masks)

        out, latent_loss, _ = model(img)
        # out, latent_loss, _ = model(img, masks)

        # recon_loss = criterion(out, img)
        # 计算 img * mask
        # masked_img = img * masks
        # 仅对掩码覆盖的正常区域计算重建损失
        # masked_output = out * masks
        # recon_loss = criterion(masked_output, masked_img)
        #使用修复的图片和生成的图片计算损失
        recon_loss = criterion(out, repaired_img)
        # recon_loss = criterion(out, img)

        latent_loss = latent_loss.mean()

        # Train discriminator
        real_labels = torch.ones(img.size(0), 1).to(device)
        fake_labels = torch.zeros(img.size(0), 1).to(device)

        real_output = model.discriminator(img)
        real_output = real_output.squeeze(dim=2).squeeze(dim=2)  # 明确去除第 2 和第 3 维度
        fake_output = model.discriminator(out.detach())
        fake_output = fake_output.squeeze(dim=2).squeeze(dim=2)  # 明确去除第 2 和第 3 维度

        discriminator_real_loss = discriminator_criterion(real_output, real_labels)
        discriminator_fake_loss = discriminator_criterion(fake_output, fake_labels)
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss
        # 计算疵点区域的损失
        defect_loss = torch.nn.functional.mse_loss(
            repaired_img * masks, out * masks, reduction='mean'
        )

        # 总损失 = 正常区域损失 + 权重 * 疵点区域损失

        # Calculate total loss
        loss = recon_loss + latent_loss_weight * latent_loss + discriminator_loss_weight * discriminator_loss + alpha * defect_loss
        # loss = recon_loss + latent_loss_weight * latent_loss + discriminator_loss_weight * discriminator_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        part_discriminator_loss_sum = discriminator_loss.item() * img.shape[0]
        part_discriminator_loss_n = img.shape[0]

        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n, "discriminator_loss_sum": part_discriminator_loss_sum,
                "discriminator_loss_n": part_discriminator_loss_n, }
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]
            discriminator_loss_sum += part["discriminator_loss_sum"]
            discriminator_loss_n += part["discriminator_loss_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"discriminator loss: {discriminator_loss.item():.5f}; avg discriminator loss: {discriminator_loss_sum / discriminator_loss_n:.5f}; "
                    f"defect_loss: {defect_loss.item():.3f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:
                model.eval()

                sample = img[:sample_size]
                with torch.no_grad():
                    out, latent_loss, _ = model(sample)

                utils.save_image(
                    torch.cat([sample, out], 2),
                    f"sample_MD_repair/all_img2/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    value_range=(-1, 1),
                )


                model.train()
from torchvision.datasets import ImageFolder

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)  # 调用父类方法获取图像和标签
        path = self.imgs[index][0]  # 获取路径信息
        return img, label, path


import os
from torchvision.utils import save_image


def save_repaired_images(epoch, batch_idx, img, repaired_img, masks, output_dir="repaired_images"):
    """
    保存修复图像以及原始图像和掩码，用于查看修复效果。

    Args:
        epoch (int): 当前训练轮次。
        batch_idx (int): 当前 batch 索引。
        img (Tensor): 原始图像 (B, C, H, W)。
        repaired_img (Tensor): 修复后的图像 (B, C, H, W)。
        masks (Tensor): 掩码图像 (B, 1, H, W)。
        output_dir (str): 保存图像的目录。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 取前4张图像进行保存
    batch_size = img.size(0)
    sample_count = min(batch_size, 4)

    for i in range(sample_count):
        # 获取单张图像、修复图像和掩码
        original = img[i:i + 1]
        repaired = repaired_img[i:i + 1]
        mask = masks[i:i + 1]

        # 拼接图像：原图 | 修复图 | 掩码
        combined = torch.cat([original, repaired, mask.repeat(1, 3, 1, 1)], dim=0)

        # 保存
        save_path = os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}_img_{i}.png")
        save_image(combined, save_path, nrow=3, normalize=True, value_range=(-1, 1))
        print(f"Saved repaired image to: {save_path}")


def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.Resize((args.size, args.size)),
            # transforms.Resize(args.size),
            # transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    # 加载正常数据集
    normal_dataset = ImageFolderWithPaths(args.normal_path, transform=transform)

    # 加载带疵点数据集
    defect_dataset = ImageFolderWithPaths(args.defect_path, transform=transform)
    # 加载修复数据集
    repair_dataset = ImageFolderWithPaths(args.repair_path, transform=transform)

    # 合并数据集
    combined_dataset = ConcatDataset([normal_dataset, defect_dataset])
    # 合并数据集
    combined_repair_dataset = ConcatDataset([normal_dataset, repair_dataset])

    sampler = dist.data_sampler(combined_dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(combined_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    re_loader = DataLoader(combined_repair_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)

    # dataset = datasets.ImageFolder(args.path, transform=transform)
    # sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    # loader = DataLoader(
    #     dataset, batch_size=8 // args.n_gpu, sampler=sampler, num_workers=2
    # )

    annotations = load_annotations(args.annotation_path)

    model = VQVAE().to(device)
    model.load_state_dict(torch.load("checkpoint_MD_repair/normal/vqvae_068.pt"))

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        # train(i, loader, model, optimizer, scheduler, device)
        train(i, loader, re_loader, model, optimizer, scheduler, device, annotations, args.size, args.orig_width, args.orig_height)
        if dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint_MD_repair/all_img2/vqvae_{str(i + 1).zfill(3)}.pt")

from torchvision.utils import save_image

def main1(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # 加载数据集
    defect_dataset = ImageFolderWithPaths(args.defect_path, transform=transform)
    loader = DataLoader(defect_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 加载标注文件
    annotations = load_annotations(args.annotation_path)

    # 加载模型
    model = VQVAE().to(device)
    # model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    # 创建保存结果的目录
    output_dir = "repaired_images"
    os.makedirs(output_dir, exist_ok=True)

    # 处理图像并生成对比图
    with torch.no_grad():
        for i, (img, label, paths) in enumerate(loader):
            img = img.to(device)

            # 根据路径和标注信息生成掩码
            masks = get_mask_from_annotations(paths, annotations, args.size, args.orig_width, args.orig_height, device)

            # 生成修复图像
            repaired_img = repair_image(img, masks)

            # 合成对比图并保存
            for j in range(img.size(0)):
                original_img = (img[j] * 0.5 + 0.5).cpu()  # 去归一化
                mask_img = (masks[j].repeat(3,1,1) * 0.5 + 0.5).cpu()
                repaired_img_tensor = (repaired_img[j] * 0.5 + 0.5).cpu()

                # 拼接三张图片：原图 | 掩码图 | 修复图
                # comparison = torch.cat([original_img, mask_img, repaired_img_tensor], dim=2)  # 在宽度方向拼接

                save_image(
                    original_img,
                    f"{output_dir}/comparison_{i * args.batch_size + j}.png",
                )

            print(f"Batch {i + 1}/{len(loader)} processed and saved in {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
            2 ** 15
            + 2 ** 14
            + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--orig_width", type=int, default=2446)
    parser.add_argument("--orig_height", type=int, default=1000)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sched", type=str)
    # parser.add_argument("--path", type=str, default="dataset/")
    # parser.add_argument("--normal_path", type=str, default="dataset/normal_dataset",help="Path to normal fabric dataset")
    parser.add_argument("--normal_path", type=str, default="dataset/normal_dataset",help="Path to normal fabric dataset")
    # parser.add_argument("--normal_path", type=str, default="dataset/other_dafect",help="Path to normal fabric dataset")
    parser.add_argument("--defect_path", type=str, default="dataset/defect_dataset",help="Path to defect fabric dataset")
    # parser.add_argument("--defect_path", type=str, default="dataset/normal_dataset2",help="Path to defect fabric dataset")
    parser.add_argument("--repair_path", type=str, default="dataset/defect_repair_img",help="Path to repaired fabric dataset")
    parser.add_argument("--annotation_path", type=str,default="dataset/anno_train.json", help="Path to defect annotation JSON file")

    args = parser.parse_args()

    print(args)

    # dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
    dist.launch(main, args.n_gpu, 1, 0, f"tcp://127.0.0.1:{2 ** 15 + 2 ** 14 + hash(os.getpid()) % 2 ** 14}", args=(args,))
    # dist.launch(main1, args.n_gpu, 1, 0, f"tcp://127.0.0.1:{2 ** 15 + 2 ** 14 + hash(os.getpid()) % 2 ** 14}", args=(args,))
