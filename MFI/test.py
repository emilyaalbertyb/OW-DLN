import argparse
import sys
import os

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist



def test(loader, model, device):
    if dist.is_primary():
        loader = tqdm(loader)
    model.eval()

    criterion = nn.MSELoss()
    latent_loss_weight=0.25
    sample_size = 128
    mse_sum = 0
    mse_n = 0

    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            img = img.to(device)
            sample = img[:sample_size]

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
    # # Save reconstructed images
    # utils.save_image(out, f"recon/reconstructed_{i}.png")

    # Create a grid to display original and reconstructed images
    grid = utils.make_grid(torch.cat([img, out], dim=0), nrow=img.shape[0])

    # Save the grid image
    utils.save_image(grid, f"small256/original_reconstructed_{i}.png",
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
        dataset,batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
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