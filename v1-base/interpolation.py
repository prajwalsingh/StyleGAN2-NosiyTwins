# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import pandas as pd
import PIL.Image
import torch
import sys
from tqdm import tqdm

import legacy

import numpy as np
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
from tsnecuda import TSNE
from matplotlib import style
import seaborn as sns
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

style.use('seaborn')

def show(imgs, batch_size, outdir, idx):

    imgs = make_grid(imgs, nrow=1)

    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig('{}/{}.png'.format(outdir, idx), bbox_inches='tight', dpi=300)
    plt.close()

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--batch_size', type=int, help='Batch Size')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--all_class', type=int, help='Total number of classes')
@click.option('--class_a', 'class_a', type=int, help='Class label (unconditional if not specified)')
@click.option('--class_b', 'class_b', type=int, help='Class label (unconditional if not specified)')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--interp', 'interp_counts', type=int, help='Total number of images to generate')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    batch_size: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    interp_counts: int,
    class_idx: Optional[int],
    class_a: Optional[int],
    class_b: Optional[int],
    all_class: Optional[int],
    projected_w: Optional[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    
    label_a = torch.zeros([batch_size, G.c_dim], device=device)
    label_a[:, class_a] = 1

    label_b = torch.zeros([batch_size, G.c_dim], device=device)
    label_b[:, class_b] = 1

    os.makedirs(outdir, exist_ok=True)

    z_a   = torch.from_numpy(np.random.RandomState(seeds[0]).randn(batch_size, G.z_dim)).to(device)        
    ws_a  = G.mapping(z_a, label_a, truncation_psi=1, mode=None)

    z_b   = torch.from_numpy(np.random.RandomState(seeds[0]).randn(batch_size, G.z_dim)).to(device)        
    ws_b  = G.mapping(z_b, label_b, truncation_psi=1, mode=None)

    for idx, alpha in enumerate(tqdm(range(1, interp_counts))):
        alpha    = (alpha - 1.0)/interp_counts
        ws_inter = (ws_a * alpha) + (ws_b * (1-alpha))
        img      = G.synthesis(ws_inter, noise_mode=noise_mode)
        img      = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        show(img, batch_size, outdir, idx)
    # for seed_idx, seed in enumerate(tqdm(seeds)):
    #     sys.stderr.write('\rGenerating image of class %d for seed %d (%d/%d) ...' % (class_idx, seed, seed_idx, len(seeds)))

    #     if new_run:
    #         embeddings = ws[:,0,:].detach().cpu().numpy()
    #         labels     = np.argmax(label.detach().cpu().numpy(), axis=-1)
    #         new_run    = False
    #     else:
    #         embeddings = np.concatenate([embeddings, ws[:,0,:].cpu().detach().numpy()],axis=0)
    #         labels = np.concatenate([labels, np.argmax(label.detach().cpu().numpy(), axis=-1)], axis=0)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
