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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import sys
from tqdm import tqdm

import legacy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import style
style.use('seaborn')

import seaborn as sns

#----------------------------------------------------------------------------

class TsnePlot:
    def __init__(self, perplexity=40, learning_rate=200, n_iter=1000):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        
    def plot(self, embedding, labels):
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(perplexity=self.perplexity, learning_rate=self.learning_rate, n_iter=self.n_iter)
        reduced_embedding = tsne.fit_transform(embedding)

        max_val = np.max(reduced_embedding)
        min_val = np.min(reduced_embedding)
        reduced_embedding = (reduced_embedding - min_val)/(max_val - min_val)
        
        # Create scatter plot with different colors for different labels
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab20b')(np.linspace(0, 1, len(unique_labels)))
        plt.figure(figsize=(3,4))
        fig, ax = plt.subplots()
        ax.tick_params(axis='both', labelsize=11)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(reduced_embedding[mask, 0], reduced_embedding[mask, 1], c=colors[i], label=label, alpha=0.6)
        # ax.legend(loc='center left', fancybox=True, shadow=True, ncol=2, bbox_to_anchor=(1, 0.5))
        ax.legend(ncol=2)
        plt.tight_layout()
        plt.savefig('generated_images/tsne_plot.png', bbox_inches='tight', dpi=300)
        plt.close()
        return reduced_embedding




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
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--all_class', type=int, help='Total number of classes')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
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

    embeddings = []
    labels     = []
    w1         = []
    w2         = []

    # Generate images.
    for class_idx in range(all_class):
        label = torch.zeros([1, G.c_dim], device=device)
        label[:, class_idx] = 1
        os.makedirs(outdir+'/{}'.format(class_idx), exist_ok=True)
        for seed_idx, seed in enumerate(tqdm(seeds)):
            sys.stderr.write('\rGenerating image of class %d for seed %d (%d/%d) ...' % (class_idx, seed, seed_idx, len(seeds)))
            z   = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            
            ws  = G.mapping(z, label, truncation_psi=truncation_psi)

            # img = img = G.synthesis(ws, noise_mode=noise_mode)
            # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{class_idx}/seed{seed:04d}.png')
            ws  = ws[0, -1].detach().cpu().numpy()
            w1.append(ws[10])
            w2.append(ws[500])
        break

    sns.jointplot(x=w1, y=w2, kind="scatter", color="#4CB391")
    plt.show()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
