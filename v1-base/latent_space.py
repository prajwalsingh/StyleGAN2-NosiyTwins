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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
style.use('seaborn')

# from umap import UMAP
#----------------------------------------------------------------------------

class TsnePlot:
    def __init__(self, perplexity=40, learning_rate='auto', n_iter=300):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        
    def plot(self, embedding, labels, fignum=None):
        # Perform t-SNE dimensionality reduction
        tsne = TSNE(perplexity=self.perplexity, learning_rate=self.learning_rate, n_iter=self.n_iter)
        reduced_embedding = tsne.fit_transform(embedding)
        # reduced_embedding = reduced_embedding.detach().cpu().numpy()

        max_val = np.max(reduced_embedding)
        min_val = np.min(reduced_embedding)
        reduced_embedding = (reduced_embedding - min_val)/(max_val - min_val)
        
        df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two'])
        df["tsne-2d-one"] = reduced_embedding[:, 0]
        df["tsne-2d-two"] = reduced_embedding[:, 1]
        df['labels'] = labels

        total_labels = np.unique(labels)

        palette = sns.color_palette("tab20",total_labels.shape[0])

        plt.figure(figsize=(16, 12))
        sns.scatterplot(x="tsne-2d-one",
                y="tsne-2d-two",
                hue="labels",
                palette=palette,
                data=df,
                legend="full",
                alpha=0.5).legend(fontsize=15, loc="upper right")
        plt.title("TSNE result of {flag} images".format(flag='fake'), fontsize=25)
        plt.xlabel("", fontsize=7)
        plt.ylabel("", fontsize=7)
        plt.tight_layout()
        # plt.savefig('generated_images/plots/tsne_plot.png')
        plt.savefig('generated_images/plots/tsne_plot.png', bbox_inches='tight', dpi=300)
        # plt.close()
        return reduced_embedding

class UmapPlot:
    def __init__(self,):
        pass
        
    def plot(self, embedding, labels, fignum=None):
        # Perform t-SNE dimensionality reduction
        umap_2d = UMAP(n_components=2, init='random', random_state=0)
        reduced_embedding = umap_2d.fit_transform(embedding)

        max_val = np.max(reduced_embedding)
        min_val = np.min(reduced_embedding)
        reduced_embedding = (reduced_embedding - min_val)/(max_val - min_val)

        df = pd.DataFrame(columns=['tsne-2d-one', 'tsne-2d-two'])
        df["tsne-2d-one"] = reduced_embedding[:, 0]
        df["tsne-2d-two"] = reduced_embedding[:, 1]
        df['labels'] = labels

        total_labels = np.unique(labels)

        palette = sns.color_palette("tab20",total_labels.shape[0])

        plt.figure(figsize=(16, 12))
        sns.scatterplot(x="tsne-2d-one",
                y="tsne-2d-two",
                hue="labels",
                palette=palette,
                data=df,
                legend="full",
                alpha=0.5).legend(fontsize=15, loc="upper right")
        plt.title("TSNE result of {flag} images".format(flag='fake'), fontsize=25)
        plt.xlabel("", fontsize=7)
        plt.ylabel("", fontsize=7)
        plt.tight_layout()
        # plt.savefig('generated_images/plots/tsne_plot.png')
        plt.savefig('generated_images/plots/umap_plot.png', bbox_inches='tight', dpi=300)
        # plt.close()
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
@click.option('--batch_size', type=int, help='Batch Size')
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
    batch_size: int,
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

    new_run    = True
    # Generate images.
    # for class_idx in [0, 1, 2, 3, 16, 17, 18, 19]:
    os.makedirs(outdir+'/{}'.format(class_idx), exist_ok=True)
    
    for class_idx in range(all_class):
        label = torch.zeros([batch_size, G.c_dim], device=device)
        label[:, class_idx] = 1
        os.makedirs(outdir+'/{}'.format(class_idx), exist_ok=True)
        for seed_idx, seed in enumerate(tqdm(seeds)):
            sys.stderr.write('\rGenerating image of class %d for seed %d (%d/%d) ...' % (class_idx, seed, seed_idx, len(seeds)))
            z   = torch.from_numpy(np.random.RandomState(seed).randn(batch_size, G.z_dim)).to(device)
            
            ws  = G.mapping(z, label, truncation_psi=1, mode=None)

            # img = img = G.synthesis(ws, noise_mode=noise_mode)
            # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # for inum, img in enumerate(img):
            #     PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(f'{outdir}/{class_idx}/seed{seed:04d}_{inum}.png')

            if new_run:
                embeddings = ws[:,0,:].detach().cpu().numpy()
                labels     = np.argmax(label.detach().cpu().numpy(), axis=-1)
                new_run    = False
            else:
                embeddings = np.concatenate([embeddings, ws[:,0,:].cpu().detach().numpy()],axis=0)
                labels = np.concatenate([labels, np.argmax(label.detach().cpu().numpy(), axis=-1)], axis=0)

    rand_idx   = np.random.choice(embeddings.shape[0], size=embeddings.shape[0], replace=False)
    embeddings = torch.from_numpy(embeddings[rand_idx])#.to('cuda')
    labels     = labels[rand_idx]

    # for perp in range(1,30):
    # perp=1
    # umap_plot = UmapPlot()
    # umap_plot.plot(embeddings, labels)

    # tsne_plot = TsnePlot(perplexity=300, learning_rate='auto', n_iter=1000)
    # tsne_plot.plot(embeddings, labels)

    tsne_plot = TsnePlot(perplexity=40, learning_rate=10.0, n_iter=2000)
    tsne_plot.plot(embeddings, labels)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
