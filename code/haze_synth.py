"""Synthetic haze generation with the atmospheric scattering model (ASM).

I(x) = J(x) * t(x) + A * (1 - t(x)),   t(x) = exp(-beta * d(x))

Three depth modes
-----------------
1. 'random'  - smooth random depth field (pilot mode; no extra data needed)
2. 'gradient'- linear depth ramp (like RESIDE-OTS Synth haze)
3. 'file'    - real per-image depth map (Depth-Anything output, .npy folder)

Used by
-------
- 04_Code/pilot_clip_haze.py            (mode 'random', evaluation)
- 04_Code/train_hazclip.py              (training pairs clean<->hazy)
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F


def smooth_random_depth(h: int, w: int, grid: int | None = None,
                        d_min: float = 0.2, d_max: float = 1.0,
                        seed: int | None = None) -> torch.Tensor:
    """Smooth random depth field (1,H,W) in [d_min, d_max]."""
    g = grid or max(2, min(h, w) // 16)
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    low = d_min + (d_max - d_min) * torch.rand(1, 1, g, g, generator=gen)
    return F.interpolate(low, size=(h, w), mode='bilinear',
                         align_corners=False)[0]


def gradient_depth(h: int, w: int, d_near: float = 0.2, d_far: float = 1.0
                   ) -> torch.Tensor:
    """Horizontal depth ramp (1,H,W): near on the left, far on the right."""
    ramp = torch.linspace(0.0, 1.0, w).view(1, 1, w)
    depth = d_near + (d_far - d_near) * ramp
    return depth.expand(1, h, w)


def load_depth_map(path: str) -> torch.Tensor:
    """Load a Depth-Anything .npy depth (H,W), normalize to [0,1] -> (1,H,W)."""
    d = np.load(path).astype(np.float32)
    d = (d - d.min()) / max(d.max() - d.min(), 1e-9)
    return torch.from_numpy(d)[None]


def apply_haze(img: torch.Tensor, beta: float, depth: torch.Tensor | None = None,
               airlight: float = 1.0, depth_mode: str = 'random',
               seed: int | None = None) -> torch.Tensor:
    """Apply ASM haze to a CxHxW image tensor in [0,1].

    beta = 0 returns the image unchanged.
    """
    if beta <= 0.0:
        return img
    c, h, w = img.shape
    if depth is None:
        if depth_mode == 'gradient':
            depth = gradient_depth(h, w)
        else:  # 'random'
            depth = smooth_random_depth(h, w, seed=seed)
    t = torch.exp(-beta * depth)                      # (1,H,W)
    return img * t + airlight * (1.0 - t)


def make_training_pair(img: torch.Tensor, beta: float, airlight: float = 1.0,
                       depth_mode: str = 'random', seed: int | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (clean J, hazy I) pair at severity beta; also returns t for L_phys.

    Returns
    -------
    (hazy, transmission) : (C,H,W), (1,H,W) — transmission is the GT t(x).
    """
    c, h, w = img.shape
    if depth_mode == 'gradient':
        depth = gradient_depth(h, w)
    else:
        depth = smooth_random_depth(h, w, seed=seed)
    t = torch.exp(-beta * depth)
    hazy = img * t + airlight * (1.0 - t)
    return hazy, t


class HazyPairDataset(torch.utils.data.Dataset):
    """On-the-fly (clean, hazy, t) triples from an image folder or tensor bank.

    Parameters
    ----------
    images : (N,C,H,W) tensor in [0,1]  (small-scale sandbox: CIFAR/Flickr8k)
    betas  : list of training severities (sampled uniformly per item)
    """

    def __init__(self, images: torch.Tensor, betas: list[float],
                 depth_mode: str = 'random', out_size: int = 224,
                 labels: np.ndarray | None = None):
        self.images = images
        self.betas = [b for b in betas if b > 0.0] or [0.6]
        self.depth_mode = depth_mode
        self.out_size = out_size
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        beta = self.betas[int(torch.randint(0, len(self.betas), (1,)))]
        hazy, t = make_training_pair(img, beta, depth_mode=self.depth_mode,
                                     seed=None)
        img_r = F.interpolate(img[None], size=(self.out_size, self.out_size),
                              mode='bilinear', align_corners=False)[0]
        hazy_r = F.interpolate(hazy[None], size=(self.out_size, self.out_size),
                               mode='bilinear', align_corners=False)[0]
        t_r = F.interpolate(t[None], size=(self.out_size, self.out_size),
                            mode='bilinear', align_corners=False)
        label = -1 if self.labels is None else int(self.labels[idx])
        return img_r, hazy_r, t_r, beta, label


# ----------------------------------------------------------------- stats utils

def transmission_stats(t: torch.Tensor) -> torch.Tensor:
    """Per-image scalar stats of t used by the haze head target:
    [mean(t), std(t), min(t)] -> (3,)  (see Method_Notes §2 h_head)."""
    return torch.stack([t.mean(), t.std(unbiased=False), t.amin()], dim=0)


def depth_folder_check(folder: str) -> bool:
    """True if a Depth-Anything .npy depth folder exists and is non-empty."""
    return os.path.isdir(folder) and any(f.endswith('.npy')
                                         for f in os.listdir(folder))
