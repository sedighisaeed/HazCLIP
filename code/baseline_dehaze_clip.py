"""Baseline: dehaze-then-retrieve (two-stage) vs HazCLIP end-to-end.

Pipeline: hazy query -> DCP dehazing (He et al., classic, no training)
          -> CLIP embedding -> clean gallery.

This is the baseline reviewers always ask for: "why not just dehaze first?"
Compares against results from 05_Experiments/pilot_clip_haze (raw CLIP)
and 05_Experiments/hazclip_v0 (HazCLIP).

Usage:
    .venv/bin/python 04_Code/baseline_dehaze_clip.py --n-per-class 60 --device mps
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eval_retrieval import (SEVERITY_BETAS, SEVERITY_LABELS,
                            evaluate_protocols, protocol_table,
                            relative_drop)
from haze_synth import apply_haze


# ------------------------------------------------------------------ DCP

def dark_channel(img: torch.Tensor, k: int = 7) -> torch.Tensor:
    """Dark channel prior: (C,H,W) -> (1,H,W)."""
    return img.amin(dim=0, keepdim=True).unfold(2, k, 1).unfold(3, k, 1) \
        .amin(dim=(-1, -2))


def box_min_filter(x: torch.Tensor, k: int = 7) -> torch.Tensor:
    """Simple min filter via max_pool on negated tensor (CPU-safe ops only)."""
    return -F.max_pool2d(-x.unsqueeze(0), k, stride=1, padding=k // 2).squeeze(0)


def guided_depth_estimate(img: torch.Tensor, omega: float = 0.95) -> torch.Tensor:
    """Estimate transmission t(x) = 1 - omega * dark_channel(img)."""
    dc = box_min_filter(img.amin(dim=0, keepdim=True), k=15)
    t = (1.0 - omega * dc).clamp(min=0.05)
    return t


def estimate_airlight(img: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """A = brightest pixel among the most haze-opaque (lowest-t) pixels."""
    flat = img.reshape(3, -1)
    t_flat = t.flatten()
    idx = torch.argsort(t_flat)[:max(1, int(0.001 * t_flat.numel()))]
    return flat[:, idx].max(dim=1).values.clamp(min=0.5)[:, None, None]


def dcp_dehaze(img: torch.Tensor, omega: float = 0.95) -> torch.Tensor:
    """Full DCP on a (3,H,W) tensor in [0,1]. Vectorized, no OpenCV."""
    t = guided_depth_estimate(img, omega)
    a = estimate_airlight(img, t)
    j = (img - a) / t + a
    return j.clamp(0.0, 1.0)


# ------------------------------------------------------------------ encoder

class ClipEncoder:
    """Same as pilot (transformers CLIP, openai weights)."""

    def __init__(self, device: str):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        hf = 'openai/clip-vit-base-patch32'
        self.model = CLIPModel.from_pretrained(hf).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(hf)

    @torch.no_grad()
    def embed(self, imgs: torch.Tensor, batch: int = 128) -> torch.Tensor:
        out = []
        for i in range(0, len(imgs), batch):
            chunk = imgs[i:i + batch]
            if chunk.device.type != 'cpu':
                chunk = chunk.cpu()
            pil = [Image.fromarray((im.permute(1, 2, 0).numpy() * 255)
                                   .astype(np.uint8)) for im in chunk]
            px = self.proc(images=pil, return_tensors='pt').to(self.device)
            e = self.model.get_image_features(**px)
            out.append(F.normalize(e, dim=-1).cpu())
        return torch.cat(out)


# ------------------------------------------------------------------ main

def main(args):
    t0 = time.time()
    device = args.device or ('mps' if torch.backends.mps.is_available()
                             else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'[setup] device={device}')

    # data: CIFAR-10 test subset (same protocol as pilot)
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root=args.data_dir, train=False, download=True)
    targets = np.array(ds.targets)
    picks = []
    for cls in range(10):
        idx = np.where(targets == cls)[0][:args.n_per_class]
        picks.extend(idx.tolist())
    picks = np.array(sorted(picks))
    imgs = torch.stack([
        torch.from_numpy(np.asarray(ds[i][0]).copy()).permute(2, 0, 1)
        .float() / 255.0 for i in picks])
    labels = targets[picks]
    print(f'[data] {len(imgs)} images')

    enc = ClipEncoder(device)
    gallery = enc.embed(imgs)                      # clean gallery embeddings

    rows = {}
    for sev, beta in zip(SEVERITY_LABELS, SEVERITY_BETAS):
        if beta == 0.0:
            hazy = imgs
        else:
            hazy = torch.stack([apply_haze(im, beta) for im in imgs])
        if args.dehaze:
            print(f'[{sev}] DCP dehazing {len(hazy)} images ...')
            # DCP on CPU tensors (max_pool), batch sequential
            rec = torch.stack([dcp_dehaze(im) for im in hazy])
            q = enc.embed(rec)
        else:
            q = enc.embed(hazy)
        sim = q @ gallery.T
        from eval_retrieval import recall_at_k, map_all, r_precision
        m = recall_at_k(sim, labels)
        m['mAP'] = map_all(sim, labels)
        m['RP'] = r_precision(sim, labels)
        rows[sev] = m
        print(f'[P1] {sev:9s} R@1={m["R@1"]:.3f} mAP={m["mAP"]:.3f}')

    tag = 'dcp_dehaze_clip' if args.dehaze else 'raw_clip'
    results = {'baseline': tag, 'n': len(imgs), 'P1': rows,
               'time_sec': time.time() - t0}
    with open(os.path.join(args.out_dir, f'results_{tag}.json'), 'w') as f:
        json.dump(results, f, indent=1)

    lines = [f'# Baseline: {tag} — P1 hazy→clean', '',
             protocol_table(rows), '',
             f"Relative extreme drop R@1: {relative_drop(rows):.1f}%"]
    with open(os.path.join(args.out_dir, f'summary_{tag}.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(f'[done] {args.out_dir} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-class', type=int, default=60)
    ap.add_argument('--data-dir', default='03_Data/cifar10')
    ap.add_argument('--out-dir', default='05_Experiments/baselines')
    ap.add_argument('--device', default=None)
    ap.add_argument('--dehaze', action='store_true',
                    help='apply DCP dehazing before CLIP')
    main(ap.parse_args())
