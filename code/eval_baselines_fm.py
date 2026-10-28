"""Foundation-model degradation study under synthetic haze:
raw CLIP vs SigLIP vs DINOv2 (feature-kNN retrieval, no text head for DINO).

Paper role (§4.1): the degradation is a *general* foundation-model property
→ motivates a general adaptation recipe (HazCLIP on CLIP; recipe transfers).

DINOv2 has no text tower → retrieval by cosine kNN against clean gallery is
the same protocol; no zero-shot classification for it.

Usage:
    .venv/bin/python 04_Code/eval_baselines_fm.py --n-per-class 60 --device mps
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_retrieval import (SEVERITY_BETAS, SEVERITY_LABELS, map_all,
                            recall_at_k, r_precision)
from haze_synth import apply_haze


def load_split(n_per_class: int, data_dir: str):
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root=data_dir, train=False, download=True)
    targets = np.array(ds.targets)
    picks = []
    for cls in range(10):
        picks.extend(np.where(targets == cls)[0][:n_per_class].tolist())
    picks = np.array(sorted(picks))
    imgs = torch.stack([
        torch.from_numpy(np.asarray(ds[i][0]).copy()).permute(2, 0, 1)
        .float() / 255.0 for i in picks])
    return imgs, targets[picks]


class FMEncoder:
    """Unified wrapper: returns L2-normalized global embeddings."""

    def __init__(self, name: str, device: str):
        self.name = name
        self.device = device
        if name == 'clip':
            from transformers import CLIPModel, CLIPProcessor
            self.proc = CLIPProcessor.from_pretrained(
                'openai/clip-vit-base-patch32')
            self.model = CLIPModel.from_pretrained(
                'openai/clip-vit-base-patch32').to(device).eval()
            self.kind = 'clip'
        elif name == 'siglip':
            from transformers import (SiglipImageProcessor, SiglipModel)
            # image-only pipeline: the slow tokenizer needs protobuf, and
            # retrieval baselines embed images only — skip text entirely.
            self.proc = SiglipImageProcessor.from_pretrained(
                'google/siglip-base-patch16-224')
            self.model = SiglipModel.from_pretrained(
                'google/siglip-base-patch16-224').to(device).eval()
            self.kind = 'siglip'
        elif name == 'dinov2':
            from transformers import AutoImageProcessor, AutoModel
            self.proc = AutoImageProcessor.from_pretrained(
                'facebook/dinov2-small')
            self.model = AutoModel.from_pretrained(
                'facebook/dinov2-small').to(device).eval()
            self.kind = 'dino'
        else:
            raise ValueError(name)

    @torch.no_grad()
    def embed(self, imgs: torch.Tensor, batch: int = 96) -> torch.Tensor:
        from PIL import Image
        out = []
        for i in range(0, len(imgs), batch):
            chunk = imgs[i:i + batch].cpu()
            pil = [Image.fromarray((im.permute(1, 2, 0).numpy() * 255)
                                   .astype(np.uint8)) for im in chunk]
            px = self.proc(images=pil, return_tensors='pt').to(self.device)
            if self.kind == 'clip':
                e = self.model.get_image_features(**px)
            elif self.kind == 'siglip':
                e = self.model.get_image_features(**px)
            else:  # dino: CLS token of last hidden state
                e = self.model(**px).last_hidden_state[:, 0]
            out.append(F.normalize(e, dim=-1).cpu())
        return torch.cat(out)


def main(args):
    t0 = time.time()
    device = args.device or ('mps' if torch.backends.mps.is_available()
                             else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'[setup] device={device}')
    imgs, labels = load_split(args.n_per_class, args.data_dir)
    print(f'[data] n={len(imgs)}')

    # pre-generate hazy variants once
    hazy_bank = {}
    for sev, beta in zip(SEVERITY_LABELS, SEVERITY_BETAS):
        hazy_bank[sev] = imgs if beta == 0 else torch.stack(
            [apply_haze(im, beta) for im in imgs])

    results = {}
    for name in args.models:
        try:
            enc = FMEncoder(name, device)
        except Exception as e:
            print(f'[skip] {name}: {e}')
            continue
        print(f'[{name}] clean gallery embedding ...')
        g = enc.embed(imgs)
        p1 = {}
        for sev in SEVERITY_LABELS:
            q = enc.embed(hazy_bank[sev])
            sim = q @ g.T
            m = recall_at_k(sim, labels)
            m['mAP'] = map_all(sim, labels)
            m['RP'] = r_precision(sim, labels)
            p1[sev] = m
            print(f'  [{name}] {sev:9s} R@1={m["R@1"]:.3f} mAP={m["mAP"]:.3f}')
        drop = 100 * (p1['clean']['R@1'] - p1['extreme']['R@1']) \
            / max(p1['clean']['R@1'], 1e-9)
        results[name] = {'P1': p1, 'rel_extreme_drop_R@1': drop}
        print(f'  [{name}] relative extreme R@1 drop = {drop:.1f}%')

    with open(os.path.join(args.out_dir, 'fm_degradation.json'), 'w') as f:
        json.dump(results, f, indent=1)

    lines = ['# Foundation-model degradation under haze (P1, identical data)',
             '',
             '| model | clean R@1 | moderate | heavy | extreme | ext mAP | rel drop |',
             '|---|---|---|---|---|---|---|']
    for name, r in results.items():
        p = r['P1']
        lines.append(f"| {name} | {p['clean']['R@1']:.3f} "
                     f"| {p['moderate']['R@1']:.3f} | {p['heavy']['R@1']:.3f} "
                     f"| {p['extreme']['R@1']:.3f} | {p['extreme']['mAP']:.3f} "
                     f"| {r['rel_extreme_drop_R@1']:.1f}% |")
    with open(os.path.join(args.out_dir, 'fm_degradation.md'), 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'[done] {time.time() - t0:.0f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+',
                    default=['clip', 'siglip', 'dinov2'])
    ap.add_argument('--n-per-class', type=int, default=60)
    ap.add_argument('--data-dir', default='03_Data/cifar10')
    ap.add_argument('--out-dir', default='05_Experiments/fm_baselines')
    ap.add_argument('--device', default=None)
    main(ap.parse_args())
