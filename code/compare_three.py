"""Head-to-head comparison on ONE identical split:
raw CLIP vs DCP-dehaze->CLIP vs HazCLIP (trained ckpt).

Resolves the split-mismatch caveat in EXPERIMENTS_LOG §4.

Usage:
    .venv/bin/python 04_Code/compare_three.py --n-per-class 60 \
        --ckpt 05_Experiments/hazclip_v0/hazclip_trainable.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_retrieval import (SEVERITY_BETAS, SEVERITY_LABELS,
                            evaluate_protocols, protocol_table)
from haze_synth import apply_haze


def load_split(n_per_class: int, data_dir: str):
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root=data_dir, train=True, download=True)
    targets = np.array(ds.targets)
    rng = np.random.RandomState(0)
    ev_idx = []
    for cls in range(10):
        idx = np.where(targets == cls)[0]
        rng.shuffle(idx)
        k = int(0.7 * len(idx))
        ev_idx.extend(idx[k:k + n_per_class])   # same tail split as train_hazclip
    ev_idx = np.array(sorted(ev_idx))
    imgs = torch.stack([
        torch.from_numpy(np.asarray(ds[i][0]).copy()).permute(2, 0, 1)
        .float() / 255.0 for i in ev_idx])
    return imgs, targets[ev_idx]


class RawCLIP:
    name = 'raw_clip'

    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained(
            'openai/clip-vit-base-patch32').to(device).eval()
        self.proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    @torch.no_grad()
    def embed(self, imgs):
        import torch.nn.functional as F
        from PIL import Image
        out = []
        for i in range(0, len(imgs), 128):
            chunk = imgs[i:i + 128].cpu()
            pil = [Image.fromarray((im.permute(1, 2, 0).numpy() * 255)
                                   .astype(np.uint8)) for im in chunk]
            px = self.proc(images=pil, return_tensors='pt').to(self.device)
            e = self.model.get_image_features(**px)
            out.append(F.normalize(e, dim=-1).cpu())
        return torch.cat(out)


class DCPCLIP(RawCLIP):
    name = 'dcp_dehaze_clip'

    @torch.no_grad()
    def embed(self, imgs):
        from baseline_dehaze_clip import dcp_dehaze
        rec = torch.stack([dcp_dehaze(im) for im in imgs])
        return super().embed(rec)


class HazCLIPWrap(RawCLIP):
    name = 'hazclip'

    def __init__(self, device, ckpt: str):
        super().__init__(device)
        from models.hazclip import HazCLIP
        self.hz = HazCLIP(device=device).to(device)
        self.hz.load_trainable(ckpt)
        self.hz.eval()

    @torch.no_grad()
    def embed(self, imgs):
        import torch.nn.functional as F
        from train_hazclip import clip_pixel_batch
        out = []
        for i in range(0, len(imgs), 128):
            px = clip_pixel_batch(imgs[i:i + 128].cpu(), self.proc,
                                  self.device)
            e = self.hz.encode_images_eval(px)
            out.append(F.normalize(e, dim=-1).cpu())
        return torch.cat(out)


def main(args):
    t0 = time.time()
    device = args.device or ('mps' if torch.backends.mps.is_available()
                             else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    imgs, labels = load_split(args.n_per_class, args.data_dir)
    print(f'[data] eval split n={len(imgs)} (identical for all systems)')

    # pre-generate hazy variants once (shared)
    hazy_bank = {}
    for sev, beta in zip(SEVERITY_LABELS, SEVERITY_BETAS):
        hazy_bank[sev] = imgs if beta == 0 else torch.stack(
            [apply_haze(im, beta) for im in imgs])

    systems = [RawCLIP(device), DCPCLIP(device)]
    if os.path.exists(args.ckpt):
        systems.append(HazCLIPWrap(device, args.ckpt))
    else:
        print(f'[warn] ckpt not found: {args.ckpt} — skipping HazCLIP')

    all_protos = {}
    gallery_cache = {}
    for sys_ in systems:
        print(f'[{sys_.name}] embedding ...')
        q_embs = {}
        for sev in SEVERITY_LABELS:
            q_embs[sev] = sys_.embed(hazy_bank[sev])
        protos = evaluate_protocols(q_embs, labels)
        all_protos[sys_.name] = protos
        p1 = protos['P1_hazy_query_clean_gallery']
        print(f"  P1 extreme R@1={p1['extreme']['R@1']:.3f} "
              f"mAP={p1['extreme']['mAP']:.3f}")

    with open(os.path.join(args.out_dir, 'compare_three.json'), 'w') as f:
        json.dump(all_protos, f, indent=1)

    # combined table
    lines = ['# Head-to-head (identical split, P1 hazy→clean)', '']
    head = ('| system | ' + ' | '.join(f'{s} R@1' for s in SEVERITY_LABELS)
            + ' | ext mAP | clean mAP |')
    lines += [head, '|' + '---|' * (len(SEVERITY_LABELS) + 3)]
    for name, protos in all_protos.items():
        p1 = protos['P1_hazy_query_clean_gallery']
        row = [name] + [f"{p1[s]['R@1']:.3f}" for s in SEVERITY_LABELS] + [
            f"{p1['extreme']['mAP']:.3f}", f"{p1['clean']['mAP']:.3f}"]
        lines.append('| ' + ' | '.join(row) + ' |')
    with open(os.path.join(args.out_dir, 'compare_three.md'), 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'[done] {time.time() - t0:.0f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-class', type=int, default=60)
    ap.add_argument('--data-dir', default='03_Data/cifar10')
    ap.add_argument('--out-dir', default='05_Experiments/compare_three')
    ap.add_argument('--device', default=None)
    ap.add_argument('--ckpt', default='05_Experiments/hazclip_v0/hazclip_trainable.pt')
    main(ap.parse_args())
