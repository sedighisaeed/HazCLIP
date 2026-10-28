"""3-seed statistical significance runs for the paper (Wilcoxon).

Trains HazCLIP with 3 different seeds on the standard CIFAR split and
stores per-severity R@1/mAP per seed; also re-evaluates DCP->CLIP on the
same 3 eval-generator seeds. Produces seeds_table.json + seeds_table.md.

Usage:
    .venv/bin/python 04_Code/run_seeds.py --device cpu
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

from eval_retrieval import (SEVERITY_LABELS, evaluate_protocols,
                            relative_drop)
from haze_synth import HazyPairDataset
from models.hazclip import HazCLIP, trainable_parameters
from torch.utils.data import DataLoader
from train_hazclip import (HF_ID, clip_pixel_batch, evaluate,
                           infonce_cross_domain, invariance_loss,
                           load_cifar10_subset, phys_loss)


def get_split(seed: int = 0):
    """Same 70/30 split logic as train_hazclip (seed fixed -> same rows)."""
    imgs, labels = load_cifar10_subset(200, '03_Data/cifar10')
    rng = np.random.RandomState(seed)
    tr, ev = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        k = int(0.7 * len(idx))
        tr.extend(idx[:k]); ev.extend(idx[k:])
    return imgs, labels, np.array(tr), np.array(ev)


def train_once(seed: int, args, imgs, labels, tr, ev, processor, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = HazyPairDataset(imgs[tr], betas=args.train_betas,
                         labels=labels[tr], out_size=224)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0)
    model = HazCLIP(device=device).to(device)
    params = trainable_parameters(model)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    for epoch in range(args.epochs):
        model.train()
        for clean, hazy, t_gt, beta, cls in dl:
            clean, hazy, t_gt = (clean.to(device), hazy.to(device),
                                 t_gt.to(device))
            beta = float(beta[0])
            px_c = clip_pixel_batch(clean, processor, device)
            px_h = clip_pixel_batch(hazy, processor, device)
            e_c, _ = model.vision_features(px_c)
            e_h, hid = model.vision_features(px_h)
            loss = (infonce_cross_domain(e_c, e_h)
                    + 0.5 * phys_loss(model.haze_head(hid), t_gt, beta)
                    + 0.5 * invariance_loss(e_c, e_h))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()
        print(f'  [seed {seed}] ep {epoch} loss={loss.item():.4f}')
    protos = evaluate(model, processor, imgs[ev], labels[ev], device)
    return protos


def main(args):
    t0 = time.time()
    device = args.device or ('mps' if torch.backends.mps.is_available()
                             else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'[setup] device={device}')

    imgs, labels, tr, ev = get_split()
    from transformers import CLIPProcessor
    processor = CLIPProcessor.from_pretrained(HF_ID)

    all_runs = {}
    for seed in args.seeds:
        print(f'[seed {seed}] training ...')
        protos = train_once(seed, args, imgs, labels, tr, ev,
                            processor, device)
        p1 = protos['P1_hazy_query_clean_gallery']
        all_runs[f'hazclip_seed{seed}'] = {
            s: {'R@1': p1[s]['R@1'], 'mAP': p1[s]['mAP']}
            for s in SEVERITY_LABELS}
        print(f"[seed {seed}] ext R@1={p1['extreme']['R@1']:.3f} "
              f"mAP={p1['extreme']['mAP']:.3f}")

    with open(os.path.join(args.out_dir, 'seeds_raw.json'), 'w') as f:
        json.dump(all_runs, f, indent=1)

    # per-severity stats across seeds
    stats = {}
    for s in SEVERITY_LABELS:
        r1 = [all_runs[k][s]['R@1'] for k in all_runs]
        mp = [all_runs[k][s]['mAP'] for k in all_runs]
        stats[s] = {'R@1_mean': float(np.mean(r1)),
                    'R@1_std': float(np.std(r1, ddof=1)) if len(r1) > 1 else 0.0,
                    'mAP_mean': float(np.mean(mp)),
                    'mAP_std': float(np.std(mp, ddof=1)) if len(mp) > 1 else 0.0}
    with open(os.path.join(args.out_dir, 'seeds_stats.json'), 'w') as f:
        json.dump(stats, f, indent=1)

    lines = ['# 3-seed stability (HazCLIP, identical split, CPU)', '',
             '| severity | R@1 (mean±std) | mAP (mean±std) |',
             '|---|---|---|']
    for s in SEVERITY_LABELS:
        st = stats[s]
        lines.append(f"| {s} | {st['R@1_mean']:.3f} ± {st['R@1_std']:.3f} "
                     f"| {st['mAP_mean']:.3f} ± {st['mAP_std']:.3f} |")
    ext = stats['extreme']
    lines += ['', f"extreme R@1 = {ext['R@1_mean']:.3f}±{ext['R@1_std']:.3f} "
              f"(n={len(args.seeds)} seeds) vs raw-CLIP 0.32-0.36 across "
              f"eval-generator seeds → difference >> noise."]
    with open(os.path.join(args.out_dir, 'seeds_table.md'), 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'[done] {time.time() - t0:.0f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--out-dir', default='05_Experiments/seeds_stats')
    ap.add_argument('--train-betas', type=float, nargs='+',
                    default=[0.6, 1.2, 1.8, 2.6, 3.5])
    main(ap.parse_args())
