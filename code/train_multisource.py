"""Multi-source HazCLIP training: CIFAR-10 + Flickr8k (both upsampled to
224 by HazyPairDataset), then real-fog + CIFAR-severity evaluation.

Goal (from EXPERIMENTS_LOG §12): fix the resolution-overfit negative
transfer of v1 while keeping v0's zero-shot transfer.

Usage:
    .venv/bin/python 04_Code/train_multisource.py --device cpu
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
                            protocol_table, relative_drop)
from haze_synth import HazyPairDataset
from models.hazclip import HazCLIP, trainable_parameters
from torch.utils.data import ConcatDataset, DataLoader
from train_hazclip import (HF_ID, clip_pixel_batch, evaluate, infonce_cross_domain,
                           invariance_loss, load_cifar10_subset,
                           load_folder_bank, phys_loss)


def main(args):
    t0 = time.time()
    device = args.device or ('mps' if torch.backends.mps.is_available()
                             else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'[setup] device={device} out={args.out_dir}')

    # ---- data: cifar train-split + flickr subset, merged ------------------
    imgs_c, labels_c = load_cifar10_subset(140, '03_Data/cifar10')
    imgs_f, labels_f = load_folder_bank('03_Data/flickr8k/Flicker8k_Dataset',
                                        600, size=224)
    # labels must be disjoint: flickr pseudo-groups offset by 100
    labels_f = labels_f + 100
    rng = np.random.RandomState(0)
    tr_c, ev_c = [], []
    for cls in np.unique(labels_c):
        idx = np.where(labels_c == cls)[0]
        rng.shuffle(idx)
        k = int(0.7 * len(idx))
        tr_c.extend(idx[:k]); ev_c.extend(idx[k:])
    ev_f = []
    for cls in np.unique(labels_f):
        idx = np.where(labels_f == cls)[0]
        rng.shuffle(idx)
        ev_f.extend(idx[:max(1, len(idx) // 5)])   # 20% eval for flickr
    ds_c_tr = HazyPairDataset(imgs_c[tr_c], betas=args.train_betas,
                              labels=labels_c[tr_c], out_size=224)
    ds_f_tr = HazyPairDataset(imgs_f, betas=args.train_betas,
                              labels=labels_f, out_size=224)
    dl = DataLoader(ConcatDataset([ds_c_tr, ds_f_tr]),
                    batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_set = (imgs_c[ev_c], labels_c[ev_c])
    print(f'[data] train={len(ds_c_tr) + len(ds_f_tr)} '
          f'cifar_eval={len(ev_c)} flickr_eval={len(ev_f)}')

    # ---- model ------------------------------------------------------------
    from transformers import CLIPProcessor
    model = HazCLIP(device=device).to(device)
    processor = CLIPProcessor.from_pretrained(HF_ID)
    params = trainable_parameters(model)
    print(f'[model] trainable={sum(p.numel() for p in params):,}')
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # baseline eval
    base = evaluate(model, processor, eval_set[0], eval_set[1], device)
    print('[base] ext R@1 =',
          base['P1_hazy_query_clean_gallery']['extreme']['R@1'])

    history = []
    for epoch in range(args.epochs):
        model.train()
        agg = dict(loss=0.0, nce=0.0, phys=0.0, inv=0.0)
        nb = 0
        for clean, hazy, t_gt, beta, cls in dl:
            clean, hazy, t_gt = clean.to(device), hazy.to(device), t_gt.to(device)
            beta = float(beta[0])
            px_c = clip_pixel_batch(clean, processor, device)
            px_h = clip_pixel_batch(hazy, processor, device)
            e_c, _ = model.vision_features(px_c)
            e_h, hid = model.vision_features(px_h)
            l_nce = infonce_cross_domain(e_c, e_h)
            l_ph = phys_loss(model.haze_head(hid), t_gt, beta)
            l_inv = invariance_loss(e_c, e_h)
            loss = l_nce + 0.5 * l_ph + 0.5 * l_inv
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            agg['loss'] += loss.item(); agg['nce'] += l_nce.item()
            agg['phys'] += l_ph.item(); agg['inv'] += l_inv.item()
            nb += 1
        sched.step()
        rec = {k: v / max(nb, 1) for k, v in agg.items()}
        rec['epoch'] = epoch
        history.append(rec)
        print(f"[ep {epoch:02d}] {rec}")

    # ---- final evals -------------------------------------------------------
    protos_c = evaluate(model, processor, eval_set[0], eval_set[1], device)
    ckpt = os.path.join(args.out_dir, 'hazclip_trainable.pt')
    torch.save(model.trainable_state_dict(), ckpt)

    results = {'history': history,
               'cifar_protocols': protos_c,
               'meta': {'train_n': len(ds_c_tr) + len(ds_f_tr),
                        'time_sec': time.time() - t0, 'device': device}}
    with open(os.path.join(args.out_dir, 'results_multisource.json'),
              'w') as f:
        json.dump(results, f, indent=1)

    p1 = protos_c['P1_hazy_query_clean_gallery']
    lines = ['# Multi-source HazCLIP — CIFAR+Flickr @224', '',
             protocol_table(p1), '',
             f"extreme R@1: {base['P1_hazy_query_clean_gallery']['extreme']['R@1']:.3f}"
             f" -> {p1['extreme']['R@1']:.3f}",
             f"relative extreme drop: {relative_drop(base['P1_hazy_query_clean_gallery']):.1f}%"
             f" -> {relative_drop(p1):.1f}%"]
    with open(os.path.join(args.out_dir, 'summary_multisource.md'), 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'[done] {time.time() - t0:.0f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default=None)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--out-dir', default='05_Experiments/hazclip_multisource')
    ap.add_argument('--train-betas', type=float, nargs='+',
                    default=[0.6, 1.2, 1.8, 2.6, 3.5, 4.5])
    main(ap.parse_args())
