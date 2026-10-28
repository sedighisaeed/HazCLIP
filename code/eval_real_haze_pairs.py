"""REAL-HAZE pairwise retrieval benchmark (replaces ACDC while pending).

Data: 03_Data/real_haze — 1,623 real-haze / clean-GT pairs (max1024).
Protocol (paper §4.5, upgraded):
  - Query  = real haze image  i
  - Gallery= all 1,623 clean GT images (incl. its own pair)
  - Relevant = the GT pair of the SAME scene (exactly one per query)
  - Metrics: R@1 / R@10 / MRR / mean rank of the true pair
Systems: raw CLIP vs HazCLIP ckpt (optional).

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python 04_Code/eval_real_haze_pairs.py \
      [--ckpt 05_Experiments/hazclip_v0/hazclip_trainable.pt] [--n 500] [--device mps]
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

ROOT = '03_Data/real_haze'
GT_DIR = os.path.join(ROOT, 'gt', 'GT_max1024')
HZ_DIR = os.path.join(ROOT, 'hazy', 'HAZE_max1024')


def pair_lists():
    g = sorted(os.listdir(GT_DIR))
    h = sorted(os.listdir(HZ_DIR))
    # match by basename
    gmap = {os.path.splitext(f)[0]: f for f in g}
    pairs = []
    for f in h:
        base = os.path.splitext(f)[0]
        if base in gmap:
            pairs.append((f, gmap[base]))
    return pairs


@torch.no_grad()
def encode(model_enc, pil_list, batch=64):
    out = []
    for i in range(0, len(pil_list), batch):
        px = model_enc.proc(images=pil_list[i:i + batch],
                            return_tensors='pt').to(model_enc.device)
        if getattr(model_enc, 'hz', None) is not None:
            e = model_enc.hz.encode_images_eval(px['pixel_values'])
        else:
            e = model_enc.model.get_image_features(**px)
        out.append(F.normalize(e, dim=-1).cpu())
    return torch.cat(out)


def main(args):
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    t0 = time.time()
    device = args.device or ('mps' if torch.backends.mps.is_available()
                             else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'[setup] device={device}')

    pairs = pair_lists()
    if args.n > 0:
        step = max(1, len(pairs) // args.n)
        pairs = pairs[::step][:args.n]
    print(f'[data] {len(pairs)} real-haze pairs')

    hz = None
    if args.ckpt and os.path.exists(args.ckpt):
        from models.hazclip import HazCLIP
        hz = HazCLIP(device=device).to(device)
        hz.load_trainable(args.ckpt)
        hz.eval()
        print('[enc] HazCLIP loaded:', args.ckpt)

    proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip = CLIPModel.from_pretrained(
        'openai/clip-vit-base-patch32').to(device).eval()

    class E:  # simple namespace encoder
        pass
    enc = E()
    enc.proc, enc.model, enc.hz, enc.device = proc, clip, hz, device

    names_h = [p[0] for p in pairs]
    names_g = [p[1] for p in pairs]
    gt_index = {os.path.splitext(f)[0]: i for i, f in enumerate(names_g)}

    print('[enc] gallery (clean GT) ...')
    g_imgs = [Image.open(os.path.join(GT_DIR, f)).convert('RGB')
              for f in names_g]
    G = encode(enc, g_imgs)

    print('[enc] queries (real haze) ...')
    q_imgs = [Image.open(os.path.join(HZ_DIR, f)).convert('RGB')
              for f in names_h]
    Q = encode(enc, q_imgs)

    sim = Q @ G.T
    order = sim.argsort(dim=1, descending=True).numpy()

    r1, r10, mrr, ranks = [], [], [], []
    for qi, (hname, gname) in enumerate(pairs):
        base = os.path.splitext(gname)[0]
        true_rank = int(np.where(order[qi] == gt_index[base])[0][0]) + 1
        ranks.append(true_rank)
        r1.append(true_rank == 1)
        r10.append(true_rank <= 10)
        mrr.append(1.0 / true_rank)
    res = {'n_pairs': len(pairs),
           'R@1': float(np.mean(r1)), 'R@10': float(np.mean(r10)),
           'MRR': float(np.mean(mrr)),
           'median_rank': float(np.median(ranks)),
           'ckpt': args.ckpt, 'time_sec': time.time() - t0,
           'per_query': {'ranks': [int(x) for x in ranks],
                         'recip_rank': [float(x) for x in mrr]}}
    print(json.dumps(res, indent=1))

    os.makedirs(args.out_dir, exist_ok=True)
    tag = 'hazclip' if hz is not None else 'raw_clip'
    with open(os.path.join(args.out_dir, f'real_haze_pairs_{tag}.json'),
              'w') as f:
        json.dump(res, f, indent=1)
    print('[done]', args.out_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--n', type=int, default=500,
                    help='0 = all pairs')
    ap.add_argument('--out-dir', default='05_Experiments/real_haze_pairs')
    ap.add_argument('--device', default=None)
    main(ap.parse_args())
