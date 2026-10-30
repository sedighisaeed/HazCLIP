"""Evaluate HazCLIP variants (flickr8k / hard-beta / multisource) on the
same self-excluded n=600 split and fixed depth realizations used by
eval_harmonized.py, so the paper's variant paragraph shares one protocol.

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python 04_Code/eval_variants_selfexcl.py --device mps
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

from eval_harmonized import (  # noqa: E402
    EXP, SEVERITY_BETAS, SEVERITY_LABELS, _rank_stats, get_split, hazy_views)


class HazClipEnc:
    """HazCLIP-backed encoder exposing the same embed() API."""

    def __init__(self, device, ckpt):
        from transformers import CLIPModel, CLIPProcessor
        from models.hazclip import HazCLIP
        hf = 'openai/clip-vit-base-patch32'
        self.device = device
        self.proc = CLIPProcessor.from_pretrained(hf)
        self.model = CLIPModel.from_pretrained(hf).to(device).eval()
        self.hz = HazCLIP(device=device).to(device)
        self.hz.load_trainable(ckpt)
        self.hz.eval()

    @torch.no_grad()
    def embed(self, pil_list, batch=64):
        from PIL import Image
        if isinstance(pil_list, torch.Tensor):
            pil_list = [Image.fromarray(
                (im.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                for im in pil_list]
        out = []
        for i in range(0, len(pil_list), batch):
            px = self.proc(images=pil_list[i:i + batch],
                           return_tensors='pt').to(self.device)
            e = self.hz.encode_images_eval(px['pixel_values'])
            out.append(F.normalize(e, dim=-1).cpu())
        return torch.cat(out)


def main(args):
    t0 = time.time()
    device = args.device or ('mps' if torch.backends.mps.is_available()
                             else 'cpu')
    out_dir = os.path.join(EXP, 'variants_selfexcl')
    os.makedirs(out_dir, exist_ok=True)

    imgs, labels, tr, ev = get_split(0)
    ev_imgs, ev_labels = imgs[ev], labels[ev]
    print(f'[data] eval split n={len(ev_imgs)}')

    views = {}
    for si, (sev, beta) in enumerate(zip(SEVERITY_LABELS, SEVERITY_BETAS)):
        views[sev] = hazy_views(ev_imgs, si, beta)

    ckpts = {
        'hazclip_v0': os.path.join(EXP, 'hazclip_v0', 'hazclip_trainable.pt'),
        'hazclip_flickr8k': os.path.join(EXP, 'hazclip_v1_flickr8k_full',
                                         'hazclip_trainable.pt'),
        'hazclip_hardbeta': os.path.join(EXP, 'hazclip_v0.1_hardbetas',
                                         'hazclip_trainable.pt'),
        'hazclip_multisource': os.path.join(EXP, 'hazclip_multisource',
                                            'hazclip_trainable.pt'),
    }
    ckpts = {k: v for k, v in ckpts.items() if os.path.exists(v)}
    results = {}
    for name, ckpt in ckpts.items():
        enc = HazClipEnc(device, ckpt)
        g = enc.embed(ev_imgs)
        p1 = {}
        for si, sev in enumerate(SEVERITY_LABELS):
            q = enc.embed(views[sev])
            p1[sev] = _rank_stats(q @ g.T, ev_labels, exclude_diag=True)
            print(f'  [{name}] {sev:9s} R@1={p1[sev]["R@1"]:.3f} '
                  f'mAP={p1[sev]["mAP"]:.3f}')
        results[name] = p1
        del enc
    results['_meta'] = {'n': len(ev_imgs), 'self_exclusion': True,
                        'time_sec': time.time() - t0}
    with open(os.path.join(out_dir, 'variants_selfexcl.json'), 'w') as f:
        json.dump(results, f, indent=1)

    lines = ['# Variants on the harmonized self-excluded split (n=600)', '',
             '| variant | ' + ' | '.join(SEVERITY_LABELS) + ' |',
             '|---|' + '---|' * len(SEVERITY_LABELS)]
    for name in results:
        if name.startswith('_'):
            continue
        row = [f'{results[name][s]["R@1"]:.3f}' for s in SEVERITY_LABELS]
        lines.append(f'| {name} | ' + ' | '.join(row) + ' |')
    with open(os.path.join(out_dir, 'variants_selfexcl.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('[done]', out_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default=None)
    main(ap.parse_args())
