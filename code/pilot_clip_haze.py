"""Pilot experiment: How much does CLIP retrieval degrade under atmospheric haze?

Design
------
1. Clean gallery: 1000 CIFAR-10 test images (100 per class).
2. Queries: the SAME 1000 images, degraded with synthetic haze at 5 severities
   (atmospheric scattering model: I = J*t + A*(1-t), t = exp(-beta*d)).
3. Encoder: OpenCLIP / CLIP image+text embeddings (ViT-B/32 by default).
4. Retrieval metrics: Recall@1/5/10, mAP@all (class-level relevance).
5. Protocols:
   - P1 "hazy-query -> clean-gallery": realistic CBIR scenario.
   - P2 "hazy-query -> hazy-gallery": same-severity gallery (upper bound).
   - P3 zero-shot CLASSIFICATION accuracy drop (secondary evidence).

Outputs (05_Experiments/pilot_clip_haze/)
    results_pilot.json      - all raw numbers
    plot_recall_curve.png   - Recall@1/5/10 vs haze severity
    plot_map_curve.png      - mAP vs haze severity
    summary_report.md       - human-readable summary

Usage
-----
    python 04_Code/pilot_clip_haze.py --model ViT-B-32 --pretrained openai --n 1000
    # add --device mps on Apple Silicon (auto-detected by default)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

# ----------------------------------------------------------------------------- haze model

def apply_haze(img: torch.Tensor, beta: float, airlight: float = 1.0,
               depth: torch.Tensor | None = None) -> torch.Tensor:
    """Atmospheric scattering model on a CxHxW tensor in [0,1].

    I(x) = J(x) * t(x) + A * (1 - t(x)),  t(x) = exp(-beta * d(x))
    If depth is None, a smooth random depth in [0.2, 1.0] is generated so haze
    is non-uniform (more realistic than a flat transmission map).
    """
    c, h, w = img.shape
    if depth is None:
        # smooth random depth field via low-res upsampling (N,1,gr,gc) -> (1,H,W)
        gr, gc = max(2, h // 16), max(2, w // 16)
        rng = torch.Generator().manual_seed(int(beta * 10_000) + h * w)
        low = 0.2 + 0.8 * torch.rand(1, 1, gr, gc, generator=rng)
        depth = torch.nn.functional.interpolate(
            low, size=(h, w), mode='bilinear', align_corners=False)[0]
    t = torch.exp(-beta * depth)
    return img * t + airlight * (1.0 - t)


# Severity ladder: beta chosen to span light fog -> dense haze
SEVERITY_BETAS = [0.0, 0.6, 1.2, 1.8, 2.6, 3.5]
SEVERITY_LABELS = ['clean', 'light', 'mild', 'moderate', 'heavy', 'extreme']


# ----------------------------------------------------------------------------- data

def load_cifar10_subset(n_per_class: int = 100, data_dir: str = '03_Data/cifar10'
                        ) -> tuple[torch.Tensor, np.ndarray]:
    """Download CIFAR-10 test split; return (N,3,32,32) float tensor in [0,1] + labels."""
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root=data_dir, train=False, download=True)
    targets = np.array(ds.targets)
    picks = []
    for cls in range(10):
        idx = np.where(targets == cls)[0][:n_per_class]
        picks.extend(idx.tolist())
    picks = np.array(sorted(picks))
    imgs = torch.stack([
        torch.from_numpy(np.asarray(ds[i][0])).permute(2, 0, 1).float() / 255.0
        for i in picks
    ])
    labels = targets[picks]
    return imgs, labels


# ----------------------------------------------------------------------------- metrics

def recall_at_k(sim: torch.Tensor, labels: np.ndarray, ks=(1, 5, 10)) -> dict:
    """sim: (Nq, Ng) higher=better. Query i correct if top-k contains same class."""
    order = sim.argsort(dim=1, descending=True)
    out = {}
    for k in ks:
        topk = order[:, :k]
        hits = (labels[topk.cpu().numpy()] == labels[:, None]).any(axis=1)
        out[f'R@{k}'] = float(hits.mean())
    return out


def map_all(sim: torch.Tensor, labels: np.ndarray) -> float:
    """Class-level mAP: relevant = same class. Standard AP over ranked list."""
    order = sim.argsort(dim=1, descending=True).cpu().numpy()
    aps = []
    for qi in range(sim.shape[0]):
        rel = (labels[order[qi]] == labels[qi]).astype(float)
        rel_sum = rel.cumsum()
        prec = rel_sum / (np.arange(len(rel)) + 1)
        denom = rel_sum[-1]  # total relevant
        aps.append(float((prec * rel).sum() / denom))
    return float(np.mean(aps))


# ----------------------------------------------------------------------------- encoder

class ClipEncoder:
    def __init__(self, model: str, pretrained: str, device: str):
        import torch.nn.functional as F
        from transformers import CLIPModel, CLIPProcessor
        self.F = F
        hf_id = 'openai/clip-vit-base-patch32'  # v0 uses openai weights only
        self.device = device
        self.model = CLIPModel.from_pretrained(hf_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(hf_id)

    @torch.no_grad()
    def image_embeddings(self, imgs: torch.Tensor, batch: int = 128) -> torch.Tensor:
        embs = []
        pil = [Image.fromarray((im.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
               for im in imgs]
        for i in range(0, len(pil), batch):
            inputs = self.processor(images=pil[i:i + batch], return_tensors='pt'
                                    ).to(self.device)
            e = self.model.get_image_features(**inputs)
            embs.append(self.F.normalize(e, dim=-1).cpu())
        return torch.cat(embs)


# ----------------------------------------------------------------------------- main

def run(args):
    t0 = time.time()
    device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'[setup] device={device}')

    print('[data] loading CIFAR-10 subset ...')
    imgs, labels = load_cifar10_subset(args.n_per_class, args.data_dir)
    n = len(imgs)
    print(f'[data] {n} images, classes={np.unique(labels)}')

    enc = ClipEncoder(args.model, args.pretrained, device)

    results = {'meta': vars(args) | {'device': device, 'n_images': n,
                                     'model': 'openai/clip-vit-base-patch32'},
               'protocols': {}}

    # --- encode hazy queries per severity (reused across protocols) ---
    print('[encode] hazy queries per severity ...')
    q_embs, viz_samples = {}, []
    for sev_label, beta in zip(SEVERITY_LABELS, SEVERITY_BETAS):
        if beta == 0.0:
            hazy = imgs
        else:
            hazy = torch.stack([apply_haze(im, beta) for im in imgs])
            if sev_label in ('clean', 'light', 'moderate', 'extreme') and len(viz_samples) < 4:
                viz_samples.append((sev_label, hazy[0].clone()))
        q_embs[sev_label] = enc.image_embeddings(hazy)
        print(f'    {sev_label:9s} beta={beta:.1f} done')

    print('[encode] clean galleries ...')
    g_clean = q_embs['clean']  # same images, clean embeddings

    # P1: hazy query -> clean gallery
    p1 = {}
    for sev_label in SEVERITY_LABELS:
        sim = q_embs[sev_label] @ g_clean.T
        m = recall_at_k(sim, labels)
        m['mAP'] = map_all(sim, labels)
        p1[sev_label] = m
        print(f'[P1 hazy->clean] {sev_label:9s} R@1={m["R@1"]:.3f} mAP={m["mAP"]:.3f}')
    results['protocols']['P1_hazy_query_clean_gallery'] = p1

    # P2: hazy query -> same-severity gallery
    p2 = {}
    for sev_label in SEVERITY_LABELS:
        sim = q_embs[sev_label] @ q_embs[sev_label].T
        m = recall_at_k(sim, labels)
        m['mAP'] = map_all(sim, labels)
        p2[sev_label] = m
        print(f'[P2 hazy->hazy ] {sev_label:9s} R@1={m["R@1"]:.3f} mAP={m["mAP"]:.3f}')
    results['protocols']['P2_hazy_query_same_severity_gallery'] = p2

    # P3: zero-shot classification drop (secondary)
    prompts = [f'a photo of a {c}' for c in
               ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']]
    with torch.no_grad():
        txt = enc.processor(text=prompts, return_tensors='pt', padding=True).to(device)
        t_emb = enc.F.normalize(enc.model.get_text_features(**txt), dim=-1).cpu()
    p3 = {}
    for sev_label in SEVERITY_LABELS:
        pred = (q_embs[sev_label] @ t_emb.T).argmax(dim=1).numpy()
        p3[sev_label] = float((pred == labels).mean())
        print(f'[P3 zeroshot   ] {sev_label:9s} acc={p3[sev_label]:.3f}')
    results['protocols']['P3_zeroshot_classification'] = p3

    # --- save ---
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'results_pilot.json'), 'w') as f:
        json.dump(results, f, indent=1)

    # plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs = range(len(SEVERITY_LABELS))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in (1, 5, 10):
        ys = [p1[s][f'R@{k}'] for s in SEVERITY_LABELS]
        ax.plot(xs, ys, marker='o', label=f'Recall@{k}')
    ax.set_xticks(xs, SEVERITY_LABELS)
    ax.set_xlabel('Haze severity (atmospheric scattering, beta)')
    ax.set_ylabel('Recall')
    ax.set_title('CLIP retrieval degradation under haze\n(hazy query vs clean gallery, CIFAR-10 subset)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'plot_recall_curve.png'), dpi=150)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(xs, [p1[s]['mAP'] for s in SEVERITY_LABELS], marker='s',
             color='tab:red', label='P1 hazy->clean')
    ax2.plot(xs, [p2[s]['mAP'] for s in SEVERITY_LABELS], marker='^',
             color='tab:blue', label='P2 hazy->hazy')
    ax2.set_xticks(xs, SEVERITY_LABELS)
    ax2.set_xlabel('Haze severity')
    ax2.set_ylabel('mAP')
    ax2.set_title('mAP under haze — both protocols')
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'plot_map_curve.png'), dpi=150)

    # markdown summary
    def row(sev):
        p, q = p1[sev], p2[sev]
        return (f"| {sev} | {p['R@1']:.3f} | {p['R@5']:.3f} | {p['R@10']:.3f} "
                f"| {p['mAP']:.3f} | {q['R@1']:.3f} | {q['mAP']:.3f} | {p3[sev]:.3f} |")
    drop_r1 = 100 * (p1['clean']['R@1'] - p1['extreme']['R@1']) / max(p1['clean']['R@1'], 1e-9)
    lines = [
        '# Pilot: CLIP retrieval under haze — results',
        '',
        f'- Model: openai/clip-vit-base-patch32 | Images: {n} (CIFAR-10, balanced) '
        f'| Device: {device} | Time: {time.time()-t0:.0f}s',
        f'- Haze: atmospheric scattering model, betas={SEVERITY_BETAS}',
        '',
        '| severity | R@1 | R@5 | R@10 | mAP (P1) | R@1 (P2) | mAP (P2) | zeroshot acc |',
        '|---|---|---|---|---|---|---|---|',
        *[row(s) for s in SEVERITY_LABELS],
        '',
        f'**Headline: Recall@1 drops {drop_r1:.1f}% (relative) from clean to extreme haze (P1).**',
        '',
        'Interpretation for the gap analysis:',
        '1. If R@1/mAP fall steeply with severity -> strong empirical evidence for gap G1/G2.',
        '2. P2 > P1 consistently would show gallery-side degradation matters (design insight).',
        '3. Zero-shot accuracy drop triangulates that semantic content survives but alignment degrades.',
    ]
    with open(os.path.join(out_dir, 'summary_report.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(f'[done] outputs in {out_dir} ({time.time()-t0:.0f}s total)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-class', type=int, default=100)
    ap.add_argument('--data-dir', default='03_Data/cifar10')
    ap.add_argument('--out-dir', default='05_Experiments/pilot_clip_haze')
    ap.add_argument('--model', default='ViT-B-32')
    ap.add_argument('--pretrained', default='openai')
    ap.add_argument('--device', default=None, help='cpu | mps | cuda')
    run(ap.parse_args())
