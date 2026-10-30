"""HARMONIZED re-evaluation for the camera-grade paper tables.

Fixes the reviewer-flagged inconsistencies by evaluating EVERY system on
ONE identical split (the 70/30 CIFAR split, eval n=600), ONE fixed
per-image deterministic depth realization per severity, and — critically —
with SELF-EXCLUSION: the query's own base image is removed from the
gallery column before ranking, so R@1 at beta=0 is no longer a trivial
1.000 and P2 ("hazy -> same-severity") is not bitwise identity matching.

Systems evaluated here:
  raw CLIP            (frozen, openai ViT-B/32)
  HazCLIP v0          (ckpt: 05_Experiments/hazclip_v0)
  DCP -> CLIP         (classical two-stage baseline, DCP at 224 px)
  InfoNCE-only        (ckpt: 05_Experiments/ablation_infonce_only)
  w/o L_phys          (ckpt: 05_Experiments/ablation_no_phys)
  w/o L_inv           (ckpt: 05_Experiments/ablation_no_inv)
  SigLIP B/16, DINOv2-small (frozen FM baselines)

Also runs the zero-shot weather-prompt probe (with/without the learned
weather-prompt-bank offsets) on CLIP and HazCLIP embeddings -> the
missing Section IV-F prompt-bank evidence.

Outputs:
  05_Experiments/harmonized_eval/harmonized.json   (all numbers)
  05_Experiments/harmonized_eval/harmonized.md     (paper-ready tables)

Usage:
  PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python 04_Code/eval_harmonized.py --device mps
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

from eval_retrieval import SEVERITY_BETAS, SEVERITY_LABELS
from haze_synth import apply_haze

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, '05_Experiments')


# ------------------------------------------------------------------ metrics
def _rank_stats(sim: torch.Tensor, labels: np.ndarray,
                exclude_diag: bool, ks=(1, 5, 10)) -> dict:
    """Class-level metrics with optional self-exclusion of the diagonal."""
    sim = sim.clone()
    if exclude_diag:
        n = min(sim.shape[0], sim.shape[1])
        idx = torch.arange(n)
        sim[idx, idx] = -float('inf')
    order = sim.argsort(dim=1, descending=True)
    out = {}
    for k in ks:
        topk = order[:, :k]
        hits = (labels[topk.cpu().numpy()] == labels[:, None]).any(axis=1)
        out[f'R@{k}'] = float(hits.mean())
    order_np = order.cpu().numpy()
    aps = []
    for qi in range(sim.shape[0]):
        rel = (labels[order_np[qi]] == labels[qi]).astype(float)
        rel_sum = rel.cumsum()
        prec = rel_sum / (np.arange(len(rel)) + 1)
        aps.append(float((prec * rel).sum() / max(rel_sum[-1], 1e-9)))
    out['mAP'] = float(np.mean(aps))
    return out


# ------------------------------------------------------------------ data
def get_split(seed: int = 0):
    """Identical 70/30 CIFAR split as train_hazclip/run_seeds (n_eval=600)."""
    from train_hazclip import load_cifar10_subset
    imgs, labels = load_cifar10_subset(200, '03_Data/cifar10')
    rng = np.random.RandomState(seed)
    tr, ev = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        k = int(0.7 * len(idx))
        tr.extend(idx[:k])
        ev.extend(idx[k:])
    return imgs, labels, np.array(tr), np.array(ev)


def depth_seed(img_idx: int, sev_idx: int) -> int:
    """Deterministic per-image, per-severity depth realization."""
    return (int(img_idx) * 100_003 + sev_idx * 7_919) % (2 ** 31 - 1)


def hazy_views(imgs: torch.Tensor, sev_idx: int, beta: float
               ) -> torch.Tensor:
    if beta <= 0:
        return imgs
    return torch.stack([
        apply_haze(im, beta, seed=depth_seed(i, sev_idx))
        for i, im in enumerate(imgs)])


# ------------------------------------------------------------------ encoders
class ClipEnc:
    kind = 'clip'

    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        hf = 'openai/clip-vit-base-patch32'
        self.proc = CLIPProcessor.from_pretrained(hf)
        self.model = CLIPModel.from_pretrained(hf).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, imgs_32: torch.Tensor, batch=128) -> torch.Tensor:
        from PIL import Image
        out = []
        for i in range(0, len(imgs_32), batch):
            chunk = imgs_32[i:i + batch].cpu()
            pil = [Image.fromarray((im.permute(1, 2, 0).numpy() * 255)
                                   .astype(np.uint8)) for im in chunk]
            px = self.proc(images=pil, return_tensors='pt').to(self.device)
            e = self.model.get_image_features(**px)
            out.append(F.normalize(e, dim=-1).cpu())
        return torch.cat(out)


class HazClipEnc(ClipEnc):
    kind = 'hazclip'

    def __init__(self, device, ckpt):
        super().__init__(device)
        from models.hazclip import HazCLIP
        self.hz = HazCLIP(device=device).to(device)
        self.hz.load_trainable(ckpt)
        self.hz.eval()

    @torch.no_grad()
    def embed(self, imgs_32: torch.Tensor, batch=128) -> torch.Tensor:
        from PIL import Image
        out = []
        for i in range(0, len(imgs_32), batch):
            chunk = imgs_32[i:i + batch].cpu()
            pil = [Image.fromarray((im.permute(1, 2, 0).numpy() * 255)
                                   .astype(np.uint8)) for im in chunk]
            px = self.proc(images=pil, return_tensors='pt').to(self.device)
            e = self.hz.encode_images_eval(px['pixel_values'])
            out.append(F.normalize(e, dim=-1).cpu())
        return torch.cat(out)


class FMClsEnc:
    """Frozen SigLIP / DINOv2 baseline encoder (same protocol)."""

    def __init__(self, name: str, device):
        from eval_baselines_fm import FMEncoder
        self.enc = FMEncoder(name, device)

    @torch.no_grad()
    def embed(self, imgs_32: torch.Tensor, batch=96) -> torch.Tensor:
        return self.enc.embed(imgs_32, batch=batch)


# ------------------------------------------------------------------ DCP
def dcp_dehaze_t(img: torch.Tensor) -> torch.Tensor:
    from baseline_dehaze_clip import dcp_dehaze
    return dcp_dehaze(img)


# ------------------------------------------------------------------ probe
@torch.no_grad()
def zero_shot_probe(embs: dict, labels: np.ndarray, enc: ClipEnc,
                    hz_model=None, device='cpu', bank: bool = False):
    """CIFAR-10 zero-shot accuracy per severity; optionally apply the
    learned weather-prompt-bank offsets (hz_model.weather_bank)."""
    from models.hazclip import SEV2IDX
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    from models.hazclip import WEATHER_TOKENS
    accs = {}
    for sev, q in embs.items():
        tmpl = WEATHER_TOKENS.get(sev, WEATHER_TOKENS['clear'] if sev == 'clean' else None)
        if tmpl is None:
            continue
        prompts = [tmpl.format(f'a {c}') for c in classes]
        txt = enc.proc(text=prompts, return_tensors='pt', padding=True)
        t = enc.model.get_text_features(
            input_ids=txt['input_ids'].to(device),
            attention_mask=txt['attention_mask'].to(device))
        if bank and hz_model is not None:
            idx = SEV2IDX.get(sev, SEV2IDX['clear'] if sev == 'clean' else None)
            if idx is not None:
                t = t + hz_model.weather_bank(torch.tensor(idx, device=device))
        t = F.normalize(t, dim=-1).cpu()
        pred = (q @ t.T).argmax(dim=1).numpy()
        accs[sev] = float((pred == labels).mean())
    return accs


# ------------------------------------------------------------------ main
def main(args):
    t0 = time.time()
    device = args.device or ('mps' if torch.backends.mps.is_available()
                             else 'cpu')
    out_dir = os.path.join(EXP, 'harmonized_eval')
    os.makedirs(out_dir, exist_ok=True)
    print(f'[setup] device={device}')

    imgs, labels, tr, ev = get_split(0)
    ev_imgs, ev_labels = imgs[ev], labels[ev]
    n = len(ev_imgs)
    print(f'[data] eval split n={n} (identical for all systems)')

    # precompute hazy views once (shared by every system)
    views = {}
    for si, (sev, beta) in enumerate(zip(SEVERITY_LABELS, SEVERITY_BETAS)):
        views[sev] = hazy_views(ev_imgs, si, beta)
        print(f'[data] views ready: {sev}')

    def build_systems():
        systems = {}
        systems['raw_clip'] = ClipEnc(device)
        systems['hazclip_v0'] = HazClipEnc(
            device, os.path.join(EXP, 'hazclip_v0', 'hazclip_trainable.pt'))
        systems['dcp_clip'] = systems['raw_clip']          # same tower
        systems['ablation_infonce_only'] = HazClipEnc(
            device, os.path.join(EXP, 'ablation_infonce_only',
                                 'hazclip_trainable.pt'))
        systems['ablation_no_phys'] = HazClipEnc(
            device, os.path.join(EXP, 'ablation_no_phys',
                                 'hazclip_trainable.pt'))
        systems['ablation_no_inv'] = HazClipEnc(
            device, os.path.join(EXP, 'ablation_no_inv',
                                 'hazclip_trainable.pt'))
        if not args.clip_only:
            systems['siglip'] = FMClsEnc('siglip', device)
            systems['dinov2'] = FMClsEnc('dinov2', device)
        return systems

    results = {'meta': {
        'device': device, 'n_eval': n, 'split_seed': 0,
        'self_exclusion': True,
        'depth': 'deterministic per-image smooth random field '
                 '(smooth_random_depth, grid=max(2,size//16), [0.2,1.0])',
        'betas': SEVERITY_BETAS,
        'dcp': 'He et al. 2011 DCP at 224px, omega=0.95',
        'time_sec': None,
    }, 'systems': {}, 'probe': {}}

    systems = build_systems()
    probe_pair = None
    for name, enc in list(systems.items()):
        print(f'[enc] {name}: clean gallery ...')
        g = enc.embed(ev_imgs)
        p1, p2, embs = {}, {}, {}
        for si, sev in enumerate(SEVERITY_LABELS):
            q_in = views[sev]
            if name == 'dcp_clip':
                q_in = torch.stack([
                    dcp_dehaze_t(im) for im in q_in]) if si > 0 else q_in
            q = enc.embed(q_in)
            embs[sev] = q
            p1[sev] = _rank_stats(q @ g.T, ev_labels, exclude_diag=True)
            p2[sev] = _rank_stats(q @ q.T, ev_labels, exclude_diag=True)
            print(f'  [{name}] {sev:9s} P1 R@1={p1[sev]["R@1"]:.3f} '
                  f'mAP={p1[sev]["mAP"]:.3f} | P2 R@1={p2[sev]["R@1"]:.3f}')
        results['systems'][name] = {'P1_selfexcl': p1, 'P2_selfexcl': p2}
        if name == 'raw_clip':
            results['probe']['raw_clip_nobank'] = zero_shot_probe(
                embs, ev_labels, enc, None, device, bank=False)
            probe_pair = (embs, enc, None)
        if name == 'hazclip_v0':
            results['probe']['hazclip_nobank'] = zero_shot_probe(
                embs, ev_labels, enc, enc.hz, device, bank=False)
            results['probe']['hazclip_bank'] = zero_shot_probe(
                embs, ev_labels, enc, enc.hz, device, bank=True)
            probe_pair = (embs, enc, enc.hz)
        del systems[name]
        if device == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    with open(os.path.join(out_dir, 'harmonized.json'), 'w') as f:
        results['meta']['time_sec'] = time.time() - t0
        json.dump(results, f, indent=1)

    # ------------------------------------------------- markdown tables
    def fmt_row(p, sev):
        return (f"| {sev} | {p[sev]['R@1']:.3f} | {p[sev]['mAP']:.3f} "
                f"| {p[sev]['R@5']:.3f} | {p[sev]['R@10']:.3f} |")

    lines = ['# Harmonized self-excluded evaluation (n=600, identical '
             'split, fixed depth realizations)', '']
    lines += ['## P1 hazy -> clean (self-excluded): R@1 / mAP', '']
    head = '| system | ' + ' | '.join(f'{s} R@1' for s in SEVERITY_LABELS) \
        + ' | ' + ' | '.join(f'{s} mAP' for s in SEVERITY_LABELS) + ' |'
    lines += [head, '|' + '---|' * (2 * len(SEVERITY_LABELS) + 1)]
    for name in results['systems']:
        p1 = results['systems'][name]['P1_selfexcl']
        row = f"| {name} | " + ' | '.join(
            f"{p1[s]['R@1']:.3f}" for s in SEVERITY_LABELS) + ' | ' \
            + ' | '.join(f"{p1[s]['mAP']:.3f}" for s in SEVERITY_LABELS) \
            + ' |'
        lines.append(row)
    lines += ['', '## P2 hazy -> same-severity (self-excluded): R@1 / mAP',
              '']
    lines += [head, '|' + '---|' * (2 * len(SEVERITY_LABELS) + 1)]
    for name in results['systems']:
        p2 = results['systems'][name]['P2_selfexcl']
        row = f"| {name} | " + ' | '.join(
            f"{p2[s]['R@1']:.3f}" for s in SEVERITY_LABELS) + ' | ' \
            + ' | '.join(f"{p2[s]['mAP']:.3f}" for s in SEVERITY_LABELS) \
            + ' |'
        lines.append(row)
    lines += ['', '## Zero-shot weather-prompt probe (CIFAR-10)', '',
              '| variant | ' + ' | '.join(SEVERITY_LABELS) + ' |',
              '|' + '---|' * 7]
    for k, acc in results['probe'].items():
        lines.append(f"| {k} | " + ' | '.join(
            f"{acc[s]:.3f}" for s in SEVERITY_LABELS) + ' |')
    with open(os.path.join(out_dir, 'harmonized.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(f'[done] {out_dir} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default=None)
    ap.add_argument('--clip-only', action='store_true',
                    help='skip SigLIP/DINOv2 (faster CLIP-only rerun)')
    main(ap.parse_args())
