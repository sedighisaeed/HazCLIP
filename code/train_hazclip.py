"""Train HazCLIP with the three-term objective (Method_Notes §2).

    L = InfoNCE_cross_domain(clean_i <-> hazy_i(beta_j))
        + lambda1 * L_phys  (haze-head vs synthetic GT: beta, t-stats)
        + lambda2 * L_inv   (||e(clean) - e(hazy)||^2 pull)

Training data: on-the-fly synthetic haze (haze_synth.HazyPairDataset)
over a clean image bank (CIFAR-10 subset now; Flickr8k/ROxford later).

Usage (smoke test on this machine):
    .venv/bin/python 04_Code/train_hazclip.py --n-per-class 40 --epochs 2
Full v0 run:
    .venv/bin/python 04_Code/train_hazclip.py --n-per-class 200 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eval_retrieval import (SEVERITY_BETAS, SEVERITY_LABELS,
                            evaluate_protocols, protocol_table,
                            relative_drop)
from haze_synth import HazyPairDataset, transmission_stats
from models.hazclip import HazCLIP, trainable_parameters

HF_ID = 'openai/clip-vit-base-patch32'

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# ------------------------------------------------------------------ data

def load_cifar10_subset(n_per_class: int, data_dir: str):
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root=data_dir, train=True, download=True)
    targets = np.array(ds.targets)
    picks = []
    for cls in range(10):
        idx = np.where(targets == cls)[0][:n_per_class]
        picks.extend(idx.tolist())
    picks = np.array(sorted(picks))
    imgs = torch.stack([
        torch.from_numpy(np.asarray(ds[i][0]).copy()).permute(2, 0, 1)
        .float() / 255.0 for i in picks])
    labels = targets[picks]
    return imgs, labels


def load_folder_bank(folder: str, n_images: int, size: int = 64,
                     seed: int = 0):
    """Flat image-folder bank (e.g. Flickr8k): resize to (size,size),
    return (N,3,H,W) in [0,1] + pseudo-labels via filename clustering.

    For the sandbox we assign pseudo-ids by sorting filenames in groups of
    `group_size` — retrieval relevance = same group. Real evaluation uses
    ROxford gnd / MIRFlickr categories later.
    """
    from PIL import Image as PILImage
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    rng = np.random.RandomState(seed)
    rng.shuffle(files)
    files = files[:n_images]
    group_size = max(2, n_images // 100)  # ~100 groups
    imgs, labels = [], []
    for gi, fname in enumerate(files):
        im = PILImage.open(os.path.join(folder, fname)).convert('RGB')
        im = im.resize((size, size), PILImage.BILINEAR)
        imgs.append(torch.from_numpy(np.asarray(im).copy())
                    .permute(2, 0, 1).float() / 255.0)
        labels.append(gi // group_size)
    return torch.stack(imgs), np.array(labels)


def clip_pixel_batch(imgs: torch.Tensor, processor, device: str
                     ) -> torch.Tensor:
    """(B,3,32,32) [0,1] -> CLIP-normalized pixel_values at 224."""
    from PIL import Image
    if imgs.device.type != 'cpu':
        imgs = imgs.cpu()
    pil = [Image.fromarray((im.permute(1, 2, 0).numpy() * 255)
                           .astype(np.uint8)) for im in imgs]
    inputs = processor(images=pil, return_tensors='pt').to(device)
    return inputs['pixel_values']


# ------------------------------------------------------------------ losses

def infonce_cross_domain(e_clean: torch.Tensor, e_hazy: torch.Tensor,
                         temperature: float = 0.07) -> torch.Tensor:
    """Batchwise InfoNCE: positive pair = same index (clean_i, hazy_i).

    Symmetric loss over both directions; in-batch negatives are other indices.
    """
    logits = e_clean @ e_hazy.T / temperature
    targets = torch.arange(len(e_clean), device=e_clean.device)
    return (F.cross_entropy(logits, targets)
            + F.cross_entropy(logits.T, targets)) / 2


def phys_loss(pred: torch.Tensor, t_gt: torch.Tensor, beta: float
              ) -> torch.Tensor:
    """Haze-head supervision vs synthetic GT.

    pred: (B,4) = [beta_hat, t_mean_hat, t_std_hat, t_min_hat]
    GT   : [beta, mean(t), std(t), min(t)]  computed per-image from t(x).
    """
    # NOTE: manual std + MSE instead of smooth_l1/std kernels — torch 1.13
    # MPS crashes on smooth_l1 backward (bool/int64 cast in the MPS graph).
    b = t_gt.shape[0]
    t_flat = t_gt.view(b, -1)
    t_mean = t_flat.mean(dim=1)
    t_std = ((t_flat - t_mean[:, None]) ** 2).mean(dim=1).sqrt()
    gt = torch.stack([
        torch.full((b,), beta, device=t_gt.device, dtype=pred.dtype),
        t_mean, t_std, t_flat.amin(dim=1)], dim=1)
    return F.mse_loss(pred, gt)


def invariance_loss(e_clean: torch.Tensor, e_hazy: torch.Tensor
                    ) -> torch.Tensor:
    return ((e_clean - e_hazy) ** 2).sum(dim=1).mean()


def weather_prompt_alignment(model, processor, e_hazy: torch.Tensor,
                             cls: torch.Tensor, beta: float, device: str,
                             class_names: list[str]) -> torch.Tensor:
    """Pull weather-conditioned class text embeddings toward the mean
    hazy-vision embedding of the corresponding class in the batch.

    beta -> nearest SEVERITY index picks which bank offset to train.
    Only classes present in the batch contribute (label=-1 is skipped).
    """
    from models.hazclip import SEV2IDX
    if (cls < 0).all():
        return torch.zeros((), device=e_hazy.device)
    # nearest severity for this batch's beta
    sev = min(SEVERITY_LABELS, key=lambda s: abs(
        SEVERITY_BETAS[SEVERITY_LABELS.index(s)] - beta))
    sev_idx = SEV2IDX[sev]
    # text embeddings WITH offset (offset is trainable -> gradient flows)
    prompts = [f'a photo of a {c}, {_weather_phrase(sev)}'
               for c in class_names]
    txt = processor(text=prompts, return_tensors='pt', padding=True)
    t_emb = model.clip.get_text_features(
        input_ids=txt['input_ids'].to(device),
        attention_mask=txt['attention_mask'].to(device))
    t_emb = model.text_features(txt['input_ids'].to(device),
                                txt['attention_mask'].to(device),
                                sev_idx=sev_idx)
    # class-mean of hazy vision embeddings
    losses = []
    for c in cls.unique():
        if int(c) < 0 or int(c) >= len(class_names):
            continue
        mask = cls == c
        if mask.sum() < 2:
            continue
        losses.append(-torch.cosine_similarity(
            t_emb[int(c)][None], e_hazy[mask].mean(dim=0)[None]).mean())
    if not losses:
        return torch.zeros((), device=e_hazy.device)
    return torch.stack(losses).mean()


def _weather_phrase(sev: str) -> str:
    return {
        'clean': 'clear weather',
        'light': 'slightly hazy weather',
        'mild': 'hazy weather',
        'moderate': 'foggy weather with moderate visibility',
        'heavy': 'dense fog with poor visibility',
        'extreme': 'extremely dense haze, visibility near zero',
    }[sev]


# ------------------------------------------------------------------ eval

@torch.no_grad()
def evaluate(model: HazCLIP, processor, imgs: torch.Tensor,
             labels: np.ndarray, device: str, batch: int = 256) -> dict:
    model.eval()
    q_embs = {}
    for sev, beta in zip(SEVERITY_LABELS, SEVERITY_BETAS):
        embs = []
        for i in range(0, len(imgs), batch):
            chunk = imgs[i:i + batch]
            if beta > 0:
                from haze_synth import apply_haze
                chunk = torch.stack([apply_haze(im, beta) for im in chunk])
            px = clip_pixel_batch(chunk, processor, device)
            embs.append(model.encode_images_eval(px).cpu())
        q_embs[sev] = torch.cat(embs)
    protos = evaluate_protocols(q_embs, labels)
    return protos


# ------------------------------------------------------------------ main

def main(args):
    t0 = time.time()
    device = args.device or ('mps' if torch.backends.mps.is_available()
                             else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'[setup] device={device} | out={args.out_dir}')

    # data ---------------------------------------------------------------
    print('[data] building training bank ...')
    if args.dataset == 'folder':
        imgs, labels = load_folder_bank(args.data_dir, args.folder_n,
                                        size=args.folder_size)
    else:
        imgs, labels = load_cifar10_subset(args.n_per_class, args.data_dir)
    # split: train 70% / eval 30% per label group (class or pseudo-group)
    rng = np.random.RandomState(0)
    tr_idx, ev_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        k = int(0.7 * len(idx))
        tr_idx.extend(idx[:k])
        ev_idx.extend(idx[k:])
    tr_idx, ev_idx = np.array(tr_idx), np.array(ev_idx)
    train_bank = imgs[tr_idx]
    eval_imgs, eval_labels = imgs[ev_idx], labels[ev_idx]
    print(f'[data] train={len(train_bank)} eval={len(eval_imgs)}')

    ds = HazyPairDataset(train_bank, betas=args.train_betas,
                         depth_mode=args.depth_mode, out_size=224,
                         labels=labels[tr_idx])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0)

    # model --------------------------------------------------------------
    print('[model] loading CLIP + building HazCLIP ...')
    from transformers import CLIPProcessor
    model = HazCLIP(lora_r=args.lora_r, lora_last_n=args.lora_last_n,
                    device=device).to(device)
    processor = CLIPProcessor.from_pretrained(HF_ID)
    params = trainable_parameters(model)
    n_train = sum(p.numel() for p in params)
    print(f'[model] trainable params: {n_train:,}')
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(dl))

    # NOTE: weather-conditioned text prompts are used only at evaluation
    # (zero-shot probing); the training loss is purely image-side.

    # baseline (pre-training) eval ---------------------------------------
    print('[eval] baseline (before training) ...')
    base_protos = evaluate(model, processor, eval_imgs, eval_labels, device)
    base_r1 = base_protos['P1_hazy_query_clean_gallery']['extreme']['R@1']
    print(f'[eval] baseline P1 extreme R@1 = {base_r1:.3f}')

    # train loop ----------------------------------------------------------
    history = []
    for epoch in range(args.epochs):
        model.train()
        ep = {'epoch': epoch, 'loss': 0.0, 'infonce': 0.0,
              'phys': 0.0, 'inv': 0.0}
        nb = 0
        for clean, hazy, t_gt, beta, cls in dl:
            clean = clean.to(device)
            hazy = hazy.to(device)
            t_gt = t_gt.to(device)
            beta = float(beta[0])
            px_clean = clip_pixel_batch(clean, processor, device)
            px_hazy = clip_pixel_batch(hazy, processor, device)
            # forward
            e_clean, _ = model.vision_features(px_clean)
            e_hazy, hidden_hazy = model.vision_features(px_hazy)
            pred = model.haze_head(hidden_hazy)
            # losses
            l_nce = infonce_cross_domain(e_clean, e_hazy)
            l_ph = phys_loss(pred, t_gt, beta)
            l_inv = invariance_loss(e_clean, e_hazy)
            loss = l_nce + args.lambda1 * l_ph + args.lambda2 * l_inv
            # --- weather prompt bank alignment (optional term) -----------
            if args.lambda3 > 0:
                l_w = weather_prompt_alignment(
                    model, processor, e_hazy.detach(), cls, beta, device,
                    class_names=args.class_names)
                loss = loss + args.lambda3 * l_w
                ep['wprompt'] = ep.get('wprompt', 0.0) + float(l_w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            ep['loss'] += loss.item()
            ep['infonce'] += l_nce.item()
            ep['phys'] += l_ph.item()
            ep['inv'] += l_inv.item()
            nb += 1
        for k in ('loss', 'infonce', 'phys', 'inv'):
            ep[k] /= max(nb, 1)
        history.append(ep)
        print(f"[ep {epoch:02d}] loss={ep['loss']:.4f} "
              f"nce={ep['infonce']:.4f} phys={ep['phys']:.4f} "
              f"inv={ep['inv']:.4f} | lr={sched.get_last_lr()[0]:.2e}")

        # mid-training quick eval every `eval_every` epochs
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            protos = evaluate(model, processor, eval_imgs, eval_labels,
                              device)
            r1x = protos['P1_hazy_query_clean_gallery']['extreme']['R@1']
            print(f'[eval] P1 extreme R@1 = {r1x:.3f} '
                  f'(baseline {base_r1:.3f})')

    # final eval + save ----------------------------------------------------
    print('[eval] final ...')
    protos = evaluate(model, processor, eval_imgs, eval_labels, device)
    ckpt = os.path.join(args.out_dir, 'hazclip_trainable.pt')
    torch.save(model.trainable_state_dict(), ckpt)

    results = {
        'meta': vars(args) | {'device': device, 'n_trainable': n_train,
                              'train_n': len(train_bank),
                              'eval_n': len(eval_imgs),
                              'hf_id': HF_ID},
        'baseline_protocols': base_protos,
        'final_protocols': protos,
        'history': history,
        'time_sec': time.time() - t0,
    }
    with open(os.path.join(args.out_dir, 'results_train.json'), 'w') as f:
        json.dump(results, f, indent=1)

    # summary markdown
    b1 = base_protos['P1_hazy_query_clean_gallery']
    f1 = protos['P1_hazy_query_clean_gallery']
    lines = [
        '# HazCLIP training v0 — summary',
        '',
        f'- device={device}, trainable={n_train:,} params, '
        f'time={results["time_sec"]:.0f}s',
        f'- betas(train)={args.train_betas}, l1={args.lambda1}, '
        f'l2={args.lambda2}, lr={args.lr}, epochs={args.epochs}',
        '',
        '## P1 hazy→clean — BEFORE',
        protocol_table(b1),
        '',
        '## P1 hazy→clean — AFTER',
        protocol_table(f1),
        '',
        f"- extreme R@1: {b1['extreme']['R@1']:.3f} → "
        f"{f1['extreme']['R@1']:.3f}",
        f"- extreme mAP: {b1['extreme']['mAP']:.3f} → "
        f"{f1['extreme']['mAP']:.3f}",
        f"- clean R@1  : {b1['clean']['R@1']:.3f} → "
        f"{f1['clean']['R@1']:.3f}  (sanity: should not degrade much)",
        '',
        f"Relative extreme-drop: "
        f"{relative_drop(b1):.1f}% → {relative_drop(f1):.1f}%",
    ]
    with open(os.path.join(args.out_dir, 'summary_train.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(f'[done] {args.out_dir} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-class', type=int, default=40)
    ap.add_argument('--dataset', default='cifar10', choices=['cifar10', 'folder'])
    ap.add_argument('--data-dir', default='03_Data/cifar10')
    ap.add_argument('--folder-n', type=int, default=2000)
    ap.add_argument('--folder-size', type=int, default=64)
    ap.add_argument('--out-dir', default='05_Experiments/hazclip_v0')
    ap.add_argument('--device', default=None)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--lora-r', type=int, default=8)
    ap.add_argument('--lora-last-n', type=int, default=4)
    ap.add_argument('--lambda1', type=float, default=0.5)
    ap.add_argument('--lambda2', type=float, default=0.5)
    ap.add_argument('--lambda3', type=float, default=0.0,
                    help='weather prompt alignment weight (0 = off)')
    ap.add_argument('--class-names', nargs='+', default=CLASS_NAMES,
                    help='class names for prompt alignment (labeled banks)')
    ap.add_argument('--eval-every', type=int, default=2)
    ap.add_argument('--depth-mode', default='random',
                    choices=['random', 'gradient'])
    ap.add_argument('--train-betas', type=float, nargs='+',
                    default=[0.6, 1.2, 1.8, 2.6, 3.5])
    main(ap.parse_args())
