"""Real-fog generalization test: ImageNet-C fog (severity 5) queries ->
zero-shot ImageNet class probing (1000-way), raw CLIP vs HazCLIP.

Data is streamed via the HF datasets-server /rows API (JSON+base64), so no
pyarrow is required. The local parquet (03_Data/imagenet_c_fog_s5.parquet)
is kept as an offline backup for a future pyarrow install.

Deliverables (paper §4.5):
- zero-shot accuracy / top-5 hit under REAL fog, before vs after adaptation
Usage:
    .venv/bin/python 04_Code/eval_real_fog.py --n 1000 --device mps \
        --ckpt 05_Experiments/hazclip_v0/hazclip_trainable.pt
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASET = 'Tsomaros/ImageNet-C-fog-severity_5'
API = 'https://datasets-server.huggingface.co'


def load_fog_rows(n: int):
    """Stream fog-s5 rows -> (list[PIL.Image], labels np.int64)."""
    from PIL import Image
    q = urllib.parse.quote(DATASET, safe='')
    feat = json.loads(urllib.request.urlopen(
        f'{API}/first-rows?dataset={q}&config=default&split=validation',
        timeout=30).read())
    names = feat['features'][1]['type']['names']
    wnid2idx = {w: i for i, w in enumerate(names)}

    imgs, labels = [], []
    offset = 0
    while len(imgs) < n:
        url = (f'{API}/rows?dataset={q}&config=default&split=validation'
               f'&offset={offset}&length=100')
        page = None
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'hazclip-research/0.1'})
                page = json.loads(urllib.request.urlopen(
                    req, timeout=90).read())
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f'    rows fetch failed ({e}); retry in {wait}s')
                time.sleep(wait)
        if page is None:
            print('    giving up on this page (server busy)')
            break
        rows = page.get('rows', [])
        if not rows:
            break
        for r in rows:
            rec = r['row']['image']
            im = None
            if isinstance(rec, dict):
                src = rec.get('src', '')
                if src.startswith('data:image'):
                    im = Image.open(io.BytesIO(
                        base64.b64decode(src.split(',', 1)[1])))
                elif src:
                    im = Image.open(urllib.request.urlopen(src, timeout=60))
            if im is None:
                continue
            imgs.append(im.convert('RGB'))
            lab = r['row']['label']
            labels.append(wnid2idx[lab] if isinstance(lab, str)
                          else int(lab))
            if len(imgs) >= n:
                break
        offset += len(rows)
        print(f'    fetched {len(imgs)}/{n}')
    return imgs, np.array(labels)


class Encoder:
    def __init__(self, device, ckpt: str | None):
        from transformers import CLIPModel, CLIPProcessor
        from models.hazclip import HazCLIP
        self.device = device
        self.proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.model = CLIPModel.from_pretrained(
            'openai/clip-vit-base-patch32').to(device).eval()
        self.hz = None
        if ckpt and os.path.exists(ckpt):
            self.hz = HazCLIP(device=device).to(device)
            self.hz.load_trainable(ckpt)
            self.hz.eval()
            print('[enc] HazCLIP loaded from', ckpt)

    @torch.no_grad()
    def img_emb(self, pil_list, batch=128, use_hz=True):
        out = []
        for i in range(0, len(pil_list), batch):
            px = self.proc(images=pil_list[i:i + batch],
                           return_tensors='pt').to(self.device)
            if use_hz and self.hz is not None:
                e = self.hz.encode_images_eval(px['pixel_values'])
            else:
                e = self.model.get_image_features(**px)
            out.append(F.normalize(e, dim=-1).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def txt_emb(self, prompts):
        txt = self.proc(text=prompts, return_tensors='pt', padding=True)
        e = self.model.get_text_features(
            input_ids=txt['input_ids'].to(self.device),
            attention_mask=txt['attention_mask'].to(self.device))
        return F.normalize(e, dim=-1).cpu()


def main(args):
    t0 = time.time()
    device = args.device or ('mps' if torch.backends.mps.is_available()
                             else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'[setup] device={device}')

    imgs, labels = load_fog_rows(args.n)
    print(f'[data] {len(imgs)} real-fog queries, '
          f'{len(np.unique(labels))} classes')

    enc = Encoder(device, args.ckpt)
    try:
        url = ('https://raw.githubusercontent.com/pytorch/hub/master/'
               'imagenet_classes.txt')
        names = urllib.request.urlopen(url, timeout=15).read().decode() \
            .splitlines()
        prompts = [f'a photo of a {nm}.' for nm in names]
        print('[data] readable ImageNet class names loaded')
    except Exception as e:
        print('[warn] class-name download failed; numeric prompts:', e)
        prompts = [f'a photo of class {i}.' for i in range(1000)]
    t_emb = enc.txt_emb(prompts)

    res = {}
    for tag, use_hz in (('raw_clip', False), ('hazclip', True)):
        if use_hz and enc.hz is None:
            print('[skip] hazclip (no ckpt)')
            continue
        q = enc.img_emb(imgs, use_hz=use_hz)
        logits = q @ t_emb.T
        pred = logits.argmax(dim=1).numpy()
        acc = float((pred == labels).mean())
        top5 = logits.topk(5, dim=1).indices.numpy()
        hit5 = float(np.mean([labels[i] in top5[i]
                              for i in range(len(labels))]))
        res[tag] = {'zeroshot_acc': acc, 'top5_hit': hit5}
        print(f'[{tag}] REAL-FOG acc={acc:.3f} top5={hit5:.3f}')

    res['meta'] = {'n': len(imgs), 'dataset': DATASET, 'ckpt': args.ckpt,
                   'time_sec': time.time() - t0}
    with open(os.path.join(args.out_dir, 'real_fog_results.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print('[done]', args.out_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--ckpt', default='05_Experiments/hazclip_v0/hazclip_trainable.pt')
    ap.add_argument('--out-dir', default='05_Experiments/real_fog')
    ap.add_argument('--device', default=None)
    main(ap.parse_args())
