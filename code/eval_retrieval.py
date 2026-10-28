"""Shared retrieval evaluation utilities for the HazCLIP project.

Functions
---------
recall_at_k(sim, labels, ks)      - class-level Recall@{1,5,10,...}
map_all(sim, labels)              - class-level mAP@all
r_precision(sim, labels)          - class-level R-precision
evaluate_protocols(embs, labels)  - P1 / P2 protocol table used everywhere

These mirror the pilot (04_Code/pilot_clip_haze.py) so numbers stay
comparable across all experiments (05_Experiments/*).
"""

from __future__ import annotations

import numpy as np
import torch

# Severity ladder shared by pilot + training + eval
SEVERITY_BETAS = [0.0, 0.6, 1.2, 1.8, 2.6, 3.5]
SEVERITY_LABELS = ['clean', 'light', 'mild', 'moderate', 'heavy', 'extreme']


# ---------------------------------------------------------------- metrics

def recall_at_k(sim: torch.Tensor, labels: np.ndarray, ks=(1, 5, 10)) -> dict:
    """sim: (Nq, Ng), higher=better. Query i correct if top-k contains same class."""
    order = sim.argsort(dim=1, descending=True)
    out = {}
    for k in ks:
        topk = order[:, :k]
        hits = (labels[topk.cpu().numpy()] == labels[:, None]).any(axis=1)
        out[f'R@{k}'] = float(hits.mean())
    return out


def map_all(sim: torch.Tensor, labels: np.ndarray) -> float:
    """Class-level mAP: relevant = same class. Standard AP over the ranked list."""
    order = sim.argsort(dim=1, descending=True).cpu().numpy()
    aps = []
    for qi in range(sim.shape[0]):
        rel = (labels[order[qi]] == labels[qi]).astype(float)
        rel_sum = rel.cumsum()
        prec = rel_sum / (np.arange(len(rel)) + 1)
        aps.append(float((prec * rel).sum() / max(rel_sum[-1], 1e-9)))
    return float(np.mean(aps))


def r_precision(sim: torch.Tensor, labels: np.ndarray) -> float:
    """Class-level R-precision: precision within the top-(#relevant) ranks."""
    order = sim.argsort(dim=1, descending=True).cpu().numpy()
    vals = []
    for qi in range(sim.shape[0]):
        rel = (labels[order[qi]] == labels[qi]).astype(float)
        r = int(rel.sum())
        vals.append(float(rel[:r].mean()) if r > 0 else 0.0)
    return float(np.mean(vals))


# ---------------------------------------------------------------- protocols

def evaluate_protocols(q_embs: dict, labels: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Pilot-style protocol evaluation from per-severity embeddings.

    q_embs: {'clean': (N,d) tensor, 'light': ..., ...}
    Returns
    -------
    {
      'P1_hazy_query_clean_gallery': {severity: {R@k..., mAP, RP}},
      'P2_hazy_query_same_severity_gallery': {...}
    }
    """
    out = {'P1_hazy_query_clean_gallery': {},
           'P2_hazy_query_same_severity_gallery': {}}
    g_clean = q_embs['clean']
    for sev, q in q_embs.items():
        for proto, gallery in (('P1_hazy_query_clean_gallery', g_clean),
                               ('P2_hazy_query_same_severity_gallery', q)):
            sim = q @ gallery.T
            m = recall_at_k(sim, labels, ks)
            m['mAP'] = map_all(sim, labels)
            m['RP'] = r_precision(sim, labels)
            out[proto][sev] = m
    return out


def protocol_table(proto: dict, sevs=SEVERITY_LABELS) -> str:
    """Markdown table for one protocol dict (as returned by evaluate_protocols)."""
    head = ('| severity | ' + ' | '.join(f'R@{k}' for k in (1, 5, 10))
            + ' | mAP | RP |')
    sep = '|' + '---|' * 6
    rows = []
    for s in sevs:
        m = proto[s]
        rows.append(f"| {s} | {m['R@1']:.3f} | {m['R@5']:.3f} "
                    f"| {m['R@10']:.3f} | {m['mAP']:.3f} | {m['RP']:.3f} |")
    return '\n'.join([head, sep, *rows])


def relative_drop(proto: dict, sev_a: str = 'clean', sev_b: str = 'extreme',
                  metric: str = 'R@1') -> float:
    """Relative metric drop (percent) between two severities of one protocol."""
    a, b = proto[sev_a][metric], proto[sev_b][metric]
    return 100.0 * (a - b) / max(a, 1e-9)


# ------------------------------------------------------- zero-shot with weather prompts

ZERO_SHOT_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck']

WEATHER_PHRASES = {
    'clear': 'clear weather',
    'light': 'slightly hazy weather',
    'mild': 'hazy weather',
    'moderate': 'foggy weather with moderate visibility',
    'heavy': 'dense fog with poor visibility',
    'extreme': 'extremely dense haze, visibility near zero',
}


@torch.no_grad()
def zero_shot_weather_probe(model, processor, q_embs: dict,
                            labels: np.ndarray, device: str,
                            classes=ZERO_SHOT_CLASSES) -> dict:
    """P3 diagnostic: zero-shot accuracy of class prompts per severity.

    Uses the frozen text tower; when `model` is a HazCLIP, the learned
    weather offsets are applied so training can be validated: after
    training, the 'extreme' offset should rescue most of the drop.
    """
    from models.hazclip import WEATHER_TOKENS, SEV2IDX
    accs = {}
    for sev, q in q_embs.items():
        prompts = [f'a photo of a {c}, {WEATHER_PHRASES[sev]}'
                   for c in classes]
        txt = processor(text=prompts, return_tensors='pt', padding=True)
        t_emb = model.clip.get_text_features(
            input_ids=txt['input_ids'].to(device),
            attention_mask=txt['attention_mask'].to(device))
        if hasattr(model, 'weather_bank'):
            sev_idx = SEV2IDX[sev]
            t_emb = t_emb + model.weather_bank(
                torch.tensor(sev_idx, device=device))
        t_emb = torch.nn.functional.normalize(t_emb, dim=-1).cpu()
        pred = (q @ t_emb.T).argmax(dim=1).numpy()
        accs[sev] = float((pred == labels).mean())
    return accs
