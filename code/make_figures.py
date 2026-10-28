"""Generate paper figures from saved experiment JSONs (§4 of the draft).

Outputs (06_Paper/figures/):
  fig2_fm_severity.png     - R@1 vs severity for CLIP/SigLIP/DINOv2 (motivation)
  fig3_p1_p2_paradox.png   - P1 vs P2 mAP curves (pilot) - domain-mismatch paradox
  fig4_head2head.png       - raw vs DCP->CLIP vs HazCLIP (identical split)
  fig5_realhaze_ranks.png  - rank histogram / cumulative on real_haze pairs

Usage: .venv/bin/python 04_Code/make_figures.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, '05_Experiments')
OUT = os.path.join(ROOT, '06_Paper', 'figures')
os.makedirs(OUT, exist_ok=True)

SEVS = ['clean', 'light', 'mild', 'moderate', 'heavy', 'extreme']


def load(path):
    with open(path) as f:
        return json.load(f)


# ---------------- Fig 2: FM severity ----------------
fm = load(os.path.join(EXP, 'fm_baselines', 'fm_degradation.json'))
fig, ax = plt.subplots(figsize=(6.2, 4.2))
xs = range(len(SEVS))
for name, label, ls in (('clip', 'CLIP B/32', '-'),
                        ('siglip', 'SigLIP B/16', '--'),
                        ('dinov2', 'DINOv2-small', ':')):
    ys = [fm[name]['P1'][s]['R@1'] for s in SEVS]
    ax.plot(xs, ys, ls, marker='o', label=label)
ax.set_xticks(xs, SEVS)
ax.set_xlabel('Haze severity (ASM β)')
ax.set_ylabel('R@1 (hazy query → clean gallery)')
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2_fm_severity.png'), dpi=200)
plt.close(fig)

# ---------------- Fig 3: P1 vs P2 paradox (pilot) ----------------
pilot = load(os.path.join(EXP, 'pilot_clip_haze', 'results_pilot.json'))
p1 = pilot['protocols']['P1_hazy_query_clean_gallery']
p2 = pilot['protocols']['P2_hazy_query_same_severity_gallery']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
for axx, prot, ttl in ((a1, p1, 'P1: hazy → clean gallery'),
                       (a2, p2, 'P2: hazy → same-severity gallery')):
    axx.plot(xs, [prot[s]['R@1'] for s in SEVS], 'o-', label='R@1')
    axx.plot(xs, [prot[s]['mAP'] for s in SEVS], 's--', label='mAP')
    axx.set_xticks(xs, SEVS, rotation=30)
    axx.set_title(ttl); axx.set_ylim(-0.05, 1.05)
    axx.grid(alpha=.3); axx.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3_p1_p2_paradox.png'), dpi=200)
plt.close(fig)

# ---------------- Fig 4: head-to-head (identical split) ----------------
h2h = load(os.path.join(EXP, 'compare_three', 'compare_three.json'))
fig, ax = plt.subplots(figsize=(6.8, 4.2))
w = 0.26
import numpy as np
x = np.arange(len(SEVS))
for i, (name, label, col) in enumerate((
        ('raw_clip', 'raw CLIP', '#9aa0a6'),
        ('dcp_dehaze_clip', 'DCP→CLIP', '#e8833a'),
        ('hazclip', 'HazCLIP (ours)', '#1a73e8'))):
    ys = [h2h[name]['P1_hazy_query_clean_gallery'][s]['R@1'] for s in SEVS]
    ax.bar(x + (i - 1) * w, ys, w, label=label, color=col)
ax.set_xticks(x, SEVS)
ax.set_ylim(0, 1.1)
ax.set_ylabel('R@1')
ax.grid(axis='y', alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig4_head2head.png'), dpi=200)
plt.close(fig)

# ---------------- Fig 5: real_haze cumulative rank (n=1623) ----------------
hp = load(os.path.join(EXP, 'real_haze_pairs',
                       'real_haze_pairs_raw_clip.json'))
hz = load(os.path.join(EXP, 'real_haze_pairs',
                       'real_haze_pairs_hazclip.json'))
fig, ax = plt.subplots(figsize=(6.0, 4.0))
for d, label, col, ls in ((hp, 'raw CLIP', '#9aa0a6', '--'),
                          (hz, 'HazCLIP (ours)', '#1a73e8', '-')):
    ranks = np.asarray(d['per_query']['ranks'])
    ks = np.arange(1, 51)
    frac = [(ranks <= k).mean() for k in ks]
    ax.plot(ks, frac, ls, color=col, lw=2,
            label=f"{label} (R@1={d['R@1']:.3f}, MRR={d['MRR']:.3f})")
ax.set_xlabel('Rank k of the true clean pair')
ax.set_ylabel('Fraction of queries with rank ≤ k')
ax.set_xlim(1, 50); ax.set_ylim(0.75, 1.005)
ax.axhline(hp['R@1'], color='#9aa0a6', lw=.6, alpha=.5)
ax.axhline(hz['R@1'], color='#1a73e8', lw=.6, alpha=.5)
ax.grid(alpha=.3); ax.legend(loc='lower right', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig5_realhaze_pairs.png'), dpi=200)
plt.close(fig)

print('figures written to', OUT)
for f in sorted(os.listdir(OUT)):
    print(' -', f)
