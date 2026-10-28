"""Paired statistical tests for real-haze pairwise retrieval.

Compares HazCLIP vs raw CLIP per-query using the per_query dumps:
  - Wilcoxon signed-rank test on reciprocal ranks (primary)
  - McNemar test on R@1 hits (secondary, exact binomial on discordant pairs)

Usage:
  .venv/bin/python 04_Code/stat_tests.py \
      [--raw 05_Experiments/real_haze_pairs/real_haze_pairs_raw_clip.json] \
      [--haz 05_Experiments/real_haze_pairs/real_haze_pairs_hazclip.json] \
      [--out 05_Experiments/real_haze_pairs/stat_tests.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np


def load_rr(path):
    with open(path) as f:
        d = json.load(f)
    return np.asarray(d['per_query']['recip_rank'], dtype=np.float64), \
        np.asarray(d['per_query']['ranks'], dtype=np.int64)


def wilcoxon_signed_rank(x, y):
    """Manual Wilcoxon signed-rank (two-sided, normal approximation with
    tie correction). Returns (W, z, p, n_nonzero, n_ties)."""
    d = x - y
    nz = d != 0
    d = d[nz]
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 1.0, 0, 0
    sgn = np.sign(d)
    a = np.abs(d)
    order = np.argsort(a, kind='mergesort')
    sorted_a = a[order]
    avg_r = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg_r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    W = float((sgn * avg_r).sum())
    mu = 0.0
    # tie correction: group sizes
    grp_sizes = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        grp_sizes.append(j - i + 1)
        i = j + 1
    tie_term = sum((g ** 3 - g) for g in grp_sizes if g > 1)
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 6.0 - tie_term / 2.0)
    z = (W - mu) / sigma if sigma > 0 else 0.0
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return W, z, p, n, tie_term


def mcnemar_exact(r1_raw, r1_haz):
    """Exact McNemar via binomial on discordant pairs."""
    b = int(np.sum((r1_raw == 1) & (r1_haz == 0)))  # raw wins
    c = int(np.sum((r1_raw == 0) & (r1_haz == 1)))  # haz wins
    n = b + c
    if n == 0:
        return b, c, 1.0
    # two-sided exact binomial, p=0.5
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2.0 ** (n - 1)
    return b, c, min(1.0, p)


def main(args):
    rr_raw, ranks_raw = load_rr(args.raw)
    rr_haz, ranks_haz = load_rr(args.haz)
    assert len(rr_raw) == len(rr_haz), 'n mismatch between dumps'
    W, z, p_w, n, _ = wilcoxon_signed_rank(rr_haz, rr_raw)
    b, c, p_m = mcnemar_exact((ranks_raw == 1).astype(int),
                              (ranks_haz == 1).astype(int))
    out = {
        'n_pairs': int(len(rr_raw)),
        'wilcoxon_recip_rank': {'W': W, 'z': z, 'p_two_sided': p_w,
                                'n_nonzero': n,
                                'note': 'HazCLIP minus raw CLIP; '
                                        'positive z favors HazCLIP'},
        'mcnemar_R@1': {'b_raw_only_hits': b, 'c_haz_only_hits': c,
                        'p_exact_two_sided': p_m},
        'mean_recip_rank': {'raw': float(rr_raw.mean()),
                            'hazclip': float(rr_haz.mean())},
    }
    print(json.dumps(out, indent=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print('[done]', args.out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw',
                    default='05_Experiments/real_haze_pairs/'
                            'real_haze_pairs_raw_clip.json')
    ap.add_argument('--haz',
                    default='05_Experiments/real_haze_pairs/'
                            'real_haze_pairs_hazclip.json')
    ap.add_argument('--out',
                    default='05_Experiments/real_haze_pairs/'
                            'stat_tests.json')
    main(ap.parse_args())
