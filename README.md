# HazCLIP — Physics-Guided LoRA Adaptation of CLIP for Haze-Robust Image Retrieval

Official repository for the paper:

> **Haze-Robust Image Retrieval: Physics-Guided Adaptation of Vision–Language
> Models for Content-Based Retrieval in Hazy Domains**
> Ali Hosseini, Saeed Sedighi (Corresponding member)
> Dept. of Computer Engineering, Shi.C., Islamic Azad University, Shiraz, Iran

## TL;DR

Haze collapses query→gallery retrieval for CLIP/SigLIP/DINOv2 (−50–65% R@1
relative). We show the failure is **cross-domain misalignment**, not
information loss, and close it with **HazCLIP**: a 0.198%-parameter
physics-guided LoRA adaptation of a frozen CLIP trained with a three-term
objective (cross-domain InfoNCE + ASM physics head + invariance). It
recovers 73% of the synthetic-haze loss and lifts instance-level R@1 from
0.811 → **0.930** on 1,623 real haze photos (McNemar p = 3.9e-57).

## Results at a glance

| Benchmark | Metric | raw CLIP | HazCLIP |
|---|---|---|---|
| Synthetic, extreme haze (CIFAR-10) | R@1 | 0.353 | **0.814 ± 0.008** (3 seeds) |
| Flickr8k real photos, extreme | R@1 | 0.297 | **0.872** |
| **real_haze (1,623 pairs, instance)** | R@1 | 0.811 | **0.930** |
| real_haze | MRR | 0.858 | **0.953** |
| ImageNet-C fog s5 (zero-shot) | top-5 | 0.512 | **0.588** |

Efficiency: 300,804 trainable params (0.198%) · ≤7 CPU-minutes training ·
≈1.2% extra ViT FLOPs at query time · no dehazing pass.

## Repository layout

```
├── code/                  # full pipeline (python)
│   ├── haze_synth.py          # ASM haze synthesis w/ random depth fields
│   ├── models/hazclip.py      # Haze-LoRA + physics head + prompt bank
│   ├── train_hazclip.py       # training loop (Eq. 5 of the paper)
│   ├── eval_retrieval.py      # R@k / mAP protocols (P1, P2, zero-shot)
│   ├── eval_real_haze_pairs.py# instance-level pairwise benchmark
│   ├── eval_real_fog.py       # ImageNet-C fog zero-shot probe
│   ├── compare_three.py       # raw vs DCP→CLIP vs HazCLIP (same split)
│   ├── run_seeds.py           # 3-seed variance study
│   ├── stat_tests.py          # Wilcoxon signed-rank + exact McNemar
│   └── make_figures.py        # paper figures from result JSONs
├── figures/               # paper figures (Fig. 1–5)
├── results/               # per-run JSONs incl. per-query dumps
│   ├── real_haze_pairs_raw_clip.json
│   ├── real_haze_pairs_hazclip.json
│   ├── stat_tests.json        # Wilcoxon + McNemar outputs
│   └── ...
└── EXPERIMENTS_LOG.md     # par. 1-18: every experiment, verbatim numbers

> **Note:** the manuscript itself is not distributed here while under review; this repository provides code, per-run results, and figures.
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) train HazCLIP v0 (laptop CPU, ~7 min)
python code/train_hazclip.py --device cpu

# 2) synthetic severity evaluation (P1/P2 + zero-shot)
python code/eval_retrieval.py --ckpt <ckpt.pt>

# 3) real-haze pairwise benchmark (needs the real_haze corpus, see below)
python code/eval_real_haze_pairs.py --ckpt <ckpt.pt> --n 0 --device mps

# 4) paired significance tests (Wilcoxon + exact McNemar)
python code/stat_tests.py
```

## Data

| Dataset | Used for | Access |
|---|---|---|
| CIFAR-10 | synthetic severity sandbox (train bank + eval) | auto-download |
| Flickr8k | real-photo bank (2,000 imgs) | manual download |
| real_haze (1,623 hazy/GT pairs) | instance-level real benchmark | HuggingFace `gujintao/real_haze` |
| ImageNet-C fog s5 | zero-shot fog probe (different fog generator) | HuggingFace |

Set `03_Data/real_haze/` with `hazy/HAZE_max1024/` and `gt/GT_max1024/`.

## Reproducibility

- Every number in the paper maps to a § in `EXPERIMENTS_LOG.md`.
- 3 training seeds reported (std ≤ 0.008); per-query paired tests shipped.
- Deterministic splits; per-query rank dumps included in `results/`.

## Citation

If you use this code, please cite (BibTeX in `CITATION.cff`):

```bibtex
@article{hosseini2026hazclip,
  title  = {Haze-Robust Image Retrieval: Physics-Guided Adaptation of
            Vision-Language Models for Content-Based Retrieval in Hazy Domains},
  author = {Hosseini, Ali and Sedighi, Saeed},
  journal= {IEEE Transactions on Image Processing (under review)},
  year   = {2026}
}
```

## License

Released under the MIT License (see `LICENSE`). Datasets follow their own
licenses/terms of use.
