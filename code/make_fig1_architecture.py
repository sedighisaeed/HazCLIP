"""Clean Fig.1 architecture diagram (IEEE style, publication quality).

v2: generous box sizes (no text overflow), shorter labels, wider canvas.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, '06_Paper', 'figures', 'fig1_architecture.png')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
})

fig, ax = plt.subplots(figsize=(7.16, 3.2), dpi=300)
ax.set_xlim(0, 112)
ax.set_ylim(0, 34)
ax.axis('off')

C_FROZEN = '#e8eef7'
C_LORA = '#1a73e8'
C_HEAD = '#e8833a'
C_BANK = '#7a4fd0'
C_IMG = '#f4f4f4'
C_LOSS = '#2e7d32'


def box(x, y, w, h, text, fc, ec='#333', lw=0.9, fs=9,
        text_color='black', bold=False):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.15,rounding_size=1.2',
                       fc=fc, ec=ec, lw=lw, mutation_aspect=0.5)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=text_color,
            fontweight='bold' if bold else 'normal')


def arrow(x1, y1, x2, y2, color='#333', lw=1.1):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                        mutation_scale=10, color=color, lw=lw,
                        shrinkA=1, shrinkB=1)
    ax.add_patch(a)


# ---------------- inputs ----------------
box(0.5, 19.5, 13, 7, 'Hazy query\n$I_q$', C_IMG)
box(0.5, 6.0, 13, 7, 'Clean gallery\n$I_g$', C_IMG)

# ---------------- frozen CLIP + LoRA ----------------
box(19.5, 14.5, 26, 11, '', C_FROZEN)
ax.text(32.5, 23.0, 'CLIP ViT-B/32 (frozen)', ha='center',
        fontsize=9, fontweight='bold')
box(22, 15.8, 21, 5.4, 'Haze-LoRA\nQ/V, last 4 blocks, $r{=}8$',
    '#ffffff', ec=C_LORA, lw=1.4, fs=8)

arrow(13.8, 23.0, 19.3, 21.5)
arrow(13.8, 9.5, 19.3, 17.5)
ax.text(16.2, 15.6, 'shared\nweights', ha='center', fontsize=6.8,
        style='italic', color='#555')

# ---------------- embeddings ----------------
box(51, 19.0, 10, 6, '$e_q$', '#fff', ec='#555')
box(51, 7.0, 10, 6, '$e_g$', '#fff', ec='#555')
arrow(45.8, 20.5, 50.8, 21.8)
arrow(45.8, 17.0, 50.8, 10.2)

# ---------------- physics head ----------------
box(51, 27.4, 22, 5.2, 'ASM head:  $\\hat{\\beta}$, $\\hat{t}$',
    C_HEAD, text_color='white', fs=9, bold=True)
arrow(42, 25.6, 55, 27.4, color=C_HEAD)

# ---------------- prompt bank ----------------
box(51, 0.6, 22, 5.0, 'Weather prompt bank', C_BANK, text_color='white',
    fs=9, bold=True)

# ---------------- similarity / loss ----------------
box(67, 12.0, 12.5, 8.5, 'cos-sim\nranking', '#fff', ec='#333')
arrow(61.2, 22.0, 67, 18.5)
arrow(61.2, 10.0, 67, 14.0)

box(85, 10.5, 25, 11.5,
    '$\\mathcal{L} = \\mathcal{L}_{\\mathrm{xNCE}}$'
    '\n$+\\,\\lambda_1\\,\\mathcal{L}_{\\mathrm{phys}}$'
    '\n$+\\,\\lambda_2\\,\\mathcal{L}_{\\mathrm{inv}}$',
    '#eef7ee', ec=C_LOSS, lw=1.4, fs=9.5)
arrow(79.8, 16.2, 84.8, 16.2)

# physics head / bank connect to loss (dashed, thin)
a = FancyArrowPatch((73, 27.4), (93, 22.2), arrowstyle='-|>',
                    mutation_scale=9, color=C_HEAD, lw=0.9,
                    linestyle=(0, (3, 2)), shrinkA=1, shrinkB=1)
ax.add_patch(a)
a = FancyArrowPatch((62, 5.6), (88, 10.3), arrowstyle='-|>',
                    mutation_scale=9, color=C_BANK, lw=0.9,
                    linestyle=(0, (3, 2)), shrinkA=1, shrinkB=1)
ax.add_patch(a)

fig.savefig(OUT, bbox_inches='tight', dpi=300, facecolor='white')
print('saved', OUT)
