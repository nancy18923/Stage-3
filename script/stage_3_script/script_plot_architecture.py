"""
Draw the CNN architecture diagram used in the Stage 3 report.

Top row: the shared processing pipeline, where the conv block
(Conv2d 3x3 -> BatchNorm -> ReLU -> MaxPool2d) is repeated `depth` times.
Bottom: a per-dataset table of the configuration differences.

usage:
    python script/stage_3_script/script_plot_architecture.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGURE_FOLDER = os.path.join(PROJECT_ROOT, 'result', 'stage_3_result', 'figures') + os.sep

C_INPUT = '#8C8C8C'
C_CONV = '#4C78A8'
C_POOL = '#54A24B'
C_FC = '#B279A2'
C_OUT = '#E45756'


def box(ax, x, y, w, h, text, color, fs=10, tc='white'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                                facecolor=color, edgecolor='#3a3a3a', linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc, linespacing=1.35)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>',
                                 mutation_scale=13, color='#555555', linewidth=1.1,
                                 shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis('off')

    ax.text(50, 44, 'Shared CNN pipeline (all three datasets)',
            ha='center', va='center', fontsize=13, fontweight='bold')

    # ---------------- pipeline row ----------------
    y, h = 29, 8
    boxes = [
        (2.0, 13.0, 'Input\nimage', C_INPUT, 10),
        (17.5, 28.0, 'Conv Block\n(Conv2d 3x3 -> BN -> ReLU\n-> MaxPool2d 2x2) x depth', C_CONV, 9),
        (50.0, 14.5, 'AdaptiveAvgPool\n2x2 + Flatten', C_POOL, 9),
        (69.0, 13.0, 'Linear 128\nReLU + Dropout', C_FC, 9),
        (86.5, 11.5, 'Linear\nn_classes', C_OUT, 10),
    ]
    for i, (x, w, text, color, fs) in enumerate(boxes):
        box(ax, x, y, w, h, text, color, fs=fs)
        if i + 1 < len(boxes):
            arrow(ax, x + w, boxes[i + 1][0], y + h / 2)

    # ---------------- per-dataset table ----------------
    ax.text(2.0, 24.6, 'Per-dataset configuration (only these values differ)',
            fontsize=11, fontweight='bold', color='#333333')

    cols = [(2.0, 12, 'Dataset'), (14.0, 18, 'Input shape'),
            (32.0, 22, 'Conv channels (depth)'), (54.0, 12, 'Dropout'),
            (66.0, 12, 'Classes'), (78.0, 20, 'Epochs / Batch')]
    rows = [
        ('MNIST', '1 x 28 x 28 (gray)', '32, 64 (2 blocks)', '0.25', '10', '10 / 64'),
        ('ORL', '1 x 112 x 92 (gray)', '32, 64 (2 blocks)', '0.5', '40', '40 / 32'),
        ('CIFAR', '3 x 32 x 32 (RGB)', '32, 64, 128 (3 blocks)', '0.3', '10', '20 / 64'),
    ]
    rh = 4.2
    ty = 19.8
    # header
    for x, w, label in cols:
        box(ax, x, ty, w, rh, label, '#4a4a4a', fs=9)
    yy = ty - 1.0
    for row in rows:
        yy -= rh + 0.8
        for (x, w, _), val in zip(cols, row):
            box(ax, x, yy, w, rh, val, '#EAEAEA', fs=9, tc='#222222')

    # shared training setup note
    ax.text(2.0, 1.2,
            'Shared training setup: CrossEntropyLoss on raw logits | Adam (lr = 1e-3) | '
            'mini-batch gradient descent | weight reset + seed = 2 for every run | PyTorch (CPU)',
            fontsize=9, color='#333333', va='center')

    out_path = os.path.join(FIGURE_FOLDER, 'cnn_architecture.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', out_path)


if __name__ == '__main__':
    main()
