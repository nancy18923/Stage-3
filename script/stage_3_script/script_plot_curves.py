"""
Plot the learning curves of the Stage 3 CNN experiments

This script only consumes the result files written by Result_Saver, so the
figures can be regenerated at any time without retraining anything.

usage (run from anywhere, the project root is added to sys.path below):
    python script/stage_3_script/script_plot_curves.py                 # every CNN_* result file
    python script/stage_3_script/script_plot_curves.py MNIST_base      # only the given tags
    python script/stage_3_script/script_plot_curves.py --prefix ABL    # ablation result files
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import glob

# 'Agg' must be selected before pyplot is imported: it renders to file only,
# so the script never tries to open a window (works on a headless machine / over ssh)
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from local_code.stage_3_code.Result_Loader import Result_Loader

RESULT_FOLDER = os.path.join(PROJECT_ROOT, 'result', 'stage_3_result') + os.sep
FIGURE_FOLDER = os.path.join(PROJECT_ROOT, 'result', 'stage_3_result', 'figures') + os.sep

# Result_Saver writes folder + prefix + '_' + fold_count, so the prefix is the
# part shared by all files of one experiment family
DEFAULT_PREFIX = 'CNN'

C_TRAIN = '#1f77b4'
C_TEST = '#d62728'


def with_sep(path):
    """Result_Saver / Result_Loader concatenate strings, a trailing separator is required"""
    return path if path.endswith(os.sep) else path + os.sep


def list_result_tags(folder, prefix):
    """recover the fold_count (= tag) of every '<prefix>_<tag>' file in the folder"""
    tags = []
    for path in sorted(glob.glob(os.path.join(folder, prefix + '_*'))):
        if os.path.isdir(path) or not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        tags.append(name[len(prefix) + 1:])
    return tags


def load_result(folder, tag, prefix):
    loader = Result_Loader('result', '')
    loader.result_destination_folder_path = with_sep(folder)
    loader.result_destination_file_name = prefix
    loader.fold_count = tag
    loader.load()
    return loader.data


def get_history(data):
    """the method returns 'loss_history'; 'history' is kept as a fallback for older dumps"""
    for key in ('loss_history', 'history'):
        history = data.get(key)
        if history and history.get('train_loss'):
            return history
    return None


def metrics_line(data):
    metrics = data.get('metrics') or {}
    if not metrics:
        return ''
    return (f'Acc {metrics["accuracy"]:.4f} | P {metrics["precision"]:.4f} | '
            f'R {metrics["recall"]:.4f} | F1 {metrics["f1"]:.4f}')


def format_epoch_axis(ax, num_epochs):
    """integer epoch ticks, and a sane range even when only one epoch was recorded"""
    ax.set_xlabel('epoch')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # the default autoscale would zoom into 0.945 - 1.05 for a single point
    ax.set_xlim(0.5, max(num_epochs, 1) + 0.5)


def plot_single(tag, data, history, out_path):
    """one figure per experiment: loss on the left, accuracy on the right"""
    epochs = list(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    ax.plot(epochs, history['train_loss'], color=C_TRAIN, marker='o', markersize=3, label='train loss')
    if history.get('test_loss'):
        ax.plot(epochs, history['test_loss'], color=C_TEST, marker='s', markersize=3, label='test loss')
    ax.set_ylabel('loss')
    ax.set_title('loss')
    ax.grid(alpha=0.3)
    ax.legend()
    format_epoch_axis(ax, len(epochs))

    ax = axes[1]
    ax.plot(epochs, history['train_acc'], color=C_TRAIN, marker='o', markersize=3, label='train acc')
    if history.get('test_acc'):
        ax.plot(epochs, history['test_acc'], color=C_TEST, marker='s', markersize=3, label='test acc')
        best_epoch = int(np.argmax(history['test_acc']))
        best_acc = history['test_acc'][best_epoch]
        ax.scatter([epochs[best_epoch]], [best_acc], s=60, facecolor='none',
                   edgecolor=C_TEST, linewidth=1.5, zorder=5)
        # flip the label inward when the best epoch sits near the right/top edge
        near_right = epochs[best_epoch] > 0.7 * len(epochs)
        offset = (-10 if near_right else 10, -16 if best_acc >= 0.85 else 8)
        ax.annotate(f'best {best_acc:.4f} @ep{epochs[best_epoch]}',
                    xy=(epochs[best_epoch], best_acc),
                    xytext=offset, textcoords='offset points', fontsize=9, color=C_TEST,
                    ha='right' if near_right else 'left')
    ax.set_ylabel('accuracy')
    ax.set_title('accuracy')
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    format_epoch_axis(ax, len(epochs))

    subtitle = metrics_line(data)
    fig.suptitle(f'{tag}' + (f'\n{subtitle}' if subtitle else ''), fontsize=11)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {os.path.basename(out_path)}')


def plot_combined(series, out_path):
    """overlay every experiment: train loss and test accuracy side by side"""
    if len(series) < 2:
        return False

    num_epochs = max(len(history['train_loss']) for _, history in series)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))

    # the marker matters here: a single recorded epoch would otherwise plot as an
    # invisible one-point line
    ax = axes[0]
    for tag, history in series:
        epochs = list(range(1, len(history['train_loss']) + 1))
        ax.plot(epochs, history['train_loss'], linewidth=1.6, marker='o', markersize=3, label=tag)
    ax.set_ylabel('train loss')
    ax.set_title('train loss')
    ax.grid(alpha=0.3)
    format_epoch_axis(ax, num_epochs)

    ax = axes[1]
    for tag, history in series:
        if not history.get('test_acc'):
            continue
        epochs = list(range(1, len(history['test_acc']) + 1))
        ax.plot(epochs, history['test_acc'], linewidth=1.6, marker='o', markersize=3, label=tag)
    ax.set_ylabel('test accuracy')
    ax.set_title('test accuracy')
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    format_epoch_axis(ax, num_epochs)

    # one shared legend below both panels: with a dozen experiments an in-axes
    # legend would cover the curves it is supposed to explain
    handles, labels = axes[1].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=min(4, max(len(labels), 1)), fontsize=8, frameon=False)
    fig.suptitle('Comparison of all experiments', fontsize=11)
    # keep a strip at the bottom free for the legend, otherwise it lands on the x label
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {os.path.basename(out_path)}')
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='plot the training curves stored in the result files')
    parser.add_argument('tags', nargs='*',
                        help='result tags to plot, e.g. MNIST_base; empty means every file of the prefix')
    parser.add_argument('--prefix', default=DEFAULT_PREFIX,
                        help=f'file name prefix written by Result_Saver (default: {DEFAULT_PREFIX})')
    parser.add_argument('--result-folder', default=RESULT_FOLDER,
                        help='folder holding the result files')
    parser.add_argument('--out-folder', default=None,
                        help='folder for the png files, default: <result-folder>/figures/')
    parser.add_argument('--no-combined', action='store_true',
                        help='skip the overlay figure')
    return parser.parse_args()


def main():
    args = parse_args()

    result_folder = with_sep(os.path.abspath(args.result_folder))
    figure_folder = with_sep(os.path.abspath(args.out_folder or os.path.join(result_folder, 'figures')))
    os.makedirs(figure_folder, exist_ok=True)

    tags = args.tags if args.tags else list_result_tags(result_folder, args.prefix)
    if not tags:
        print(f'no result file found as {result_folder}{args.prefix}_*')
        print('run script_cnn.py (or script_ablation.py) first, the curves come from the saved results.')
        return 1

    print(f'result folder: {result_folder}')
    print(f'figure folder: {figure_folder}')
    print(f'{len(tags)} result file(s) to plot: {", ".join(tags)}')
    print()

    series = []
    for tag in tags:
        try:
            data = load_result(result_folder, tag, args.prefix)
        except FileNotFoundError:
            print(f'  skip {tag}: file not found')
            continue

        history = get_history(data)
        if history is None:
            print(f'  skip {tag}: no loss_history in the result file')
            continue

        print(f'  {tag}: {len(history["train_loss"])} epochs'
              + (f' | best test acc {max(history["test_acc"]):.4f}' if history.get('test_acc') else ''))
        plot_single(tag, data, history, os.path.join(figure_folder, f'curve_{tag}.png'))
        series.append((tag, history))

    if not args.no_combined:
        plot_combined(series, os.path.join(figure_folder, 'curve_comparison.png'))

    print()
    print(f'done, {len(series)} figure(s) written to {figure_folder}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
