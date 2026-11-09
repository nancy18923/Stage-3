"""
Ablation driver for the Stage 3 CNN experiments

Every configuration varies exactly one knob of the baseline architecture, so the
resulting table can answer "which component actually helps" (report section 3.7).

The method classes are sub-classed on the fly (type(...)), which means the
configuration lives in a dict here and Method_CNN_*.py is never edited during
the sweep. Each run goes through the same Setting / Result / Evaluate objects as
script_cnn.py, so the ablation curves are saved as normal result files too and
can be plotted with:  script_plot_curves.py --prefix ABL

usage:
    python script/stage_3_script/script_ablation.py                    # all datasets, quick defaults
    python script/stage_3_script/script_ablation.py ORL                # one dataset only
    python script/stage_3_script/script_ablation.py MNIST --epochs 8 --subset 0   # full data
    python script/stage_3_script/script_ablation.py MNIST --configs baseline,kernel5,avgpool
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import csv
import time

import numpy as np
import torch

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics

DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'stage_3_data')
RESULT_FOLDER = os.path.join(PROJECT_ROOT, 'result', 'stage_3_result') + os.sep

METHOD_CLASSES = {'MNIST': Method_CNN_MNIST, 'ORL': Method_CNN_ORL, 'CIFAR': Method_CNN_CIFAR}
DATASETS = list(METHOD_CLASSES.keys())

# Result_Saver writes folder + prefix + '_' + tag, a different prefix keeps the
# ablation dumps apart from the main experiment dumps (CNN_*)
RESULT_PREFIX = 'ABL'

SEED = 2

# quick defaults for the sweep on a CPU-only machine. The numbers are only meant
# to rank the configurations against each other; the final result table should be
# re-run with --epochs <real value> --subset 0 on the full data.
DEFAULT_EPOCHS = {'MNIST': 3, 'ORL': 20, 'CIFAR': 3}
DEFAULT_SUBSET_PER_CLASS = {'MNIST': 300, 'ORL': 9, 'CIFAR': 300}


def reset_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)


class Subset_Loader(Dataset_Loader):
    """same loader, but keeps at most N samples per class on the training split.

    Sub-classing the loader instead of touching Setting_Train_Test keeps the
    template objects untouched: Setting still calls load() and gets {'X', 'Y'} back.
    """

    max_per_class = None  # None / 0 -> use every sample

    def load(self):
        data = super().load()
        if not self.max_per_class:
            return data

        Y = data['Y']
        keep = [np.flatnonzero(Y == c)[:self.max_per_class] for c in np.unique(Y)]
        keep = np.sort(np.concatenate(keep))

        data = {'X': np.ascontiguousarray(data['X'][keep]), 'Y': data['Y'][keep]}
        print(f'  subset: {len(keep)} samples '
              f'({self.max_per_class} per class, {len(np.unique(data["Y"]))} classes)')
        return data


def build_configs(base_cls):
    """one knob changed per configuration, everything else inherits from the baseline class"""
    channels = list(base_cls.conv_channels)
    return [
        ('baseline', {}),
        ('kernel5', {'kernel_size': 5, 'padding': 2}),          # same output size, bigger receptive field
        ('padding0', {'kernel_size': 3, 'padding': 0}),         # spatial shrink, adaptive pool absorbs it
        ('depth_up', {'conv_channels': channels + [channels[-1] * 2]}),
        ('wider', {'conv_channels': [c * 2 for c in channels]}),
        ('avgpool', {'pool_type': 'avg'}),
        ('no_bn', {'use_batch_norm': False}),
        ('dropout0', {'dropout': 0.0}),
        ('hidden256', {'hidden_dims': [256]}),
        ('sgd', {'optimizer_name': 'sgd'}),
        # log_softmax + NLLLoss is mathematically the same as CrossEntropyLoss, so
        # this row is expected to match the baseline: it is an implementation check
        ('nll', {'loss_func': 'nll'}),
        ('lr_dec', {'learning_rate': base_cls.learning_rate / 10}),
    ]


def make_method(dataset_name, config_name, overrides, epochs):
    """sub-class the method so that the overrides become class attributes"""
    base_cls = METHOD_CLASSES[dataset_name]
    attrs = dict(overrides)
    attrs['max_epoch'] = epochs
    method_cls = type(f'{base_cls.__name__}_{config_name}', (base_cls,), attrs)
    return method_cls(f'cnn-{dataset_name.lower()}-{config_name}', '')


def run_one_config(dataset_name, config_name, overrides, epochs, subset_per_class, result_folder):
    reset_seed()
    method_obj = make_method(dataset_name, config_name, overrides, epochs)

    train_obj = Subset_Loader(dataset_name.lower(), '')
    train_obj.dataset_source_folder_path = DATA_FOLDER
    train_obj.dataset_source_file_name = dataset_name
    train_obj.split = 'train'
    train_obj.max_per_class = subset_per_class

    test_obj = Subset_Loader(dataset_name.lower(), '')
    test_obj.dataset_source_folder_path = DATA_FOLDER
    test_obj.dataset_source_file_name = dataset_name
    test_obj.split = 'test'
    test_obj.max_per_class = None  # the test split is always used in full

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = result_folder
    result_obj.result_destination_file_name = RESULT_PREFIX

    evaluate_obj = Evaluate_Metrics('metrics', '')

    setting_obj = Setting_Train_Test('train test', '')
    setting_obj.prepare(train_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.test_dataset = test_obj
    setting_obj.result_tag = f'{dataset_name}_{config_name}'
    setting_obj.verbose = False

    metrics = setting_obj.load_run_save_evaluate()

    history = method_obj.loss_history or {}
    return {
        'dataset': dataset_name,
        'config': config_name,
        'overrides': ', '.join(f'{k}={v}' for k, v in overrides.items()) or '(baseline)',
        'epochs': epochs,
        'train_samples': len(method_obj.data['train']['Y']),
        'batch_size': method_obj.batch_size,
        'learning_rate': method_obj.learning_rate,
        'optimizer': method_obj.optimizer_name,
        'loss_func': method_obj.loss_func,
        'conv_channels': '-'.join(str(c) for c in method_obj.conv_channels),
        'kernel': method_obj.kernel_size,
        'padding': method_obj.padding,
        'pool': f'{method_obj.pool_type}{method_obj.pool_size}',
        'hidden_dims': '-'.join(str(h) for h in method_obj.hidden_dims),
        'dropout': method_obj.dropout,
        'batch_norm': int(bool(method_obj.use_batch_norm)),
        'accuracy': round(float(metrics['accuracy']), 4),
        'precision': round(float(metrics['precision']), 4),
        'recall': round(float(metrics['recall']), 4),
        'f1': round(float(metrics['f1']), 4),
        'best_test_acc': round(float(max(history['test_acc'])), 4) if history.get('test_acc') else '',
        'final_test_acc': round(float(history['test_acc'][-1]), 4) if history.get('test_acc') else '',
        'train_time_s': round(float(getattr(method_obj, 'method_running_time', 0.0)), 1),
    }


CSV_FIELDS = ['dataset', 'config', 'overrides', 'epochs', 'train_samples', 'batch_size',
              'learning_rate', 'optimizer', 'loss_func', 'conv_channels', 'kernel', 'padding',
              'pool', 'hidden_dims', 'dropout', 'batch_norm', 'accuracy', 'precision',
              'recall', 'f1', 'best_test_acc', 'final_test_acc', 'train_time_s']


def print_table(rows, dataset_name):
    header = (f'{"config":<12}{"acc":>8}{"prec":>8}{"rec":>8}{"f1":>8}'
              f'{"best":>8}{"time":>9}')
    print()
    print('=' * 72)
    print(f'Ablation result - {dataset_name}')
    print('=' * 72)
    print(header)
    print('-' * 72)
    for row in rows:
        print(f'{row["config"]:<12}{row["accuracy"]:>8.4f}{row["precision"]:>8.4f}'
              f'{row["recall"]:>8.4f}{row["f1"]:>8.4f}{row["best_test_acc"]:>8.4f}'
              f'{row["train_time_s"]:>8.0f}s')


def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        # extrasaction='ignore': a failed run carries an extra 'note' field that
        # is not part of the table, it only shows up in the console log
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'csv written: {os.path.basename(path)}')


def parse_args():
    parser = argparse.ArgumentParser(description='single-factor ablation sweep of the Stage 3 CNN')
    parser.add_argument('datasets', nargs='*', default=None,
                        help=f'datasets to sweep, any of {DATASETS}; empty means all of them')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override the epoch count used for every configuration')
    parser.add_argument('--subset', type=int, default=None,
                        help='max training samples per class, 0 = full data (default: quick subset)')
    parser.add_argument('--configs', default=None,
                        help='comma separated configuration names, empty means all of them')
    parser.add_argument('--out-folder', default=RESULT_FOLDER,
                        help='folder for the result files and the csv')
    return parser.parse_args()


def main():
    args = parse_args()

    datasets = [d.upper() for d in args.datasets] if args.datasets else DATASETS
    unknown = [d for d in datasets if d not in METHOD_CLASSES]
    if unknown:
        print(f'unknown dataset(s) {unknown}, choose from {DATASETS}')
        return 1

    # Result_Saver concatenates folder + file name, the trailing separator is required;
    # abspath also normalises the mixed '/' and '\' of a hand written path
    result_folder = os.path.abspath(args.out_folder)
    result_folder = result_folder if result_folder.endswith(os.sep) else result_folder + os.sep
    os.makedirs(result_folder, exist_ok=True)

    only = [c.strip() for c in args.configs.split(',')] if args.configs else None

    all_rows = []
    start_time = time.time()

    for dataset_name in datasets:
        base_cls = METHOD_CLASSES[dataset_name]
        configs = build_configs(base_cls)
        if only:
            configs = [c for c in configs if c[0] in only]
        if not configs:
            print(f'{dataset_name}: no configuration selected, skip')
            continue

        epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS[dataset_name]
        subset = args.subset if args.subset is not None else DEFAULT_SUBSET_PER_CLASS[dataset_name]

        print()
        print('=' * 72)
        print(f'### ablation on {dataset_name} | {len(configs)} configs | {epochs} epochs | '
              f'subset/class {subset if subset else "full"}')
        print('=' * 72)

        rows = []
        dataset_start = time.time()
        for index, (config_name, overrides) in enumerate(configs, start=1):
            print()
            print(f'--- [{index}/{len(configs)}] {dataset_name} / {config_name} ---')
            try:
                row = run_one_config(dataset_name, config_name, overrides,
                                     epochs, subset, result_folder)
            except Exception as error:  # a broken config must not kill the whole sweep
                print(f'  FAILED: {type(error).__name__}: {error}')
                rows.append({field: '' for field in CSV_FIELDS}
                            | {'dataset': dataset_name, 'config': config_name,
                               'overrides': ', '.join(f'{k}={v}' for k, v in overrides.items()),
                               'note': str(error)})
                continue

            rows.append(row)
            print(f'  acc {row["accuracy"]:.4f} | f1 {row["f1"]:.4f} | '
                  f'best {row["best_test_acc"]} | {row["train_time_s"]:.0f}s')

        valid_rows = [r for r in rows if r.get('accuracy') != '']
        print_table(valid_rows, dataset_name)
        print(f'{dataset_name} sweep finished in {(time.time() - dataset_start) / 60:.1f} min')

        write_csv(rows, os.path.join(result_folder, f'ablation_{dataset_name}.csv'))
        all_rows.extend(rows)

    if not all_rows:
        print('nothing was run')
        return 1

    write_csv(all_rows, os.path.join(result_folder, 'ablation_all.csv'))
    print()
    print(f'total wall time: {(time.time() - start_time) / 60:.1f} min')
    print(f'result files and csv are in {result_folder}')
    print(f'plot the sweep curves with: script_plot_curves.py --prefix {RESULT_PREFIX}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
