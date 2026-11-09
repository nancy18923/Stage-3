"""
image classification for object recognition with CNN models
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import sys
import os
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'stage_3_data')
# Result_Saver concatenates folder + file name, so keep the trailing separator
RESULT_FOLDER = os.path.join(PROJECT_ROOT, 'result', 'stage_3_result') + os.sep
SEED = 2

def reset_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

# ---- build the four template objects for one dataset and run a full train/test round ----
def run_one(dataset_name, method_obj, result_tag):
    print()
    print('=' * 72)
    print(f'### {dataset_name} | {method_obj.__class__.__name__} | result tag: {result_tag}')
    print('=' * 72)

    # ------------ objection initialization ----------------
    train_obj = Dataset_Loader(dataset_name.lower(), '')
    train_obj.dataset_source_folder_path = DATA_FOLDER
    train_obj.dataset_source_file_name = dataset_name
    train_obj.split = 'train'

    test_obj = Dataset_Loader(dataset_name.lower(), '')
    test_obj.dataset_source_folder_path = DATA_FOLDER
    test_obj.dataset_source_file_name = dataset_name
    test_obj.split = 'test'

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = RESULT_FOLDER
    result_obj.result_destination_file_name = 'CNN'

    evaluate_obj = Evaluate_Metrics('metrics', '')

    setting_obj = Setting_Train_Test('train test', '')
    setting_obj.prepare(train_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.test_dataset = test_obj     # 注入测试数据 dataset
    setting_obj.result_tag = result_tag

    setting_obj.print_setup_summary()
    return setting_obj.load_run_save_evaluate()

if __name__ == '__main__':
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    results = {}
    start_time = time.time()

    # comment out a line to skip that dataset while debugging
    reset_seed()
    results['MNIST'] = run_one('MNIST', Method_CNN_MNIST('cnn-mnist', ''), 'MNIST_base')

    reset_seed()
    results['ORL'] = run_one('ORL', Method_CNN_ORL('cnn-orl', ''), 'ORL_base')

    reset_seed()
    results['CIFAR'] = run_one('CIFAR', Method_CNN_CIFAR('cnn-cifar', ''), 'CIFAR_base')

    # ---- summary, useful for section 3.6 of the report -----------------
    elapsed = time.time() - start_time
    print()
    print('=' * 72)
    print('Overall performance')
    print('=' * 72)
    print(f'{"dataset":<8}{"accuracy":>10}{"precision":>11}{"recall":>9}{"f1":>9}')
    for name, metrics in results.items():
        print(f'{name:<8}{metrics["accuracy"]:>10.4f}{metrics["precision"]:>11.4f}'
              f'{metrics["recall"]:>9.4f}{metrics["f1"]:>9.4f}')
    print(f'total wall time: {elapsed / 60:.1f} min')