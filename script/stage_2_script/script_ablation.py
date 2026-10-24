from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch
import csv

# ---- Ablation study script (report section 3.7) ----
# 每组只改一个变量，其余保持与基线一致；每组固定 seed=2 保证公平对比。

# 类属性默认值，用于每组实验前恢复
DEFAULTS = dict(hidden_dims=[256], learning_rate=1e-3,
                batch_size=64, loss_func='ce', optimizer_name='adam')

CONFIGS = [
    # (配置名, 相对基线的改动)
    ('baseline [256] adam lr=1e-3', {}),
    ('hidden [128]',                dict(hidden_dims=[128])),
    ('hidden [256,128] deeper',     dict(hidden_dims=[256, 128])),
    ('optimizer SGD momentum=0.9',  dict(optimizer_name='sgd')),
    ('learning rate 1e-2',          dict(learning_rate=1e-2)),
]

MAX_EPOCH = 20


def run_config(config_name, overrides):
    # 每组都重置种子与类属性，保证组间可比
    np.random.seed(2)
    torch.manual_seed(2)
    for k, v in DEFAULTS.items():
        setattr(Method_MLP, k, v)
    for k, v in overrides.items():
        setattr(Method_MLP, k, v)

    train_obj = Dataset_Loader('mnist', '')
    train_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    train_obj.dataset_source_file_name = 'train.csv'

    test_obj = Dataset_Loader('mnist', '')
    test_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    test_obj.dataset_source_file_name = 'test.csv'

    method_obj = Method_MLP(config_name, '')
    method_obj.max_epoch = MAX_EPOCH

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/ablation_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('train test', '')
    setting_obj.test_dataset = test_obj

    evaluate_obj = Evaluate_Metrics('metrics', '')

    setting_obj.prepare(train_obj, method_obj, result_obj, evaluate_obj)
    metrics = setting_obj.load_run_save_evaluate()
    metrics['final_train_loss'] = method_obj.loss_history[-1]
    return metrics


if 1:
    print('************ Ablation Study Start ************')
    rows = []
    for name, overrides in CONFIGS:
        print(f'\n>>>> config: {name} <<<<')
        metrics = run_config(name, overrides)
        rows.append((name, metrics))
        print(f'>>>> done: acc={metrics["accuracy"]:.4f} '
              f'f1={metrics["f1"]:.4f} final_loss={metrics["final_train_loss"]:.4f}')

    # 写 CSV 方便直接贴进报告
    csv_path = '../../result/stage_2_result/ablation_results.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'accuracy', 'precision', 'recall', 'f1', 'final_train_loss'])
        for name, m in rows:
            writer.writerow([name,
                             f'{m["accuracy"]:.4f}', f'{m["precision"]:.4f}',
                             f'{m["recall"]:.4f}', f'{m["f1"]:.4f}',
                             f'{m["final_train_loss"]:.4f}'])

    print('\n************ Ablation Summary ************')
    print(f'{"config":<32}{"acc":>8}{"P":>8}{"R":>8}{"F1":>8}{"loss":>8}')
    for name, m in rows:
        print(f'{name:<32}{m["accuracy"]:>8.4f}{m["precision"]:>8.4f}'
              f'{m["recall"]:>8.4f}{m["f1"]:>8.4f}{m["final_train_loss"]:>8.4f}')
    print('saved to:', csv_path)
    print('************ Ablation Finish ************')
