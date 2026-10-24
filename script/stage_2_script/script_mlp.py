from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
# from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
# from local_code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
# from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 无界面环境下也能出图
import matplotlib.pyplot as plt

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ------------ objection initialization ----------------
    train_obj = Dataset_Loader('mnist', '')
    train_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    train_obj.dataset_source_file_name = 'train.csv'

    test_obj = Dataset_Loader('mnist', '')
    test_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    test_obj.dataset_source_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')
    method_obj.max_epoch = 20

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    # setting_obj = Setting_KFold_CV('k fold cross validation', '')
    setting_obj = Setting_Train_Test('train test', '')
    setting_obj.test_dataset = test_obj     # 注入测试数据 dataset

    # in_Test_Split('train test split', "")

    evaluate_obj = Evaluate_Metrics('metrics', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(train_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    # mean_score, std_score = setting_obj.load_run_save_evaluate()
    metrics = setting_obj.load_run_save_evaluate()  # 返回值从两个数变成了 dict
    print('************ Overall Performance ************')
    # print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print(metrics)
    print('************ Finish ************')

    # ---- training convergence curve (report section 3.5) ----
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, method_obj.max_epoch + 1), method_obj.loss_history,
             marker='o', markersize=3, linewidth=1.2, color='#185FA5', label='training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MLP Training Convergence on MNIST')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    curve_path = '../../result/stage_2_result/MLP_convergence_curve.png'
    plt.savefig(curve_path, dpi=200)
    print('convergence curve saved to:', curve_path)
    # ------------------------------------------------------
    

    