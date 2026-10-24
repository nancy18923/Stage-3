'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset

import numpy as np

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        # 打开 .csv 文件
        raw = np.loadtxt(self.dataset_source_folder_path + self.dataset_source_file_name, delimiter=',', dtype=np.float32)
        X = raw[:, 1:] / 255.0              # 首列是标签，其余是像素
        Y = raw[:, 0].astype(np.int64)      # LongTensor 需要整型
        print('X:', X.shape, 'Y:', Y.shape, 'range:', X.min(), X.max())
        return {'X': X, 'Y': Y}

# Example usage
if __name__ == "__main__":
    data_obj = Dataset_Loader('mnist', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_source_file_name = 'train.csv'
    Dataset_Loader.load(data_obj)
