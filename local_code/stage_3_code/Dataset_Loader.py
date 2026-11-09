"""
Concrete IO class for the three datasets (MNIST / ORL / CIFAR-10)
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset

import os
import pickle
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    # which part of the pickle file to read: 'train' or 'test'
    split = None
    # ---- image preprocessing settings -------------------------------
    # 'auto' : drop the duplicated channels when R == G == B (ORL), keep 3 channels otherwise (CIFAR)
    # 'gray' : use a single channel        'rgb' : use all three channels
    channel_mode = 'auto'
    # value subtracted from the raw labels so that classes become 0 .. C-1
    # 'auto' : use the minimum label over train+test (ORL 1..40 -> 0..39, MNIST / CIFAR stay 0..9)
    label_offset = 'auto'
    # divide the pixel values by this factor; skipped automatically if data is already in [0, 1]
    pixel_scale = 255.0

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        raw = self._read_pickle()  # 原始数据

        if self.split not in raw:
            raise KeyError(f"split '{self.split}' not found, available: {list(raw.keys())}")

        instances = raw[self.split]
        # specify float32 right here: MNIST is stored as float64 and CIFAR would
        # otherwise be expanded to float64 first, which costs ~1.2 GB
        X = np.asarray([inst['image'] for inst in instances], dtype=np.float32)
        Y = np.asarray([inst['label'] for inst in instances], dtype=np.int64)

        X = self._to_nchw(X)
        X = self._normalize(X)
        Y, num_classes = self._shift_labels(Y, raw)

        print(f'[{self.dataset_name}] split={self.split} | X={X.shape} ({X.dtype}) | '
              f'Y={Y.shape} | classes={num_classes} | pixel_range=[{X.min():.3f}, {X.max():.3f}]')

        self.data = {'X': X, 'Y': Y}
        return self.data

    # ---- internals ---------------------------------------------------
    """把磁盘字节变成Python字典，返回原始结构 {'train': [{'image':…, 'label':…}, …], 'test': […]}"""
    """pickle 是 Python 内置序列化库，用来把之前用pickle.dump()存到文件里的 Python 对象（列表、字典、numpy 数组等）从文件还原回来。"""
    def _read_pickle(self):
        path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        print(f'loading data from {os.path.normpath(path)} ...')
        with open(path, 'rb') as f:     # 'rb' = read binary，以二进制只读模式打开文件,不能用普通文本模式 `r` 打开
            return pickle.load(f)

    """ 把 (N,H,W) 或 (N,H,W,C) 统一成 nn.Conv2d 要求的 (N,C,H,W) """
    """ N = size：图片/样本的数量         C = channels：通道数
        H = height：图像高度（像素行数）   W = width：图像宽度（像素列数）"""
    def _to_nchw(self, X):
        mode = self.channel_mode
        if mode == 'auto':
            mode = self._detect_channel_mode(X)

        if mode == 'gray':
            if X.ndim == 4:  # (N, H, W, C) -> keep one channel
                X = X[:, :, :, 0]
            return np.ascontiguousarray(X[:, None, :, :])  # (N, 1, H, W)

        if mode == 'rgb':
            if X.ndim != 4:
                raise ValueError(f"channel_mode='rgb' expects (N, H, W, C) images, got {X.shape}")
            return np.ascontiguousarray(X.transpose(0, 3, 1, 2))  # (N, 3, H, W)

        raise ValueError(f'unsupported channel_mode: {mode}')

    def _detect_channel_mode(self, X):
        if X.ndim == 3:  # (N, H, W): already single channel
            return 'gray'
        sample = X[:20]
        if np.array_equal(sample[..., 0], sample[..., 1]) and np.array_equal(sample[..., 0], sample[..., 2]):
            return 'gray'
        return 'rgb'

    """ 把 0–255 压到 0–1 """
    def _normalize(self, X):
        if self.pixel_scale is None or X.max() <= 1.0:
            return X
        return X / float(self.pixel_scale)

    """ 把标签值平移到 0..C-1，并算出类别数。CrossEntropyLoss 要求 target 落在 [0, C) """
    def _shift_labels(self, Y, raw):
        # the offset is computed over train+test so that both splits always share
        # the same label space (using the min of the current split alone is unsafe)
        all_labels = np.asarray([inst['label'] for split in raw.values() for inst in split], dtype=np.int64)
        offset = self.label_offset
        if offset == 'auto':    # 如果配置是 "auto"，自动用全部标签里的最小值作为偏移
            offset = int(all_labels.min())

        Y = Y - int(offset)
        # 计算平移之后理论上一共多少个类别 最大标签 - 偏移 +1 → 平移后：max_label - offset +1
        num_classes = int(all_labels.max()) - int(offset) + 1

        if Y.min() < 0:
            raise ValueError(f'negative labels after shifting by {offset}, check label_offset')
        if len(np.unique(Y)) != num_classes:
            print(f'warning: this split only covers {len(np.unique(Y))} of the {num_classes} classes')
        return Y, num_classes


# ---- self test: load all three datasets and print their shapes -------
# run from the project root:  python -m local_code.stage_3_code.Dataset_Loader
if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_folder = os.path.join(project_root, 'data', 'stage_3_data')

    settings = {
        'MNIST': {'channel_mode': 'gray'},
        'ORL': {'channel_mode': 'gray'},
        'CIFAR': {'channel_mode': 'rgb'},
    }

    for name, cfg in settings.items():
        for split in ('train', 'test'):
            loader = Dataset_Loader(name.lower(), '')
            loader.dataset_source_folder_path = data_folder
            loader.dataset_source_file_name = name
            loader.split = split
            for key, value in cfg.items():
                setattr(loader, key, value)
            loader.load()
