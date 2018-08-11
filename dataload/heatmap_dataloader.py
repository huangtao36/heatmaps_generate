# _*_coding:utf-8 _*_
# Author  : Tao
"""
Code for load heatmaps data
"""

import torch.utils.data as data
import os
import torch


class HeatmapsData(data.Dataset):
    def __init__(self, train_or_test, data_path):
        super(HeatmapsData, self).__init__()
        if train_or_test == 'train':
            self.data_dir = os.path.join(data_path, 'train')
        else:
            self.data_dir = os.path.join(data_path, 'test')

        self.train_files = os.listdir(self.data_dir)

    def __getitem__(self, index_):
        data_ = os.path.join(self.data_dir, self.train_files[index_])
        heatmaps, category_index = torch.load(data_)

        return heatmaps, category_index

    def __len__(self):
        return len(self.train_files)
