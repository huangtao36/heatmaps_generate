# _*_coding:utf-8 _*_
# Author  : Tao
"""
Code for load UCF_Sport dataset
"""

import os
import torch.utils.data as data
from .data_utils import *


sport = ['Diving', 'Golf', 'Kicking',
         'Lifting', 'Riding', 'Run',
         'SkateBoarding', 'Swing1',
         'Swing2', 'Walk']      # len(sport) --> 10


def read_data_file(file_dir):

    lists = []
    with open(file_dir, 'r') as fp:
        line = fp.readline()
        while line:
            path = line.strip()
            lists.append(path)
            line = fp.readline()

    return lists


def list_process(lists_):

    train_list, test_list = [], []
    for i in lists_:
        i_split = i.split(' ')
        while '' in i_split:
            i_split.remove('')
        if 'avi' in i_split[6]:
            if i_split[2] == 'train':
                train_list.append(i_split)
            else:
                test_list.append(i_split)

    return train_list, test_list


class UCFSportData(data.Dataset):
    def __init__(self, train_or_test, data_root_path, transform=None):
        super(UCFSportData, self).__init__()
        self.data_root_path = data_root_path

        file = os.path.join(data_root_path, 'ucf_train_test_split.txt')
        self.train_list, self.test_list = list_process(read_data_file(file))

        self.path_list = self.train_list \
            if train_or_test == 'train' else self.test_list

        self.tranforms = transform

    def __getitem__(self, index_):
        category = self.path_list[index_][1]
        category_index = sport.index(category)
        category_index = torch.tensor(category_index)
        
        video_subpath = self.path_list[index_][6]
        video_path = os.path.join(self.data_root_path, video_subpath)
        sequence = read_video_sequance(video_path, 480, 280)

        return sequence, category_index

    def __len__(self):
        return len(self.path_list)
