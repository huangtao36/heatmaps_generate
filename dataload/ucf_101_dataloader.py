# _*_coding:utf-8 _*_
# Author  : Tao
"""
Code for load UCF101 dataset
"""

import os
import torch.utils.data as data
from .data_utils import *


class UCF101Data(data.Dataset):
    def __init__(self, train_or_test, data_root_path, transform=None):
        super(UCF101Data, self).__init__()

        self.video_dir_path = os.path.join(data_root_path, 'UCF-101')
        txt_path = os.path.join(data_root_path, 'ucfTrainTestlist')

        splitter = UCF101Splitter(txt_path, '01')
        self.train_video, self.test_video = splitter.split_video()

        self.path_and_label_list = self.train_video \
            if train_or_test == 'train' else self.test_video

        self.tranforms = transform

    def __getitem__(self, index_):
        video_subpath = self.path_and_label_list[index_][0]
        video_path = os.path.join(self.video_dir_path, video_subpath)

        category_index = self.path_and_label_list[index_][1]
        category_index = torch.tensor(int(category_index) - 1)

        sequence = read_video_sequance(video_path, 480, 280)

        return sequence, category_index

    def __len__(self):
        return len(self.path_and_label_list)


class UCF101Splitter:
    def __init__(self, path_, split_):
        self.path = path_
        self.split = split_
        self.action_label = {}
        self.train_video = []
        self.test_video = []

    def get_action_index(self):

        with open(os.path.join(self.path, 'classInd.txt')) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        for line in content:
            label, action = line.split(' ')
            if action not in self.action_label.keys():
                self.action_label[action] = label

    def split_video(self):
        self.get_action_index()

        for path1, subdir, files in os.walk(self.path):
            for filename in files:
                if filename.split('.')[0] == 'trainlist' + self.split:
                    # trainlist01
                    self.train_video = self.file2_dic(
                        os.path.join(self.path, filename))
                if filename.split('.')[0] == 'testlist' + self.split:
                    # testlist01
                    self.test_video = self.file2_dic(
                        os.path.join(self.path, filename))

        # print('==> Training_video: {}, Test_video: {}'.format(
        #     len(self.train_video), len(self.test_video)))

        return self.train_video, self.test_video

    def file2_dic(self, fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        data_list = []
        for line in content:
            data_sublist = []
            video_path = line.split(' ', 1)[0]
            label = self.action_label[line.split('/')[0]]
            data_sublist.append(video_path)
            data_sublist.append(label)
            data_list.append(data_sublist)
        return data_list
