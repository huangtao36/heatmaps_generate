# _*_coding:utf-8 _*_
# Author  : Tao
"""
Code for load Penn Active dataset
"""

import os
from scipy.io import loadmat
import collections
import torch.utils.data as data
from .data_utils import *


def action_statistical(labels_path, num):
    class_list = []
    train_or_test_list = []
    for i in range(num):
        mat_path = os.path.join(labels_path, "%04d.mat" % (i + 1))
        data_ = loadmat(mat_path)
        action = str(data_['action'][0])
        train_or_test = int(data_['train'][0][0])  # train: 1 or test: -1

        class_list.append(action)
        train_or_test_list.append(train_or_test)

    class_counter = collections.Counter(class_list)
    split_counter = collections.Counter(train_or_test_list)

    dict_split = dict(split_counter)
    dict_split['train'] = dict_split.pop(int(1))
    dict_split['test'] = dict_split.pop(int(-1))

    return dict(class_counter), dict_split


def image_category_list(sign, labels_path, video_dir_path, num):

    image_path_category_list = []  # list for save image path and its category
    for i in range(num):
        cache = []
        mat_path = os.path.join(labels_path, "%04d.mat" % (i + 1))
        ann_data = loadmat(mat_path)
        action_category = ann_data['action'][0]

        frames_path = os.path.join(video_dir_path, "%04d" % (i + 1))

        if sign == int(ann_data['train'][0][0]):
            cache.append(frames_path)
            cache.append(action_category)
            image_path_category_list.append(cache)
        else:
            continue

    return image_path_category_list


def get_seq(path):
    image_name_list = os.listdir(path)
    frames = len(image_name_list)
    image_list = []
    for i in range(frames):
        n = i + 1
        if frames > 400:
            n = i + 1 + int((frames - 400)/2)
            image_path = os.path.join(path, "%06d.jpg" % n)
            if i > 400:
                continue
        else:
            image_path = os.path.join(path, "%06d.jpg" % n)
        image = cv2.imread(image_path)
        resize_img = cv2.resize(image, (480, 320), interpolation=cv2.INTER_LINEAR)
        tensor_img = normalize(to_tensor(resize_img),
                                   [128.0, 128.0, 128.0],
                                   [256.0, 256.0, 256.0])
        image_list.append(tensor_img[:, :, :])
    
    return image_list


class PennActionData(data.Dataset):
    def __init__(self, train_or_test, data_root_path):
        super(PennActionData, self).__init__()

        self.video_dir_path = os.path.join(data_root_path, 'frames')
        self.labels_path = os.path.join(data_root_path, 'labels')
        self.num = len(os.listdir(self.labels_path))
        
        self.category_stat, self.split = action_statistical(
            self.labels_path,
            self.num
        )

        if train_or_test == 'train':
            self.label = 1
        else:
            self.label = -1
        
        self.image_list = image_category_list(
            self.label,
            self.labels_path,
            self.video_dir_path,
            self.num
        )

        self.classify_Action = {'baseball_pitch': 0, 'baseball_swing': 1,
                                'bench_press': 2, 'bowl': 3,
                                'clean_and_jerk': 4, 'golf_swing': 5,
                                'jump_rope': 6, 'jumping_jacks': 7,
                                'pullup': 8, 'pushup': 9, 'situp': 10,
                                'squat': 11, 'strum_guitar': 12,
                                'tennis_forehand': 13, 'tennis_serve': 14}

    def __getitem__(self, index_):
        image_path = self.image_list[index_][0]
        category = self.image_list[index_][1]
        category_index = int(self.classify_Action.get(category))
        seq = get_seq(image_path)

        return seq, category_index

    def __len__(self):
        return len(self.image_list)
