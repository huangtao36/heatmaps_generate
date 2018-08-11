# _*_coding:utf-8 _*_
# Author  : Tao
"""
This Code is to generate heatmaps use a model pretrain by Cao's PAFs paper
(Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields)\

run: python heatmap_generate.py --dataset xxx --gpu 0_or_1
"""


import torch.utils.data as data
import torch
import argparse
import os
from dataload import ucf_sport_dataloader as ucfs
from dataload import ucf_101_dataloader as ucf1
from dataload import penn_action_dataloader as penn
from model import heat_vec_network
from utils import util

parser = argparse.ArgumentParser(description='Heatmaps Generate')
parser.add_argument('--dataset', type=str, default='ucfsport')
parser.add_argument('--save_dir', type=str, default='heatmaps',
                    help='path to save generated heatmaps')
parser.add_argument("--gpu", type=int, default=1,
                    help='number of GPU')

opt = parser.parse_args()

device = torch.device('cuda: {}'.format(opt.gpu)) \
    if torch.cuda.is_available else torch.device('cpu')

project_par_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
save_train_path = os.path.join(project_par_path, 'processed_dataset', 
                               opt.dataset, opt.save_dir, "train")
save_test_path = os.path.join(project_par_path, 'processed_dataset', 
                              opt.dataset, opt.save_dir, "test")

util.mkdirs([save_train_path, save_test_path])


def load_data(train_or_test):
    dataset = None

    if opt.dataset == 'ucfsport':
        data_root_path = os.path.join(project_par_path,
                                      'source_dataset/UCFSport')
        dataset = ucfs.UCFSportData(train_or_test, data_root_path)
    elif opt.dataset == 'ucf101':
        data_root_path = os.path.join(project_par_path,
                                      'source_dataset/UCF101')
        dataset = ucf1.UCF101Data(train_or_test, data_root_path)
    elif opt.dataset == 'pennactive':
        data_root_path = os.path.join(project_par_path,
                                      'source_dataset/Penn_Action')
        dataset = penn.PennActionData(train_or_test, data_root_path)
    else:
        assert "Dataset Wrong!"

    dataload = data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available)

    return dataload


def load_model(pth_path):
    model = heat_vec_network.define_net()
    model = model.to(device)
    model.load_state_dict(torch.load(pth_path))
    model.eval()

    return model


def process(data_, train_or_test, model):
    save_path = save_train_path \
        if train_or_test == 'train' else save_test_path
    print('save_path: ', save_path)

    for i, (video_seq, category_index) in enumerate(data_):
        print('\r{0}: {1}/{2}'.format(train_or_test, i + 1, len(data_)),
              end='', flush=True)

        heatmaps = []
        for n in range(len(video_seq)):
            img_tensor = video_seq[n]
            img_tensor = img_tensor.to(device)
            vecmap, heatmap = model(img_tensor)
            heatmaps.append(heatmap[0, :, :, :].data.cpu())

        torch.save((heatmaps, category_index.data),
                   os.path.join(save_path, 'heatmaps_%d.c' % i))
    print('\n')


if __name__ == '__main__':
    data_tr = load_data('train')
    data_te = load_data('test')
    print("train: {}, test: {}".format(len(data_tr), len(data_te)))
    heat_model = load_model('./pose_model.pth')

    process(data_tr, 'train', heat_model)
    process(data_te, 'test', heat_model)
    print("Done!")
