# _*_coding:utf-8 _*_
# Author  : Tao
"""
Some basic Code, for any Project
"""

import os


def mkdirs(paths):
    """
    Create folder
    :param paths: folder path 'str_list' or only a 'str' type path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
