# _*_coding:utf-8 _*_
# Author  : Tao
"""
some basic code for data processing
"""

import torch
import cv2


def normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def to_tensor(pic):
    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    return img.float()


def read_video_sequance(video_path, video_w, video_h):
    """
    read video to image sequence

    :param video_path: video path
    :param get_frames: int, Number of frames to get
    :param video_w: image width resize
    :param video_h: image height resize
    :return: video sequance with tensor type, normalize readied
    """

    video_capture = cv2.VideoCapture(video_path)
    frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)  # 帧数

    seq = []
    for i in range(int(frames)):
        """
        Loading 500-frames image at one time may overflow
        OSError: [Errno 24] Too many open files
        """
        n = i
        if frames > 400:    # Take the middle 400 frames
            n = i + int((frames - 400)/2)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, n)
            if i > 400:
                continue
        else:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, n)
        bool_, img = video_capture.read()
        if bool_ is True:
            resize_img = cv2.resize(img, (video_w, video_h))
            tensor_img = normalize(to_tensor(resize_img),
                                   [128.0, 128.0, 128.0],
                                   [256.0, 256.0, 256.0])
            seq.append(tensor_img[:, :, :])

    return seq
