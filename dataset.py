import os
import glob
import numpy as np
import cv2
import pickle
import torch
import torch.utils.data as data


def random_hue_saturation_value(sat, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255), val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        sat = cv2.cvtColor(sat, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(sat)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        sat = cv2.merge((h, s, v))
        sat = cv2.cvtColor(sat, cv2.COLOR_HSV2BGR)
    return sat


def random_shift_scale_rotate(sat, svm, dxy, shift_limit=(-0.0, 0.0), scale_limit=(-0.0, 0.0), rotate_limit=(-0.0, 0.0), aspect_limit=(-0.0, 0.0), border_mode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = sat.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.cos(angle / 180 * np.pi) * sx
        ss = np.sin(angle / 180 * np.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        sat = cv2.warpPerspective(sat, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(0, 0, 0,))
        svm = cv2.warpPerspective(svm, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(0, 0, 0,))
        dxy = cv2.warpPerspective(dxy * scale, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=border_mode, borderValue=(0, 0, 0,))
    return sat, svm, dxy


def random_horizontal_flip(sat, svm, dxy, u=0.5):
    if np.random.random() < u:
        sat = np.flip(sat, 1)
        svm = np.flip(svm, 1)
        dxy = np.flip(dxy, 1)
        dxy[:, :, 1] = - dxy[:, :, 1]
        dxy[:, :, 3] = - dxy[:, :, 3]
        tmp = dxy[:, :, [2, 3, 0, 1]]
        dxy[dxy[:, :, 0] == 0, :] = tmp[dxy[:, :, 0] == 0, :]
    return sat, svm, dxy


def random_vertical_flip(sat, svm, dxy, u=0.5):
    if np.random.random() < u:
        sat = np.flip(sat, 0)
        svm = np.flip(svm, 0)
        dxy = np.flip(dxy, 0)
        dxy[:, :, 0] = - dxy[:, :, 0]
        dxy[:, :, 2] = - dxy[:, :, 2]
        tmp = dxy[:, :, [2, 3, 0, 1]]
        dxy[dxy[:, :, 0] != 0, :] = tmp[dxy[:, :, 0] != 0, :]
    return sat, svm, dxy


def random_rotate_90(sat, svm, dxy, u=0.5):
    if np.random.random() < u:
        svm = np.rot90(svm)
        sat = np.rot90(sat)
        dxy = np.rot90(dxy)
        dxy[:, :, 1] = - dxy[:, :, 1]
        dxy[:, :, 3] = - dxy[:, :, 3]
        dxy = dxy[:, :, [1, 0, 3, 2]]
        tmp = dxy[:, :, [2, 3, 0, 1]]
        dxy[dxy[:, :, 0] > 0, :] = tmp[dxy[:, :, 0] > 0, :]
        dxy[(dxy[:, :, 0] == 0) * (dxy[:, :, 1] > 0), :] = tmp[(dxy[:, :, 0] == 0) * (dxy[:, :, 1] > 0), :]
    return sat, svm, dxy


class TopoRoadDataset(data.Dataset):
    def __init__(self, root_dir):
        self.sample_list = list(map(lambda x: x[:-8], glob.glob(root_dir + '*_sat.png')))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, item):
        sat = cv2.imread(os.path.join('{}_sat.png').format(self.sample_list[item]))
        svm = cv2.imread(os.path.join('{}_svm.png').format(self.sample_list[item]))
        dxy = pickle.load(open(os.path.join('{}_dxy.pkl').format(self.sample_list[item]), 'rb'))

        sat = random_hue_saturation_value(sat, hue_shift_limit=(-30, 30), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15))
        sat, svm, dxy = random_shift_scale_rotate(sat, svm, dxy, shift_limit=(-0.1, 0.1), scale_limit=(-0, 0), aspect_limit=(-0, 0), rotate_limit=(-0, 0))
        sat, svm, dxy = random_horizontal_flip(sat, svm, dxy)
        sat, svm, dxy = random_vertical_flip(sat, svm, dxy)
        sat, svm, dxy = random_rotate_90(sat, svm, dxy)

        sat = torch.Tensor(np.array(sat, np.float32).transpose((2, 0, 1))) / 255.0 * 3.2 - 1.6
        seg = torch.Tensor(np.array(np.expand_dims(svm[:, :, 0], 2), np.float32).transpose((2, 0, 1))) / 255.0
        ver = torch.Tensor(np.array(np.expand_dims(svm[:, :, 1], 2), np.float32).transpose((2, 0, 1))) / 255.0
        mid = torch.Tensor(np.array(np.expand_dims(svm[:, :, 2], 2), np.float32).transpose((2, 0, 1))) / 255.0
        dxy = torch.Tensor(np.array(dxy, np.float32).transpose((2, 0, 1))) / 12.0

        return {'sat': sat, 'seg': seg, 'ver': ver, 'mid': mid, 'dxy': dxy}
