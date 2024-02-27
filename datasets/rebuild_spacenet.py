import cv2
import pickle
import json
import numpy as np


def neighbor_to_integer(n_in):
    n_out = {}
    for k, v in n_in.items():
        nk = (int(k[0]), int(k[1]))
        if nk in n_out:
            nv = n_out[nk]
        else:
            nv = []
        for _v in v:
            new_n_k = (int(_v[0]), int(_v[1]))
            if new_n_k in nv:
                pass
            else:
                nv.append(new_n_k)
        n_out[nk] = nv
    return n_out


dataset = json.load(open('./spacenet/dataset.json', 'r'))
dataset_image_size = 400
rebuild_image_size = 512

for item in dataset['train']:
    sat = cv2.resize(cv2.flip(cv2.imread('./spacenet/' + item + '__rgb.png'), 0), (rebuild_image_size, rebuild_image_size))
    neighbors = neighbor_to_integer(pickle.load(open('./spacenet/' + item + '__gt_graph_dense.p', 'rb')))

    seg = np.zeros((rebuild_image_size, rebuild_image_size))
    svm = np.zeros((rebuild_image_size, rebuild_image_size, 3))
    dxy = np.zeros((rebuild_image_size, rebuild_image_size, 4))
    for loc, n_locs in neighbors.items():
        x0 = round(loc[0] * rebuild_image_size / dataset_image_size)
        y0 = round(loc[1] * rebuild_image_size / dataset_image_size)
        if x0 - 1 < 0 or x0 + 2 > rebuild_image_size or y0 - 1 < 0 or y0 + 2 > rebuild_image_size:
            continue
        svm[x0 - 1: x0 + 2, y0 - 1: y0 + 2, 1] = np.ones((3, 3)) * 255
        for n_loc in n_locs:
            x = round(n_loc[0] * rebuild_image_size / dataset_image_size)
            y = round(n_loc[1] * rebuild_image_size / dataset_image_size)
            if x - 1 < 0 or x + 2 > rebuild_image_size or y - 1 < 0 or y + 2 > rebuild_image_size:
                continue
            cv2.line(seg, (y0, x0), (y, x), 255, 2)
            mid_x = round((x0 + x) / 2)
            mid_y = round((y0 + y) / 2)
            svm[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 2] = np.ones((3, 3)) * 255
            dx = x0 - mid_x
            dy = y0 - mid_y
            if dx < 0:
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 0] = np.ones((3, 3)) * dx
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 1] = np.ones((3, 3)) * dy
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 2] = - np.ones((3, 3)) * dx
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 3] = - np.ones((3, 3)) * dy
            elif dx > 0:
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 0] = - np.ones((3, 3)) * dx
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 1] = - np.ones((3, 3)) * dy
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 2] = np.ones((3, 3)) * dx
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 3] = np.ones((3, 3)) * dy
            elif dx == 0 and dy < 0:
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 0] = np.ones((3, 3)) * dx
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 1] = np.ones((3, 3)) * dy
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 2] = - np.ones((3, 3)) * dx
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 3] = - np.ones((3, 3)) * dy
            elif dx == 0 and dy > 0:
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 0] = - np.ones((3, 3)) * dx
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 1] = - np.ones((3, 3)) * dy
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 2] = np.ones((3, 3)) * dx
                dxy[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 3] = np.ones((3, 3)) * dy

    svm[:, :, 0] = seg
    cv2.imwrite('./spacenet/train/' + '{}_sat.png'.format(item), sat)
    cv2.imwrite('./spacenet/train/' + '{}_svm.png'.format(item), svm)
    pickle.dump(dxy, open('./spacenet/train/' + '{}_dxy.pkl'.format(item), 'wb'))


for item in dataset['validation']:
    sat = cv2.resize(cv2.flip(cv2.imread('./spacenet/' + item + '__rgb.png'), 0), (rebuild_image_size, rebuild_image_size))
    neighbors = neighbor_to_integer(pickle.load(open('./spacenet/' + item + '__gt_graph_dense.p', 'rb')))
    rebuild_neighbors = {}

    seg = np.zeros((rebuild_image_size, rebuild_image_size))
    svm = np.zeros((rebuild_image_size, rebuild_image_size, 3))
    for loc, n_locs in neighbors.items():
        x0 = round(loc[0] * rebuild_image_size / dataset_image_size)
        y0 = round(loc[1] * rebuild_image_size / dataset_image_size)
        if x0 - 1 < 0 or x0 + 2 > rebuild_image_size or y0 - 1 < 0 or y0 + 2 > rebuild_image_size:
            continue
        svm[x0 - 1: x0 + 2, y0 - 1: y0 + 2, 1] = np.ones((3, 3)) * 255
        rebuild_neighbors[(x0, y0)] = []
        for n_loc in n_locs:
            x = round(n_loc[0] * rebuild_image_size / dataset_image_size)
            y = round(n_loc[1] * rebuild_image_size / dataset_image_size)
            if x - 1 < 0 or x + 2 > rebuild_image_size or y - 1 < 0 or y + 2 > rebuild_image_size:
                continue
            cv2.line(seg, (y0, x0), (y, x), 255, 2)
            mid_x = round((x0 + x) / 2)
            mid_y = round((y0 + y) / 2)
            svm[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 2] = np.ones((3, 3)) * 255
            rebuild_neighbors[(x0, y0)].append((x, y))

    svm[:, :, 0] = seg
    cv2.imwrite('./spacenet/valid/' + '{}_sat.png'.format(item), sat)
    cv2.imwrite('./spacenet/valid/' + '{}_svm.png'.format(item), svm)
    pickle.dump(rebuild_neighbors, open('./spacenet/valid/' + '{}__gt_graph_rebuild.p'.format(item), 'wb'))


for item in dataset['test']:
    sat = cv2.resize(cv2.flip(cv2.imread('./spacenet/' + item + '__rgb.png'), 0), (rebuild_image_size, rebuild_image_size))
    neighbors = neighbor_to_integer(pickle.load(open('./spacenet/' + item + '__gt_graph_dense.p', 'rb')))
    rebuild_neighbors = {}

    seg = np.zeros((rebuild_image_size, rebuild_image_size))
    svm = np.zeros((rebuild_image_size, rebuild_image_size, 3))
    for loc, n_locs in neighbors.items():
        x0 = round(loc[0] * rebuild_image_size / dataset_image_size)
        y0 = round(loc[1] * rebuild_image_size / dataset_image_size)
        if x0 - 1 < 0 or x0 + 2 > rebuild_image_size or y0 - 1 < 0 or y0 + 2 > rebuild_image_size:
            continue
        svm[x0 - 1: x0 + 2, y0 - 1: y0 + 2, 1] = np.ones((3, 3)) * 255
        rebuild_neighbors[(x0, y0)] = []
        for n_loc in n_locs:
            x = round(n_loc[0] * rebuild_image_size / dataset_image_size)
            y = round(n_loc[1] * rebuild_image_size / dataset_image_size)
            if x - 1 < 0 or x + 2 > rebuild_image_size or y - 1 < 0 or y + 2 > rebuild_image_size:
                continue
            cv2.line(seg, (y0, x0), (y, x), 255, 2)
            mid_x = round((x0 + x) / 2)
            mid_y = round((y0 + y) / 2)
            svm[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 2] = np.ones((3, 3)) * 255
            rebuild_neighbors[(x0, y0)].append((x, y))

    svm[:, :, 0] = seg
    cv2.imwrite('./spacenet/test/' + '{}_sat.png'.format(item), sat)
    cv2.imwrite('./spacenet/test/' + '{}_svm.png'.format(item), svm)
    pickle.dump(rebuild_neighbors, open('./spacenet/test/' + '{}__gt_graph_rebuild.p'.format(item), 'wb'))
