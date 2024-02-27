import pickle
import cv2
import numpy as np
from tqdm import tqdm
import shutil


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


input_dir = './cityscale/'
output_root = './cityscale/'
output_dir = ''
dataset_image_size = 2048
size = 512
stride = 256

for i in tqdm(range(180)):
    sat = cv2.imread(input_dir + "region_%d_sat.png" % i)
    seg = cv2.imread(input_dir + "region_%d_gt.png" % i, 0)
    neighbors = neighbor_to_integer(pickle.load(open(input_dir + 'region_%d_refine_gt_graph.p' % i, 'rb')))

    svm = np.zeros((dataset_image_size, dataset_image_size, 3))
    dxy = np.zeros((dataset_image_size, dataset_image_size, 4))
    svm[:, :, 0] = seg
    for loc, n_locs in neighbors.items():
        svm[loc[0] - 1: loc[0] + 2, loc[1] - 1: loc[1] + 2, 1] = np.ones((3, 3)) * 255
        for n_loc in n_locs:
            mid_x = round((loc[0] + n_loc[0]) / 2)
            mid_y = round((loc[1] + n_loc[1]) / 2)
            svm[mid_x - 1: mid_x + 2, mid_y - 1: mid_y + 2, 2] = np.ones((3, 3)) * 255
            dx = loc[0] - mid_x
            dy = loc[1] - mid_y
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

    if i % 10 < 8:
        output_dir = output_root + 'train/'
    if i % 20 == 18:
        output_dir = output_root + 'valid/'
        shutil.copyfile(input_dir + "region_%d_sat.png" % i, output_dir + "region_%d_sat.png" % i)
        shutil.copyfile(input_dir + "region_%d_refine_gt_graph.p" % i, output_dir + "region_%d_refine_gt_graph.p" % i)
        cv2.imwrite(output_dir + 'region_%d_svm.png' % i, svm)
        continue
    if i % 20 == 8 or i % 10 == 9:
        output_dir = output_root + 'test/'
        shutil.copyfile(input_dir + "region_%d_sat.png" % i, output_dir + "region_%d_sat.png" % i)
        shutil.copyfile(input_dir + "region_%d_refine_gt_graph.p" % i, output_dir + "region_%d_refine_gt_graph.p" % i)
        cv2.imwrite(output_dir + 'region_%d_svm.png' % i, svm)
        continue

    maxx = int((dataset_image_size - size) / stride)
    maxy = int((dataset_image_size - size) / stride)
    for x in range(maxx + 1):
        for y in range(maxy + 1):
            sat_block = sat[x * stride:x * stride + size, y * stride:y * stride + size, :]
            svm_block = svm[x * stride:x * stride + size, y * stride:y * stride + size, :]
            dxy_block = dxy[x * stride:x * stride + size, y * stride:y * stride + size, :]
            cv2.imwrite(output_dir + '{}_{}_{}_sat.png'.format(i, x, y), sat_block)
            cv2.imwrite(output_dir + '{}_{}_{}_svm.png'.format(i, x, y), svm_block)
            pickle.dump(dxy_block, open(output_dir + '{}_{}_{}_dxy.pkl'.format(i, x, y), 'wb'))
