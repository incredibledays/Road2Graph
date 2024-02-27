import numpy as np
import cv2
import os
import pickle
import math
import json
from rtree import index
import scipy
import scipy.ndimage.morphology as morphology
import scipy.ndimage.filters as filters
import torch
from torch.autograd import Variable as V
from douglasPeucker import simplifyGraph


def distance(p1, p2):
    a = p1[0] - p2[0]
    b = p1[1] - p2[1]
    return np.sqrt(a * a + b * b)


def vNorm(v1):
    l = distance(v1,(0,0))+0.0000001
    return v1[0] / l, v1[1] / l


def anglediff(v1, v2):
    v1 = vNorm(v1)
    v2 = vNorm(v2)
    return v1[0]*v2[0] + v1[1] * v2[1]


def cosine_similarity(k1, k2, k3):
    vec1 = distance_norm(k2, k1)
    vec2 = distance_norm(k3, k1)
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]


def distance_norm(k1, k2):
    l = distance(k1, k2)
    a = k1[0] - k2[0]
    b = k1[1] - k2[1]
    return a/l, b/l


def point2lineDistance(p, n1, n2):
    length = distance(n1, n2)

    v1 = [n1[0] - p[0], n1[1] - p[1]]
    v2 = [n2[0] - p[0], n2[1] - p[1]]

    area = abs(v1[0] * v2[1] - v1[1] * v2[0])

    if n1 == n2:
        return 0

    return area / length


def graph_refine(graph, isolated_thr=150, spurs_thr=30):
    neighbors = graph

    gid = 0
    grouping = {}

    for k, v in neighbors.items():
        if k not in grouping:
            # start a search
            queue = [k]

            while len(queue) > 0:
                n = queue.pop(0)

                if n not in grouping:
                    grouping[n] = gid
                    for nei in neighbors[n]:
                        queue.append(nei)

            gid += 1

    group_count = {}

    for k, v in grouping.items():
        if v not in group_count:
            group_count[v] = (1, 0)
        else:
            group_count[v] = (group_count[v][0] + 1, group_count[v][1])

        for nei in neighbors[k]:
            a = k[0] - nei[0]
            b = k[1] - nei[1]

            d = np.sqrt(a * a + b * b)

            group_count[v] = (group_count[v][0], group_count[v][1] + d / 2)

    # short spurs
    remove_list = []
    for k, v in neighbors.items():
        if len(v) == 1:
            if len(neighbors[v[0]]) >= 3:
                a = k[0] - v[0][0]
                b = k[1] - v[0][1]

                d = np.sqrt(a * a + b * b)

                if d < spurs_thr:
                    remove_list.append(k)

    remove_list2 = []
    remove_counter = 0
    new_neighbors = {}

    def isRemoved(k):
        gid = grouping[k]
        if group_count[gid][0] <= 1:
            return True
        elif group_count[gid][1] <= isolated_thr:
            return True
        elif k in remove_list:
            return True
        elif k in remove_list2:
            return True
        else:
            return False

    for k, v in neighbors.items():
        if isRemoved(k):
            remove_counter += 1
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    return new_neighbors


def graph_shave(graph, spurs_thr=50):
    neighbors = graph

    # short spurs
    remove_list = []
    for k, v in neighbors.items():
        if len(v) == 1:
            d = distance(k, v[0])
            cur = v[0]
            l = [k]
            while True:
                if len(neighbors[cur]) >= 3:
                    break
                elif len(neighbors[cur]) == 1:
                    l.append(cur)
                    break

                else:

                    if neighbors[cur][0] == l[-1]:
                        next_node = neighbors[cur][1]
                    else:
                        next_node = neighbors[cur][0]

                    d += distance(cur, next_node)
                    l.append(cur)

                    cur = next_node

            if d < spurs_thr:
                for n in l:
                    if n not in remove_list:
                        remove_list.append(n)

    def isRemoved(k):
        if k in remove_list:
            return True
        else:
            return False

    new_neighbors = {}
    remove_counter = 0

    for k, v in neighbors.items():
        if isRemoved(k):
            remove_counter += 1
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    # print("shave", len(new_neighbors), "remove", remove_counter, "nodes")

    return new_neighbors


def graph_refine_deloop(neighbors, max_step=10, max_length=200, max_diff=5):
    removed = []
    impact = []

    remove_edge = []
    new_edge = []

    for k, v in neighbors.items():
        if k in removed:
            continue

        if k in impact:
            continue

        if len(v) < 2:
            continue

        for nei1 in v:
            if nei1 in impact:
                continue

            if k in impact:
                continue

            for nei2 in v:
                if nei2 in impact:
                    continue
                if nei1 == nei2:
                    continue

                if cosine_similarity(k, nei1, nei2) > 0.984:
                    l1 = distance(k, nei1)
                    l2 = distance(k, nei2)

                    # print("candidate!", l1,l2,neighbors_cos(neighbors, k, nei1, nei2))

                    if l2 < l1:
                        nei1, nei2 = nei2, nei1

                    remove_edge.append((k, nei2))
                    remove_edge.append((nei2, k))

                    new_edge.append((nei1, nei2))

                    impact.append(k)
                    impact.append(nei1)
                    impact.append(nei2)

                    break

    new_neighbors = {}

    def isRemoved(k):
        if k in removed:
            return True
        else:
            return False

    for k, v in neighbors.items():
        if isRemoved(k):
            pass
        else:
            new_nei = []
            for nei in v:
                if isRemoved(nei):
                    pass
                elif (nei, k) in remove_edge:
                    pass
                else:
                    new_nei.append(nei)

            new_neighbors[k] = list(new_nei)

    for new_e in new_edge:
        nk1 = new_e[0]
        nk2 = new_e[1]

        if nk2 not in new_neighbors[nk1]:
            new_neighbors[nk1].append(nk2)
        if nk1 not in new_neighbors[nk2]:
            new_neighbors[nk2].append(nk1)

    # print("remove %d edges" % len(remove_edge))

    return new_neighbors, len(remove_edge)


def detect_local_minima(arr, mask, threshold=0.5):
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood) == arr)
    background = (arr == 0)
    eroded_background = morphology.binary_erosion(background, structure=neighborhood, border_value=1)
    detected_minima = local_min ^ eroded_background
    return np.where((detected_minima & (mask > threshold)))


def line_pooling(src, x0, y0, x, y):
    step = round(math.sqrt((x - x0) ** 2 + (y - y0) ** 2))
    sample = np.linspace(np.array([x0, y0]), np.array([x, y]), step, dtype=int)
    mean = np.mean(src[sample[round(step / 4):round(3 * step / 4), 0], sample[round(step / 4):round(3 * step / 4), 1]])
    return mean


def decodingc(ver_map, mid_map, dxy_map, seg_map, thr=0.05, rad=100):
    smooth_ver = scipy.ndimage.filters.gaussian_filter(ver_map, 1)
    smooth_ver = smooth_ver / max(np.amax(smooth_ver), 0.001)
    vertices = detect_local_minima(-smooth_ver, smooth_ver, thr)

    ver_idx = index.Index()
    for i in range(len(vertices[0])):
        x0, y0 = vertices[0][i], vertices[1][i]
        if len(list(ver_idx.intersection((x0 - 5, y0 - 5, x0 + 5, y0 + 5)))) == 0:
            ver_idx.insert(i, (x0 - 1, y0 - 1, x0 + 1, y0 + 1))

        for j in range(len(vertices[0])):
            x, y = vertices[0][j], vertices[1][j]
            if abs(x - x0) < 5 and abs(y - y0) < 5:
                continue
            if abs(x - x0) > rad or abs(y - y0) > rad:
                continue

            line_mean = line_pooling(seg_map, x0, y0, x, y)
            if line_mean < 0.5:
                continue

            xm, ym = round((x0 + x) / 2), round((y0 + y) / 2)
            mid_map[xm, ym] = 1

    smooth_mid = scipy.ndimage.filters.gaussian_filter(mid_map, 1)
    smooth_mid = smooth_mid / max(np.amax(smooth_mid), 0.001)
    middle_points = detect_local_minima(-smooth_mid, smooth_mid, thr)

    neighbors = {}
    for i in range(len(middle_points[0])):
        x0 = middle_points[0][i]
        y0 = middle_points[1][i]

        best_candidate = -1
        min_distance = 15
        x1 = round(dxy_map[x0, y0, 0] + x0)
        y1 = round(dxy_map[x0, y0, 1] + y0)
        candidates = list(ver_idx.intersection((x1 - 20, y1 - 20, x1 + 20, y1 + 20)))
        for candidate in candidates:
            x_c = vertices[0][candidate]
            y_c = vertices[1][candidate]
            d = distance((x_c, y_c), (x1, y1))

            v1 = (x_c - x0, y_c - y0)
            v2 = (x1 - x0, y1 - y0)
            ad = 1.0 - anglediff(v1, v2)
            d = d + ad * 50
            if d < min_distance:
                min_distance = d
                best_candidate = candidate

        if best_candidate != -1:
            x1 = vertices[0][best_candidate]
            y1 = vertices[1][best_candidate]
        else:
            continue

        best_candidate = -1
        min_distance = 15
        x2 = round(dxy_map[x0, y0, 2] + x0)
        y2 = round(dxy_map[x0, y0, 3] + y0)
        candidates = list(ver_idx.intersection((x2 - 20, y2 - 20, x2 + 20, y2 + 20)))
        for candidate in candidates:
            x_c = vertices[0][candidate]
            y_c = vertices[1][candidate]
            d = distance((x_c, y_c), (x2, y2))

            v1 = (x_c - x0, y_c - y0)
            v2 = (x2 - x0, y2 - y0)
            ad = 1.0 - anglediff(v1, v2)
            d = d + ad * 50
            if d < min_distance:
                min_distance = d
                best_candidate = candidate

        if best_candidate != -1:
            x2 = vertices[0][best_candidate]
            y2 = vertices[1][best_candidate]
        else:
            continue

        if (x1, y1) in neighbors:
            if (x2, y2) in neighbors[(x1, y1)]:
                pass
            else:
                neighbors[(x1, y1)].append((x2, y2))
        else:
            neighbors[(x1, y1)] = [(x2, y2)]

        if (x2, y2) in neighbors:
            if (x1, y1) in neighbors[(x2, y2)]:
                pass
            else:
                neighbors[(x2, y2)].append((x1, y1))
        else:
            neighbors[(x2, y2)] = [(x1, y1)]

    spurs_thr = 50
    isolated_thr = 200
    graph = graph_refine(neighbors, isolated_thr=isolated_thr, spurs_thr=spurs_thr)
    rc = 100
    while rc > 0:
        graph, rc = graph_refine_deloop(graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr))
    graph = graph_shave(graph, spurs_thr=spurs_thr)
    # graph = neighbors
    return graph


def infer_cityscale():
    from extractor import Extractor
    from network import TopoDLATransformer as Net

    input_dir = './datasets/cityscale/test/'
    output_dir = './results/cityscale/'
    weight_dir = './checkpoints/cityscale/best.th'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = Extractor(Net, eval_mode=True)
    model.load(weight_dir)

    dataset_image_size = 2560
    size = 512
    stride = 256

    maxx = int((dataset_image_size - size) / stride)
    maxy = int((dataset_image_size - size) / stride)

    for i in range(180):
        if i % 10 < 8 or i % 20 == 18:
            continue

        sat_ori = cv2.imread(input_dir + "region_%d_sat.png" % i)
        sat = np.pad(sat_ori, ((stride, stride), (stride, stride), (0, 0)), 'reflect')
        seg_pre = np.zeros((dataset_image_size, dataset_image_size))
        ver_pre = np.zeros((dataset_image_size, dataset_image_size))
        mid_pre = np.zeros((dataset_image_size, dataset_image_size))
        dxy_pre = np.zeros((dataset_image_size, dataset_image_size, 4))
        for x in range(maxx + 1):
            for y in range(maxy + 1):
                sat_img = np.array(sat[x * stride:x * stride + size, y * stride:y * stride + size, :], np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
                sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
                pre = model.predict(sat_img)
                seg = torch.sigmoid(pre['seg']).squeeze().cpu().data.numpy()
                ver = torch.sigmoid(pre['ver']).squeeze().cpu().data.numpy()
                mid = torch.sigmoid(pre['mid']).squeeze().cpu().data.numpy()
                dxy = pre['dxy'].squeeze().cpu().data.numpy().transpose((1, 2, 0)) * 12
                pre = None
                sat_img = np.flip(np.array(sat[x * stride:x * stride + size, y * stride:y * stride + size, :], np.float32), 0).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
                sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
                pre = model.predict(sat_img)
                seg_v = torch.sigmoid(pre['seg']).squeeze().cpu().data.numpy()
                pre = None
                sat_img = np.flip(np.array(sat[x * stride:x * stride + size, y * stride:y * stride + size, :], np.float32), 1).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
                sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
                pre = model.predict(sat_img)
                seg_h = torch.sigmoid(pre['seg']).squeeze().cpu().data.numpy()
                pre = None
                sat_img = np.rot90(np.array(sat[x * stride:x * stride + size, y * stride:y * stride + size, :], np.float32)).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
                sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
                pre = model.predict(sat_img)
                seg_r = torch.sigmoid(pre['seg']).squeeze().cpu().data.numpy()
                pre = None
                seg = np.clip(seg + np.flip(seg_v, 0) + np.flip(seg_h, 1) + np.rot90(seg_r, k=-1), a_min=0, a_max=1)

                seg_pre[x * stride + 128:x * stride + size - 128, y * stride + 128:y * stride + size - 128] = seg[128:384, 128:384]
                ver_pre[x * stride + 128:x * stride + size - 128, y * stride + 128:y * stride + size - 128] = ver[128:384, 128:384]
                mid_pre[x * stride + 128:x * stride + size - 128, y * stride + 128:y * stride + size - 128] = mid[128:384, 128:384]
                dxy_pre[x * stride + 128:x * stride + size - 128, y * stride + 128:y * stride + size - 128, :] = dxy[128:384, 128:384, :]

        seg_pre = seg_pre[stride:dataset_image_size - stride, stride:dataset_image_size - stride]
        ver_pre = ver_pre[stride:dataset_image_size - stride, stride:dataset_image_size - stride]
        mid_pre = mid_pre[stride:dataset_image_size - stride, stride:dataset_image_size - stride]
        dxy_pre = dxy_pre[stride:dataset_image_size - stride, stride:dataset_image_size - stride, :]
        graph = decodingc(ver_pre, mid_pre, dxy_pre, seg_pre)
        pickle.dump(graph, open(output_dir + "region_%d_graph.p" % i, "wb"))
        graph = simplifyGraph(graph)
        seg_map = np.zeros((dataset_image_size - size, dataset_image_size - size))
        for u, v in graph.items():
            n1 = u
            for n2 in v:
                cv2.line(sat_ori, (int(n1[1]), int(n1[0])), (int(n2[1]), int(n2[0])), (0, 128, 255), 3)
                cv2.line(seg_map, (int(n1[1]), int(n1[0])), (int(n2[1]), int(n2[0])), 255, 3)
        for u, v in graph.items():
            n1 = u
            if len(v) < 3:
                cv2.circle(sat_ori, (int(n1[1]), int(n1[0])), 3, (0, 255, 255), -1)
            else:
                cv2.circle(sat_ori, (int(n1[1]), int(n1[0])), 3, (0, 255, 255), -1)
        cv2.imwrite(output_dir + "region_%d_vis.png" % i, sat_ori)
        cv2.imwrite(output_dir + "region_%d_seg.png" % i, seg_map)


def decodings(ver_map, mid_map, dxy_map, seg_map, thr=0.05, rad=100):
    smooth_ver = scipy.ndimage.filters.gaussian_filter(ver_map, 1)
    smooth_ver = smooth_ver / max(np.amax(smooth_ver), 0.001)
    vertices = detect_local_minima(-smooth_ver, smooth_ver, thr)

    ver_idx = index.Index()
    for i in range(len(vertices[0])):
        x0, y0 = vertices[0][i], vertices[1][i]
        if len(list(ver_idx.intersection((x0 - 5, y0 - 5, x0 + 5, y0 + 5)))) == 0:
            ver_idx.insert(i, (x0 - 1, y0 - 1, x0 + 1, y0 + 1))

        for j in range(len(vertices[0])):
            x, y = vertices[0][j], vertices[1][j]
            if abs(x - x0) < 5 and abs(y - y0) < 5:
                continue
            if abs(x - x0) > rad or abs(y - y0) > rad:
                continue

            line_mean = line_pooling(seg_map, x0, y0, x, y)
            if line_mean < 0.5:
                continue

            xm, ym = round((x0 + x) / 2), round((y0 + y) / 2)
            mid_map[xm, ym] = 1

    smooth_mid = scipy.ndimage.filters.gaussian_filter(mid_map, 1)
    smooth_mid = smooth_mid / max(np.amax(smooth_mid), 0.001)
    middle_points = detect_local_minima(-smooth_mid, smooth_mid, thr)

    neighbors = {}
    for i in range(len(middle_points[0])):
        x0 = middle_points[0][i]
        y0 = middle_points[1][i]

        best_candidate = -1
        min_distance = 15
        x1 = round(dxy_map[x0, y0, 0] + x0)
        y1 = round(dxy_map[x0, y0, 1] + y0)
        candidates = list(ver_idx.intersection((x1 - 20, y1 - 20, x1 + 20, y1 + 20)))
        for candidate in candidates:
            x_c = vertices[0][candidate]
            y_c = vertices[1][candidate]
            d = distance((x_c, y_c), (x1, y1))

            v1 = (x_c - x0, y_c - y0)
            v2 = (x1 - x0, y1 - y0)
            ad = 1.0 - anglediff(v1, v2)
            d = d + ad * 50
            if d < min_distance:
                min_distance = d
                best_candidate = candidate

        if best_candidate != -1:
            x1 = vertices[0][best_candidate]
            y1 = vertices[1][best_candidate]
        else:
            continue

        best_candidate = -1
        min_distance = 15
        x2 = round(dxy_map[x0, y0, 2] + x0)
        y2 = round(dxy_map[x0, y0, 3] + y0)
        candidates = list(ver_idx.intersection((x2 - 20, y2 - 20, x2 + 20, y2 + 20)))
        for candidate in candidates:
            x_c = vertices[0][candidate]
            y_c = vertices[1][candidate]
            d = distance((x_c, y_c), (x2, y2))

            v1 = (x_c - x0, y_c - y0)
            v2 = (x2 - x0, y2 - y0)
            ad = 1.0 - anglediff(v1, v2)
            d = d + ad * 50
            if d < min_distance:
                min_distance = d
                best_candidate = candidate

        if best_candidate != -1:
            x2 = vertices[0][best_candidate]
            y2 = vertices[1][best_candidate]
        else:
            continue

        if (x1, y1) in neighbors:
            if (x2, y2) in neighbors[(x1, y1)]:
                pass
            else:
                neighbors[(x1, y1)].append((x2, y2))
        else:
            neighbors[(x1, y1)] = [(x2, y2)]

        if (x2, y2) in neighbors:
            if (x1, y1) in neighbors[(x2, y2)]:
                pass
            else:
                neighbors[(x2, y2)].append((x1, y1))
        else:
            neighbors[(x2, y2)] = [(x1, y1)]

    spurs_thr = 25
    isolated_thr = 100
    graph = graph_refine(neighbors, isolated_thr=isolated_thr, spurs_thr=spurs_thr)
    rc = 100
    while rc > 0:
        graph, rc = graph_refine_deloop(graph_refine(graph, isolated_thr=isolated_thr, spurs_thr=spurs_thr))
    graph = graph_shave(graph, spurs_thr=spurs_thr)
    # graph = neighbors
    return graph


def infer_spacenet():
    from extractor import Extractor
    from network import TopoDLATransformer as Net

    input_dir = './datasets/spacenet/test/'
    output_dir = './results/spacenet/'
    weight_dir = './checkpoints/spacenet/best.th'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = Extractor(Net, eval_mode=True)
    model.load(weight_dir)

    dataset = json.load(open('./datasets/spacenet/dataset.json', 'r'))
    for item in dataset['test']:
        sat = cv2.imread(input_dir + item + '_sat.png')
        sat_img = np.array(sat, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre = torch.sigmoid(pre['seg']).squeeze().cpu().data.numpy()
        ver_pre = torch.sigmoid(pre['ver']).squeeze().cpu().data.numpy()
        mid_pre = torch.sigmoid(pre['mid']).squeeze().cpu().data.numpy()
        dxy_pre = pre['dxy'].squeeze().cpu().data.numpy().transpose((1, 2, 0)) * 12
        pre = None
        sat_img = np.flip(sat, 0)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_v = torch.sigmoid(pre['seg']).squeeze().cpu().data.numpy()
        pre = None
        sat_img = np.flip(sat, 1)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_h = torch.sigmoid(pre['seg']).squeeze().cpu().data.numpy()
        pre = None
        sat_img = np.rot90(sat)
        sat_img = np.array(sat_img, np.float32).transpose((2, 0, 1)) / 255.0 * 3.2 - 1.6
        sat_img = V(torch.Tensor(np.expand_dims(sat_img, axis=0)).cuda())
        pre = model.predict(sat_img)
        seg_pre_r = torch.sigmoid(pre['seg']).squeeze().cpu().data.numpy()
        pre = None
        seg_pre = np.clip(seg_pre + np.flip(seg_pre_v, 0) + np.flip(seg_pre_h, 1) + np.rot90(seg_pre_r, k=-1), a_min=0, a_max=1)

        graph = decodings(ver_pre, mid_pre, dxy_pre, seg_pre)
        pickle.dump(graph, open(output_dir + "{}_graph.p".format(item), "wb"))
        graph = simplifyGraph(graph)
        seg = np.zeros_like(seg_pre)
        for u, v in graph.items():
            n1 = u
            for n2 in v:
                cv2.line(sat, (int(n1[1]), int(n1[0])), (int(n2[1]), int(n2[0])), (0, 128, 255), 3)
                cv2.line(seg, (int(n1[1]), int(n1[0])), (int(n2[1]), int(n2[0])), 255, 3)
        for u, v in graph.items():
            n1 = u
            if len(v) < 3:
                cv2.circle(sat, (int(n1[1]), int(n1[0])), 3, (0, 255, 255), -1)
            else:
                cv2.circle(sat, (int(n1[1]), int(n1[0])), 3, (0, 255, 255), -1)
        cv2.imwrite(output_dir + "{}_vis.png".format(item), sat)
        cv2.imwrite(output_dir + "{}_seg.png".format(item), seg)
