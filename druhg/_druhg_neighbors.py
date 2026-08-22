"""KD-tree and Ball-tree k-nearest neighbor queries for DRUHG.
Author: AI generated
License: 3-clause BSD
"""
import math

import numpy as np
from numba import njit, prange

INF = np.inf

TREE_KD = 0
TREE_BALL = 1

MET_EUCLIDEAN = 0
MET_MANHATTAN = 1
MET_CHEBYSHEV = 2
MET_MINKOWSKI = 3
MET_SEUCLIDEAN = 4
MET_MAHALANOBIS = 5
MET_HAMMING = 6
MET_CANBERRA = 7
MET_BRAYCURTIS = 8
MET_JACCARD = 9
MET_DICE = 10
MET_ROGERSTANIMOTO = 11
MET_RUSSELLRAO = 12
MET_SOKALMICHENER = 13
MET_SOKALSNEATH = 14
MET_HAVERSINE = 15
MET_COSINE = 16
MET_ARCCOS = 17

METRIC_CODES = {
    'euclidean': MET_EUCLIDEAN,
    'l2': MET_EUCLIDEAN,
    'manhattan': MET_MANHATTAN,
    'cityblock': MET_MANHATTAN,
    'l1': MET_MANHATTAN,
    'chebyshev': MET_CHEBYSHEV,
    'infinity': MET_CHEBYSHEV,
    'minkowski': MET_MINKOWSKI,
    'p': MET_MINKOWSKI,
    'seuclidean': MET_SEUCLIDEAN,
    'mahalanobis': MET_MAHALANOBIS,
    'hamming': MET_HAMMING,
    'canberra': MET_CANBERRA,
    'braycurtis': MET_BRAYCURTIS,
    'jaccard': MET_JACCARD,
    'dice': MET_DICE,
    'rogerstanimoto': MET_ROGERSTANIMOTO,
    'russellrao': MET_RUSSELLRAO,
    'sokalmichener': MET_SOKALMICHENER,
    'sokalsneath': MET_SOKALSNEATH,
    'haversine': MET_HAVERSINE,
    'cosine': MET_COSINE,
    'arccos': MET_ARCCOS,
}

KDTREE_VALID_METRICS = [
    'euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1',
    'chebyshev', 'infinity', 'cosine', 'arccos',
]
BALLTREE_VALID_METRICS = [
    'euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1',
    'chebyshev', 'infinity', 'seuclidean', 'mahalanobis', 'hamming',
    'canberra', 'braycurtis', 'jaccard', 'dice', 'rogerstanimoto',
    'russellrao', 'sokalmichener', 'sokalsneath', 'haversine',
    'cosine', 'arccos',
]


def _as_sample_matrix(X, n_features=None):
    """Turn 1-d vectors into a 2-d sample matrix.

    Training data of shape (n,) is n points with one feature.
    A 1-d query against a 1-d tree is n query points; otherwise it is one
    query with ``n_features`` coordinates (sklearn-style).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 0:
        X = X.reshape(1, 1)
    elif X.ndim == 1:
        if n_features is None or n_features == 1:
            X = X.reshape(-1, 1)
        else:
            X = X.reshape(1, -1)
    elif X.ndim != 2:
        raise ValueError('X must be a 1- or 2-dimensional array')
    if not X.flags['C_CONTIGUOUS']:
        X = np.ascontiguousarray(X)
    return X


def _resolve_metric(metric, p):
    name = metric.lower()
    if name not in METRIC_CODES:
        raise ValueError('Unknown metric: %s' % metric)
    code = METRIC_CODES[name]
    if code == MET_MINKOWSKI:
        if p == 1:
            return MET_MANHATTAN
        if p == 2:
            return MET_EUCLIDEAN
        if p == INF:
            return MET_CHEBYSHEV
    return code


@njit(cache=True)
def _rdist(x1, x2, metric_id, p, w, V, VI, has_w, has_V, has_VI):
    n = x1.shape[0]
    d = 0.0

    if metric_id == MET_EUCLIDEAN:
        for j in range(n):
            diff = x1[j] - x2[j]
            d += diff * diff
        return d

    if metric_id == MET_MANHATTAN:
        if has_w:
            for j in range(n):
                d += w[j] * abs(x1[j] - x2[j])
        else:
            for j in range(n):
                d += abs(x1[j] - x2[j])
        return d

    if metric_id == MET_CHEBYSHEV:
        for j in range(n):
            diff = abs(x1[j] - x2[j])
            if diff > d:
                d = diff
        return d

    if metric_id == MET_MINKOWSKI:
        if has_w:
            for j in range(n):
                d += w[j] * abs(x1[j] - x2[j]) ** p
        else:
            for j in range(n):
                d += abs(x1[j] - x2[j]) ** p
        return d

    if metric_id == MET_SEUCLIDEAN:
        for j in range(n):
            diff = x1[j] - x2[j]
            d += diff * diff / V[j]
        return d

    if metric_id == MET_MAHALANOBIS:
        for i in range(n):
            tmp = 0.0
            for j in range(n):
                tmp += VI[i, j] * (x1[j] - x2[j])
            d += tmp * (x1[i] - x2[i])
        return d

    if metric_id == MET_HAMMING:
        for j in range(n):
            if x1[j] != x2[j]:
                d += 1.0
        return d / n

    if metric_id == MET_CANBERRA:
        for j in range(n):
            denom = abs(x1[j]) + abs(x2[j])
            if denom > 0.0:
                d += abs(x1[j] - x2[j]) / denom
        return d

    if metric_id == MET_BRAYCURTIS:
        acc = 0.0
        for j in range(n):
            d += abs(x1[j] - x2[j])
            acc += abs(x1[j] + x2[j])
        if acc == 0.0:
            return 0.0
        return d / acc

    if metric_id == MET_HAVERSINE:
        tmp = math.sin(0.5 * (x1[0] - x2[0]))
        diff = math.sin(0.5 * (x1[1] - x2[1]))
        d = tmp * tmp + math.cos(x1[0]) * math.cos(x2[0]) * diff * diff
        if d < 0.0:
            d = 0.0
        elif d > 1.0:
            d = 1.0
        return 2.0 * math.asin(math.sqrt(d))

    if metric_id == MET_COSINE or metric_id == MET_ARCCOS:
        acc = 0.0
        n1 = 0.0
        n2 = 0.0
        for j in range(n):
            acc += x1[j] * x2[j]
            n1 += x1[j] * x1[j]
            n2 += x2[j] * x2[j]
        denom = math.sqrt(n1) * math.sqrt(n2)
        if denom == 0.0:
            sim = 1.0 if n1 == 0.0 and n2 == 0.0 else 0.0
        else:
            sim = acc / denom
            if sim > 1.0:
                sim = 1.0
            elif sim < -1.0:
                sim = -1.0
        if metric_id == MET_ARCCOS:
            return math.acos(sim)
        return 1.0 - sim

    tt = 0
    tf = 0
    ft = 0
    for j in range(n):
        a = x1[j] != 0.0
        b = x2[j] != 0.0
        if a and b:
            tt += 1
        elif a:
            tf += 1
        elif b:
            ft += 1
    n_neq = tf + ft

    if metric_id == MET_JACCARD:
        denom = tt + n_neq
        return 0.0 if denom == 0 else n_neq / denom

    if metric_id == MET_DICE:
        denom = 2 * tt + n_neq
        return 0.0 if denom == 0 else n_neq / denom

    if metric_id == MET_ROGERSTANIMOTO or metric_id == MET_SOKALMICHENER:
        denom = n + n_neq
        return 0.0 if denom == 0 else (2.0 * n_neq) / denom

    if metric_id == MET_RUSSELLRAO:
        return (n - tt) / n

    if metric_id == MET_SOKALSNEATH:
        denom = 0.5 * tt + n_neq
        return 0.0 if denom == 0 else n_neq / denom

    return 0.0


@njit(cache=True)
def _rdist_to_dist(r, metric_id, p):
    if metric_id == MET_EUCLIDEAN or metric_id == MET_SEUCLIDEAN or metric_id == MET_MAHALANOBIS:
        return math.sqrt(r)
    if metric_id == MET_MINKOWSKI:
        return r ** (1.0 / p)
    return r


@njit(cache=True)
def _dist_to_rdist(d, metric_id, p):
    if metric_id == MET_EUCLIDEAN or metric_id == MET_SEUCLIDEAN or metric_id == MET_MAHALANOBIS:
        return d * d
    if metric_id == MET_MINKOWSKI:
        return d ** p
    return d


@njit(cache=True)
def _dist(x1, x2, metric_id, p, w, V, VI, has_w, has_V, has_VI):
    return _rdist_to_dist(_rdist(x1, x2, metric_id, p, w, V, VI, has_w, has_V, has_VI),
                          metric_id, p)


@njit(cache=True)
def _kd_min_rdist(bounds, i_node, pt, metric_id, p, w, has_w):
    n = pt.shape[0]
    d = 0.0
    for j in range(n):
        v = pt[j]
        lo = bounds[0, i_node, j]
        hi = bounds[1, i_node, j]
        if v < lo:
            delta = lo - v
        elif v > hi:
            delta = v - hi
        else:
            continue
        if metric_id == MET_CHEBYSHEV:
            if delta > d:
                d = delta
        elif metric_id == MET_MANHATTAN:
            d += delta if not has_w else w[j] * delta
        elif metric_id == MET_MINKOWSKI:
            d += delta ** p if not has_w else w[j] * (delta ** p)
        else:
            d += delta * delta
    return d


@njit(cache=True)
def _ball_min_rdist(bounds, radius, i_node, pt, metric_id, p, w, V, VI, has_w, has_V, has_VI):
    d = _dist(pt, bounds[0, i_node], metric_id, p, w, V, VI, has_w, has_V, has_VI) - radius[i_node]
    if d < 0.0:
        d = 0.0
    return _dist_to_rdist(d, metric_id, p)


@njit(cache=True)
def _min_rdist(tree_kind, bounds, radius, i_node, pt, metric_id, p, w, V, VI, has_w, has_V, has_VI):
    if tree_kind == TREE_KD:
        return _kd_min_rdist(bounds, i_node, pt, metric_id, p, w, has_w)
    return _ball_min_rdist(bounds, radius, i_node, pt, metric_id, p, w, V, VI, has_w, has_V, has_VI)


_STACK_CAP = 64


@njit(inline='always', cache=True)
def _heap_push(dist, ind, row, val, i_val):
    size = dist.shape[1]
    if val > dist[row, 0]:
        return dist[row, 0]
    dist[row, 0] = val
    ind[row, 0] = i_val
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1
        if ic1 >= size:
            break
        if ic2 >= size:
            if dist[row, ic1] > val:
                i_swap = ic1
            else:
                break
        elif dist[row, ic1] >= dist[row, ic2]:
            if val < dist[row, ic1]:
                i_swap = ic1
            else:
                break
        else:
            if val < dist[row, ic2]:
                i_swap = ic2
            else:
                break
        dist[row, i] = dist[row, i_swap]
        ind[row, i] = ind[row, i_swap]
        i = i_swap
    dist[row, i] = val
    ind[row, i] = i_val
    return dist[row, 0]


@njit(cache=True)
def _sort_row(dist, ind, row, size):
    for i in range(1, size):
        dkey = dist[row, i]
        ikey = ind[row, i]
        j = i
        while j > 0 and dist[row, j - 1] > dkey:
            dist[row, j] = dist[row, j - 1]
            ind[row, j] = ind[row, j - 1]
            j -= 1
        dist[row, j] = dkey
        ind[row, j] = ikey


@njit(cache=True)
def _split_dim(data, idx_array, idx_start, n_points, n_features):
    j_max = 0
    max_spread = 0.0
    for j in range(n_features):
        min_val = data[idx_array[idx_start], j]
        max_val = min_val
        for i in range(1, n_points):
            val = data[idx_array[idx_start + i], j]
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
        spread = max_val - min_val
        if spread > max_spread:
            max_spread = spread
            j_max = j
    return j_max


@njit(cache=True)
def _partition_indices(data, idx_array, idx_start, n_points, split_dim, split_index):
    left = 0
    right = n_points - 1
    while True:
        mid = left
        for i in range(left, right):
            d1 = data[idx_array[idx_start + i], split_dim]
            d2 = data[idx_array[idx_start + right], split_dim]
            if d1 < d2:
                tmp = idx_array[idx_start + i]
                idx_array[idx_start + i] = idx_array[idx_start + mid]
                idx_array[idx_start + mid] = tmp
                mid += 1
        tmp = idx_array[idx_start + mid]
        idx_array[idx_start + mid] = idx_array[idx_start + right]
        idx_array[idx_start + right] = tmp
        if mid == split_index:
            return
        if mid < split_index:
            left = mid + 1
        else:
            right = mid - 1
        if left > right:
            return


@njit(cache=True)
def _init_node(data, idx_array, idx_start_arr, idx_end_arr, radius, bounds,
               i_node, idx_start, idx_end, tree_kind, metric_id, p, w, V, VI,
               has_w, has_V, has_VI):
    n = data.shape[1]
    idx_start_arr[i_node] = idx_start
    idx_end_arr[i_node] = idx_end
    n_points = idx_end - idx_start

    if tree_kind == TREE_KD:
        idx0 = idx_array[idx_start]
        for j in range(n):
            bounds[0, i_node, j] = data[idx0, j]
            bounds[1, i_node, j] = data[idx0, j]
        for i in range(idx_start + 1, idx_end):
            idx = idx_array[i]
            for j in range(n):
                val = data[idx, j]
                if val < bounds[0, i_node, j]:
                    bounds[0, i_node, j] = val
                if val > bounds[1, i_node, j]:
                    bounds[1, i_node, j] = val
        radius[i_node] = 0.5 * _dist(
            bounds[0, i_node], bounds[1, i_node],
            metric_id, p, w, V, VI, has_w, has_V, has_VI)
    else:
        for j in range(n):
            bounds[0, i_node, j] = 0.0
        for i in range(idx_start, idx_end):
            idx = idx_array[i]
            for j in range(n):
                bounds[0, i_node, j] += data[idx, j]
        inv = 1.0 / n_points
        for j in range(n):
            bounds[0, i_node, j] *= inv
        r = 0.0
        for i in range(idx_start, idx_end):
            idx = idx_array[i]
            d = _dist(bounds[0, i_node], data[idx], metric_id, p, w, V, VI,
                      has_w, has_V, has_VI)
            if d > r:
                r = d
        radius[i_node] = r


@njit(cache=True)
def _build_tree(data, idx_array, idx_start_arr, idx_end_arr, is_leaf, radius, bounds,
                n_nodes, tree_kind, metric_id, p, w, V, VI, has_w, has_V, has_VI):
    n_samples = data.shape[0]
    n_features = data.shape[1]
    sn = np.empty(n_nodes + 1, dtype=np.intp)
    ss = np.empty(n_nodes + 1, dtype=np.intp)
    se = np.empty(n_nodes + 1, dtype=np.intp)
    sn[0] = 0
    ss[0] = 0
    se[0] = n_samples
    sp = 1
    while sp > 0:
        sp -= 1
        i_node = sn[sp]
        a = ss[sp]
        b = se[sp]
        _init_node(data, idx_array, idx_start_arr, idx_end_arr, radius, bounds,
                   i_node, a, b, tree_kind, metric_id, p, w, V, VI,
                   has_w, has_V, has_VI)
        n_points = b - a
        if 2 * i_node + 1 >= n_nodes or n_points < 2:
            is_leaf[i_node] = 1
            continue
        is_leaf[i_node] = 0
        n_mid = n_points // 2
        i_max = _split_dim(data, idx_array, a, n_points, n_features)
        _partition_indices(data, idx_array, a, n_points, i_max, n_mid)
        sn[sp] = 2 * i_node + 2
        ss[sp] = a + n_mid
        se[sp] = b
        sp += 1
        sn[sp] = 2 * i_node + 1
        ss[sp] = a
        se[sp] = a + n_mid
        sp += 1


@njit(cache=True)
def _fill_leaf_of(idx_array, idx_start, idx_end, is_leaf, leaf_of):
    n_nodes = is_leaf.shape[0]
    for i_node in range(n_nodes):
        if is_leaf[i_node] != 0:
            for i in range(idx_start[i_node], idx_end[i_node]):
                leaf_of[idx_array[i]] = i_node


@njit(inline='always', fastmath=True, cache=True)
def _kd_l2_rdist(X, i_pt, data, idx, n_feat):
    if n_feat == 2:
        d0 = X[i_pt, 0] - data[idx, 0]
        d1 = X[i_pt, 1] - data[idx, 1]
        return d0 * d0 + d1 * d1
    if n_feat == 3:
        d0 = X[i_pt, 0] - data[idx, 0]
        d1 = X[i_pt, 1] - data[idx, 1]
        d2 = X[i_pt, 2] - data[idx, 2]
        return d0 * d0 + d1 * d1 + d2 * d2
    d = 0.0
    for j in range(n_feat):
        diff = X[i_pt, j] - data[idx, j]
        d += diff * diff
    return d


@njit(inline='always', fastmath=True, cache=True)
def _kd_l2_min_max_rdist(bounds, i_node, X, i_pt, n_feat):
    min_d = 0.0
    max_d = 0.0
    if n_feat == 2:
        for j in range(2):
            v = X[i_pt, j]
            lo = bounds[0, i_node, j]
            hi = bounds[1, i_node, j]
            d_lo = lo - v
            d_hi = v - hi
            d = max(d_lo, 0.0) + max(d_hi, 0.0)
            min_d += d * d
            d = max(abs(d_lo), abs(d_hi))
            max_d += d * d
        return min_d, max_d
    if n_feat == 3:
        for j in range(3):
            v = X[i_pt, j]
            lo = bounds[0, i_node, j]
            hi = bounds[1, i_node, j]
            d_lo = lo - v
            d_hi = v - hi
            d = max(d_lo, 0.0) + max(d_hi, 0.0)
            min_d += d * d
            d = max(abs(d_lo), abs(d_hi))
            max_d += d * d
        return min_d, max_d
    for j in range(n_feat):
        v = X[i_pt, j]
        lo = bounds[0, i_node, j]
        hi = bounds[1, i_node, j]
        d_lo = lo - v
        d_hi = v - hi
        d = max(d_lo, 0.0) + max(d_hi, 0.0)
        min_d += d * d
        d = max(abs(d_lo), abs(d_hi))
        max_d += d * d
    return min_d, max_d


@njit(inline='always', fastmath=True, cache=True)
def _kd_l1_rdist(X, i_pt, data, idx, n_feat):
    d = 0.0
    for j in range(n_feat):
        d += abs(X[i_pt, j] - data[idx, j])
    return d


@njit(inline='always', fastmath=True, cache=True)
def _kd_l1_min_max_rdist(bounds, i_node, X, i_pt, n_feat):
    min_d = 0.0
    max_d = 0.0
    for j in range(n_feat):
        v = X[i_pt, j]
        lo = bounds[0, i_node, j]
        hi = bounds[1, i_node, j]
        d_lo = lo - v
        d_hi = v - hi
        min_d += max(d_lo, 0.0) + max(d_hi, 0.0)
        max_d += max(abs(d_lo), abs(d_hi))
    return min_d, max_d


@njit(inline='always', cache=True)
def _scan_node_kd_l2(X, i_pt, data, idx_array, idx_start, idx_end, i_node, dist, ind, tau):
    n_feat = data.shape[1]
    for i in range(idx_start[i_node], idx_end[i_node]):
        idx = idx_array[i]
        if idx == i_pt:
            continue
        d = _kd_l2_rdist(X, i_pt, data, idx, n_feat)
        if d <= tau:
            tau = _heap_push(dist, ind, i_pt, d, idx)
    return tau


@njit(inline='always', cache=True)
def _scan_node_kd_l1(X, i_pt, data, idx_array, idx_start, idx_end, i_node, dist, ind, tau):
    n_feat = data.shape[1]
    for i in range(idx_start[i_node], idx_end[i_node]):
        idx = idx_array[i]
        if idx == i_pt:
            continue
        d = _kd_l1_rdist(X, i_pt, data, idx, n_feat)
        if d <= tau:
            tau = _heap_push(dist, ind, i_pt, d, idx)
    return tau


@njit(cache=True)
def _dfs_kd_l2(X, i_pt, data, idx_array, idx_start, idx_end, is_leaf, bounds,
               dist, ind, tau, root, stack_node, stack_lb):
    n_feat = data.shape[1]
    mn, mx = _kd_l2_min_max_rdist(bounds, root, X, i_pt, n_feat)
    if mn > tau:
        return tau
    stack_node[0] = root
    stack_lb[0] = mn
    sp = 1
    while sp > 0:
        sp -= 1
        i_node = stack_node[sp]
        if stack_lb[sp] > tau:
            continue
        mn, mx = _kd_l2_min_max_rdist(bounds, i_node, X, i_pt, n_feat)
        if mn > tau:
            continue
        if mx <= tau or is_leaf[i_node]:
            tau = _scan_node_kd_l2(X, i_pt, data, idx_array, idx_start, idx_end,
                                   i_node, dist, ind, tau)
            continue
        i1 = 2 * i_node + 1
        i2 = i1 + 1
        lb1, _ = _kd_l2_min_max_rdist(bounds, i1, X, i_pt, n_feat)
        lb2, _ = _kd_l2_min_max_rdist(bounds, i2, X, i_pt, n_feat)
        if lb1 <= lb2:
            if lb2 <= tau:
                stack_node[sp] = i2
                stack_lb[sp] = lb2
                sp += 1
            if lb1 <= tau:
                stack_node[sp] = i1
                stack_lb[sp] = lb1
                sp += 1
        else:
            if lb1 <= tau:
                stack_node[sp] = i1
                stack_lb[sp] = lb1
                sp += 1
            if lb2 <= tau:
                stack_node[sp] = i2
                stack_lb[sp] = lb2
                sp += 1
    return tau


@njit(cache=True)
def _dfs_kd_l1(X, i_pt, data, idx_array, idx_start, idx_end, is_leaf, bounds,
               dist, ind, tau, root, stack_node, stack_lb):
    n_feat = data.shape[1]
    mn, mx = _kd_l1_min_max_rdist(bounds, root, X, i_pt, n_feat)
    if mn > tau:
        return tau
    stack_node[0] = root
    stack_lb[0] = mn
    sp = 1
    while sp > 0:
        sp -= 1
        i_node = stack_node[sp]
        if stack_lb[sp] > tau:
            continue
        mn, mx = _kd_l1_min_max_rdist(bounds, i_node, X, i_pt, n_feat)
        if mn > tau:
            continue
        if mx <= tau or is_leaf[i_node]:
            tau = _scan_node_kd_l1(X, i_pt, data, idx_array, idx_start, idx_end,
                                   i_node, dist, ind, tau)
            continue
        i1 = 2 * i_node + 1
        i2 = i1 + 1
        lb1, _ = _kd_l1_min_max_rdist(bounds, i1, X, i_pt, n_feat)
        lb2, _ = _kd_l1_min_max_rdist(bounds, i2, X, i_pt, n_feat)
        if lb1 <= lb2:
            if lb2 <= tau:
                stack_node[sp] = i2
                stack_lb[sp] = lb2
                sp += 1
            if lb1 <= tau:
                stack_node[sp] = i1
                stack_lb[sp] = lb1
                sp += 1
        else:
            if lb1 <= tau:
                stack_node[sp] = i1
                stack_lb[sp] = lb1
                sp += 1
            if lb2 <= tau:
                stack_node[sp] = i2
                stack_lb[sp] = lb2
                sp += 1
    return tau


@njit(cache=True)
def _query_one_kd_l2(X, i_pt, data, idx_array, idx_start, idx_end, is_leaf, bounds,
                     leaf_of, dist, ind, stack_node, stack_lb):
    tau = dist[i_pt, 0]
    start = leaf_of[i_pt] if i_pt < leaf_of.shape[0] else -1
    if start < 0:
        return _dfs_kd_l2(X, i_pt, data, idx_array, idx_start, idx_end, is_leaf, bounds,
                          dist, ind, tau, 0, stack_node, stack_lb)
    tau = _scan_node_kd_l2(X, i_pt, data, idx_array, idx_start, idx_end,
                           start, dist, ind, tau)
    i_node = start
    while i_node != 0:
        sibling = i_node + 1 if i_node % 2 == 1 else i_node - 1
        tau = _dfs_kd_l2(X, i_pt, data, idx_array, idx_start, idx_end, is_leaf, bounds,
                         dist, ind, tau, sibling, stack_node, stack_lb)
        i_node = (i_node - 1) // 2
    return tau


@njit(cache=True)
def _query_one_kd_l1(X, i_pt, data, idx_array, idx_start, idx_end, is_leaf, bounds,
                     leaf_of, dist, ind, stack_node, stack_lb):
    tau = dist[i_pt, 0]
    start = leaf_of[i_pt] if i_pt < leaf_of.shape[0] else -1
    if start < 0:
        return _dfs_kd_l1(X, i_pt, data, idx_array, idx_start, idx_end, is_leaf, bounds,
                          dist, ind, tau, 0, stack_node, stack_lb)
    tau = _scan_node_kd_l1(X, i_pt, data, idx_array, idx_start, idx_end,
                           start, dist, ind, tau)
    i_node = start
    while i_node != 0:
        sibling = i_node + 1 if i_node % 2 == 1 else i_node - 1
        tau = _dfs_kd_l1(X, i_pt, data, idx_array, idx_start, idx_end, is_leaf, bounds,
                         dist, ind, tau, sibling, stack_node, stack_lb)
        i_node = (i_node - 1) // 2
    return tau


@njit(cache=True)
def _dfs_generic(X, i_pt, pt, data, idx_array, idx_start, idx_end, is_leaf, radius, bounds,
                 tree_kind, metric_id, p, w, V, VI, has_w, has_V, has_VI, dist, ind,
                 tau, root, stack_node, stack_lb):
    lb = _min_rdist(tree_kind, bounds, radius, root, pt, metric_id, p,
                    w, V, VI, has_w, has_V, has_VI)
    if lb > tau:
        return tau
    stack_node[0] = root
    stack_lb[0] = lb
    sp = 1
    while sp > 0:
        sp -= 1
        i_node = stack_node[sp]
        if stack_lb[sp] > tau:
            continue
        if is_leaf[i_node]:
            for i in range(idx_start[i_node], idx_end[i_node]):
                idx = idx_array[i]
                if idx == i_pt:
                    continue
                d = _rdist(pt, data[idx], metric_id, p, w, V, VI, has_w, has_V, has_VI)
                if d <= tau:
                    tau = _heap_push(dist, ind, i_pt, d, idx)
            continue
        i1 = 2 * i_node + 1
        i2 = i1 + 1
        lb1 = _min_rdist(tree_kind, bounds, radius, i1, pt, metric_id, p,
                         w, V, VI, has_w, has_V, has_VI)
        lb2 = _min_rdist(tree_kind, bounds, radius, i2, pt, metric_id, p,
                         w, V, VI, has_w, has_V, has_VI)
        if lb1 <= lb2:
            if lb2 <= tau:
                stack_node[sp] = i2
                stack_lb[sp] = lb2
                sp += 1
            if lb1 <= tau:
                stack_node[sp] = i1
                stack_lb[sp] = lb1
                sp += 1
        else:
            if lb1 <= tau:
                stack_node[sp] = i1
                stack_lb[sp] = lb1
                sp += 1
            if lb2 <= tau:
                stack_node[sp] = i2
                stack_lb[sp] = lb2
                sp += 1
    return tau


@njit(cache=True)
def _query_one(X, i_pt, data, idx_array, idx_start, idx_end, is_leaf, radius, bounds,
               tree_kind, metric_id, p, w, V, VI, has_w, has_V, has_VI, dist, ind,
               leaf_of, stack_node, stack_lb):
    n_feat = data.shape[1]
    pt = np.empty(n_feat, dtype=np.float64)
    for j in range(n_feat):
        pt[j] = X[i_pt, j]
    tau = dist[i_pt, 0]
    start = leaf_of[i_pt] if i_pt < leaf_of.shape[0] else -1
    if start < 0:
        return _dfs_generic(X, i_pt, pt, data, idx_array, idx_start, idx_end, is_leaf, radius, bounds,
                            tree_kind, metric_id, p, w, V, VI, has_w, has_V, has_VI, dist, ind,
                            tau, 0, stack_node, stack_lb)
    for i in range(idx_start[start], idx_end[start]):
        idx = idx_array[i]
        if idx == i_pt:
            continue
        d = _rdist(pt, data[idx], metric_id, p, w, V, VI, has_w, has_V, has_VI)
        if d <= tau:
            tau = _heap_push(dist, ind, i_pt, d, idx)
    i_node = start
    while i_node != 0:
        sibling = i_node + 1 if i_node % 2 == 1 else i_node - 1
        tau = _dfs_generic(X, i_pt, pt, data, idx_array, idx_start, idx_end, is_leaf, radius, bounds,
                           tree_kind, metric_id, p, w, V, VI, has_w, has_V, has_VI, dist, ind,
                           tau, sibling, stack_node, stack_lb)
        i_node = (i_node - 1) // 2
    return tau


@njit(parallel=True, cache=True)
def _query_all_kd_l2(X, data, idx_array, idx_start, idx_end, is_leaf, bounds, leaf_of,
                     dist, ind, do_sort):
    n_queries = X.shape[0]
    k_nbrs = dist.shape[1]
    stack_node = np.empty((n_queries, _STACK_CAP), dtype=np.intp)
    stack_lb = np.empty((n_queries, _STACK_CAP), dtype=np.float64)
    for i in prange(n_queries):
        _query_one_kd_l2(X, i, data, idx_array, idx_start, idx_end, is_leaf, bounds,
                         leaf_of, dist, ind, stack_node[i], stack_lb[i])
        if do_sort:
            _sort_row(dist, ind, i, k_nbrs)
        for j in range(k_nbrs):
            dist[i, j] = math.sqrt(dist[i, j])


@njit(cache=True)
def _query_all_kd_l2_seq(X, data, idx_array, idx_start, idx_end, is_leaf, bounds, leaf_of,
                         dist, ind, do_sort):
    n_queries = X.shape[0]
    k_nbrs = dist.shape[1]
    stack_node = np.empty((_STACK_CAP,), dtype=np.intp)
    stack_lb = np.empty((_STACK_CAP,), dtype=np.float64)
    for i in range(n_queries):
        _query_one_kd_l2(X, i, data, idx_array, idx_start, idx_end, is_leaf, bounds,
                         leaf_of, dist, ind, stack_node, stack_lb)
        if do_sort:
            _sort_row(dist, ind, i, k_nbrs)
        for j in range(k_nbrs):
            dist[i, j] = math.sqrt(dist[i, j])


@njit(parallel=True, cache=True)
def _query_all_kd_l1(X, data, idx_array, idx_start, idx_end, is_leaf, bounds, leaf_of,
                     dist, ind, do_sort):
    n_queries = X.shape[0]
    k_nbrs = dist.shape[1]
    stack_node = np.empty((n_queries, _STACK_CAP), dtype=np.intp)
    stack_lb = np.empty((n_queries, _STACK_CAP), dtype=np.float64)
    for i in prange(n_queries):
        _query_one_kd_l1(X, i, data, idx_array, idx_start, idx_end, is_leaf, bounds,
                         leaf_of, dist, ind, stack_node[i], stack_lb[i])
        if do_sort:
            _sort_row(dist, ind, i, k_nbrs)


@njit(cache=True)
def _query_all_kd_l1_seq(X, data, idx_array, idx_start, idx_end, is_leaf, bounds, leaf_of,
                         dist, ind, do_sort):
    n_queries = X.shape[0]
    k_nbrs = dist.shape[1]
    stack_node = np.empty((_STACK_CAP,), dtype=np.intp)
    stack_lb = np.empty((_STACK_CAP,), dtype=np.float64)
    for i in range(n_queries):
        _query_one_kd_l1(X, i, data, idx_array, idx_start, idx_end, is_leaf, bounds,
                         leaf_of, dist, ind, stack_node, stack_lb)
        if do_sort:
            _sort_row(dist, ind, i, k_nbrs)


@njit(parallel=True, cache=True)
def _query_all(X, data, idx_array, idx_start, idx_end, is_leaf, radius, bounds,
               tree_kind, metric_id, p, w, V, VI, has_w, has_V, has_VI,
               leaf_of, dist, ind, do_sort):
    n_queries = X.shape[0]
    k_nbrs = dist.shape[1]
    stack_node = np.empty((n_queries, _STACK_CAP), dtype=np.intp)
    stack_lb = np.empty((n_queries, _STACK_CAP), dtype=np.float64)
    for i in prange(n_queries):
        _query_one(X, i, data, idx_array, idx_start, idx_end, is_leaf, radius, bounds,
                   tree_kind, metric_id, p, w, V, VI, has_w, has_V, has_VI, dist, ind,
                   leaf_of, stack_node[i], stack_lb[i])
        if do_sort:
            _sort_row(dist, ind, i, k_nbrs)
        for j in range(k_nbrs):
            dist[i, j] = _rdist_to_dist(dist[i, j], metric_id, p)


@njit(cache=True)
def _query_all_seq(X, data, idx_array, idx_start, idx_end, is_leaf, radius, bounds,
                   tree_kind, metric_id, p, w, V, VI, has_w, has_V, has_VI,
                   leaf_of, dist, ind, do_sort):
    n_queries = X.shape[0]
    k_nbrs = dist.shape[1]
    stack_node = np.empty((_STACK_CAP,), dtype=np.intp)
    stack_lb = np.empty((_STACK_CAP,), dtype=np.float64)
    for i in range(n_queries):
        _query_one(X, i, data, idx_array, idx_start, idx_end, is_leaf, radius, bounds,
                   tree_kind, metric_id, p, w, V, VI, has_w, has_V, has_VI, dist, ind,
                   leaf_of, stack_node, stack_lb)
        if do_sort:
            _sort_row(dist, ind, i, k_nbrs)
        for j in range(k_nbrs):
            dist[i, j] = _rdist_to_dist(dist[i, j], metric_id, p)


class NeighborTree:
    def __init__(self, X, leaf_size=40, metric='minkowski', tree_kind=TREE_KD, **kwargs):
        if leaf_size < 1:
            raise ValueError('leaf_size must be greater than or equal to 1')

        X = _as_sample_matrix(X)
        n_samples, n_features = X.shape
        if n_samples == 0:
            raise ValueError('X must contain at least one sample')

        p = kwargs.get('p', 2.0)
        if p is None:
            p = 2.0
        p = float(p)
        if p <= 0:
            raise ValueError('p must be positive')

        self.metric = metric
        self.metric_id = _resolve_metric(metric, p)
        self.angular_mode = 0
        self.p = p
        self.leaf_size = int(leaf_size)
        self.tree_kind = tree_kind
        self.n_samples = n_samples
        self.n_features = n_features
        self.data = X
        self._metric_params = {}

        if self.metric_id in (MET_COSINE, MET_ARCCOS):
            self.angular_mode = self.metric_id
            self.metric_id = MET_EUCLIDEAN
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            tree_data = np.ascontiguousarray(X / np.maximum(norms, 1e-15), dtype=np.float64)
        else:
            tree_data = X
        self._tree_data = tree_data

        if tree_kind == TREE_KD and self.metric_id not in (
                MET_EUCLIDEAN, MET_MANHATTAN, MET_CHEBYSHEV, MET_MINKOWSKI):
            raise ValueError('Metric: %s\nCannot be used with KDTree' % metric)

        if self.metric_id == MET_HAVERSINE and n_features != 2:
            raise ValueError('Haversine metric requires 2 features (lat, lon in radians)')

        w = kwargs.get('w', None)
        V = kwargs.get('V', None)
        VI = kwargs.get('VI', None)
        self.has_weight = 0
        self.has_V = 0
        self.has_VI = 0

        if w is not None:
            self._weight = np.ascontiguousarray(w, dtype=np.float64).reshape(-1)
            if self._weight.shape[0] != n_features:
                raise ValueError('w must have length n_features')
            self.has_weight = 1
            self._metric_params['w'] = self._weight
        else:
            self._weight = np.zeros(1, dtype=np.float64)

        if self.metric_id == MET_SEUCLIDEAN:
            if V is None:
                raise ValueError('Must provide V for seuclidean distance')
            self._V = np.ascontiguousarray(V, dtype=np.float64).reshape(-1)
            if self._V.shape[0] != n_features:
                raise ValueError('V must have length n_features')
            self.has_V = 1
            self._metric_params['V'] = self._V
        else:
            self._V = np.zeros(1, dtype=np.float64)

        if self.metric_id == MET_MAHALANOBIS:
            if VI is None:
                if V is None:
                    raise ValueError('Must provide either V or VI for Mahalanobis distance')
                VI = np.linalg.inv(np.asarray(V, dtype=np.float64))
            VI = np.ascontiguousarray(VI, dtype=np.float64)
            if VI.ndim != 2 or VI.shape[0] != VI.shape[1]:
                raise ValueError('V/VI must be square')
            if VI.shape[0] != n_features:
                raise ValueError('Mahalanobis V/VI size does not match n_features')
            self._VI = VI
            self.has_VI = 1
            self._metric_params['VI'] = self._VI
        else:
            self._VI = np.zeros((1, 1), dtype=np.float64)

        if metric.lower() in ('minkowski', 'p'):
            self._metric_params['p'] = p

        n_levels = 1
        if n_samples > self.leaf_size:
            n_levels = int(math.log2(max(1.0, (n_samples - 1.0) / self.leaf_size))) + 1
        self.n_nodes = (1 << n_levels) - 1

        self._idx_array = np.arange(n_samples, dtype=np.intp)
        self._idx_start = np.zeros(self.n_nodes, dtype=np.intp)
        self._idx_end = np.zeros(self.n_nodes, dtype=np.intp)
        self._is_leaf = np.zeros(self.n_nodes, dtype=np.intp)
        self._radius = np.zeros(self.n_nodes, dtype=np.float64)
        self._bounds = np.zeros((2, self.n_nodes, n_features), dtype=np.float64)
        self._leaf_of = np.zeros(n_samples, dtype=np.intp)

        _build_tree(
            self._tree_data, self._idx_array, self._idx_start, self._idx_end,
            self._is_leaf, self._radius, self._bounds, self.n_nodes,
            self.tree_kind, self.metric_id, self.p, self._weight, self._V, self._VI,
            self.has_weight, self.has_V, self.has_VI,
        )
        _fill_leaf_of(self._idx_array, self._idx_start, self._idx_end, self._is_leaf, self._leaf_of)

    def __reduce__(self):
        cls = KDTree if self.tree_kind == TREE_KD else BallTree
        kwargs = dict(self._metric_params)
        kwargs['metric'] = self.metric
        kwargs['leaf_size'] = self.leaf_size
        return (_rebuild_tree, (cls, np.asarray(self.data), kwargs))

    def query(self, X, k=1, return_distance=True, dualtree=False, breadth_first=False,
              sort_results=True):
        """Return k nearest neighbors for each row in X, excluding the point itself.

        Training sample ``i`` is omitted from the neighbors of query row ``i``.
        ``dualtree`` and ``breadth_first`` are accepted for API compatibility;
        queries use a single-tree depth-first search.
        """
        X = _as_sample_matrix(X, n_features=self.n_features)
        if X.shape[1] != self.n_features:
            raise ValueError('query data dimension must match training data dimension')
        if k < 1:
            raise ValueError('k must be at least 1')
        if k > self.n_samples - 1:
            raise ValueError('k must be less than the number of training points')

        if self.angular_mode:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = np.ascontiguousarray(X / np.maximum(norms, 1e-15), dtype=np.float64)

        n_queries = X.shape[0]
        dist_arr = np.full((n_queries, k), INF, dtype=np.float64)
        ind_arr = np.zeros((n_queries, k), dtype=np.intp)
        do_sort = 1 if sort_results else 0
        parallel = n_queries >= 128
        args = (
            X, self._tree_data, self._idx_array, self._idx_start, self._idx_end,
            self._is_leaf, self._bounds, self._leaf_of, dist_arr, ind_arr, do_sort,
        )
        if self.tree_kind == TREE_KD and self.metric_id == MET_EUCLIDEAN and not self.has_weight:
            (_query_all_kd_l2 if parallel else _query_all_kd_l2_seq)(*args)
        elif self.tree_kind == TREE_KD and self.metric_id == MET_MANHATTAN and not self.has_weight:
            (_query_all_kd_l1 if parallel else _query_all_kd_l1_seq)(*args)
        else:
            generic = (
                X, self._tree_data, self._idx_array, self._idx_start, self._idx_end,
                self._is_leaf, self._radius, self._bounds, self.tree_kind, self.metric_id,
                self.p, self._weight, self._V, self._VI, self.has_weight, self.has_V,
                self.has_VI, self._leaf_of, dist_arr, ind_arr, do_sort,
            )
            (_query_all if parallel else _query_all_seq)(*generic)

        if self.angular_mode == MET_COSINE:
            dist_arr = 0.5 * dist_arr * dist_arr
        elif self.angular_mode == MET_ARCCOS:
            dist_arr = np.arccos(np.clip(1.0 - 0.5 * dist_arr * dist_arr, -1.0, 1.0))

        if return_distance:
            return dist_arr, ind_arr
        return ind_arr


class KDTree(NeighborTree):
    """KD-tree for fast k-nearest neighbor queries (Lp / Minkowski metrics)."""
    valid_metrics = KDTREE_VALID_METRICS

    def __init__(self, X, leaf_size=40, metric='minkowski', **kwargs):
        super().__init__(X, leaf_size=leaf_size, metric=metric, tree_kind=TREE_KD, **kwargs)


class BallTree(NeighborTree):
    """Ball-tree for fast k-nearest neighbor queries (broader metric set)."""
    valid_metrics = BALLTREE_VALID_METRICS

    def __init__(self, X, leaf_size=40, metric='minkowski', **kwargs):
        super().__init__(X, leaf_size=leaf_size, metric=metric, tree_kind=TREE_BALL, **kwargs)


def _rebuild_tree(cls, data, kwargs):
    return cls(data, **kwargs)
