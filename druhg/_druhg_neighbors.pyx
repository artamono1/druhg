# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# KD-tree and Ball-tree k-nearest neighbor queries for DRUHG.
# Author: Pavel Artamonov
# License: 3-clause BSD

from libc.math cimport fabs, sqrt, pow, fmax, fmin, sin, cos, asin, acos, log2
from libc.math cimport INFINITY as C_INF

import numpy as np
cimport numpy as np

cdef np.float64_t INF = C_INF

cdef int TREE_KD = 0
cdef int TREE_BALL = 1

cdef int MET_EUCLIDEAN = 0
cdef int MET_MANHATTAN = 1
cdef int MET_CHEBYSHEV = 2
cdef int MET_MINKOWSKI = 3
cdef int MET_SEUCLIDEAN = 4
cdef int MET_MAHALANOBIS = 5
cdef int MET_HAMMING = 6
cdef int MET_CANBERRA = 7
cdef int MET_BRAYCURTIS = 8
cdef int MET_JACCARD = 9
cdef int MET_DICE = 10
cdef int MET_ROGERSTANIMOTO = 11
cdef int MET_RUSSELLRAO = 12
cdef int MET_SOKALMICHENER = 13
cdef int MET_SOKALSNEATH = 14
cdef int MET_HAVERSINE = 15
cdef int MET_COSINE = 16
cdef int MET_ARCCOS = 17

cdef dict METRIC_CODES = {
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


cdef inline void _swap_i(np.intp_t* arr, np.intp_t i, np.intp_t j) noexcept nogil:
    cdef np.intp_t tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp


cdef int _resolve_metric(str metric, np.float64_t p) except -1:
    cdef str name = metric.lower()
    cdef int code
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


cdef inline np.float64_t _boolean_counts(const np.float64_t* x1,
                                         const np.float64_t* x2,
                                         np.intp_t n,
                                         np.intp_t* ntt,
                                         np.intp_t* ntf,
                                         np.intp_t* nft) noexcept nogil:
    cdef np.intp_t j, tt = 0, tf = 0, ft = 0
    cdef bint a, b
    for j in range(n):
        a = x1[j] != 0
        b = x2[j] != 0
        if a and b:
            tt += 1
        elif a:
            tf += 1
        elif b:
            ft += 1
    ntt[0] = tt
    ntf[0] = tf
    nft[0] = ft
    return 0


cdef inline np.float64_t _rdist(const np.float64_t* x1, const np.float64_t* x2,
                                np.intp_t n, int metric_id, np.float64_t p,
                                const np.float64_t* w, const np.float64_t* V,
                                const np.float64_t* VI) noexcept nogil:
    cdef np.intp_t j, i
    cdef np.float64_t d = 0, diff, tmp, denom, acc, n1, n2, sim
    cdef np.intp_t ntt, ntf, nft, n_neq

    if metric_id == MET_EUCLIDEAN:
        for j in range(n):
            diff = x1[j] - x2[j]
            d += diff * diff
        return d

    if metric_id == MET_MANHATTAN:
        if w != NULL:
            for j in range(n):
                d += w[j] * fabs(x1[j] - x2[j])
        else:
            for j in range(n):
                d += fabs(x1[j] - x2[j])
        return d

    if metric_id == MET_CHEBYSHEV:
        for j in range(n):
            diff = fabs(x1[j] - x2[j])
            if diff > d:
                d = diff
        return d

    if metric_id == MET_MINKOWSKI:
        if w != NULL:
            for j in range(n):
                d += w[j] * pow(fabs(x1[j] - x2[j]), p)
        else:
            for j in range(n):
                d += pow(fabs(x1[j] - x2[j]), p)
        return d

    if metric_id == MET_SEUCLIDEAN:
        for j in range(n):
            diff = x1[j] - x2[j]
            d += diff * diff / V[j]
        return d

    if metric_id == MET_MAHALANOBIS:
        for i in range(n):
            tmp = 0
            for j in range(n):
                tmp += VI[i * n + j] * (x1[j] - x2[j])
            d += tmp * (x1[i] - x2[i])
        return d

    if metric_id == MET_HAMMING:
        for j in range(n):
            if x1[j] != x2[j]:
                d += 1
        return d / n

    if metric_id == MET_CANBERRA:
        for j in range(n):
            denom = fabs(x1[j]) + fabs(x2[j])
            if denom > 0:
                d += fabs(x1[j] - x2[j]) / denom
        return d

    if metric_id == MET_BRAYCURTIS:
        acc = 0
        for j in range(n):
            d += fabs(x1[j] - x2[j])
            acc += fabs(x1[j] + x2[j])
        if acc == 0:
            return 0
        return d / acc

    if metric_id == MET_HAVERSINE:
        tmp = sin(0.5 * (x1[0] - x2[0]))
        diff = sin(0.5 * (x1[1] - x2[1]))
        d = tmp * tmp + cos(x1[0]) * cos(x2[0]) * diff * diff
        if d < 0:
            d = 0
        elif d > 1:
            d = 1
        return 2 * asin(sqrt(d))

    if metric_id == MET_COSINE or metric_id == MET_ARCCOS:
        acc = 0
        n1 = 0
        n2 = 0
        for j in range(n):
            acc += x1[j] * x2[j]
            n1 += x1[j] * x1[j]
            n2 += x2[j] * x2[j]
        denom = sqrt(n1) * sqrt(n2)
        if denom == 0:
            sim = 1 if n1 == 0 and n2 == 0 else 0
        else:
            sim = acc / denom
            if sim > 1:
                sim = 1
            elif sim < -1:
                sim = -1
        if metric_id == MET_ARCCOS:
            return acos(sim)
        return 1 - sim

    _boolean_counts(x1, x2, n, &ntt, &ntf, &nft)
    n_neq = ntf + nft

    if metric_id == MET_JACCARD:
        denom = ntt + n_neq
        return 0 if denom == 0 else n_neq / denom

    if metric_id == MET_DICE:
        denom = 2 * ntt + n_neq
        return 0 if denom == 0 else n_neq / denom

    if metric_id == MET_ROGERSTANIMOTO or metric_id == MET_SOKALMICHENER:
        denom = n + n_neq
        return 0 if denom == 0 else (2 * n_neq) / denom

    if metric_id == MET_RUSSELLRAO:
        return (n - ntt) / n

    if metric_id == MET_SOKALSNEATH:
        denom = 0.5 * ntt + n_neq
        return 0 if denom == 0 else n_neq / denom

    return 0


cdef inline np.float64_t _dist(const np.float64_t* x1, const np.float64_t* x2,
                               np.intp_t n, int metric_id, np.float64_t p,
                               const np.float64_t* w, const np.float64_t* V,
                               const np.float64_t* VI) noexcept nogil:
    cdef np.float64_t r = _rdist(x1, x2, n, metric_id, p, w, V, VI)
    return _rdist_to_dist(r, metric_id, p)


cdef inline np.float64_t _rdist_to_dist(np.float64_t r, int metric_id,
                                        np.float64_t p) noexcept nogil:
    if metric_id == MET_EUCLIDEAN or metric_id == MET_SEUCLIDEAN or metric_id == MET_MAHALANOBIS:
        return sqrt(r)
    if metric_id == MET_MINKOWSKI:
        return pow(r, 1.0 / p)
    return r


cdef inline np.float64_t _dist_to_rdist(np.float64_t d, int metric_id,
                                        np.float64_t p) noexcept nogil:
    if metric_id == MET_EUCLIDEAN or metric_id == MET_SEUCLIDEAN or metric_id == MET_MAHALANOBIS:
        return d * d
    if metric_id == MET_MINKOWSKI:
        return pow(d, p)
    return d


cdef inline np.float64_t _kd_min_rdist(np.float64_t[:, :, ::1] bounds,
                                       np.intp_t i_node, const np.float64_t* pt,
                                       np.intp_t n, int metric_id, np.float64_t p,
                                       const np.float64_t* w) noexcept nogil:
    cdef np.intp_t j
    cdef np.float64_t d = 0, delta, lo, hi, v
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
            d += delta if w == NULL else w[j] * delta
        elif metric_id == MET_MINKOWSKI:
            d += pow(delta, p) if w == NULL else w[j] * pow(delta, p)
        else:
            d += delta * delta
    return d


cdef inline np.float64_t _ball_min_rdist(np.float64_t[:, :, ::1] bounds,
                                         np.float64_t[::1] radius,
                                         np.intp_t i_node, const np.float64_t* pt,
                                         np.intp_t n, int metric_id, np.float64_t p,
                                         const np.float64_t* w, const np.float64_t* V,
                                         const np.float64_t* VI) noexcept nogil:
    cdef np.float64_t d
    d = _dist(pt, &bounds[0, i_node, 0], n, metric_id, p, w, V, VI) - radius[i_node]
    if d < 0:
        d = 0
    return _dist_to_rdist(d, metric_id, p)


cdef inline int _heap_push(np.float64_t[:, ::1] dist, np.intp_t[:, ::1] ind,
                           np.intp_t row, np.float64_t val, np.intp_t i_val) noexcept nogil:
    cdef np.intp_t i, ic1, ic2, i_swap, size
    cdef np.float64_t* dist_arr
    cdef np.intp_t* ind_arr

    dist_arr = &dist[row, 0]
    ind_arr = &ind[row, 0]
    size = dist.shape[1]

    if val > dist_arr[0]:
        return 0

    dist_arr[0] = val
    ind_arr[0] = i_val
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1
        if ic1 >= size:
            break
        if ic2 >= size:
            if dist_arr[ic1] > val:
                i_swap = ic1
            else:
                break
        elif dist_arr[ic1] >= dist_arr[ic2]:
            if val < dist_arr[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if val < dist_arr[ic2]:
                i_swap = ic2
            else:
                break
        dist_arr[i] = dist_arr[i_swap]
        ind_arr[i] = ind_arr[i_swap]
        i = i_swap
    dist_arr[i] = val
    ind_arr[i] = i_val
    return 0


cdef void _sort_row(np.float64_t* dist, np.intp_t* idx, np.intp_t size) noexcept nogil:
    cdef np.intp_t i, j
    cdef np.float64_t dkey
    cdef np.intp_t ikey
    for i in range(1, size):
        dkey = dist[i]
        ikey = idx[i]
        j = i
        while j > 0 and dist[j - 1] > dkey:
            dist[j] = dist[j - 1]
            idx[j] = idx[j - 1]
            j -= 1
        dist[j] = dkey
        idx[j] = ikey


cdef np.intp_t _split_dim(const np.float64_t* data, const np.intp_t* node_indices,
                          np.intp_t n_features, np.intp_t n_points) noexcept nogil:
    cdef np.intp_t i, j, j_max = 0
    cdef np.float64_t min_val, max_val, val, spread, max_spread = 0
    for j in range(n_features):
        min_val = data[node_indices[0] * n_features + j]
        max_val = min_val
        for i in range(1, n_points):
            val = data[node_indices[i] * n_features + j]
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
        spread = max_val - min_val
        if spread > max_spread:
            max_spread = spread
            j_max = j
    return j_max


cdef void _partition_indices(const np.float64_t* data, np.intp_t* node_indices,
                             np.intp_t split_dim, np.intp_t split_index,
                             np.intp_t n_features, np.intp_t n_points) noexcept nogil:
    cdef np.intp_t left = 0, right = n_points - 1, mid, i
    cdef np.float64_t d1, d2
    while True:
        mid = left
        for i in range(left, right):
            d1 = data[node_indices[i] * n_features + split_dim]
            d2 = data[node_indices[right] * n_features + split_dim]
            if d1 < d2:
                _swap_i(node_indices, i, mid)
                mid += 1
        _swap_i(node_indices, mid, right)
        if mid == split_index:
            return
        if mid < split_index:
            left = mid + 1
        else:
            right = mid - 1
        if left > right:
            return


cdef class NeighborTree:
    cdef readonly np.ndarray data
    cdef readonly np.intp_t n_samples
    cdef readonly np.intp_t n_features
    cdef readonly np.intp_t leaf_size
    cdef readonly object metric

    cdef np.ndarray _idx_arr
    cdef np.ndarray _idx_start_arr
    cdef np.ndarray _idx_end_arr
    cdef np.ndarray _is_leaf_arr
    cdef np.ndarray _radius_arr
    cdef np.ndarray _bounds_arr
    cdef np.ndarray _weight_arr
    cdef np.ndarray _V_arr
    cdef np.ndarray _VI_arr
    cdef np.ndarray _tree_data_arr

    cdef np.float64_t[:, ::1] data_m
    cdef np.intp_t[::1] idx_array
    cdef np.intp_t[::1] idx_start
    cdef np.intp_t[::1] idx_end
    cdef np.intp_t[::1] is_leaf
    cdef np.float64_t[::1] radius
    cdef np.float64_t[:, :, ::1] bounds
    cdef np.float64_t[::1] weight
    cdef np.float64_t[::1] V
    cdef np.float64_t[:, ::1] VI

    cdef np.intp_t n_nodes
    cdef np.intp_t tree_kind
    cdef int metric_id
    cdef np.float64_t p
    cdef bint has_weight
    cdef bint has_V
    cdef bint has_VI
    cdef int angular_mode
    cdef dict _metric_params

    def __init__(self, X, leaf_size=40, metric='minkowski', tree_kind=TREE_KD, **kwargs):
        cdef np.intp_t n_samples, n_features, n_levels, i
        cdef np.float64_t p
        cdef object w, V, VI

        if leaf_size < 1:
            raise ValueError('leaf_size must be greater than or equal to 1')

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError('X must be a 2-dimensional array')
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)
        n_samples = X.shape[0]
        n_features = X.shape[1]
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

        if self.metric_id == MET_COSINE or self.metric_id == MET_ARCCOS:
            self.angular_mode = self.metric_id
            self.metric_id = MET_EUCLIDEAN
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            tree_data = np.ascontiguousarray(X / norms, dtype=np.float64)
        else:
            tree_data = X
        self._tree_data_arr = tree_data
        self.data_m = self._tree_data_arr

        if tree_kind == TREE_KD and self.metric_id not in (
                MET_EUCLIDEAN, MET_MANHATTAN, MET_CHEBYSHEV, MET_MINKOWSKI):
            raise ValueError('Metric: %s\nCannot be used with KDTree' % metric)

        if self.metric_id == MET_HAVERSINE and n_features != 2:
            raise ValueError('Haversine metric requires 2 features (lat, lon in radians)')

        self.has_weight = False
        self.has_V = False
        self.has_VI = False
        w = kwargs.get('w', None)
        V = kwargs.get('V', None)
        VI = kwargs.get('VI', None)

        if w is not None:
            self._weight_arr = np.ascontiguousarray(w, dtype=np.float64).reshape(-1)
            if self._weight_arr.shape[0] != n_features:
                raise ValueError('w must have length n_features')
            self.weight = self._weight_arr
            self.has_weight = True
            self._metric_params['w'] = self._weight_arr
        else:
            self._weight_arr = np.zeros(1, dtype=np.float64)
            self.weight = self._weight_arr

        if self.metric_id == MET_SEUCLIDEAN:
            if V is None:
                raise ValueError('Must provide V for seuclidean distance')
            self._V_arr = np.ascontiguousarray(V, dtype=np.float64).reshape(-1)
            if self._V_arr.shape[0] != n_features:
                raise ValueError('V must have length n_features')
            self.V = self._V_arr
            self.has_V = True
            self._metric_params['V'] = self._V_arr
        else:
            self._V_arr = np.zeros(1, dtype=np.float64)
            self.V = self._V_arr

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
            self._VI_arr = VI
            self.VI = self._VI_arr
            self.has_VI = True
            self._metric_params['VI'] = self._VI_arr
        else:
            self._VI_arr = np.zeros((1, 1), dtype=np.float64)
            self.VI = self._VI_arr

        if metric.lower() in ('minkowski', 'p'):
            self._metric_params['p'] = p

        n_levels = 1
        if n_samples > self.leaf_size:
            n_levels = <np.intp_t>log2(fmax(1.0, (n_samples - 1.0) / self.leaf_size)) + 1
        self.n_nodes = (1 << n_levels) - 1

        self._idx_arr = np.arange(n_samples, dtype=np.intp)
        self._idx_start_arr = np.zeros(self.n_nodes, dtype=np.intp)
        self._idx_end_arr = np.zeros(self.n_nodes, dtype=np.intp)
        self._is_leaf_arr = np.zeros(self.n_nodes, dtype=np.intp)
        self._radius_arr = np.zeros(self.n_nodes, dtype=np.float64)
        self._bounds_arr = np.zeros((2, self.n_nodes, n_features), dtype=np.float64)

        self.idx_array = self._idx_arr
        self.idx_start = self._idx_start_arr
        self.idx_end = self._idx_end_arr
        self.is_leaf = self._is_leaf_arr
        self.radius = self._radius_arr
        self.bounds = self._bounds_arr

        self._recursive_build(0, 0, n_samples)

    def __reduce__(self):
        cls = KDTree if self.tree_kind == TREE_KD else BallTree
        kwargs = dict(self._metric_params)
        kwargs['metric'] = self.metric
        kwargs['leaf_size'] = self.leaf_size
        return (_rebuild_tree, (cls, np.asarray(self.data), kwargs))

    cdef const np.float64_t* _wptr(self) noexcept nogil:
        if self.has_weight:
            return &self.weight[0]
        return NULL

    cdef const np.float64_t* _Vptr(self) noexcept nogil:
        if self.has_V:
            return &self.V[0]
        return NULL

    cdef const np.float64_t* _VIptr(self) noexcept nogil:
        if self.has_VI:
            return &self.VI[0, 0]
        return NULL

    cdef void _init_node(self, np.intp_t i_node, np.intp_t idx_start, np.intp_t idx_end) noexcept nogil:
        cdef np.intp_t i, j, n_points, idx
        cdef np.intp_t n = self.n_features
        cdef np.float64_t val, r, d
        cdef const np.float64_t* data = &self.data_m[0, 0]
        cdef np.float64_t* centroid
        cdef const np.float64_t* w = self._wptr()
        cdef const np.float64_t* V = self._Vptr()
        cdef const np.float64_t* VI = self._VIptr()

        self.idx_start[i_node] = idx_start
        self.idx_end[i_node] = idx_end
        n_points = idx_end - idx_start

        if self.tree_kind == TREE_KD:
            for j in range(n):
                val = data[self.idx_array[idx_start] * n + j]
                self.bounds[0, i_node, j] = val
                self.bounds[1, i_node, j] = val
            for i in range(idx_start + 1, idx_end):
                idx = self.idx_array[i]
                for j in range(n):
                    val = data[idx * n + j]
                    if val < self.bounds[0, i_node, j]:
                        self.bounds[0, i_node, j] = val
                    if val > self.bounds[1, i_node, j]:
                        self.bounds[1, i_node, j] = val
            self.radius[i_node] = 0.5 * _dist(&self.bounds[0, i_node, 0],
                                              &self.bounds[1, i_node, 0],
                                              n, self.metric_id, self.p, w, V, VI)
        else:
            centroid = &self.bounds[0, i_node, 0]
            for j in range(n):
                centroid[j] = 0
            for i in range(idx_start, idx_end):
                idx = self.idx_array[i]
                for j in range(n):
                    centroid[j] += data[idx * n + j]
            for j in range(n):
                centroid[j] /= n_points
            r = 0
            for i in range(idx_start, idx_end):
                idx = self.idx_array[i]
                d = _dist(centroid, data + idx * n, n, self.metric_id, self.p, w, V, VI)
                if d > r:
                    r = d
            self.radius[i_node] = r

    cdef void _recursive_build(self, np.intp_t i_node, np.intp_t idx_start, np.intp_t idx_end) noexcept nogil:
        cdef np.intp_t n_points = idx_end - idx_start
        cdef np.intp_t n_mid = n_points // 2
        cdef np.intp_t i_max
        cdef np.intp_t* idx_ptr = &self.idx_array[idx_start]
        cdef const np.float64_t* data = &self.data_m[0, 0]

        self._init_node(i_node, idx_start, idx_end)

        if 2 * i_node + 1 >= self.n_nodes or n_points < 2:
            self.is_leaf[i_node] = 1
            return

        self.is_leaf[i_node] = 0
        i_max = _split_dim(data, idx_ptr, self.n_features, n_points)
        _partition_indices(data, idx_ptr, i_max, n_mid, self.n_features, n_points)
        self._recursive_build(2 * i_node + 1, idx_start, idx_start + n_mid)
        self._recursive_build(2 * i_node + 2, idx_start + n_mid, idx_end)

    cdef np.float64_t _min_rdist(self, np.intp_t i_node, const np.float64_t* pt) noexcept nogil:
        if self.tree_kind == TREE_KD:
            return _kd_min_rdist(self.bounds, i_node, pt, self.n_features,
                                 self.metric_id, self.p, self._wptr())
        return _ball_min_rdist(self.bounds, self.radius, i_node, pt, self.n_features,
                               self.metric_id, self.p, self._wptr(), self._Vptr(), self._VIptr())

    cdef void _query_depthfirst(self, np.intp_t i_node, const np.float64_t* pt,
                                np.intp_t i_pt, np.float64_t[:, ::1] dist,
                                np.intp_t[:, ::1] ind,
                                np.float64_t reduced_lb) noexcept nogil:
        cdef np.intp_t i, i1, i2, idx
        cdef np.float64_t d, lb1, lb2
        cdef const np.float64_t* data = &self.data_m[0, 0]
        cdef const np.float64_t* w = self._wptr()
        cdef const np.float64_t* V = self._Vptr()
        cdef const np.float64_t* VI = self._VIptr()

        if reduced_lb > dist[i_pt, 0]:
            return

        if self.is_leaf[i_node]:
            for i in range(self.idx_start[i_node], self.idx_end[i_node]):
                idx = self.idx_array[i]
                d = _rdist(pt, data + idx * self.n_features, self.n_features,
                           self.metric_id, self.p, w, V, VI)
                if d <= dist[i_pt, 0]:
                    _heap_push(dist, ind, i_pt, d, idx)
            return

        i1 = 2 * i_node + 1
        i2 = i1 + 1
        lb1 = self._min_rdist(i1, pt)
        lb2 = self._min_rdist(i2, pt)
        if lb1 <= lb2:
            self._query_depthfirst(i1, pt, i_pt, dist, ind, lb1)
            self._query_depthfirst(i2, pt, i_pt, dist, ind, lb2)
        else:
            self._query_depthfirst(i2, pt, i_pt, dist, ind, lb2)
            self._query_depthfirst(i1, pt, i_pt, dist, ind, lb1)

    def query(self, X, k=1, return_distance=True, dualtree=False, breadth_first=False,
              sort_results=True):
        """Return k nearest neighbors for each row in X.

        ``dualtree`` and ``breadth_first`` are accepted for API compatibility;
        queries use a single-tree depth-first search.
        """
        cdef np.intp_t n_queries, i, j, k_nbrs
        cdef np.float64_t[:, ::1] Xarr
        cdef np.float64_t[:, ::1] dist_m
        cdef np.intp_t[:, ::1] ind_m
        cdef np.ndarray dist_arr, ind_arr
        cdef const np.float64_t* pt
        cdef np.float64_t lb
        cdef bint do_sort = sort_results

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[X.ndim - 1] != self.n_features:
            raise ValueError('query data dimension must match training data dimension')
        if k < 1:
            raise ValueError('k must be at least 1')
        if k > self.n_samples:
            raise ValueError('k must be less than or equal to the number of training points')
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)
        if self.angular_mode:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = np.ascontiguousarray(X / np.maximum(norms, 1e-15), dtype=np.float64)

        Xarr = X
        n_queries = Xarr.shape[0]
        k_nbrs = k
        dist_arr = np.full((n_queries, k_nbrs), INF, dtype=np.float64)
        ind_arr = np.zeros((n_queries, k_nbrs), dtype=np.intp)
        dist_m = dist_arr
        ind_m = ind_arr

        with nogil:
            for i in range(n_queries):
                pt = &Xarr[i, 0]
                lb = self._min_rdist(0, pt)
                self._query_depthfirst(0, pt, i, dist_m, ind_m, lb)
                if do_sort:
                    _sort_row(&dist_m[i, 0], &ind_m[i, 0], k_nbrs)
                for j in range(k_nbrs):
                    dist_m[i, j] = _rdist_to_dist(dist_m[i, j], self.metric_id, self.p)

        if self.angular_mode == MET_COSINE:
            dist_arr *= dist_arr
            dist_arr *= 0.5
        elif self.angular_mode == MET_ARCCOS:
            dist_arr = np.arccos(np.clip(1.0 - 0.5 * dist_arr * dist_arr, -1.0, 1.0))

        if return_distance:
            return dist_arr, ind_arr
        return ind_arr


cdef class KDTree(NeighborTree):
    """KD-tree for fast k-nearest neighbor queries (Lp / Minkowski metrics)."""
    valid_metrics = KDTREE_VALID_METRICS

    def __init__(self, X, leaf_size=40, metric='minkowski', **kwargs):
        super().__init__(X, leaf_size=leaf_size, metric=metric, tree_kind=TREE_KD, **kwargs)


cdef class BallTree(NeighborTree):
    """Ball-tree for fast k-nearest neighbor queries (broader metric set)."""
    valid_metrics = BALLTREE_VALID_METRICS

    def __init__(self, X, leaf_size=40, metric='minkowski', **kwargs):
        super().__init__(X, leaf_size=leaf_size, metric=metric, tree_kind=TREE_BALL, **kwargs)


def _rebuild_tree(cls, data, kwargs):
    return cls(data, **kwargs)
