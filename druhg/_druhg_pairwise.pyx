# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# kNN over a full or sparse precomputed pairwise distance matrix
# Author: Pavel Artamonov
# License: 3-clause BSD

import sys
import logging

import numpy as np
cimport numpy as np

cdef np.double_t INF = sys.float_info.max


def _resolve_knn_scope(knn_scope, query_ids, n_samples):
    n_q = query_ids.shape[0]
    scope_in = np.asarray(knn_scope, dtype=np.intp)
    if scope_in.ndim == 0:
        return np.full(n_q, int(scope_in), dtype=np.intp)
    scope = np.ascontiguousarray(scope_in.reshape(-1))
    if scope.shape[0] == n_samples:
        return np.ascontiguousarray(scope[query_ids])
    if scope.shape[0] == n_q:
        return scope
    raise ValueError('knn_scope must have length n_samples or match indices')


def _pad_extend_rows(rows_d, rows_i, knn_skip, max_k):
    n_q = len(rows_d)
    width = max_k
    for q in range(n_q):
        width = max(width, len(rows_d[q]))
    dist = np.zeros((n_q, width), dtype=np.float64)
    ind = np.zeros((n_q, width), dtype=np.intp)
    for q in range(n_q):
        n = len(rows_d[q])
        if n:
            dist[q, :n] = rows_d[q]
            ind[q, :n] = rows_i[q]
    return knn_skip, dist, ind


cdef class PairwiseDistanceTreeSparse(object):
    cdef object data_arr
    cdef int data_size

    def __init__(self, N, d):
        self.data_size = N
        self.data_arr = d

    cpdef tuple query_init(self, k=1, dualtree=0, breadth_first=0, sort_results=1, return_distance=1):
        # TODO: Reciprocity of absent link is not the same as INF. Do reciprocity with graphs!
        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices
        cdef np.double_t val
        cdef np.intp_t i, j, pos, q, n_q

        n_q = self.data_size
        knn_dist = INF * np.ones((n_q, k))
        knn_indices = np.zeros((n_q, k), dtype=np.intp)
        warning = 0

        q = n_q
        while q:
            q -= 1
            i = q
            row = self.data_arr.getrow(i)
            idx, data = row.indices, row.data
            sorted = np.argsort(data)
            pos = 0
            for s in sorted:
                j = idx[s]
                if j == i:
                    warning += 1
                    continue
                val = data[s]
                if pos >= k:
                    break
                knn_dist[q][pos] = val
                knn_indices[q][pos] = j
                pos += 1

        if warning:
            logging.getLogger(__package__).warning(
                'Attention!: Sparse matrix has an edge that forms a loop! They were zeroed. ' + str(warning))
        return knn_dist, knn_indices

    cpdef tuple query_extend(self, indices, skip_radii, knn_scope):
        cdef np.ndarray[np.intp_t, ndim=1] query_ids, k_q, knn_skip
        cdef np.ndarray[np.double_t, ndim=1] skip_arr
        cdef np.double_t r_i, val, tau
        cdef np.intp_t i, j, q, n_q, k, warning

        query_ids = np.ascontiguousarray(np.asarray(indices, dtype=np.intp).reshape(-1))
        n_q = query_ids.shape[0]
        skip_arr = np.asarray(skip_radii, dtype=np.float64).reshape(-1)
        if skip_arr.shape[0] != self.data_size:
            raise ValueError('skip_radii must have length n_samples')
        k_q = _resolve_knn_scope(knn_scope, query_ids, self.data_size)
        if n_q == 0:
            width = 1 if k_q.size == 0 else max(int(k_q.max()), 1)
            return (np.empty(0, dtype=np.intp),
                    np.empty((0, width), dtype=np.float64),
                    np.empty((0, width), dtype=np.intp))
        if int(k_q.min()) < 1:
            raise ValueError('knn_scope values must be at least 1')
        if int(k_q.max()) > self.data_size - 1:
            raise ValueError('knn_scope values must be less than the number of training points')

        knn_skip = np.zeros(n_q, dtype=np.intp)
        rows_d = []
        rows_i = []
        warning = 0
        for q in range(n_q):
            i = query_ids[q]
            r_i = skip_arr[i]
            k = int(k_q[q])
            row = self.data_arr.getrow(i)
            idx, data = row.indices, row.data
            keep_d = []
            keep_i = []
            yi = 0
            for s in np.argsort(data):
                j = idx[s]
                if j == i:
                    warning += 1
                    continue
                val = data[s]
                if val < r_i:
                    yi += 1
                    continue
                keep_d.append(val)
                keep_i.append(j)
            knn_skip[q] = yi
            if len(keep_d) > k:
                tau = keep_d[k - 1]
                n_keep = k
                while n_keep < len(keep_d) and keep_d[n_keep] <= tau:
                    n_keep += 1
                keep_d = keep_d[:n_keep]
                keep_i = keep_i[:n_keep]
            rows_d.append(keep_d)
            rows_i.append(keep_i)
        if warning:
            logging.getLogger(__package__).warning(
                'Attention!: Sparse matrix has an edge that forms a loop! They were zeroed. ' + str(warning))
        return _pad_extend_rows(rows_d, rows_i, knn_skip, int(k_q.max()))


cdef class PairwiseDistanceTreeGeneric(object):
    cdef object data_arr
    cdef int data_size

    def __init__(self, N, d):
        self.data_size = N
        self.data_arr = d

    cpdef tuple query_init(self, k=1, dualtree=0, breadth_first=0, sort_results=1, return_distance=1):
        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices
        cdef np.double_t val
        cdef np.intp_t i, j, pos, q, n_q

        n_q = self.data_size
        knn_dist = INF * np.ones((n_q, k))
        knn_indices = np.zeros((n_q, k), dtype=np.intp)

        q = n_q
        while q:
            q -= 1
            i = q
            row = self.data_arr[i]
            sorted = np.argsort(row)
            pos = 0
            for j in sorted:
                if j == i:
                    continue
                val = row[j]
                if pos >= k:
                    break
                knn_dist[q][pos] = val
                knn_indices[q][pos] = j
                pos += 1
        return knn_dist, knn_indices

    cpdef tuple query_extend(self, indices, skip_radii, knn_scope):
        cdef np.ndarray[np.intp_t, ndim=1] query_ids, k_q, knn_skip
        cdef np.ndarray[np.double_t, ndim=1] skip_arr
        cdef np.double_t r_i, val, tau
        cdef np.intp_t i, j, q, n_q, k

        query_ids = np.ascontiguousarray(np.asarray(indices, dtype=np.intp).reshape(-1))
        n_q = query_ids.shape[0]
        skip_arr = np.asarray(skip_radii, dtype=np.float64).reshape(-1)
        if skip_arr.shape[0] != self.data_size:
            raise ValueError('skip_radii must have length n_samples')
        k_q = _resolve_knn_scope(knn_scope, query_ids, self.data_size)
        if n_q == 0:
            width = 1 if k_q.size == 0 else max(int(k_q.max()), 1)
            return (np.empty(0, dtype=np.intp),
                    np.empty((0, width), dtype=np.float64),
                    np.empty((0, width), dtype=np.intp))
        if int(k_q.min()) < 1:
            raise ValueError('knn_scope values must be at least 1')
        if int(k_q.max()) > self.data_size - 1:
            raise ValueError('knn_scope values must be less than the number of training points')

        knn_skip = np.zeros(n_q, dtype=np.intp)
        rows_d = []
        rows_i = []
        for q in range(n_q):
            i = query_ids[q]
            r_i = skip_arr[i]
            k = int(k_q[q])
            row = self.data_arr[i]
            keep_d = []
            keep_i = []
            yi = 0
            for j in np.argsort(row):
                if j == i:
                    continue
                val = row[j]
                if val < r_i:
                    yi += 1
                    continue
                keep_d.append(val)
                keep_i.append(j)
            knn_skip[q] = yi
            if len(keep_d) > k:
                tau = keep_d[k - 1]
                n_keep = k
                while n_keep < len(keep_d) and keep_d[n_keep] <= tau:
                    n_keep += 1
                keep_d = keep_d[:n_keep]
                keep_i = keep_i[:n_keep]
            rows_d.append(keep_d)
            rows_i.append(keep_i)
        return _pad_extend_rows(rows_d, rows_i, knn_skip, int(k_q.max()))
