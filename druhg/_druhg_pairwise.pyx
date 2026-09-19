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


cdef class PairwiseDistanceTreeSparse(object):
    cdef object data_arr
    cdef int data_size

    def __init__(self, N, d):
        self.data_size = N
        self.data_arr = d

    cpdef tuple query(self, d, k, dualtree=0, breadth_first=0, n_jobs=None):
        # TODO: actually we need to consider replacing INF with something else.
        # Reciprocity of absent link is not the same as the INF. Do reciprocity with graphs!
        # n_jobs is accepted for API parity with NeighborTree.query; unused here.
        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices

        knn_dist = INF*np.ones((self.data_size, k))
        knn_indices = np.zeros((self.data_size, k), dtype=np.intp)

        warning = 0

        i = self.data_size
        while i:
            i -= 1
            row = self.data_arr.getrow(i)
            idx, data = row.indices, row.data
            sorted = np.argsort(data)
            pos = 0
            for s in sorted:
                j = idx[s]
                if j == i:
                    warning += 1
                    continue
                if pos >= k:
                    break
                knn_dist[i][pos] = data[s]
                knn_indices[i][pos] = j
                pos += 1

        if warning:
            logging.getLogger(__package__).warning('Attention!: Sparse matrix has an edge that forms a loop! They were zeroed. '+str(warning))

        return knn_dist, knn_indices


cdef class PairwiseDistanceTreeGeneric(object):
    cdef object data_arr
    cdef int data_size

    def __init__(self, N, d):
        self.data_size = N
        self.data_arr = d

    cpdef tuple query(self, d, k, dualtree=0, breadth_first=0, n_jobs=None):
        # n_jobs is accepted for API parity with NeighborTree.query; unused here.
        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices

        knn_dist = np.zeros((self.data_size, k))
        knn_indices = np.zeros((self.data_size, k), dtype=np.intp)

        i = self.data_size
        while i:
            i -= 1
            row = self.data_arr[i]
            sorted = np.argsort(row)
            pos = 0
            for j in sorted:
                if j == i:
                    continue
                knn_dist[i][pos] = row[j]
                knn_indices[i][pos] = j
                pos += 1
                if pos == k:
                    break

        return knn_dist, knn_indices
