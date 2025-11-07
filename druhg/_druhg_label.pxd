# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Author: Pavel Artamonov
# License: 3-clause BSD

import numpy as np
cimport numpy as np

from ._druhg_unionfind import UnionFind
from ._druhg_unionfind cimport UnionFind

cdef class Clusterizer:
    cdef UnionFind _U
    cdef np.ndarray _values_arr
    cdef np.ndarray _data_arr # for placement only
    cdef np.ndarray group_arr
    cdef np.ndarray ret_sizes
    cdef np.ndarray ret_clusters

    cpdef emerge(self, precision=?, run_motion=?)
    cdef emerge_still(self)
    cdef emerge_placement(self)

    cpdef np.ndarray label(self, np.ndarray ret_labels, list exclude=?, size_range=?, np.intp_t fix_outliers=?, edgepairs_arr=?, precision=?)
    cdef void _fixem(self, np.ndarray edges_arr, np.intp_t num_edges, np.ndarray result)
    cdef _mark_labels(self, ret_labels, list exclude=?, np.intp_t limitL=?, np.intp_t limitH=?, )
