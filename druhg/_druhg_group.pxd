# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# group structure that can become a cluster
# Author: Pavel Artamonov
# License: 3-clause BSD

import numpy as np
cimport numpy as np

cdef set_precision(np.double_t prec)

cdef struct GroupNode:
    np.double_t sum_reciprocals  # 1 / di sum per linked

cdef class Group:
    cdef:
        np.intp_t _size
        np.intp_t _neg_uniq_edges  # negative means it didn't cluster
        np.intp_t points(self)
        np.intp_t uniq_edges(self)  # returns absolute of _neg_uniq_edges

    @staticmethod
    cdef np.intp_t will_cluster(np.intp_t size, np.intp_t edges, GroupNode* node,
                                 np.double_t border,
                                 np.intp_t osize, np.intp_t oedges, GroupNode* onode)

    @staticmethod
    cdef np.intp_t aggregate(np.intp_t size, np.intp_t edges, GroupNode* node,
                              np.double_t v,
                              np.intp_t osize, np.intp_t oedges, GroupNode* onode)
