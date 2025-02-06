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

from libc.math cimport fabs, pow

# def allocate_buffer_groups(np.intp_t size, np.intp_t n_dim)

cdef set_precision(np.double_t prec)

cdef class Group (object):
    cdef:
        data
        __data_length
        _size
        _neg_uniq_edges
        assume_data(self, data, s, ue)
        mtn_set_like(self, ogroup)

        points(self)
        uniq_edges(self)
        child(self, np.intp_t c, is_outlier)
        mtn_coords(self)
        mtn_center(self)
        mtn_weight(self)

        aggregate(self, np.double_t v, np.intp_t is_cluster1, Group group1, np.intp_t is_cluster2, Group group2)
        mtn_aggregate(self, v, is_cluster1, group1, is_cluster2, group2)
        mtn_subtract(self, ogroup, v)
        add_1_autocluster(self, np.double_t border)
        cook_outlier(self, np.double_t border)
        mtn_add_1_coords(self, coords, border)
        mtn_set_coords(self, coords)

        will_cluster(self, np.double_t border, np.double_t neg_common_coef)
        mtn_mark_cluster(self, bint is_cluster)
        mtn_need_cluster(self)
        mtn_change_sum_edges(self, v)
