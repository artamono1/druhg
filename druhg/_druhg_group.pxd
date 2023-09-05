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
cdef bint group_is_cluster(group, np.double_t *emerged_quantity_quality, np.double_t border)

cdef class Group (object):
    cdef:
        data
        __data_length
        assume_data(self, data)
        size(self)
        amount(self)
        limit(self)

        add_1(self, np.double_t border)
        aggregate(self, ogroup)
        subtract(self, ogroup)
        add_cluster(self, np.double_t emerged_quantity_quality, emerged_cluster)
        set_like(self, ogroup)
# --------- for motion only
        coords(self)
        set_coords(self, coords)
        assume_data_coords(self, group_coords_data)
        add_coords(self, e_coords)
        add_cluster_coords(self, np.double_t emerged_quantity_quality, emerged_cluster)
