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

cdef class Group:
    cdef:
        np.ndarray data
        np.intp_t __data_length
        np.intp_t _size
        np.intp_t _neg_uniq_edges # negative means it didn't cluster
        assume_data(self, data, s, ue)
        cook_outlier(self, np.double_t border)

        bint will_cluster(self, np.double_t border, Group opp) # main formula

        np.intp_t points(self)
        np.intp_t uniq_edges(self) # returns absolute of _neg_uniq_edges
        np.intp_t get_sibling_id(self, np.intp_t c)

    @staticmethod
    cdef np.intp_t aggregate(data, np.double_t v, np.intp_t is_cluster1, Group group1, np.intp_t is_cluster2, Group group2)

    @staticmethod
    cdef np.intp_t add_child_id_and_get_sibling(data, np.intp_t c)
    @staticmethod
    cdef void form_mutual_closest_2p_cluster(data, np.double_t border)
    @staticmethod
    cdef void set_outliers(data, np.intp_t count)        
