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

cdef class GroupPlacement:
    cdef:
        np.intp_t sum_ids
        np.double_t sum_edges, net_edges, net_densities
        np.ndarray sum_coords, net_coords, center_shift, side_shift

    cdef void add_single_point(self, np.double_t v, np.intp_t id, coords)
    cdef void add_node(self, np.double_t border, np.intp_t id, GroupPlacement node, np.intp_t size, bint is_cluster)
    cdef np.intp_t get_sibling_id(self, np.intp_t id)
    cdef GroupPlacement cook_outlier_coords(self, np.ndarray coords, np.ndarray center_shift, np.ndarray side_shift=?)
