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

def allocate_buffer_groups(np.intp_t size, np.intp_t n_dim=0):
    if n_dim == 0:  # no motion, only clustering
        return None
    fields = [
        ("both_children_id", np.intp),
        ("sum_edges", np.double),
        ("sum_original_edges", np.double),
        ("sum_coords", np.double, n_dim),
        ("sum_cluster_coords", np.double, n_dim),
        ("sum_vector_shift", np.double, n_dim),
        ("densities", np.double)
    ]
    dtype = np.dtype(fields, align=True)
    return np.empty(size, dtype=dtype)


cdef class GroupPlacement:
    # declarations are in pxd file
    # https://cython.readthedocs.io/en/latest/src/userguide/sharing_declarations.html

    def __init__(self, ndim):
        self.sum_ids = 0
        self.sum_edges = 0
        self.net_edges = 0
        self.net_densities = 0
        self.sum_coords = np.zeros(ndim, dtype=np.double)
        self.net_coords = np.zeros(ndim, dtype=np.double)
        self.center_shift = np.zeros(ndim, dtype=np.double)
        self.side_shift = np.zeros(ndim, dtype=np.double)

    cdef void add_single_point(self, np.double_t border, np.intp_t id, coords):
        self.sum_ids += id
        self.sum_edges += border * 0.5
        self.sum_coords += coords
        self.net_edges += border * 0.5
        self.net_coords += coords
        self.net_densities += border

    cdef void add_node(self, np.double_t border, np.intp_t id, GroupPlacement node, np.intp_t size, bint is_cluster):
        self.sum_ids += id
        self.sum_edges += border * 0.5 + node.sum_edges
        self.sum_coords += node.sum_coords
        # TODO: merge mean? https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        self.net_edges += border * 0.5 + ((border * (size - 1)) if is_cluster else node.net_edges)
        self.net_coords += (node.sum_coords / size) if is_cluster else node.net_coords
        # have to be reevaluated if another party is cluster
        self.net_densities += (border / size) if is_cluster else node.net_densities

    cdef np.intp_t get_sibling_id(self, np.intp_t id):
        return self.sum_ids - id

    cdef GroupPlacement cook_outlier_coords(self, np.ndarray coords, np.ndarray center_shift, np.ndarray side_shift=None):
        self.sum_ids = -1
        self.sum_edges = 0
        self.sum_coords = coords
        self.net_edges = 0
        self.net_coords = coords
        self.net_densities = 0
        self.center_shift = center_shift
        if side_shift is None:
            side_shift = center_shift
        self.side_shift = side_shift
        return self
