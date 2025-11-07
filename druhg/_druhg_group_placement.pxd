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

from ._druhg_group cimport Group

# EXPERIMENTAL
cdef class GroupPlacement (Group):
  # during label aggregation
    cdef void cook_outlier_coords(self, np.ndarray coords)
    @staticmethod    
    cdef void form_coords_mutual_closest_2p_cluster(data, np.double_t border, np.ndarray coords1, np.ndarray coords2)
    @staticmethod
    cdef void aggregate_coords(data, np.double_t v, bint is_cluster1, GroupPlacement group1, bint is_cluster2, GroupPlacement group2)

  # during displacement evaluation
    cdef: 
        np.double_t current_linked_edge
        np.double_t current_sum_of_linked_tree
        void restart_current_linking(self, np.ndarray e_coords)
        void current_clusterization_update(self, np.double_t v, GroupPlacement ogroup)

        np.ndarray point_place

    cdef:
        np.ndarray evaluate_and_add_center_shift(self, np.double_t v, GroupPlacement pgroup, GroupPlacement ogroup)
        np.ndarray evaluate_shift(self, GroupPlacement pgroup, GroupPlacement ogroup)
        void cook_outlier_center_shift(self, np.double_t v, GroupPlacement pgroup, GroupPlacement ogroup)
