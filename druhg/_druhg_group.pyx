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

cdef:
    int _IDX_SUM_EDGES_OVER_ESTIMATES = 0 #  ni-1⁰ / nidi sum per linked
    int _IDX_COUNT_OUTLIERS = 1 # one edge with one point
    int _IDX_UF_BOTH_CHILDREN = 2 # storing id+id

cdef np.double_t _group_PRECISION = 0.0000001

cdef set_precision(np.double_t prec):
    _group_PRECISION = prec

def allocate_buffer_groups(np.intp_t size, np.intp_t n_dim=0):
    fields = [
              ("sum_edges_over_estimates", np.double),  #  ni-1⁰ / nidi sum per linked
              ("count_outliers", np.intp),  # one edge with one point
              ("both_children_id", np.intp),
     ]
    if n_dim != 0: # for motion
        fields.append(("sum_edges", np.double))
        fields.append(("sum_original_edges", np.double))
        fields.append(("sum_coords", np.double, n_dim))
        fields.append(("sum_cluster_coords", np.double, n_dim))
        fields.append(("sum_vector_shift", np.double, n_dim))
        fields.append(("densities", np.double)) #  di/ni sum per linked

    dtype = np.dtype(fields, align=True)
    return np.empty(size, dtype=dtype)

def allocate_buffer_clusters(np.intp_t num_points):
    return np.empty((num_points - 1), dtype=np.intp)

def allocate_buffer_sizes(np.intp_t num_points):
    return np.empty((num_points - 1), dtype=np.intp)

cdef class Group:
    # declarations are in pxd file
    # https://cython.readthedocs.io/en/latest/src/userguide/sharing_declarations.html

    def __init__(self, data):
        self.data = data
        self.__data_length = len(data)
        self._size = 0
        self._neg_uniq_edges = 0 # edges are negative until proven clusters

    cdef assume_data(self, data, np.intp_t s, np.intp_t ue): # плохо сделано, надо отдельный метод
        self.data = data
        self._size = s
        self._neg_uniq_edges = ue

    cdef np.intp_t points(self):
        return self._size

    cdef np.intp_t uniq_edges(self): # edges are negative until proven clusters
        return self._neg_uniq_edges if self._neg_uniq_edges>=0 else -self._neg_uniq_edges

    @staticmethod
    cdef void set_outliers(data, np.intp_t count):
        data[_IDX_COUNT_OUTLIERS] = count

    @staticmethod
    cdef np.intp_t add_child_id_and_get_sibling(data, np.intp_t c):
        cdef np.intp_t sibling_id
        sibling_id = data[_IDX_UF_BOTH_CHILDREN]
        data[_IDX_UF_BOTH_CHILDREN] += c
        return sibling_id 

    @staticmethod
    cdef np.intp_t aggregate(data, np.double_t v, bint is_cluster1, Group group1, bint is_cluster2, Group group2):
        cdef np.intp_t i, res

        # self._size = group1._size + group2._size

        # edges are negative until proven clusters # double clusters merge
        res = (0 if is_cluster1 else group1._neg_uniq_edges) + (0 if is_cluster2 else group2._neg_uniq_edges) \
                                + (-1 if (is_cluster1 or is_cluster2) else 0)

        same_parent_points = group1._size * is_cluster1 + group2._size * is_cluster2

        i = _IDX_SUM_EDGES_OVER_ESTIMATES #  ni-1⁰ / nidi sum per connector
        data[i] = (0 if is_cluster1 else group1.data[i]) + (0 if is_cluster2 else group2.data[i]) \
                       + ((1./v) if same_parent_points==1 else 0)  \
                       + (((same_parent_points - 1.) / (v * same_parent_points)) if (is_cluster1 or is_cluster2) else 0)

        i = _IDX_COUNT_OUTLIERS  # one edge with one point
        data[i] = (0 if is_cluster1 else group1.data[i]) + (0 if is_cluster2 else group2.data[i]) \
                       + (1. if same_parent_points == 1 else 0)
        return res

    @staticmethod
    cdef void form_mutual_closest_2p_cluster(data, np.double_t border):
        data[_IDX_SUM_EDGES_OVER_ESTIMATES] = 0.5 / border if border!=0. else 0.
        data[_IDX_COUNT_OUTLIERS] = 0

    cdef cook_outlier(self, np.double_t border):
        cdef np.intp_t i
        i = self.__data_length
        while i != 0:
            i -= 1
            self.data[i] = 0
        self.data[_IDX_SUM_EDGES_OVER_ESTIMATES] = 1./border
        self.data[_IDX_COUNT_OUTLIERS] = 1
        self._size = 1
        self._neg_uniq_edges = 0

    cdef np.intp_t get_sibling_id(self, np.intp_t c):
        assert c >= 0
        return self.data[_IDX_UF_BOTH_CHILDREN] - c


        
    cdef bint will_cluster(self, np.double_t border, Group opp):
        cdef bint is_cluster

        # 1. Double cluster merge: _neg_uniq_edges != #clusters 
        # 2. _IDX_SUM_EDGES_OVER_ESTIMATES = ni-1⁰ / nidi sum per linked. 
        new_form =  self._neg_uniq_edges  * border * opp._size * self.data[_IDX_SUM_EDGES_OVER_ESTIMATES]
        old_shells = 1. * (self._neg_uniq_edges + opp._neg_uniq_edges) \
                     * (self._size + self.data[_IDX_COUNT_OUTLIERS] + self._neg_uniq_edges) # linked edges
        is_cluster = new_form <= old_shells - _group_PRECISION

        # print("   {:.2f}".format(border),
        #       'is_cluster', "{:.2f}".format(new_form / old_shells),
        #       abs(new_form) > abs(old_shells) + _group_PRECISION,
        #       "{:.1f}".format(-new_form),
        #       "> {:.1f}".format(-old_shells),
        #       'sum ni-1 / nidi', "{:.2f}".format(self.data[_IDX_SUM_EDGES_OVER_ESTIMATES]),
        #       'clusters ',  -self._neg_uniq_edges, ' SSS', self._size, ' ouls', self.data[_IDX_COUNT_OUTLIERS],
        #       ' vs opp clusters ',  -opp._neg_uniq_edges,' SSS', opp.points(), ' ouls', opp.data[_IDX_COUNT_OUTLIERS],
        #       'opp ni-1 / nidi'.format(opp.data[_IDX_SUM_EDGES_OVER_ESTIMATES]),
        #       abs(new_form), abs(old_shells)
        # )

        return is_cluster
