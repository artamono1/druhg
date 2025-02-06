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

# cdef int _IDX_POINTS = 0 # size has to be first!!!
# cdef int _IDX_UNIQ_EDGES = 1
cdef int _IDX_SUM_RECIPROCALS = 2

cdef int _IDX_UF_HAS_LEFT_CHILD = _IDX_SUM_RECIPROCALS + 1 # for clustering by UF
# cdef int _IDX_DICT_SIZES = 6

cdef int _IDX_MTN_SUM_EDGES = _IDX_UF_HAS_LEFT_CHILD + 1
cdef int _IDX_MTN_WILL_CLUSTER = _IDX_UF_HAS_LEFT_CHILD + 2
cdef int _IDX_MTN_SUM_COORDS = _IDX_UF_HAS_LEFT_CHILD + 3

cdef np.double_t _group_PRECISION = 0.0000001

cdef set_precision(np.double_t prec):
    _group_PRECISION = prec

def allocate_buffer_groups(np.intp_t size, np.intp_t n_dim=0):
    fields = [("points", np.intp),
              ("uniq_edges", np.intp),
              ("sum_reciprocals", np.double),
              ("has_left_child", np.intp),
              # ("dict_points", dict),
     ]
    if n_dim != 0: # for motion
        fields.append(("sum_edges", np.double)),
        fields.append(("will_cluster", np.intp)),
        fields.append(("sum_coords", np.double, n_dim))


    dtype = np.dtype(fields, align=True)
    return np.empty(size, dtype=dtype)

def allocate_buffer_clusters(np.intp_t num_points):
    return np.empty((num_points - 1), dtype=np.intp)

def allocate_buffer_sizes(np.intp_t num_points):
    return np.empty((num_points - 1), dtype=np.intp)

cdef class Group (object):
    # declarations are in pxd file
    # https://cython.readthedocs.io/en/latest/src/userguide/sharing_declarations.html

    def __init__(self, data):
        self.data = data
        self.__data_length = len(data)
        self._size = 0
        self._neg_uniq_edges = 0 # edges are negative until proven clusters


    cdef assume_data(self, data, s, ue):
        self.data = data
        self._size = s
        self._neg_uniq_edges = ue

    cdef points(self):
        # return self.data[_IDX_POINTS]
        return self._size
    cdef uniq_edges(self):
        # return self.data[_IDX_UNIQ_EDGES]
        return self._neg_uniq_edges

    cdef aggregate(self, np.double_t v, np.intp_t is_cluster1, Group group1, np.intp_t is_cluster2, Group group2):
    # cdef aggregate(self, v, is_cluster1, group1, is_cluster2, group2):

        cdef np.intp_t i

        self._size = group1._size + group2._size

        # edges are negative until proven clusters
        self._neg_uniq_edges = -1 + \
                               (0 if is_cluster1 else group1._neg_uniq_edges) + \
                               (0 if is_cluster2 else group2._neg_uniq_edges)

        i = _IDX_SUM_RECIPROCALS
        self.data[i] = (0 if not is_cluster1 and not is_cluster2 else 1./v) \
                       + ((group1.data[i]) if not is_cluster1 else 0) \
                       + ((group2.data[i]) if not is_cluster2 else 0)

    cdef add_1_autocluster(self, np.double_t border):
        if self._size == 1:
            self._size = 2
            self._neg_uniq_edges = -1
            self.data[_IDX_SUM_RECIPROCALS] = 1./border
            return

        self._size += 1
        self._neg_uniq_edges += -1
        self.data[_IDX_SUM_RECIPROCALS] += 1. / border

    cdef cook_outlier(self, np.double_t border):
        self._size = 1
        # self.data[_IDX_UNIQ_EDGES] = 0
        self._neg_uniq_edges = 0
        self.data[_IDX_SUM_RECIPROCALS] = 0

    cdef child(self, np.intp_t c, is_outlier):
        if self.data[_IDX_UF_HAS_LEFT_CHILD] == 0: # setter
            self.data[_IDX_UF_HAS_LEFT_CHILD] = -1 if is_outlier else c+1
            return False, False, 0
        # getter
        return True, self.data[_IDX_UF_HAS_LEFT_CHILD] == -1, self.data[_IDX_UF_HAS_LEFT_CHILD]-1

    cdef will_cluster(self, np.double_t border, np.double_t neg_common_coef):
        cdef bint is_cluster

        new_form = 1. * border * self.data[_IDX_SUM_RECIPROCALS] * self._neg_uniq_edges * neg_common_coef
        old_shells = self._size - 1 + _group_PRECISION
        is_cluster = new_form > old_shells
        # print("{:.2f}".format(border),
        #       'is_cluster', "{:.2f}".format(new_form / old_shells),
        #       new_form > old_shells,
        #       "{:.1f}".format(new_form),
        #       '>', "{:.1f}".format(old_shells),
        #       'N/K', "{:.2f}".format(-neg_common_coef),
        #       'clusters vs size', -self._neg_uniq_edges, '/', self._size,
        #       'avg.border',
        # )

        return is_cluster

# ----------------------------
    cdef mtn_mark_cluster(self, bint is_cluster):
        self.data[_IDX_MTN_WILL_CLUSTER] = is_cluster
    cdef mtn_need_cluster(self):
        return self.data[_IDX_MTN_WILL_CLUSTER]

    cdef mtn_change_sum_edges(self, v):
        self.data[_IDX_MTN_SUM_EDGES] = v * (self._size - 1)
    cdef mtn_weight(self):
        return self.data[_IDX_MTN_SUM_EDGES]

# -----------------------
    cdef mtn_set_like(self, base_group):
        cdef np.intp_t i
        i = self.__data_length
        while i != 0:
            i -= 1
            self.data[i] = base_group[i]

    cdef mtn_subtract(self, ogroup, v):
        i = self.__data_length
        while i!=0:
            i -= 1
            self.data[i] -= ogroup[i]
        self.data[_IDX_MTN_SUM_EDGES] -= v

    cdef mtn_aggregate(self, v, is_cluster1, group1, is_cluster2, group2):
        cdef np.intp_t i

        i = _IDX_MTN_SUM_EDGES
        self.data[i] = v + (((group1[0]-1) * v) if is_cluster1 else group1[i]) + \
                       (((group2[0]-1) * v) if is_cluster2 else group2[i])

        i = _IDX_MTN_SUM_COORDS
        self.data[i] += group1[i] + group2[i]
        #TODO: merge mean?  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

#----------for motion purposes------------------
    cdef mtn_coords(self):
        return self.data[_IDX_MTN_SUM_COORDS]
    cdef mtn_set_coords(self, coords):
        self.data[_IDX_MTN_SUM_COORDS] = coords

    cdef mtn_add_1_coords(self, coords, border):
        self.data[_IDX_MTN_SUM_COORDS] += coords
        if self._size == 1:
            self.data[_IDX_MTN_SUM_EDGES] = border

    cdef mtn_center(self):
        return self.data[_IDX_MTN_SUM_COORDS] / self._size

# ------------------- debug

    # cdef aggregate_dict(self, v, is_cluster1, group1, is_cluster2, group2):
    #
    #     i = _IDX_DICT_SIZES
    #     d = dict()
    #     if is_cluster1 and is_cluster2:
    #         d[self.data[_IDX_POINTS]] = 1
    #     elif is_cluster1:
    #         d = group2[i].copy()
    #         k = group1[_IDX_POINTS]
    #         d[k] = d.get(k, 0) + 1
    #     elif is_cluster2:
    #         d = group1[i].copy()
    #         k = group2[_IDX_POINTS]
    #         d[k] = d.get(k, 0) + 1
    #     else:
    #         d = group1[i].copy()
    #         for k in group2[i]:
    #             d[k] = d.get(k, 0) + group2[i][k]
    #     self.data[i] = d

