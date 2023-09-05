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
#TODO: вставить Гегеля?

cdef int _IDX_SIZE=0 # size has to be first!!!
cdef int _IDX_CLUSTERS=1
cdef int _IDX_SUM_RECIP_COUNTS=2
cdef int _IDX_SUM_AVG_ENERGY=3

cdef int _IDX_SUM_RECIP_QUALS=4
cdef int _IDX_SUM_COORDS=5
cdef int _IDX_SUM_CENTERS=6
cdef int _IDX_SUM_W_COORDS=7
cdef int _IDX_SUM_W_CENTERS=8

cdef np.double_t _group_PRECISION = 0.0000001

cdef set_precision(np.double_t prec):
    _group_PRECISION = prec

def allocate_buffer_groups(np.intp_t size, np.intp_t n_dim=0):
    fields = [("size", np.intp),
              ("clusters", np.intp),
              ("sum_recip_counts", np.double),
              ("sum_energy", np.double),
     ]
    if n_dim != 0: # for motion

        fields.append(("sum_recip_quals", np.double))
        fields.append(("sum_coords", np.double, n_dim))
        fields.append(("sum_centers", np.double, n_dim))
        fields.append(("sum_w_coords", np.double, n_dim))
        fields.append(("sum_w_centers", np.double, n_dim))

    dtype = np.dtype(fields, align=True)
    return np.empty(size, dtype=dtype)

cdef bint group_is_cluster(group, np.double_t *emerged_quantity_quality, np.double_t border):
    emerged_quantity_quality[0] = border * group[_IDX_SUM_RECIP_COUNTS]
    # print(border, group[_IDX_SUM_AVG_ENERGY] < emerged_quantity_quality[0] - _group_PRECISION * group[_IDX_SUM_RECIP_COUNTS],
    #       group[_IDX_SUM_AVG_ENERGY], '<', emerged_quantity_quality[0],
    #       'size', group[_IDX_SIZE], group[_IDX_CLUSTERS])
    return group[_IDX_SUM_AVG_ENERGY] < emerged_quantity_quality[0] - _group_PRECISION * group[_IDX_SUM_RECIP_COUNTS]

cdef class Group (object):
    # declarations are in pxd file
    # https://cython.readthedocs.io/en/latest/src/userguide/sharing_declarations.html

    def __init__(self, data):
        self.data = data
        self.__data_length = len(data)

    cdef assume_data(self, data):
        self.data = data

    cdef size(self):
        return self.data[_IDX_SIZE]
    cdef amount(self):
        return self.data[_IDX_CLUSTERS]
    cdef limit(self):
        return self.data[_IDX_SUM_AVG_ENERGY]

    # cdef set_clusterize(self, emerged_quantity_quality):
    #     cluster_size = self.data[_IDX_SIZE]
    #     self.data[_IDX_CLUSTERS] = 1.
    #     self.data[_IDX_SUM_RECIP_COUNTS] = 1. / cluster_size
    #     self.data[_IDX_SUM_AVG_ENERGY] = emerged_quantity_quality / cluster_size
    #     self.data[_IDX_SUM_RECIP_QUALS] = cluster_size / emerged_quantity_quality

    cdef add_1(self, np.double_t border):
        self.data[_IDX_SIZE] += 1
        self.data[_IDX_CLUSTERS] += 1
        self.data[_IDX_SUM_RECIP_COUNTS] += 1.
        self.data[_IDX_SUM_AVG_ENERGY] += border
        # self.data[_IDX_SUM_RECIP_QUALS] += 1. / border

    cdef aggregate(self, ogroup):
        i = self.__data_length
        while i!=0:
            i -= 1
            self.data[i] += ogroup[i]

    cdef subtract(self, ogroup):
        i = self.__data_length
        while i!=0:
            i -= 1
            self.data[i] -= ogroup[i]

    cdef add_cluster(self, np.double_t emerged_quantity_quality, emerged_cluster):
        clusters_amount = emerged_cluster[_IDX_CLUSTERS]
        self.data[_IDX_SIZE] += emerged_cluster[_IDX_SIZE]
        self.data[_IDX_CLUSTERS] += 1.
        self.data[_IDX_SUM_RECIP_COUNTS] += 1. / clusters_amount
        self.data[_IDX_SUM_AVG_ENERGY] += emerged_quantity_quality / clusters_amount

        #TODO: merge mean?  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        # print ('was', emerged_cluster[_IDX_SIZE], emerged_cluster[_IDX_CLUSTERS], emerged_cluster[_IDX_SUM_RECIP_COUNTS], emerged_cluster[_IDX_SUM_AVG_ENERGY],
        #        'as part', 1. / emerged_cluster[_IDX_CLUSTERS], emerged_quantity_quality / emerged_cluster[_IDX_CLUSTERS] )

        # self.data[_IDX_SUM_RECIP_QUALS] += cluster_size / emerged_quantity_quality

    cdef set_like(self, base_group):
        cdef np.intp_t i
        i = self.__data_length
        while i != 0:
            i -= 1
            self.data[i] = base_group[i]

#----------for motion purposes------------------
    cdef coords(self):
        return self.data[_IDX_SUM_COORDS]
    cdef set_coords(self, coords):
        self.data[_IDX_SUM_COORDS] = coords

    cdef assume_data_coords(self, group_coords_data):
        self.assume_data(group_coords_data)
    cdef add_coords(self, e_coords):
        self.data[_IDX_SUM_COORDS] += e_coords

    cdef add_cluster_coords(self, np.double_t emerged_quantity_quality, emerged_cluster):
        clusters_size = emerged_cluster[_IDX_SIZE]
        clusters_amount = emerged_cluster[_IDX_CLUSTERS]
        self.data[_IDX_SUM_CENTERS] += emerged_cluster[_IDX_SUM_COORDS] / clusters_size
        self.data[_IDX_SUM_W_COORDS] += emerged_cluster[_IDX_SUM_COORDS] * emerged_quantity_quality
        self.data[_IDX_SUM_W_CENTERS] += emerged_cluster[_IDX_SUM_COORDS] * (emerged_quantity_quality / clusters_amount)
