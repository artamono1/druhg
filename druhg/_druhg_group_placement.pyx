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

    int _IDX_MTN_SUM_EDGES = _IDX_UF_BOTH_CHILDREN + 1
    int _IDX_MTN_SUM_ORIGINAL_EDGES = _IDX_UF_BOTH_CHILDREN + 2
    int _IDX_MTN_SUM_COORDS = _IDX_UF_BOTH_CHILDREN + 3
    int _IDX_MTN_SUM_CLUSTER_COORDS = _IDX_UF_BOTH_CHILDREN + 4
    int _IDX_MTN_SUM_VECTOR = _IDX_UF_BOTH_CHILDREN + 5
    int _IDX_MTN_DENSITIES = _IDX_UF_BOTH_CHILDREN + 6

# EXPERIMENTAL
cdef class GroupPlacement (Group):
    # declarations are in pxd file
    # https://cython.readthedocs.io/en/latest/src/userguide/sharing_declarations.html

    def __init__(self, data):
        super().__init__(data)

    @staticmethod
    cdef void form_coords_mutual_closest_2p_cluster(data, np.double_t border, np.ndarray coords1, np.ndarray coords2):
        data[_IDX_MTN_SUM_EDGES] = border
        data[_IDX_MTN_SUM_ORIGINAL_EDGES] = border
        data[_IDX_MTN_SUM_COORDS] = coords1 + coords2
        data[_IDX_MTN_SUM_CLUSTER_COORDS] = data[_IDX_MTN_SUM_COORDS] # could be more than 2 points in case of 0 distance
        data[_IDX_MTN_DENSITIES] = border*0.5

    cdef void cook_outlier_coords(self, np.ndarray coords):
        self.data[_IDX_MTN_SUM_CLUSTER_COORDS] = coords
        self.data[_IDX_MTN_SUM_COORDS] = coords
        self._size = 1
        self._neg_uniq_edges = 0

    @staticmethod
    cdef void aggregate_coords(data, np.double_t v, bint is_cluster1, GroupPlacement group1, bint is_cluster2, GroupPlacement group2):
        cdef np.intp_t i

        i = _IDX_MTN_SUM_EDGES
        data[i] = v \
                       + (((group1._size-1) * v) if is_cluster1 else group1.data[i]) \
                       + (((group2._size-1) * v) if is_cluster2 else group2.data[i])

        i = _IDX_MTN_SUM_ORIGINAL_EDGES
        data[i] = v + group1.data[i] + group2.data[i]

        i = _IDX_MTN_SUM_COORDS
        data[i] = group1.data[i] + group2.data[i]
        #TODO: merge mean?  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        i = _IDX_MTN_SUM_CLUSTER_COORDS
        data[i] = (((group1.data[_IDX_MTN_SUM_COORDS]/group1._size) if is_cluster1 else group1.data[i])  \
                      + ((group2.data[_IDX_MTN_SUM_COORDS]/group2._size) if is_cluster2 else group2.data[i]))

        i = _IDX_MTN_DENSITIES #  di/(ni-1⁰) sum per connector
        same_parent_points = group1._size * is_cluster1 + group2._size * is_cluster2
        data[i] = (0 if is_cluster1 else group1.data[i]) + (0 if is_cluster2 else group2.data[i]) \
                       + ((v/same_parent_points) if (is_cluster1 or is_cluster2) else 0)

        # print(same_parent_points, self.data[i], '=',
        #       (0 if is_cluster1 else group1.data[i]), (0 if is_cluster2 else group2.data[i]) \
        #                , ((v/same_parent_points) if (is_cluster1 or is_cluster2) else 0))


# ------displacement---first_run-------------------
    cdef void restart_current_linking(self, np.ndarray e_coords):
        self.current_linked_edge = 0.
        self.current_sum_of_linked_tree = 0.
        self.point_place = e_coords
    
    cdef void current_clusterization_update(self, np.double_t v, GroupPlacement ogroup):
        print('linked', self._neg_uniq_edges < 0 , self.current_linked_edge, v)
        self.current_linked_edge = self.current_linked_edge if self._neg_uniq_edges < 0 else v
        print('og_linked_weights', self._neg_uniq_edges < 0, self.current_sum_of_linked_tree, \
                (self._size + ((ogroup._size) if ogroup._neg_uniq_edges > 0 else 0)))
        # og_linked_weights = og_linked_weights if x_group._neg_uniq_edges < 0 else v * (x_group._size + ((y_group._size) if y_group._neg_uniq_edges > 0 else 0))
        self.current_sum_of_linked_tree = self.current_sum_of_linked_tree if self._neg_uniq_edges < 0 else \
                    (self.data[_IDX_MTN_SUM_ORIGINAL_EDGES] + v \
                       + (0 if ogroup._neg_uniq_edges < 0 else ogroup.data[_IDX_MTN_SUM_ORIGINAL_EDGES]))


    cdef np.ndarray evaluate_and_add_center_shift(self, np.double_t v, GroupPlacement pgroup, GroupPlacement ogroup):
        # В ноде хранится сумма до кластеризации, координаты до кластеризации
        # и отдельно чистая сумма и чистые координаты
        # чтобы получить после кластеризации, смотрим на знак количества кластеров внутри ноды
        
        print('OGcenters (p= x, y)', pgroup.data[_IDX_MTN_SUM_COORDS] / pgroup._size, 
                                     self.data[_IDX_MTN_SUM_COORDS] / self._size, 
                                     ogroup.data[_IDX_MTN_SUM_COORDS] / ogroup._size)
        # print('BCcenters (p= x, y)', p_group.get_center(0), x_group.get_center(0), y_group.get_center(0))
        # print('ACcenters (p= x, y)', p_group.get_center(1), x_group.get_center(1), y_group.get_center(1))
        # print('OGweights (p(+v)= x, y)', p_group.get_og_weight(), x_group.get_og_weight(), y_group.get_og_weight())
        # print('BCweights (p(+v)= x, y)', p_group.get_weight(0), x_group.get_weight(0), y_group.get_weight(0))
        # print('ACweights (p(+v)= x, y)', p_group.get_weight(v), x_group.get_weight(v), y_group.get_weight(v))

        #ФОРМУЛА!!!
        # связующее ребро кластера * (++все рёбра кластера)/(++все рёбра соединения) * (++все плотности кластеров)
        # делить на Р (1/N1 + 1/N2)

        self.current_clusterization_update(v, ogroup)
        
        wormhole = 1. * self.current_linked_edge * self.current_sum_of_linked_tree * pgroup.data[_IDX_MTN_DENSITIES]
        wormhole /= pgroup.data[_IDX_MTN_SUM_ORIGINAL_EDGES] * v * (1./self._size + 1./ogroup._size)

        # Вектор
        shift = (pgroup.data[_IDX_MTN_SUM_COORDS] / pgroup._size - self.point_place) * (-1. if self._neg_uniq_edges < 0 else 1.)
        print(np.linalg.norm(shift), 'act', shift, '=', (-1. if self._neg_uniq_edges < 0 else 1.), ' * ' , '(e - ', 
            ogroup.data[_IDX_MTN_SUM_COORDS] / ogroup._size )
        shift /= np.linalg.norm(shift)
        print('norm', shift, )
        shift *= wormhole
        print('weight', shift, "wormhole {:.2f}".format(wormhole), self.current_linked_edge * self.current_sum_of_linked_tree, 
            pgroup.data[_IDX_MTN_DENSITIES])

        self.data[_IDX_MTN_SUM_VECTOR] += shift
        return shift

# ------displacement---second_run------------------
    cdef np.ndarray evaluate_shift(self, GroupPlacement pgroup, GroupPlacement ogroup):
        central_shift = ogroup.data[_IDX_MTN_SUM_VECTOR] / ogroup._size
        print(central_shift, 'central_shift')
        return central_shift * self._size / pgroup._size

    cdef void cook_outlier_center_shift(self, np.double_t v, GroupPlacement pgroup, GroupPlacement ogroup):
        cdef np.double_t wormhole
        cdef np.double_t linked_edge = v
        cdef np.double_t current_sum_of_linked_tree = v + (0 if ogroup._neg_uniq_edges < 0 else ogroup.data[_IDX_MTN_SUM_ORIGINAL_EDGES])
        
        print(v, current_sum_of_linked_tree)
        wormhole = 1. * linked_edge * current_sum_of_linked_tree * pgroup.data[_IDX_MTN_DENSITIES]
        wormhole /= pgroup.data[_IDX_MTN_SUM_ORIGINAL_EDGES] * v * (1. / ogroup._size + 1. / self._size)

        shift = pgroup.data[_IDX_MTN_SUM_COORDS] / pgroup._size - self.data[_IDX_MTN_SUM_COORDS]
        shift /= np.linalg.norm(shift)

        ogroup.data[_IDX_MTN_SUM_VECTOR] = shift*wormhole
        print(shift*wormhole, 'opp outlier', shift, wormhole)
