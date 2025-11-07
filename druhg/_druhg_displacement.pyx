# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Produces the next position of the datapoint
# uses results from the tree edges
# logarythmic climb from point to hierarchy of parents
# first run evaluates parent nodes, second run adds the displacement 

# Author: Pavel Artamonov
# License: 3-clause BSD

import numpy as np
cimport numpy as np

import copy

from ._druhg_group cimport set_precision

from ._druhg_group_placement import GroupPlacement
from ._druhg_group_placement cimport GroupPlacement

from ._druhg_unionfind import UnionFind
from ._druhg_unionfind cimport UnionFind

cdef move_points(UnionFind U, np.ndarray values_arr,
                 np.ndarray group_arr, np.ndarray data_arr,
                 np.ndarray sizes_arr, np.ndarray clusters_arr, np.ndarray ret_arr):
    cdef:
        np.intp_t e, p, i, j, \
            loop_size = U.p_size, \
            offset = U.get_offset()
        np.double_t v
        np.ndarray e_coords, shift

    print('EXPERIMENTAL!')
    print("move_points", group_arr, data_arr, sizes_arr, clusters_arr)

    # helpers
    x_group = GroupPlacement(np.zeros_like(group_arr[:1])[0])
    x_outlier = GroupPlacement(np.zeros_like(group_arr[:1])[0])
    y_group = GroupPlacement(np.zeros_like(group_arr[:1])[0])
    y_outlier = GroupPlacement(np.zeros_like(group_arr[:1])[0])
    p_group = GroupPlacement(np.zeros_like(group_arr[:1])[0])

    
    # first loop: combining pursuit to a center
    for e in range(loop_size):
        e_coords = data_arr[e]
        x_group.assume_data(x_outlier.data,1,0)
        x_group.cook_outlier_coords(e_coords)
        x_group.restart_current_linking(e_coords)

        print('===', e, 'point', e_coords)
        assert all(ret_arr[e] == e_coords)
        p = U.parent[e]
        i = e
        while p != 0:
            p -= offset
            v = values_arr[p]

            p_group.assume_data(group_arr[p], sizes_arr[p], clusters_arr[p])
            j = p_group.get_sibling_id(i)
            
            if p_group._size - x_group._size > 1: # y_is_not_outlier            
                y_group.assume_data(group_arr[j], sizes_arr[j], clusters_arr[j])
            else:
                y_group.assume_data(y_outlier.data, 1, 0)
                y_group.cook_outlier_coords(data_arr[j])

            print('id', i-1, j, 'p', p)
            print('v',"{:.2f}".format(v), 'sizes', x_group._size, '+', y_group._size, '=', p_group._size)
            print('Clusters', x_group._neg_uniq_edges, '+', y_group._neg_uniq_edges, '=', abs(p_group._neg_uniq_edges))
            
            shift = x_group.evaluate_and_add_center_shift(v, p_group, y_group)
            ret_arr[e] += shift * y_group._size * x_group._size / p_group._size

            x_group.assume_data(p_group.data, p_group._size, p_group._neg_uniq_edges)
            i = p
            p = U.parent[p + offset]
            print ('---')

        print("{:.2f}".format(np.linalg.norm(ret_arr[e]-e_coords)), 'net_shift', ret_arr[e] - e_coords)
    
    # center pursuit is combined
    # second run: adding displacement 
    for e in range(loop_size):
        e_coords = data_arr[e]
        x_group.assume_data(x_outlier.data,1,0)
        x_group.cook_outlier_coords(e_coords)

        shift = np.zeros_like(e_coords)
        print('===', e, 'point', e_coords)
        p = U.parent[e]
        i = e
        while p != 0:
            p -= offset
            v = values_arr[p]

            p_group.assume_data(group_arr[p], sizes_arr[p], clusters_arr[p])
            j = p_group.get_sibling_id(i)

            if p_group._size - x_group._size > 1: # y_is_not_outlier
                y_group.assume_data(group_arr[j], sizes_arr[j], clusters_arr[j])
            else:
                y_coords = data_arr[j]
                y_group.assume_data(y_outlier.data, 1, 0)
                y_group.cook_outlier_coords(y_coords)
                y_group.cook_outlier_center_shift(v, p_group, x_group)

            # предыдущим циклом посчитали стремление в центр всеми точками
            shift += x_group.evaluate_shift(p_group, y_group)

            x_group.assume_data(p_group.data, p_group._size, p_group._neg_uniq_edges)
            i = p
            p = U.parent[p + offset]
        ret_arr[e] += shift
        print('---', shift, 'net_shift', ret_arr[e] - e_coords)

    return ret_arr

cpdef np.ndarray develop(np.ndarray values_arr,
                         np.ndarray uf_arr, np.intp_t size,
                         np.ndarray group_arr,
                         np.ndarray data_arr,
                         np.ndarray sizes_arr,
                         np.ndarray clusters_arr,
                         np.ndarray ret_data_arr,
                         precision=0.0000001):
    """Returns modified data points.
    
    Parameters
    ----------

    Returns
    -------

    ret_data_arr : ndarray
       New coords after the development.
    """

    cdef:
        UnionFind U

    # this is only relevant if distances between datapoints are super small
    if precision is None or precision<=0:
        precision = 0.0000001
    set_precision(precision)

    if data_arr.ndim != group_arr["sum_coords"].ndim:
        print ('ERROR amal_arr data dimensions don\'t match', data_arr.ndim, group_arr["sum_coords"].ndim)
        return
    if ret_data_arr is None:
        ret_data_arr = copy.deepcopy(data_arr)
    elif len(ret_data_arr) < size:
        print ('ERROR ret_data_arr is too small', len(ret_data_arr), size)
    else:
        print('shapes', ret_data_arr.ndim, data_arr.ndim)
        ret_data_arr[:] = data_arr

    U = UnionFind(size, uf_arr)

    move_points(U, values_arr, group_arr, data_arr, sizes_arr, clusters_arr,
                ret_data_arr)

    print('ret_data_arr develop', ret_data_arr)

    return ret_data_arr
