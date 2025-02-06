# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Produces the next position of the datapoint
# uses results from the tree edges
# partly same algo as the labeling minus memory managing part

# Author: Pavel Artamonov
# License: 3-clause BSD

import numpy as np
cimport numpy as np

import copy

from ._druhg_group cimport set_precision
from ._druhg_group import Group
from ._druhg_group cimport Group

from ._druhg_unionfind import UnionFind
from ._druhg_unionfind cimport UnionFind

cdef getIndex(UnionFind U, np.intp_t p):
    return p - U.p_size - 1

cdef move_points(UnionFind U, np.ndarray values_arr,
                 np.ndarray group_arr, np.ndarray data_arr, np.ndarray ret_arr):
    cdef:
        np.intp_t e, p, i, size
        np.double_t v, e_val, limit, g
        np.ndarray i_coords

    # helpers
    x_group = Group(np.zeros_like(group_arr[:1])[0])
    y_group = Group(np.zeros_like(group_arr[:1])[0])
    p_group = Group(np.zeros_like(group_arr[:1])[0])

    loop_size = U.p_size
    for e in range(loop_size):
        i_coords = data_arr[e]
        print('===', e, 'point', i_coords, ret_arr[e])
        assert (all(ret_arr[e] == i_coords))
        p = U.parent[e]
        if p == 0:
            continue
        i = getIndex(U,p)
        v = values_arr[i]
        x_group.cook_outlier(v)
        x_group.mtn_set_coords(i_coords)
        x_group.mtn_mark_cluster(True)
        print('start data', ret_arr[e])
        while p != 0:
            i = getIndex(U,p)
            v = values_arr[i]

            p_group.assume_data(group_arr[i],-1, 0)

            if x_group.mtn_need_cluster():
                print('x is clustering')
                x_group.mtn_change_sum_edges(v)

            y_group.mtn_set_like(group_arr[i])
            y_group.mtn_subtract(x_group.data, v)

            print('v',"{:.1f}".format(v), x_group.data[0], y_group.data[0])
            print('centers (p, x, y)', p_group.mtn_center(), x_group.mtn_center(), y_group.mtn_center(), i_coords)
            # print("x_group", x_group.mtn_center(), x_group.data)
            # print("y_group", y_group.mtn_center(), y_group.data)
            # print(i, "p_group", p_group.mtn_center(), p_group.data, i)

            #ФОРМУЛА!!!
            x_weight = x_group.mtn_weight() + v
            y_weight = y_group.mtn_weight() + v
            p_weight = p_group.mtn_weight()

            shift_point = p_group.mtn_center() - i_coords
            shift_x = x_group.mtn_center() - p_group.mtn_center()
            shift_y = y_group.mtn_center() - p_group.mtn_center()

# ------------ ок - в равносторонний, но перемещается
#             v_shift_point = ((x_group.data[0] - 1) - y_group.data[0])
#             v_shift_x = -(y_group.data[0] * x_weight) / p_weight
#             v_shift_y = (x_group.data[0] - 1) * y_weight / p_weight
# ------------ the best
#             v_shift_point *= -((x_group.data[0] - 1) - y_group.data[0])
#             v_shift_x = +(y_group.data[0] * x_weight) / p_weight
#             v_shift_y = -(x_group.data[0] - 1) * y_weight / p_weight
# ------------ ок - застывает на месте
#             shift_point = ((x_group.data[0] - 1) - y_group.data[0])
#             v_shift_x = -(x_group.data[0] - 1) * x_weight / p_weight
#             v_shift_y = y_group.data[0] * y_weight / p_weight
# ------------ the best2
#             v_shift_point = -((x_group.data[0] - 1) - y_group.data[0])
#             v_shift_x = +(x_group.data[0] - 1) * x_weight / p_weight
#             v_shift_y = -y_group.data[0] * y_weight / p_weight

            # v_shift_point = ((x_group.data[0] - 1) * y_weight - y_group.data[0] * x_weight) / p_weight
            # v_shift_x = -y_group.data[0]
            # v_shift_y = -(x_group.data[0] - 1)

            v_shift_point = +((x_group.data[0] - 1) * y_weight + y_group.data[0] * x_weight) / p_weight
            v_shift_x = -(x_group.data[0]-1)
            v_shift_y = +y_group.data[0]

            print('shifts', "{:.2f}".format(np.linalg.norm(shift_point)), shift_point,
                  'x',"{:.2f}".format(np.linalg.norm(shift_x)), shift_x,
                  'y',"{:.2f}".format(np.linalg.norm(shift_y)), shift_y)
            print('multi', "{:.2f}".format(v_shift_point), "{:.2f}".format(v_shift_x), "{:.2f}".format(v_shift_y))
            print('x,y,p weights', "{:.1f}".format(x_weight), "{:.1f}".format(y_weight), "{:.1f}".format(p_weight))
            print('res p', "{:.2f}".format(np.linalg.norm(shift_point*v_shift_point)), shift_point*v_shift_point,
                  '+ xy', "{:.2f}".format(np.linalg.norm(shift_x*v_shift_x + shift_y*v_shift_y)), shift_x*v_shift_x + shift_y*v_shift_y,)
            shift = shift_point*v_shift_point + shift_x*v_shift_x + shift_y*v_shift_y

            print('=',"{:.2f}".format(np.linalg.norm(shift)), 'shift', shift)

            if not (x_weight/p_weight != 0.5 or x_weight/p_weight != 1.) \
                    and not (y_weight/p_weight != 0.5 or y_weight/p_weight != 1.):
                print('LALALALALA')


            ret_arr[e] += shift
            p = U.parent[p]
            x_group.mtn_set_like(p_group.data)

            print ('---')
        print("{:.2f}".format(np.linalg.norm(ret_arr[e]-i_coords)), 'net_shift', ret_arr[e] - i_coords)

    return ret_arr

cpdef np.ndarray develop(np.ndarray values_arr,
                         np.ndarray ranks_arr,
                         np.ndarray uf_arr, np.intp_t size,
                         np.ndarray group_arr,
                         np.ndarray data_arr,
                         np.ndarray ret_data_arr,
                         np.ndarray ret_clusters_arr,
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
        np.ndarray amals

    # this is only relevant if distances between datapoints are super small
    if precision is None:
        precision = 0.0000001
    set_precision(precision)

    if len(group_arr) < size:
        print('ERROR amal_arr is too small', len(group_arr), size)
        return
    group_arr[:size].fill(0)

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

    # emerge_clusters_from_UF(U, values_arr, group_arr, ret_sizes_arr, ret_clusters_arr,
    #                 ranks_arr = None,  # for clustering only
    #                 run_motion = True, data_arr=data_arr)
    print('motion start data_arr develop', data_arr)
    move_points(U, values_arr, group_arr, data_arr,
                ret_data_arr)
    print('ret_data_arr develop', ret_data_arr)

    return ret_data_arr
