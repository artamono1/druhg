# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Produces the next position of the datapoint
# uses results from the tree edges
# logarithmic climb from point to hierarchy of parents
# first run evaluates parent nodes, second run adds the displacement

# Author: Pavel Artamonov
# License: 3-clause BSD

import logging
import copy

import numpy as np
cimport numpy as np

from ._druhg_group cimport set_precision
from ._druhg_group_placement import GroupPlacement
from ._druhg_group_placement cimport GroupPlacement
from ._druhg_unionfind import UnionFind
from ._druhg_unionfind cimport UnionFind


cdef np.intp_t _n_coords(np.ndarray coords_arr):
    if coords_arr.ndim == 1:
        return 1
    return coords_arr.shape[1]


cdef aggregate_coords(UnionFind U, np.ndarray values_arr,
                      np.ndarray group_arr, np.ndarray coords_arr,
                      np.ndarray sizes_arr, np.ndarray clusters_arr):
    # runs similarly to labeling, except node is created for every complex parent
    cdef:
        np.intp_t p, u, i, j, x_size, y_size, p_size
        np.intp_t loop_size1 = U.p_size
        np.intp_t loop_size2 = U.p_size * 2
        np.intp_t offset = U.get_offset()
        np.double_t has_data
        GroupPlacement x_node, p_node
        np.double_t v

    for u in range(loop_size1):
        p = U.parent[u]
        if p == 0:  # point has no connection
            continue
        p = p - offset
        v = values_arr[p]
        p_size = sizes_arr[p]
        p_node = group_arr[p]
        p_node.add_single_point(v, u, coords_arr[u])
        if p_size == 2:
            p_node.net_densities = v * 0.5

    for u in range(loop_size1 + 1, loop_size2):
        i = u - offset
        p = U.parent[u]
        if p == 0:
            if sizes_arr[i] == 0:
                break
            continue
        p = p - offset

        p_size = sizes_arr[p]
        x_size = sizes_arr[i]
        y_size = p_size - x_size

        v = values_arr[p]
        x_node, p_node = group_arr[i], group_arr[p]
        has_data, j = p_node.sum_edges, p_node.sum_ids

        p_node.add_node(v, i, x_node, x_size, clusters_arr[i] > 0)

        if has_data == 0 or clusters_arr[i] > 0:  # first time passing or no clustering
            continue

        if y_size != 1 and clusters_arr[j] > 0:  # sibling is not a cluster
            continue

        p_node.net_densities = v / p_size
        if v == 0:
            # TODO: check zero-value edges
            continue

    return group_arr


cdef side_points_to_center(UnionFind U, np.ndarray values_arr,
                           np.ndarray group_arr, np.ndarray coords_arr,
                           np.ndarray sizes_arr, np.ndarray clusters_arr):
    # monad provides the orientation to a center, magnitude is separated
    cdef:
        np.intp_t p, i, j, x_size, y_size, p_size, e
        np.intp_t loop_size = U.p_size
        np.intp_t offset = U.get_offset()
        np.intp_t x_is_cluster, y_is_cluster
        GroupPlacement x_node, y_node, p_node, x_outlier, y_outlier
        np.double_t v

    x_outlier = GroupPlacement(_n_coords(coords_arr))
    y_outlier = GroupPlacement(_n_coords(coords_arr))
    empty_array = np.zeros(_n_coords(coords_arr), dtype=np.double)

    for e in range(loop_size):
        e_coords = coords_arr[e]
        e_node = x_outlier.cook_outlier_coords(e_coords, empty_array)
        x_node = e_node
        x_size = 1
        x_is_cluster = True
        e_value = 0
        # e_node is a cluster inside x_node inside p_node
        # if e_node=x_node then attraction else repulsion

        p = U.parent[e]
        i = e
        while p != 0:
            p -= offset
            v = values_arr[p]

            p_node, p_size = group_arr[p], sizes_arr[p]
            y_size = p_size - x_size

            j = p_node.get_sibling_id(i)
            y_is_cluster = True if y_size == 1 or clusters_arr[j] < 0 else False
            y_node = group_arr[j] if y_size > 1 else y_outlier.cook_outlier_coords(coords_arr[j], empty_array)

            e_value = v * (v + x_node.sum_edges + y_is_cluster * y_node.sum_edges) if x_is_cluster else e_value

            direction = p_node.sum_coords / p_size - e_coords
            direction /= np.linalg.norm(direction)
            x_node.center_shift += direction * e_value  # used by the opposite side in the next loop

            direction = y_node.sum_coords / y_size - e_coords
            direction /= np.linalg.norm(direction)
            x_node.side_shift += direction * e_value

            x_node, x_size = p_node, p_size
            x_is_cluster = clusters_arr[p] < 0

            i = p
            p = U.parent[p + offset]


cdef point_vs_other_side(UnionFind U, np.ndarray values_arr,
                         np.ndarray group_arr, np.ndarray coords_arr,
                         np.ndarray sizes_arr, np.ndarray clusters_arr,
                         np.ndarray ret_arr):
    # point shifts to a center and whole other side repels
    # monad's center is a mediator
    cdef:
        np.intp_t p, i, j, x_size, y_size, p_size, e
        np.intp_t loop_size = U.p_size
        np.intp_t offset = U.get_offset()
        np.intp_t x_is_cluster, y_is_cluster
        GroupPlacement x_node, y_node, p_node, x_outlier, y_outlier
        np.double_t v

    logger = logging.getLogger(__package__)
    debug = logger.isEnabledFor(logging.DEBUG)

    x_outlier = GroupPlacement(_n_coords(coords_arr))
    y_outlier = GroupPlacement(_n_coords(coords_arr))
    empty_array = np.zeros(_n_coords(coords_arr), dtype=np.double)

    for e in range(loop_size):
        e_coords = coords_arr[e]
        e_node = x_outlier.cook_outlier_coords(e_coords, empty_array)
        x_node = e_node
        x_size = 1
        x_is_cluster = True
        e_value = 0

        if debug:
            logger.debug('=== %s point %s', e, e_coords)
        assert all(ret_arr[e] == e_coords)
        p = U.parent[e]
        i = e
        while p != 0:
            p -= offset
            v = values_arr[p]

            p_node, p_size = group_arr[p], sizes_arr[p]
            y_size = p_size - x_size
            center = p_node.sum_coords / p_size
            center_magnitude = 1. * x_size * y_size / (v * (x_size + y_size)) * p_node.net_densities / p_node.sum_edges

            j = p_node.get_sibling_id(i)
            y_node = group_arr[j] if y_size > 1 else y_outlier.cook_outlier_coords(
                coords_arr[j],
                v * (v + x_is_cluster * x_node.sum_edges) * (center - coords_arr[j]) / np.linalg.norm(center - coords_arr[j]))
            y_is_cluster = True if y_size == 1 or clusters_arr[j] < 0 else False
            e_value = v * (v + x_node.sum_edges + y_is_cluster * y_node.sum_edges) if x_is_cluster else e_value

            alpha = x_size * y_size / (1. * x_size + y_size)
            magnitude = e_value * center_magnitude * alpha

            direction = center - e_coords
            direction /= np.linalg.norm(direction)
            direction *= -1. if x_is_cluster else 1.
            shift = direction * magnitude

            other_side_shift = y_node.center_shift * center_magnitude
            other_side_shift *= -1. if y_is_cluster else 1.

            if debug:
                logger.debug('id %s %s p %s v %.2f sizes %s + %s = %s', i, j, p, v, x_size, y_size, p_size)
                logger.debug('shift %s other_side_shift %s', shift, other_side_shift)

            ret_arr[e] += shift + other_side_shift

            x_node, x_size = p_node, p_size
            x_is_cluster = clusters_arr[p] < 0

            i = p
            p = U.parent[p + offset]

        if debug:
            logger.debug('net_shift %s', ret_arr[e] - e_coords)
    return ret_arr


cdef point_vs_all(UnionFind U, np.ndarray values_arr,
                  np.ndarray group_arr, np.ndarray coords_arr,
                  np.ndarray sizes_arr, np.ndarray clusters_arr,
                  np.ndarray ret_arr):
    # point shifts to a center and whole other side repels
    # monad's center is a mediator
    cdef:
        np.intp_t p, i, j, x_size, y_size, p_size, e
        np.intp_t loop_size = U.p_size
        np.intp_t offset = U.get_offset()
        np.intp_t x_is_cluster, y_is_cluster
        GroupPlacement x_node, y_node, p_node, x_outlier, y_outlier
        np.double_t v

    logger = logging.getLogger(__package__)
    debug = logger.isEnabledFor(logging.DEBUG)

    x_outlier = GroupPlacement(_n_coords(coords_arr))
    y_outlier = GroupPlacement(_n_coords(coords_arr))
    empty_array = np.zeros(_n_coords(coords_arr), dtype=np.double)

    for e in range(loop_size):
        e_coords = coords_arr[e]
        e_node = x_outlier.cook_outlier_coords(e_coords, empty_array)
        x_node = e_node
        x_size = 1
        x_is_cluster = True
        e_value = 0

        if debug:
            logger.debug('=== %s point %s', e, e_coords)
        assert all(ret_arr[e] == e_coords)
        p = U.parent[e]
        i = e
        while p != 0:
            p -= offset
            v = values_arr[p]

            p_node, p_size = group_arr[p], sizes_arr[p]
            y_size = p_size - x_size
            center = p_node.sum_coords / p_size
            center_magnitude = 1. * x_size * y_size / (v * (x_size + y_size)) * p_node.net_densities / p_node.sum_edges

            j = p_node.get_sibling_id(i)
            y_node = group_arr[j] if y_size > 1 else y_outlier.cook_outlier_coords(
                coords_arr[j],
                v * (v + x_is_cluster * x_node.sum_edges) * (center - coords_arr[j]) / np.linalg.norm(center - coords_arr[j]),
                v * (v + x_is_cluster * x_node.sum_edges) * (x_node.sum_coords / x_size - coords_arr[j]) / np.linalg.norm(x_node.sum_coords / x_size - coords_arr[j]),
            )
            y_is_cluster = True if (y_size == 1 or clusters_arr[j] < 0) else False
            e_value = (v * (v + x_node.sum_edges + y_is_cluster * y_node.sum_edges)) if x_is_cluster else e_value

            e_center_shift = center - e_coords
            e_center_shift /= np.linalg.norm(e_center_shift)
            e_center_shift *= e_value

            e_side_shift = y_node.sum_coords / y_size - e_coords
            e_side_shift /= np.linalg.norm(e_side_shift)
            e_side_shift *= e_value

            if x_size == 1:  # initialisation
                x_node = x_outlier.cook_outlier_coords(e_coords, e_center_shift, e_side_shift)

            alpha = 1. * y_size  # каждый с каждым
            alpha *= 1. / (1. * x_size + y_size)
            orientation = -1. if x_is_cluster else 1.
            shift = orientation * e_center_shift * center_magnitude * alpha

            all_shift = y_node.center_shift * center_magnitude
            all_shift *= -1. if y_is_cluster else 1.
            all_shift += shift

            if debug:
                logger.debug('id %s %s p %s v %.2f sizes %s + %s = %s', i, j, p, v, x_size, y_size, p_size)
                logger.debug('all_shift %s', all_shift)

            ret_arr[e] += all_shift

            x_node, x_size = p_node, p_size
            x_is_cluster = clusters_arr[p] < 0

            i = p
            p = U.parent[p + offset]

        if debug:
            logger.debug('net_shift %s', ret_arr[e] - e_coords)
    return ret_arr


cdef move_points(UnionFind U, np.ndarray values_arr,
                 np.ndarray group_arr, np.ndarray coords_arr,
                 np.ndarray sizes_arr, np.ndarray clusters_arr,
                 np.ndarray ret_arr):
    # 1. evaluating net_coords and sibling ids (traversing connections)
    # 2. evaluating center shift for each point (traversing points)
    # 3. adding other side center shift to each point (traversing points)
    cdef np.intp_t i, loop_size = U.p_size

    group_arr = np.empty(loop_size, dtype=object)
    for i in range(loop_size):
        group_arr[i] = GroupPlacement(_n_coords(coords_arr))

    group_arr = aggregate_coords(U, values_arr, group_arr, coords_arr, sizes_arr, clusters_arr)
    side_points_to_center(U, values_arr, group_arr, coords_arr, sizes_arr, clusters_arr)
    ret_arr = point_vs_all(U, values_arr, group_arr, coords_arr, sizes_arr, clusters_arr, ret_arr)
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
    cdef UnionFind U
    logger = logging.getLogger(__package__)

    if precision is None or precision <= 0:
        precision = 0.0000001
    set_precision(precision)

    if ret_data_arr is None:
        ret_data_arr = copy.deepcopy(data_arr)
    elif len(ret_data_arr) < size:
        logger.error('ret_data_arr is too small %s %s', len(ret_data_arr), size)
        return ret_data_arr
    else:
        ret_data_arr[:] = data_arr

    U = UnionFind(size, uf_arr)
    move_points(U, values_arr, group_arr, data_arr, sizes_arr, clusters_arr, ret_data_arr)
    return ret_data_arr
