# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Motion phase.
# Each union-find level merges two sides. Relation from A to B is the sum of
# clustered edge values in A divided by the sum of edge values in B.
# The connection edge enters that relation: clustering adds it to the
# numerator, and the denominator receives it always. A single point
# clusterizes, so its numerator is that same edge.
# Every point adds its own delta at that level:
#   (center of the other side - center of its side)
#     * (sum of relations toward its side) / (sum of relations toward the other side)
#   + the sum of vectors from the point to every other point of the whole
#     connection (both sides), excluding that point.
# Relations toward a side are the relations of the points on the opposite side.

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


cdef np.intp_t subtree_rep(np.intp_t label, np.intp_t[::1] rep, np.intp_t N, np.intp_t offset):
    # Smallest point index under this union-find label.
    if label < 0:
        return -1
    if label < N:
        return label
    return rep[label - offset]


cdef np.intp_t sibling_label(np.intp_t came, np.intp_t level,
                             np.intp_t[::1] child_a, np.intp_t[::1] child_b):
    if child_a[level] < 0 or child_b[level] < 0:
        return -1
    if came == child_a[level]:
        return child_b[level]
    if came == child_b[level]:
        return child_a[level]
    return -1


cdef np.intp_t other_point_at_level(np.intp_t came, np.intp_t level,
                                    np.intp_t[::1] child_a, np.intp_t[::1] child_b,
                                    np.intp_t[::1] rep, np.intp_t N, np.intp_t offset):
    # came is the child label that contains the source point.
    # The related point is the representative of the other child.
    return subtree_rep(sibling_label(came, level, child_a, child_b), rep, N, offset)


cdef np.double_t side_edge_value(np.intp_t label, np.double_t[::1] arr,
                                 np.intp_t N, np.intp_t offset):
    # A point has no internal edges.
    if label < N:
        return 0
    return arr[label - offset]


cdef np.double_t relation_a_to_b(np.double_t clustered_edges_a, np.double_t sum_edges_b):
    # Relation from A to B = sum of clustered edges of A / sum of edges of B.
    if sum_edges_b <= 0:
        return 0
    return clustered_edges_a / sum_edges_b


cdef bint side_is_cluster(np.intp_t label, np.intp_t[::1] clusters,
                          np.intp_t N, np.intp_t offset):
    # A point is connected by this edge, so it clusterizes onto it.
    if label < N:
        return True
    return clusters[label - offset] < 0


cdef np.double_t relation_across(np.intp_t own, np.intp_t other, np.double_t edge,
                                 np.intp_t[::1] clusters,
                                 np.double_t[::1] sum_edges,
                                 np.double_t[::1] clustered_edges,
                                 np.intp_t N, np.intp_t offset):
    # The connection edge enters the relation here.
    # Clustering adds it to the numerator. The denominator gets it always.
    cdef:
        np.double_t top, bottom, rel
        bint clustered

    clustered = side_is_cluster(own, clusters, N, offset)
    top = side_edge_value(own, clustered_edges, N, offset)
    if clustered:
        top += edge
    bottom = side_edge_value(other, sum_edges, N, offset) + edge
    rel = relation_a_to_b(top, bottom)
    logging.getLogger(__package__).debug(
        'relation %s -> %s edge %.4f clustered %s top %.4f bottom %.4f = %.4f',
        own, other, edge, clustered, top, bottom, rel)
    return rel


cdef void accumulate_edge_sums(UnionFind U,
                               np.ndarray values_arr,
                               np.ndarray clusters_arr,
                               np.ndarray child_a_arr,
                               np.ndarray child_b_arr,
                               np.ndarray sum_edges_arr,
                               np.ndarray clustered_edges_arr):
    # sum_edges[level] is the sum of merge values inside that component,
    # including the merge that formed it.
    # clustered_edges[level] is that sum when the component clusterized,
    # otherwise only the clustered edges of its children.
    cdef:
        np.intp_t level, N, offset, child
        np.double_t total, clustered
        np.intp_t[::1] child_a, child_b
        np.intp_t[::1] clusters
        np.double_t[::1] values, sum_edges, clustered_edges

    N = U.p_size
    offset = U.get_offset()
    child_a = child_a_arr
    child_b = child_b_arr
    clusters = clusters_arr
    values = values_arr
    sum_edges = sum_edges_arr
    clustered_edges = clustered_edges_arr

    for level in range(N - 1):
        if child_a[level] < 0:
            continue
        total = values[level]
        clustered = 0
        child = child_a[level]
        if child >= N:
            total += sum_edges[child - offset]
            clustered += clustered_edges[child - offset]
        child = child_b[level]
        if child >= N:
            total += sum_edges[child - offset]
            clustered += clustered_edges[child - offset]
        sum_edges[level] = total
        if clusters[level] < 0:
            clustered_edges[level] = total
        else:
            clustered_edges[level] = clustered
        logging.getLogger(__package__).debug(
            'level %s v %.4f children %s %s sum_edges %.4f clustered_edges %.4f cluster %s',
            level, values[level], child_a[level], child_b[level],
            sum_edges[level], clustered_edges[level], clusters[level] < 0)


cdef void accumulate_side_coords(UnionFind U, np.ndarray coords_arr,
                                 np.ndarray child_a_arr, np.ndarray child_b_arr,
                                 np.ndarray sum_coords_arr, np.ndarray size_arr):
    # sum_coords[level] / size[level] is the center of that component.
    # Points are not stored: a point's center is its own coordinate.
    cdef:
        np.intp_t level, N, offset, child
        np.intp_t[::1] child_a, child_b, sizes

    N = U.p_size
    offset = U.get_offset()
    child_a = child_a_arr
    child_b = child_b_arr
    sizes = size_arr

    for level in range(N - 1):
        if child_a[level] < 0:
            continue
        acc = np.zeros(sum_coords_arr.shape[1], dtype=np.double)
        count = 0
        child = child_a[level]
        if child >= 0:
            if child < N:
                acc += coords_arr[child]
                count += 1
            else:
                acc += sum_coords_arr[child - offset]
                count += sizes[child - offset]
        child = child_b[level]
        if child >= 0:
            if child < N:
                acc += coords_arr[child]
                count += 1
            else:
                acc += sum_coords_arr[child - offset]
                count += sizes[child - offset]
        sum_coords_arr[level] = acc
        sizes[level] = count


cdef object _side_center(np.intp_t label, np.ndarray coords_arr,
                         np.ndarray sum_coords_arr, np.intp_t[::1] sizes,
                         np.intp_t N, np.intp_t offset):
    cdef np.double_t n
    if label < N:
        return coords_arr[label], 1.0
    n = sizes[label - offset]
    return sum_coords_arr[label - offset] / n, n


cdef object point_level_delta(np.intp_t i, np.intp_t own, np.intp_t other,
                              np.double_t edge,
                              np.ndarray coords_arr,
                              np.ndarray sum_coords_arr, np.intp_t[::1] sizes,
                              np.intp_t[::1] clusters,
                              np.double_t[::1] sum_edges,
                              np.double_t[::1] clustered_edges,
                              np.intp_t N, np.intp_t offset):
    # Delta of point i, which sits on side `own`, toward side `other`.
    cdef:
        np.double_t n_own, n_other, n_all
        np.double_t rel_to_other, rel_to_own, sum_to_own, sum_to_other, scale
        object c_own, c_other, c_all

    c_own, n_own = _side_center(own, coords_arr, sum_coords_arr, sizes, N, offset)
    c_other, n_other = _side_center(other, coords_arr, sum_coords_arr, sizes, N, offset)

    rel_to_other = relation_across(
        own, other, edge, clusters, sum_edges, clustered_edges, N, offset)
    rel_to_own = relation_across(
        other, own, edge, clusters, sum_edges, clustered_edges, N, offset)
    # Each point of a side has one relation onto the opposite side.
    sum_to_own = n_other * rel_to_own
    sum_to_other = n_own * rel_to_other
    scale = sum_to_own / sum_to_other if sum_to_other > 0 else 0

    center_delta = (c_other - c_own) * scale
    # Vectors from this point to every other point of both sides.
    n_all = n_own + n_other
    c_all = (c_own * n_own + c_other * n_other) / n_all
    to_rest = n_all * (c_all - coords_arr[i])
    logging.getLogger(__package__).debug(
        'point %s own %s other %s edge %.4f n %.0f+%.0f rel %.4f / %.4f scale %.4f center %s rest %s delta %s',
        i, own, other, edge, n_own, n_other, rel_to_other, rel_to_own, scale,
        center_delta, to_rest, center_delta + to_rest)
    return center_delta + to_rest


cdef void organize_level_relations(UnionFind U,
                                   np.ndarray child_a_arr,
                                   np.ndarray child_b_arr,
                                   np.ndarray rep_arr):
    # child_*[level] is the union-find label of one side of the merge.
    # rep[level] is the smallest point index in that component.
    # Caller fills child_* and rep with -1.
    cdef:
        np.intp_t u, p, level, N, offset, ra, rb
        np.intp_t[::1] child_a, child_b, rep

    N = U.p_size
    offset = U.get_offset()
    child_a = child_a_arr
    child_b = child_b_arr
    rep = rep_arr

    for u in range(2 * N):
        p = U.parent[u]
        if p == 0:
            continue
        level = p - offset
        if child_a[level] < 0:
            child_a[level] = u
        elif child_b[level] < 0:
            child_b[level] = u

    # Internal labels are created in order, so a child's level is smaller.
    for level in range(N - 1):
        if child_a[level] < 0:
            continue
        ra = subtree_rep(child_a[level], rep, N, offset)
        if child_b[level] < 0:
            rep[level] = ra
            continue
        rb = subtree_rep(child_b[level], rep, N, offset)
        rep[level] = ra if ra < rb else rb


cdef void apply_level_relations(UnionFind U,
                                np.ndarray coords_arr, np.ndarray ret_arr,
                                np.ndarray child_a_arr, np.ndarray child_b_arr,
                                np.ndarray sum_coords_arr, np.ndarray size_arr,
                                np.ndarray values_arr, np.ndarray clusters_arr,
                                np.ndarray sum_edges_arr,
                                np.ndarray clustered_edges_arr):
    cdef:
        np.intp_t i, came, p, level, sib, N, offset
        np.intp_t[::1] child_a, child_b, sizes, clusters
        np.double_t[::1] values, sum_edges, clustered_edges

    N = U.p_size
    offset = U.get_offset()
    child_a = child_a_arr
    child_b = child_b_arr
    sizes = size_arr
    values = values_arr
    clusters = clusters_arr
    sum_edges = sum_edges_arr
    clustered_edges = clustered_edges_arr

    for i in range(N):
        came = i
        p = U.parent[i]
        while p != 0:
            level = p - offset
            sib = sibling_label(came, level, child_a, child_b)
            if sib >= 0:
                ret_arr[i] += point_level_delta(
                    i, came, sib, values[level], coords_arr, sum_coords_arr, sizes,
                    clusters, sum_edges, clustered_edges, N, offset)
            came = p
            p = U.parent[p]


cpdef list relations_by_level(np.ndarray uf_arr, np.intp_t size):
    """Relation vector of each union-find level.

    Level ``k`` is an ``(n, 2)`` int array of ``(point, other_point)``.
    Every point of that level is a source once. ``other_point`` is the
    smallest point index on the opposite side of the same level.
    """
    cdef:
        UnionFind U
        np.intp_t level, N, offset, other, i, came, p
        np.ndarray child_a_arr, child_b_arr, rep_arr
        np.intp_t[::1] child_a, child_b, rep
        list buckets

    N = size
    if N < 2:
        return []

    U = UnionFind(N, uf_arr)
    offset = U.get_offset()
    child_a_arr = np.full(N - 1, -1, dtype=np.intp)
    child_b_arr = np.full(N - 1, -1, dtype=np.intp)
    rep_arr = np.full(N - 1, -1, dtype=np.intp)
    organize_level_relations(U, child_a_arr, child_b_arr, rep_arr)
    child_a = child_a_arr
    child_b = child_b_arr
    rep = rep_arr

    buckets = [[] for _ in range(N - 1)]
    for i in range(N):
        came = i
        p = U.parent[i]
        while p != 0:
            level = p - offset
            other = other_point_at_level(came, level, child_a, child_b, rep, N, offset)
            if other >= 0:
                buckets[level].append((i, other))
            came = p
            p = U.parent[p]

    out = []
    for level in range(N - 1):
        if len(buckets[level]) == 0:
            out.append(np.empty((0, 2), dtype=np.intp))
        else:
            out.append(np.asarray(buckets[level], dtype=np.intp))
    return out


cpdef list relation_magnitudes_by_level(np.ndarray uf_arr, np.ndarray values_arr,
                                        np.ndarray clusters_arr, np.intp_t size):
    """Relation of each point at each union-find level.

    Level ``k`` is an ``(n, 3)`` array of ``(point, other_point, magnitude)``.
    ``magnitude`` is the relation from the point's side A to the opposite side B.
    The connection edge is added to A's clustered edges when A clusterizes,
    and it is always added to B's edges.
    """
    cdef:
        UnionFind U
        np.intp_t level, N, offset, other, sib, i, came, p
        np.ndarray child_a_arr, child_b_arr, rep_arr
        np.ndarray sum_edges_arr, clustered_edges_arr
        np.intp_t[::1] child_a, child_b, rep, clusters
        np.double_t[::1] values, sum_edges, clustered_edges
        np.double_t rel
        list buckets

    N = size
    if N < 2:
        return []

    U = UnionFind(N, uf_arr)
    offset = U.get_offset()
    child_a_arr = np.full(N - 1, -1, dtype=np.intp)
    child_b_arr = np.full(N - 1, -1, dtype=np.intp)
    rep_arr = np.full(N - 1, -1, dtype=np.intp)
    sum_edges_arr = np.zeros(N - 1, dtype=np.double)
    clustered_edges_arr = np.zeros(N - 1, dtype=np.double)
    organize_level_relations(U, child_a_arr, child_b_arr, rep_arr)
    accumulate_edge_sums(U, values_arr, clusters_arr, child_a_arr, child_b_arr,
                         sum_edges_arr, clustered_edges_arr)
    child_a = child_a_arr
    child_b = child_b_arr
    rep = rep_arr
    values = values_arr
    clusters = clusters_arr
    sum_edges = sum_edges_arr
    clustered_edges = clustered_edges_arr

    buckets = [[] for _ in range(N - 1)]
    for i in range(N):
        came = i
        p = U.parent[i]
        while p != 0:
            level = p - offset
            sib = sibling_label(came, level, child_a, child_b)
            other = subtree_rep(sib, rep, N, offset)
            if other >= 0:
                rel = relation_across(
                    came, sib, values[level], clusters,
                    sum_edges, clustered_edges, N, offset)
                buckets[level].append((i, other, rel))
            came = p
            p = U.parent[p]

    out = []
    for level in range(N - 1):
        if len(buckets[level]) == 0:
            out.append(np.empty((0, 3), dtype=np.double))
        else:
            out.append(np.asarray(buckets[level], dtype=np.double))
    return out


cpdef list point_deltas_by_level(np.ndarray uf_arr, np.ndarray values_arr,
                                 np.ndarray clusters_arr, np.ndarray coords_arr,
                                 np.intp_t size):
    """Delta of every point at each union-find level.

    Level ``k`` is an ``(n, 1 + ndim)`` array. Column 0 is the point index.
    The remaining columns are that point's delta at the level.
    """
    cdef:
        UnionFind U
        np.intp_t level, N, offset, sib, i, came, p, ndim
        np.ndarray child_a_arr, child_b_arr, rep_arr
        np.ndarray sum_edges_arr, clustered_edges_arr
        np.ndarray sum_coords_arr, size_arr, coords
        np.intp_t[::1] child_a, child_b, sizes, clusters
        np.double_t[::1] values, sum_edges, clustered_edges
        list buckets

    N = size
    if N < 2:
        return []

    coords = coords_arr.reshape(-1, 1) if coords_arr.ndim == 1 else coords_arr
    ndim = coords.shape[1]
    U = UnionFind(N, uf_arr)
    offset = U.get_offset()
    child_a_arr = np.full(N - 1, -1, dtype=np.intp)
    child_b_arr = np.full(N - 1, -1, dtype=np.intp)
    rep_arr = np.full(N - 1, -1, dtype=np.intp)
    sum_edges_arr = np.zeros(N - 1, dtype=np.double)
    clustered_edges_arr = np.zeros(N - 1, dtype=np.double)
    sum_coords_arr = np.zeros((N - 1, ndim), dtype=np.double)
    size_arr = np.zeros(N - 1, dtype=np.intp)
    organize_level_relations(U, child_a_arr, child_b_arr, rep_arr)
    accumulate_edge_sums(U, values_arr, clusters_arr, child_a_arr, child_b_arr,
                         sum_edges_arr, clustered_edges_arr)
    accumulate_side_coords(U, coords, child_a_arr, child_b_arr, sum_coords_arr, size_arr)
    child_a = child_a_arr
    child_b = child_b_arr
    sizes = size_arr
    values = values_arr
    clusters = clusters_arr
    sum_edges = sum_edges_arr
    clustered_edges = clustered_edges_arr

    buckets = [[] for _ in range(N - 1)]
    for i in range(N):
        came = i
        p = U.parent[i]
        while p != 0:
            level = p - offset
            sib = sibling_label(came, level, child_a, child_b)
            if sib >= 0:
                delta = point_level_delta(
                    i, came, sib, values[level], coords, sum_coords_arr, sizes,
                    clusters, sum_edges, clustered_edges, N, offset)
                buckets[level].append((i,) + tuple(np.asarray(delta, dtype=np.double).ravel()))
            came = p
            p = U.parent[p]

    out = []
    for level in range(N - 1):
        if len(buckets[level]) == 0:
            out.append(np.empty((0, 1 + ndim), dtype=np.double))
        else:
            out.append(np.asarray(buckets[level], dtype=np.double))
    return out


cdef move_points(UnionFind U, np.ndarray values_arr,
                 np.ndarray group_arr, np.ndarray coords_arr,
                 np.ndarray sizes_arr, np.ndarray clusters_arr,
                 np.ndarray ret_arr):
    # Every point, at every ancestor level, adds its own delta.
    cdef np.intp_t n = U.p_size
    cdef np.intp_t ndim
    cdef np.ndarray child_a_arr, child_b_arr, rep_arr
    cdef np.ndarray sum_edges_arr, clustered_edges_arr
    cdef np.ndarray sum_coords_arr, size_arr, coords, ret

    if n < 2:
        return ret_arr

    coords = coords_arr.reshape(-1, 1) if coords_arr.ndim == 1 else coords_arr
    ret = ret_arr.reshape(-1, 1) if ret_arr.ndim == 1 else ret_arr
    ndim = coords.shape[1]
    child_a_arr = np.full(n - 1, -1, dtype=np.intp)
    child_b_arr = np.full(n - 1, -1, dtype=np.intp)
    rep_arr = np.full(n - 1, -1, dtype=np.intp)
    sum_edges_arr = np.zeros(n - 1, dtype=np.double)
    clustered_edges_arr = np.zeros(n - 1, dtype=np.double)
    sum_coords_arr = np.zeros((n - 1, ndim), dtype=np.double)
    size_arr = np.zeros(n - 1, dtype=np.intp)
    organize_level_relations(U, child_a_arr, child_b_arr, rep_arr)
    accumulate_edge_sums(U, values_arr, clusters_arr, child_a_arr, child_b_arr,
                         sum_edges_arr, clustered_edges_arr)
    accumulate_side_coords(U, coords, child_a_arr, child_b_arr, sum_coords_arr, size_arr)
    apply_level_relations(U, coords, ret,
                          child_a_arr, child_b_arr,
                          sum_coords_arr, size_arr,
                          values_arr, clusters_arr,
                          sum_edges_arr, clustered_edges_arr)
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
