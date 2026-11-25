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

import logging

cdef np.double_t _group_PRECISION = 0.0000001

cdef set_precision(np.double_t prec):
    _group_PRECISION = prec

def allocate_buffer_clusters(np.intp_t num_points):
    return np.empty((num_points - 1), dtype=np.intp)

def allocate_buffer_sizes(np.intp_t num_points):
    return np.empty((num_points - 1), dtype=np.intp)

cdef class Group:

    cdef np.intp_t points(self):
        return self._size

    cdef np.intp_t uniq_edges(self):  # edges are negative until proven clusters
        return self._neg_uniq_edges if self._neg_uniq_edges >= 0 else -self._neg_uniq_edges

    @staticmethod
    cdef np.intp_t will_cluster(np.intp_t size, np.intp_t edges, GroupNode* node,
                                 np.double_t border,
                                 np.intp_t osize, np.intp_t oedges, GroupNode* onode):
        cdef np.intp_t is_cluster
        cdef np.double_t sum_1, sum_2, sum_12, new_form, old_shells
        assert (edges > 0 and (oedges > 0 or (oedges == 0 and osize == 1)))

        if border <= 0:
            return edges

        # sum_1(D - di) * D * N2 * sum(ni / (Ni di))  vs  N1 * sum_12(D - di)
        # sum_k(D - di) is D times that side's link count, minus the di already stored.
        sum_1 = border * edges - node.sum_uniqs
        sum_2 = border * oedges - onode.sum_uniqs
        sum_12 = sum_1 + sum_2
        new_form = sum_1 * border * osize * node.sum_weighted_reciprocals
        old_shells = 1. * size * sum_12
        is_cluster = -edges if new_form + _group_PRECISION >= old_shells else edges

        logger = logging.getLogger(__package__)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                '%.2f is_cluster %.2f %s %.1f ? %.1f sum(n/Nd) %.2f sum1 %.1f sum12 %.1f clusters %s SSS %s vs oclusters %s oSSS %s',
                border,
                new_form / (old_shells + _group_PRECISION),
                is_cluster < 0,
                new_form, old_shells,
                node.sum_weighted_reciprocals,
                sum_1, sum_12,
                edges, size, oedges, osize,
            )

        return is_cluster  # negative if clustered

    @staticmethod
    cdef np.intp_t aggregate(np.intp_t size, np.intp_t edges_and_clustering, GroupNode* node,
                              np.double_t v,
                              np.intp_t osize, np.intp_t oedges_and_clustering, GroupNode* onode):
        # clustering => edges_and_clustering are negative
        # this is parent node, it is always positive # of clusters

        cdef np.intp_t res
        cdef np.double_t n_whole, weighted_x, weighted_y

        res = (0 if edges_and_clustering < 0 else edges_and_clustering) \
            + (0 if oedges_and_clustering < 0 else oedges_and_clustering) \
            + (1 if (edges_and_clustering < 0 or oedges_and_clustering < 0) else 0)

        same_parent_points = size * (edges_and_clustering < 0) \
                           + osize * (oedges_and_clustering < 0)

        # 1 / di sum per linked
        node.sum_reciprocals = (0 if edges_and_clustering < 0 else node.sum_reciprocals) \
                             + (0 if oedges_and_clustering < 0 else onode.sum_reciprocals) \
                             + ((1. / v) if same_parent_points != 0 else 0)

        # di over the same links that res counts. A clusterized side drops its
        # previous distances; the new link contributes v once.
        node.sum_uniqs = (0 if edges_and_clustering < 0 else node.sum_uniqs) \
                       + (0 if oedges_and_clustering < 0 else onode.sum_uniqs) \
                       + (v if (edges_and_clustering < 0 or oedges_and_clustering < 0) else 0)

        # ni / (Ni di) for each side that clusterizes. Ni is the whole connection.
        # A side that does not clusterize keeps the sum it already stored.
        n_whole = <np.double_t>(size + osize)
        if edges_and_clustering < 0:
            weighted_x = size / (v * n_whole) if v != 0. else 0.
        else:
            weighted_x = node.sum_weighted_reciprocals
        if oedges_and_clustering < 0:
            weighted_y = osize / (v * n_whole) if v != 0. else 0.
        else:
            weighted_y = onode.sum_weighted_reciprocals
        node.sum_weighted_reciprocals = weighted_x + weighted_y

        return res
