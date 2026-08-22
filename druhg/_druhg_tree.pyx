# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Builds minimum spanning tree for druhg algorithm
# uses dialectics to evaluate reciprocity
# Author: Pavel Artamonov
# License: 3-clause BSD


import numpy as np
cimport numpy as np
import sys
import logging

from ._druhg_unionfind import UnionFind
from ._druhg_unionfind cimport UnionFind

import _heapq as heapq

import bisect

cdef np.double_t INF = sys.float_info.max

def allocate_buffer_values(np.intp_t num_points):
    return np.empty((num_points - 1), dtype=np.double)
def allocate_buffer_edgepairs(np.intp_t num_points):
    return np.empty((num_points*2 - 2), dtype=np.intp)
def allocate_buffer_ranks(np.intp_t num_points):
    return np.empty((num_points - 1), dtype=np.intp)

cdef class PairwiseDistanceTreeSparse(object):
    cdef object data_arr
    cdef int data_size

    def __init__(self, N, d):
        self.data_size = N
        self.data_arr = d

    cpdef tuple query(self, d, k, dualtree = 0, breadth_first = 0):
        # TODO: actually we need to consider replacing INF with something else.
        # Reciprocity of absent link is not the same as the INF. Do reciprocity with graphs!
        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices

        knn_dist = INF*np.ones((self.data_size, k))
        knn_indices = np.zeros((self.data_size, k), dtype=np.intp)

        warning = 0

        i = self.data_size
        while i:
            i -= 1
            row = self.data_arr.getrow(i)
            idx, data = row.indices, row.data
            sorted = np.argsort(data)
            pos = 0
            for s in sorted:
                j = idx[s]
                if j == i:
                    warning += 1
                    continue
                if pos >= k:
                    break
                knn_dist[i][pos] = data[s]
                knn_indices[i][pos] = j
                pos += 1

        if warning:
            logging.getLogger(__package__).warning('Attention!: Sparse matrix has an edge that forms a loop! They were zeroed. '+str(warning))

        return knn_dist, knn_indices

cdef class PairwiseDistanceTreeGeneric(object):
    cdef object data_arr
    cdef int data_size

    def __init__(self, N, d):
        self.data_size = N
        self.data_arr = d

    cpdef tuple query(self, d, k, dualtree = 0, breadth_first = 0):
        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices

        knn_dist = np.zeros((self.data_size, k))
        knn_indices = np.zeros((self.data_size, k), dtype=np.intp)

        i = self.data_size
        while i:
            i -= 1
            row = self.data_arr[i]
            sorted = np.argsort(row)
            pos = 0
            for j in sorted:
                if j == i:
                    continue
                knn_dist[i][pos] = row[j]
                knn_indices[i][pos] = j
                pos += 1
                if pos == k:
                    break

        return knn_dist, knn_indices


cdef class UniversalReciprocity (object):
    """Constructs DRUHG spanning tree and marks parents of clusters

    Parameters
    ----------

    algorithm : int
        0/1 - for KDTree/BallTree object
        2/3 - for a full/scipy.sparse precomputed pairwise squared distance matrix

    data: object
        Pass KDTree/BallTree objects or pairwise matrix.

    max_neighbors_search : int, optional (default= 16)
        The max_neighbors_search parameter of DRUHG.
        Effects performance vs precision.
        Default is more than enough.

    metric : string, optional (default='euclidean')
        The metric used to compute distances for the tree.
        Used only with KDTree/BallTree option.

    leaf_size : int, optional (default=20)
        Leaf size of the injected KDTree/BallTree. Kept for API compatibility.

    **kwargs :
        Keyword args passed to the metric.
        Used only with KDTree/BallTree option.
    """

    cdef:
        object tree
        object dist_tree

        np.double_t PRECISION

        np.intp_t num_points
        np.intp_t num_features

        np.intp_t max_neighbors_search

        np.intp_t n_jobs

        UnionFind U
        set ball

        np.intp_t result_edges
        np.ndarray result_values_arr
        np.ndarray result_pairs_arr
        np.ndarray result_rank_arr
        bint logger_debug
        object logger

    def __init__(self, algorithm, tree,
                 buffer_uf, buffer_fast, buffer_values,
                 max_neighbors_search=16, metric='euclidean', leaf_size=20, n_jobs=4,
                 buffer_ranks=None, buffer_edgepairs=None,
                 buffer_clusters=None,
                 **kwargs):

        self.logger = logging.getLogger(__package__)
        self.logger_debug = self.logger.isEnabledFor(logging.DEBUG)

        self.PRECISION = kwargs.get('double_precision', 0.0000001)  # relevant if distances are tiny
        self.n_jobs = n_jobs
        self.ball = set()

        if algorithm == 0 or algorithm == 1:
            self.dist_tree = tree
            self.tree = tree
            self.num_points = self.tree.data.shape[0]
        elif algorithm == 2:
            self.dist_tree = PairwiseDistanceTreeGeneric(tree.shape[0], tree)
            self.tree = tree
            self.num_points = self.tree.shape[0]
        elif algorithm == 3:
            self.dist_tree = PairwiseDistanceTreeSparse(tree.shape[0], tree)
            self.tree = tree
            self.num_points = self.tree.shape[0]
        else:
            raise ValueError('algorithm value '+str(algorithm)+' is not valid')

        self.max_neighbors_search = max_neighbors_search

        self.U = UnionFind(self.num_points, buffer_uf, buffer_fast)
        self.U.nullify()

        self.result_edges = 0

        self.result_values_arr = buffer_values
        if len(self.result_values_arr) < self.num_points - 1:
            self.logger.error('ERROR: values buffer is too small '+str(len(self.result_values_arr))+' '+str(self.num_points - 1))
            return

        self.result_pairs_arr = buffer_edgepairs # np.empty((self.num_points*2 - 2))
        if self.result_pairs_arr is not None and len(self.result_pairs_arr) < self.num_points*2 - 2:
            self.logger.error('ERROR: edgepairs buffer is too small '+str(len(self.result_pairs_arr))+' '+str(self.num_points*2 - 2))
            return

        self.result_rank_arr = buffer_ranks # np.empty((self.num_points - 1))
        if self.result_rank_arr is not None and len(self.result_rank_arr) < self.num_points - 1:
            self.logger.error('ERROR: ranks buffer is too small '+str(len(self.result_rank_arr))+' '+str(self.num_points - 1))
            return

        self._compute_tree_edges()

    cpdef tuple get_tree(self):
        return self.result_values_arr[:self.result_edges * 2], self.result_pairs_arr[:self.result_edges*2].astype(int)

    cpdef np.intp_t get_num_edges(self): # Small k-nn can result in missing edges
        return self.result_edges

    cpdef tuple get_buffers(self):
        return self.result_values_arr, self.U.parent_arr

    cdef void result_write(self, np.double_t v, np.intp_t a, np.intp_t b, np.double_t r):
        cdef np.intp_t i

        i = self.result_edges
        self.result_edges += 1
        self.result_values_arr[i] = v

        if self.result_pairs_arr is not None:
            self.result_pairs_arr[2 * i] = a
            self.result_pairs_arr[2 * i + 1] = b
        if self.result_rank_arr is not None:
            self.result_rank_arr[i] = r

        if self.logger_debug:
            self.logger.debug('Edge %s %s value %s rank %s', a,b, v, r)


    cdef bint _pure_reciprocity(self, np.intp_t i, np.ndarray[np.intp_t, ndim=2] knn_indices, np.ndarray[np.double_t, ndim=2] knn_dist,
                                       Relation* rel, np.intp_t* infinitesimal):
        cdef:
            np.intp_t r, j, \
                parent, \
                rank

            np.double_t dis, core_dis

            np.ndarray indices, oindices
            np.ndarray distances, odistances

        parent = self.U.mark_up(i)
        indices = knn_indices[i]
        distances = knn_dist[i]

        rel.reciprocity = INF
        core_dis = distances[0]
        for r in range(0, self.max_neighbors_search):
            j = indices[r]
            if parent == self.U.mark_up(j):
                continue

            dis = distances[r]
            if dis > core_dis + self.PRECISION:
                break

            if dis == 0.: # degenerate case.
                rel.reciprocity = 0.
                rel.endpoint = j
                rel.max_rank = bisect.bisect(distances, 0. + self.PRECISION) + 1
                return 1
            infinitesimal += dis <= self.PRECISION

            odistances = knn_dist[j]
            if odistances[0] + self.PRECISION < dis:
                return 0

            rank = r + 1
            while rank < self.max_neighbors_search and distances[rank] <= dis + self.PRECISION:
                rank += 1

            odis = odistances[rank - 1]
            if odis >= dis + self.PRECISION:
                continue
            if odis + self.PRECISION <= dis :
                continue
            if rank < self.max_neighbors_search and odistances[rank] < dis + self.PRECISION:
                continue

            rel.reciprocity = dis
            rel.endpoint = j
            rel.max_rank = rank + 1
            return 1
        return 0

    cdef bint _evaluate_reciprocity(self, np.intp_t i, np.intp_t parent, np.ndarray[np.intp_t, ndim=2] knn_indices, np.ndarray[np.double_t, ndim=2] knn_dist, Relation* rel):
        cdef:
            int rank, orank, r, inter, last, olast
            np.intp_t j, \
                res = 0

            np.double_t best, v, v1, v2, dis

            np.intp_t[:] indices
            np.intp_t[:] oindices
            np.double_t[:] distances
            np.double_t[:] odistances

        indices = knn_indices[i]
        distances = knn_dist[i]

        self.ball.clear()
        self.ball.add(i)
        best = INF
        for r in range(0, self.max_neighbors_search):

            dis = distances[r]
            if dis - self.PRECISION > best: # v всегда >= dis по построению
                break

            j = indices[r]
            self.ball.add(j)
            if self.U.is_same_parent(parent, j):
                continue
            assert(dis > self.PRECISION)

            last = r
            while last + 1 < self.max_neighbors_search and distances[last + 1] <= dis + self.PRECISION:
                last += 1
                self.ball.add(indices[last])
            rank = last + 1

            odistances = knn_dist[j]
            if odistances[last] > dis: # outlier part has more information
                continue

            oindices = knn_indices[j]

            olast = 0
            inter = 0
            while olast < self.max_neighbors_search and odistances[olast] <= dis + self.PRECISION:
                inter += oindices[olast] != i and oindices[olast] in self.ball
                olast += 1
            olast -= 1
            orank = olast + 1

            if rank > orank:
                continue
            if rank == orank and i < j:
                continue

            v1 = max(distances[orank - 1] + self.PRECISION,  dis * rank / (orank - inter)) # со своей стороны r<=oR
            v2 = max(odistances[rank - 1] + self.PRECISION,  dis * orank / (rank - inter)) # с чужой стороны
            v = min(v1, v2)

            assert(v!=0)
            assert(v+self.PRECISION>dis)

            if v >= best:
                continue

            if self.logger_debug:
                self.logger.debug('%s %s new best %s < %s', i,j, v, best)
                self.logger.debug('r %s, %s (%s) d %s (%s, %s)', rank, orank, inter, dis, distances[orank], odistances[rank])

            best = v
            rel.endpoint = j
            rel.max_rank = orank

            res = 1
        rel.reciprocity = best
        return res

    cdef _compute_tree_edges(self):
        # DRUHG
        # computes DRUHG Spanning Tree
        # uses heap
        cdef:
            np.intp_t i, \
                warn, infinitesimal

            Relation rel = Relation(0,0,0,0, 0,0)

            np.ndarray[np.double_t, ndim=2] knn_dist
            np.ndarray[np.intp_t, ndim=2] knn_indices

            list heap

        knn_dist, knn_indices = self.dist_tree.query(
                    self.tree.data,
                    k=self.max_neighbors_search,
                    dualtree=True,
                    breadth_first=True,
                    )
        heap = []
#### Initialization and pure reciprocity (ranks equal)
        warn, infinitesimal = 0, 0

        # if self.tree.data.shape[0] > 16384 and self.n_jobs > 1: # multicore 2-3x speed up for big datasets
        i = self.num_points
        while i:
            i -= 1
            if knn_dist[i][0] < 0.:
                self.logger.error('Distances cannot be negative! Exiting. '+str(i)+' '+str(knn_dist[i][0]))
                return

            if self._pure_reciprocity(i, knn_indices, knn_dist, &rel, &infinitesimal):
                self.result_write(rel.reciprocity, i, rel.endpoint, rel.max_rank - 1)
                p, op = self.U.mark_up(i), self.U.mark_up(rel.endpoint)
                self.U.union(i, rel.endpoint, p, op)

                if rel.reciprocity == 0.: # values match
                    warn += 1
                    i += 1  # need to relaunch same index
                    continue
                if rel.max_rank > 2:
                    i += 1  # need to relaunch same index
                    continue

            if self._evaluate_reciprocity(i, self.U.mark_up(i), knn_indices, knn_dist, &rel):
                heapq.heappush(heap,
                               (rel.reciprocity, i, rel.endpoint, rel.max_rank))

        if self.result_edges >= self.num_points - 1:
            self.logger.info('Two subjects only')
            return
        if warn > 0:
            self.logger.info(
            'A lot of values('+str(warn)+') are the same. Try increasing max_neighbors_search('+str(self.max_neighbors_search)+
            ') parameter.')

        if infinitesimal > 0:
            self.logger.warning('Some distances('+str(infinitesimal)+') are smaller than self.PRECISION ('+str(self.PRECISION)+
                   ') level. Try decreasing double_precision parameter.')

        self.logger.info(f'{self.result_edges:.0f} mutual edges {100.*self.result_edges/self.num_points:.2f}%')

        edge_cases = 0
############
        while self.result_edges < self.num_points - 1 and heap:
            rel.reciprocity, i, rel.endpoint, rel.max_rank = heapq.heappop(heap)

            p, op = self.U.mark_up(i), self.U.mark_up(rel.endpoint)
            if p != op:
                self.result_write(rel.reciprocity, i, rel.endpoint, rel.max_rank)
                p = self.U.union(i, rel.endpoint, p, op)
                if rel.max_rank == self.max_neighbors_search:
                    edge_cases+=1

            if self._evaluate_reciprocity(i, p, knn_indices, knn_dist, &rel):
                heapq.heappush(heap, (rel.reciprocity, i, rel.endpoint, rel.max_rank))
###############
        if self.result_edges != self.num_points - 1:
            self.logger.info('%s not connected edges of %s. It is a forest. Try increasing max_neighbors(max_ranking) value %s for a better result.',
                self.num_points - 1 - self.result_edges, self.num_points - 1, self.max_neighbors_search)
            if self.result_pairs_arr is not None:
                self.result_pairs_arr[2 * self.result_edges] = -1
                self.result_pairs_arr[2 * self.result_edges + 1] = -1
            self.result_values_arr[self.result_edges] = -1

        if self.max_neighbors_search < self.num_points - 1 and edge_cases != 0:
            # todo: may be check the actual reachability of indices?
            self.logger.info('%s edges with the max rank. Try increasing max_neighbors(max_ranking) value %s or pick the square mode (not available yet).',
                edge_cases, self.max_neighbors_search)
