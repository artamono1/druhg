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

    cpdef tuple query(self, d, k, dualtree = 0, breadth_first = 0, skip_radius=None):
        # TODO: actually we need to consider replacing INF with something else.
        # Reciprocity of absent link is not the same as the INF. Do reciprocity with graphs!
        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices
        cdef np.ndarray[np.intp_t, ndim=1] n_skipped
        cdef np.ndarray[np.double_t, ndim=1] skip_arr
        cdef np.double_t r_i, val
        cdef np.intp_t i, j, pos, yi
        cdef bint has_skip

        knn_dist = INF*np.ones((self.data_size, k))
        knn_indices = np.zeros((self.data_size, k), dtype=np.intp)
        has_skip = skip_radius is not None
        if has_skip:
            skip_arr = np.asarray(skip_radius, dtype=np.float64)
            if skip_arr.ndim == 0:
                skip_arr = np.full(self.data_size, float(skip_arr), dtype=np.float64)
            n_skipped = np.zeros(self.data_size, dtype=np.intp)
        else:
            skip_arr = np.zeros(1, dtype=np.float64)
            n_skipped = np.zeros(1, dtype=np.intp)

        warning = 0

        i = self.data_size
        while i:
            i -= 1
            row = self.data_arr.getrow(i)
            idx, data = row.indices, row.data
            sorted = np.argsort(data)
            pos = 0
            yi = 0
            r_i = skip_arr[i] if has_skip else -1.0
            for s in sorted:
                j = idx[s]
                if j == i:
                    warning += 1
                    continue
                val = data[s]
                if has_skip and val < r_i:
                    yi += 1
                    continue
                if pos >= k:
                    if not has_skip:
                        break
                    continue
                knn_dist[i][pos] = val
                knn_indices[i][pos] = j
                pos += 1
            if has_skip:
                n_skipped[i] = yi

        if warning:
            logging.getLogger(__package__).warning('Attention!: Sparse matrix has an edge that forms a loop! They were zeroed. '+str(warning))

        if has_skip:
            return n_skipped, knn_dist, knn_indices
        return knn_dist, knn_indices

cdef class PairwiseDistanceTreeGeneric(object):
    cdef object data_arr
    cdef int data_size

    def __init__(self, N, d):
        self.data_size = N
        self.data_arr = d

    cpdef tuple query(self, d, k, dualtree = 0, breadth_first = 0, skip_radius=None):
        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices
        cdef np.ndarray[np.intp_t, ndim=1] n_skipped
        cdef np.ndarray[np.double_t, ndim=1] skip_arr
        cdef np.double_t r_i, val
        cdef np.intp_t i, j, pos, yi
        cdef bint has_skip

        knn_dist = INF*np.ones((self.data_size, k))
        knn_indices = np.zeros((self.data_size, k), dtype=np.intp)
        has_skip = skip_radius is not None
        if has_skip:
            skip_arr = np.asarray(skip_radius, dtype=np.float64)
            if skip_arr.ndim == 0:
                skip_arr = np.full(self.data_size, float(skip_arr), dtype=np.float64)
            n_skipped = np.zeros(self.data_size, dtype=np.intp)
        else:
            skip_arr = np.zeros(1, dtype=np.float64)
            n_skipped = np.zeros(1, dtype=np.intp)

        i = self.data_size
        while i:
            i -= 1
            row = self.data_arr[i]
            sorted = np.argsort(row)
            pos = 0
            yi = 0
            r_i = skip_arr[i] if has_skip else -1.0
            for j in sorted:
                if j == i:
                    continue
                val = row[j]
                if has_skip and val < r_i:
                    yi += 1
                    continue
                if pos >= k:
                    if not has_skip:
                        break
                    continue
                knn_dist[i][pos] = val
                knn_indices[i][pos] = j
                pos += 1
            if has_skip:
                n_skipped[i] = yi

        if has_skip:
            return n_skipped, knn_dist, knn_indices
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
        Neighbor-query batch size. The spanning tree starts with this many
        neighbors per point and fetches another batch on demand (via
        ``skip_radius``) when the current lists cannot connect the forest.

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
        np.intp_t knn_cap

        np.intp_t n_jobs

        UnionFind U
        set ball

        np.ndarray knn_dist
        np.ndarray knn_indices
        np.ndarray knn_used
        np.ndarray skip_radius
        object query_X

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
            self.query_X = self.tree.data
        elif algorithm == 2:
            self.dist_tree = PairwiseDistanceTreeGeneric(tree.shape[0], tree)
            self.tree = tree
            self.num_points = self.tree.shape[0]
            self.query_X = self.tree
        elif algorithm == 3:
            self.dist_tree = PairwiseDistanceTreeSparse(tree.shape[0], tree)
            self.tree = tree
            self.num_points = self.tree.shape[0]
            self.query_X = self.tree
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
            self.logger.debug('+Edge %s %s value %s rank %s', a,b, v, r)


    cdef bint _pure_reciprocity(self, np.intp_t i, Relation* rel, np.intp_t* infinitesimal):
        cdef:
            np.intp_t r, j, \
                parent, \
                rank, used_i, used_j

            np.double_t dis, core_dis

            np.ndarray indices, oindices
            np.ndarray distances, odistances

        parent = self.U.mark_up(i)
        indices = self.knn_indices[i]
        distances = self.knn_dist[i]
        used_i = self.knn_used[i]

        rel.reciprocity = INF
        core_dis = distances[0]
        for r in range(0, used_i):
            j = indices[r]
            if parent == self.U.mark_up(j):
                continue

            dis = distances[r]
            if dis > core_dis + self.PRECISION:
                break

            if dis == 0.: # degenerate case.
                rel.reciprocity = 0.
                rel.endpoint = j
                rel.max_rank = bisect.bisect(distances[:used_i], 0. + self.PRECISION) + 1
                return 1
            infinitesimal += dis <= self.PRECISION

            odistances = self.knn_dist[j]
            used_j = self.knn_used[j]
            if odistances[0] + self.PRECISION < dis:
                return 0

            rank = r + 1
            while rank < used_i and distances[rank] <= dis + self.PRECISION:
                rank += 1

            if rank > used_j:
                continue
            odis = odistances[rank - 1]
            if odis >= dis + self.PRECISION:
                continue
            if odis + self.PRECISION <= dis :
                continue
            if rank < used_j and odistances[rank] < dis + self.PRECISION:
                continue

            rel.reciprocity = dis
            rel.endpoint = j
            rel.max_rank = rank + 1
            return 1
        return 0

    cdef bint _evaluate_reciprocity(self, np.intp_t i, np.intp_t parent, Relation* rel):
        cdef:
            int rank, orank, r, inter
            np.intp_t j, used_i, used_j, \
                res = 0

            np.double_t best, v, v1, v2, dis

            np.intp_t[:] indices
            np.intp_t[:] oindices
            np.double_t[:] distances
            np.double_t[:] odistances

        used_i = self.knn_used[i]
        indices = self.knn_indices[i]
        distances = self.knn_dist[i]

        self.ball.clear()
        self.ball.add(i)
        best = INF
        for r in range(0, used_i):

            dis = distances[r]
            if dis >= INF / 2.:
                break
            if dis - self.PRECISION > best: # v всегда >= dis по построению
                break

            j = indices[r]
            self.ball.add(j)
            if self.U.is_same_parent(parent, j):
                continue
            assert(dis > self.PRECISION)

            used_j = self.knn_used[j]
            odistances = self.knn_dist[j]
            if r >= used_j or odistances[r] > dis + self.PRECISION: # outlier part has more information
                continue

            rank = r + 1
            while rank < used_i and distances[rank] <= dis + self.PRECISION:
                self.ball.add(indices[rank])
                rank += 1

            if rank > used_j:
                continue
            if odistances[rank-1] > dis + self.PRECISION: # outlier part has more information
                continue

            oindices = self.knn_indices[j]
            orank = 0
            inter = 0
            while orank < used_j and odistances[orank] <= dis + self.PRECISION:
                inter += oindices[orank] != i and oindices[orank] in self.ball
                orank += 1

            if orank == 0:
                continue
            if rank > orank:
                continue
            if orank > used_i:
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
                self.logger.debug('%s-%s new best %s < %s', i,j, v, best)
                self.logger.debug('  r %s, %s (%s) d %s', rank+1, orank+1, inter, dis)

            best = v
            rel.endpoint = j
            rel.max_rank = orank

            res = 1
        rel.reciprocity = best
        return res

    cdef bint _neighbors_all_connected(self, np.intp_t i, np.intp_t parent):
        cdef np.intp_t r, used_i
        used_i = self.knn_used[i]
        for r in range(used_i):
            if self.knn_dist[i, r] >= INF / 2.:
                break
            if not self.U.is_same_parent(parent, self.knn_indices[i, r]):
                return 0
        return 1

    cdef np.intp_t _expand_knn(self, np.ndarray[np.uint8_t, ndim=1] need):
        cdef:
            np.intp_t i, t, pos, n_add, j, n, any_need, mx, extra, added, u
            np.double_t d
            np.ndarray skip
            np.ndarray n_skipped
            np.ndarray[np.double_t, ndim=2] new_dist
            np.ndarray[np.intp_t, ndim=2] new_ind
            bint seen

        n = self.num_points
        n_add = self.max_neighbors_search
        if n_add < 1:
            n_add = 1
        if n_add > n - 1:
            n_add = n - 1

        any_need = 0
        mx = 0
        for i in range(n):
            if need[i]:
                u = self.knn_used[i]
                if u >= n - 1:
                    need[i] = 0
                    continue
                any_need += 1
                if u > mx:
                    mx = u
        if any_need == 0:
            return 0

        if mx + n_add > self.knn_cap:
            extra = mx + n_add - self.knn_cap
            self.knn_dist = np.hstack((
                self.knn_dist, np.full((n, extra), INF, dtype=np.double)))
            self.knn_indices = np.hstack((
                self.knn_indices, np.zeros((n, extra), dtype=np.intp)))
            self.knn_cap += extra

        skip = np.full(n, np.inf, dtype=np.double)
        for i in range(n):
            if need[i] and self.knn_used[i] > 0:
                skip[i] = self.knn_dist[i, self.knn_used[i] - 1]
                self.skip_radius[i] = skip[i]

        self.logger.info('kNN expand: %s points +%s neighbors, skip_radius from stored prefix',
                         any_need, n_add)
        n_skipped, new_dist, new_ind = self.dist_tree.query(
            self.query_X, k=n_add, dualtree=True, breadth_first=True, skip_radius=skip)

        added = 0
        for i in range(n):
            if not need[i]:
                continue
            pos = self.knn_used[i]
            for t in range(n_add):
                d = new_dist[i, t]
                if not np.isfinite(d) or d >= INF / 2.:
                    break
                j = new_ind[i, t]
                seen = 0
                for u in range(pos):
                    if self.knn_indices[i, u] == j:
                        seen = 1
                        break
                if seen:
                    continue
                if pos >= n - 1:
                    break
                self.knn_dist[i, pos] = d
                self.knn_indices[i, pos] = j
                pos += 1
                added += 1
            self.knn_used[i] = pos
            if pos > 0:
                self.skip_radius[i] = self.knn_dist[i, pos - 1]
        self.logger.info('kNN expand: done, +%s stored', added)
        return added

    cdef _compute_tree_edges(self):
        # DRUHG
        # computes DRUHG Spanning Tree
        # uses heap
        cdef:
            np.intp_t i, \
                warn, infinitesimal, k0, edge_cases, grown

            Relation rel = Relation(0,0,0,0, 0,0)

            list heap
            np.ndarray[np.uint8_t, ndim=1] need

        k0 = self.max_neighbors_search
        if k0 < 1:
            k0 = 1
        if k0 > self.num_points - 1:
            k0 = self.num_points - 1
        self.knn_cap = k0
        self.knn_used = np.full(self.num_points, k0, dtype=np.intp)
        self.skip_radius = np.zeros(self.num_points, dtype=np.double)

        self.logger.info(f'kNN querying: %s', k0)
        self.knn_dist, self.knn_indices = self.dist_tree.query(
                    self.query_X,
                    k=k0,
                    dualtree=True,
                    breadth_first=True,
                    )
        self.logger.info('kNN querying: done')
        for i in range(self.num_points):
            self.skip_radius[i] = self.knn_dist[i, k0 - 1]

        heap = []
#### Initialization and pure reciprocity (ranks equal)
        self.logger.info(f'MSTree formation: initializing nearest connections. Pure autoconnect.')
        warn, infinitesimal = 0, 0
        i = self.num_points
        while i:
            i -= 1
            if self.knn_dist[i][0] < 0.:
                self.logger.error('Distances cannot be negative! Exiting. '+str(i)+' '+str(self.knn_dist[i][0]))
                return
            if self._pure_reciprocity(i, &rel, &infinitesimal):
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

            if self._evaluate_reciprocity(i, self.U.mark_up(i), &rel):
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

        self.logger.info(f'MSTree formation: {self.result_edges:.0f} pure edges {100.*self.result_edges/self.num_points:.2f}%. Continue with complex connections.')
        edge_cases = 0
############
        while self.result_edges < self.num_points - 1:
            if heap:
                rel.reciprocity, i, rel.endpoint, rel.max_rank = heapq.heappop(heap)

                p, op = self.U.mark_up(i), self.U.mark_up(rel.endpoint)
                if p != op:
                    self.result_write(rel.reciprocity, i, rel.endpoint, rel.max_rank)
                    p = self.U.union(i, rel.endpoint, p, op)
                    if rel.max_rank == self.knn_used[i] or rel.max_rank == self.knn_used[rel.endpoint]:
                        edge_cases += 1

                if self._evaluate_reciprocity(i, p, &rel):
                    heapq.heappush(heap, (rel.reciprocity, i, rel.endpoint, rel.max_rank))
                continue

            need = np.zeros(self.num_points, dtype=np.uint8)
            grown = 0
            for i in range(self.num_points):
                if self.knn_used[i] >= self.num_points - 1:
                    continue
                p = self.U.mark_up(i)
                if self._neighbors_all_connected(i, p):
                    need[i] = 1
                    grown += 1
            if grown == 0:
                break
            if self._expand_knn(need) == 0:
                break
            for i in range(self.num_points):
                if not need[i]:
                    continue
                if self._evaluate_reciprocity(i, self.U.mark_up(i), &rel):
                    heapq.heappush(heap, (rel.reciprocity, i, rel.endpoint, rel.max_rank))
###############
        self.logger.info(
            'MSTree formation: %s edges %.2f%%. Done.',
            self.result_edges, 100. * self.result_edges / self.num_points)
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
