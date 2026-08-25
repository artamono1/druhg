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
import queue

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

    cpdef tuple query_init(self, d, k, dualtree = 0, breadth_first = 0, skip_radius=None, indices=None):
        # TODO: actually we need to consider replacing INF with something else.
        # Reciprocity of absent link is not the same as the INF. Do reciprocity with graphs!
        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices
        cdef np.ndarray[np.intp_t, ndim=1] n_skipped
        cdef np.ndarray[np.double_t, ndim=1] skip_arr
        cdef np.ndarray[np.intp_t, ndim=1] query_ids
        cdef np.double_t r_i, val
        cdef np.intp_t i, j, pos, yi, q, n_q
        cdef bint has_skip

        if indices is None:
            query_ids = np.arange(self.data_size, dtype=np.intp)
        else:
            query_ids = np.ascontiguousarray(np.asarray(indices, dtype=np.intp).reshape(-1))
        n_q = query_ids.shape[0]

        knn_dist = INF*np.ones((n_q, k))
        knn_indices = np.zeros((n_q, k), dtype=np.intp)
        has_skip = skip_radius is not None
        if has_skip:
            skip_arr = np.asarray(skip_radius, dtype=np.float64)
            if skip_arr.ndim == 0:
                skip_arr = np.full(n_q, float(skip_arr), dtype=np.float64)
            elif skip_arr.shape[0] == self.data_size and n_q != self.data_size:
                skip_arr = skip_arr[query_ids]
            n_skipped = np.zeros(n_q, dtype=np.intp)
        else:
            skip_arr = np.zeros(1, dtype=np.float64)
            n_skipped = np.zeros(1, dtype=np.intp)

        warning = 0

        q = n_q
        while q:
            q -= 1
            i = query_ids[q]
            row = self.data_arr.getrow(i)
            idx, data = row.indices, row.data
            sorted = np.argsort(data)
            pos = 0
            yi = 0
            r_i = skip_arr[q] if has_skip else -1.0
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
                knn_dist[q][pos] = val
                knn_indices[q][pos] = j
                pos += 1
            if has_skip:
                n_skipped[q] = yi

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

    cpdef tuple query_init(self, d, k, dualtree = 0, breadth_first = 0, skip_radius=None, indices=None):
        cdef np.ndarray[np.double_t, ndim=2] knn_dist
        cdef np.ndarray[np.intp_t, ndim=2] knn_indices
        cdef np.ndarray[np.intp_t, ndim=1] n_skipped
        cdef np.ndarray[np.double_t, ndim=1] skip_arr
        cdef np.ndarray[np.intp_t, ndim=1] query_ids
        cdef np.double_t r_i, val
        cdef np.intp_t i, j, pos, yi, q, n_q
        cdef bint has_skip

        if indices is None:
            query_ids = np.arange(self.data_size, dtype=np.intp)
        else:
            query_ids = np.ascontiguousarray(np.asarray(indices, dtype=np.intp).reshape(-1))
        n_q = query_ids.shape[0]

        knn_dist = INF*np.ones((n_q, k))
        knn_indices = np.zeros((n_q, k), dtype=np.intp)
        has_skip = skip_radius is not None
        if has_skip:
            skip_arr = np.asarray(skip_radius, dtype=np.float64)
            if skip_arr.ndim == 0:
                skip_arr = np.full(n_q, float(skip_arr), dtype=np.float64)
            elif skip_arr.shape[0] == self.data_size and n_q != self.data_size:
                skip_arr = skip_arr[query_ids]
            n_skipped = np.zeros(n_q, dtype=np.intp)
        else:
            skip_arr = np.zeros(1, dtype=np.float64)
            n_skipped = np.zeros(1, dtype=np.intp)

        q = n_q
        while q:
            q -= 1
            i = query_ids[q]
            row = self.data_arr[i]
            sorted = np.argsort(row)
            pos = 0
            yi = 0
            r_i = skip_arr[q] if has_skip else -1.0
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
                knn_dist[q][pos] = val
                knn_indices[q][pos] = j
                pos += 1
            if has_skip:
                n_skipped[q] = yi

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

    max_neighbors_search : int, optional (default= n-1)
        Hard cap on stored neighbors per point (``max_ranking``).

    step_expansion : int, optional (default= 16)
        Neighbor-query batch size. Starts with this many neighbors per point
        and fetches another batch on demand (via ``skip_radius``) while
        under ``max_neighbors_search``.

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
        np.intp_t step_expansion
        np.intp_t knn_cap

        np.intp_t n_jobs

        UnionFind U
        set ball

        np.ndarray knn_dist
        np.ndarray knn_indices
        np.ndarray knn_skip
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
                 max_neighbors_search=None, step_expansion=16, metric='euclidean', leaf_size=16, n_jobs=4,
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

        if max_neighbors_search is None:
            self.max_neighbors_search = self.num_points
        else:
            self.max_neighbors_search = max_neighbors_search

        if self.step_expansion is None:
            self.step_expansion = self.num_points
        else:
            self.step_expansion = step_expansion
        if self.step_expansion < 1:
            self.step_expansion = 1

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
        arr_size = self.step_expansion

        rel.reciprocity = INF
        core_dis = distances[0]
        for r in range(0, arr_size):
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

            odistances = self.knn_dist[j]
            if odistances[0] + self.PRECISION < dis:
                return 0

            rank = r + 1
            while rank < arr_size and distances[rank] <= dis + self.PRECISION:
                rank += 1

            odis = odistances[rank - 1]
            if odis >= dis + self.PRECISION:
                continue
            if odis + self.PRECISION <= dis :
                continue
            if rank < arr_size and odistances[rank] < dis + self.PRECISION:
                continue

            rel.reciprocity = dis
            rel.endpoint = j
            rel.max_rank = rank + 1
            return 1
        return 0

    cdef bint _evaluate_reciprocity(self, np.intp_t i, np.intp_t parent, Relation* rel):
        cdef:
            int rank, orank, r, inter,\
                rr, rank_united, orank_united, \
                is_united
            np.intp_t j,  \
                res = 0

            np.double_t best, v, v1, v2, \
                    d, ddis, oddis

            np.intp_t[:] indices
            np.intp_t[:] oindices
            np.double_t[:] distances
            np.double_t[:] odistances

        rank_united = self.knn_skip[i]
        indices = self.knn_indices[i]
        distances = self.knn_dist[i]

        arr_size = self.step_expansion

        is_reachable = 0
        is_united = 0
        is_full = 1

        self.ball.clear()
        self.ball.add(i)
        best = INF
        for r in range(0, arr_size):

            d = distances[r]
            if d == 0:
                break
            if d - self.PRECISION > best: # v всегда >= dis по построению
                break

            j = indices[r]
            self.ball.add(j)
            if self.U.is_same_parent(parent, j):
                if is_united == 0:
                    self.skip_radius[i] = d
                continue
            assert(d > self.PRECISION)

            is_full = 0
            is_united = 1

            rank = r + rank_united

            orank_united = self.knn_skip[j]
            if rank >= orank_united + arr_size + 1:
                continue

            rr = r + 1
            while rr < arr_size and distances[rr] <= d + self.PRECISION: # equidistant case
                self.ball.add(indices[rr])
                rr += 1
                rank += 1

            odistances = self.knn_dist[j]
            oindices = self.knn_indices[j]

            is_reachable = -1
            orank = orank_united
            inter = 0
            rr = 0
            while rr < arr_size and odistances[rr] <= d + self.PRECISION:
                if oindices[rr] == i:
                    is_reachable = 1
                elif oindices[rr] in self.ball:
                    inter += 1
                rr += 1
                orank += 1

            ddis = d
            oddis = d
            if rank < orank:
                rr = orank - 1 - rank_united
                ddis = distances[rr] if rr>0 and rr < arr_size else d*2.
            else:
                rr = rank - 1 - orank_united
                oddis = odistances[rr] if rr>0 and rr < arr_size else d*2.

            v1 = max(ddis + self.PRECISION,  d * rank / (orank - inter)) # со своей стороны r<=oR
            v2 = max(oddis + self.PRECISION,  d * orank / (rank - inter)) # с чужой стороны
            v = min(v1, v2)

            assert(v!=0)
            assert(v+self.PRECISION>d)

            if v >= best:
                continue

            if self.logger_debug:
                self.logger.debug('%s-%s new best %s < %s', i,j, v, best)
                self.logger.debug('  r %s, %s (%s) d %s', rank+1, orank+1, inter, d)

            best = v
            rel.endpoint = j
            rel.max_rank = orank * is_reachable

            res = 1
        rel.reciprocity = best
        rel.is_full = is_full

        return res

    cdef np.intp_t _expand_knn(self, object need):
        cdef:
            np.intp_t i, t, pos, n_add, j, n, q, n_q, added, u
            np.double_t d
            np.ndarray idx
            np.ndarray n_skipped
            np.ndarray[np.double_t, ndim=2] new_dist
            np.ndarray[np.intp_t, ndim=2] new_ind
            bint seen
            object need_arr

        n = self.num_points
        n_add = self.step_expansion
        need_arr = np.asarray(need)
        if need_arr.ndim == 1 and need_arr.shape[0] == n and need_arr.dtype == np.uint8:
            idx = np.flatnonzero(need_arr).astype(np.intp)
        else:
            idx = np.ascontiguousarray(need_arr, dtype=np.intp).ravel()
        n_q = idx.shape[0]
        if n_q == 0:
            return 0

        self.logger.info('kNN expand: %s points +%s neighbors, skip_radius from stored prefix',
                         n_q, n_add)

        n_skipped, new_dist, new_ind = self.dist_tree.query_skip(indices=idx, skip_radius=self.skip_radius, k=n_add)

        added = 0
        for q in range(n_q):
            i = idx[q]
            self.knn_skip[i] = n_skipped[q]

            for t in range(n_add):
                self.knn_dist[i, t] = new_dist[q, t]
                self.knn_indices[i, t] = new_ind[q, t]

        self.logger.info('kNN expand: done, +%s stored', added)
        return added

    cdef _compute_tree_edges(self):
        # DRUHG
        # computes DRUHG Spanning Tree
        # uses heap
        cdef:
            np.intp_t i, \
                warn, infinitesimal, pure_unreachable, edge_cases, grown

            Relation rel = Relation(0,0,0,0, 0,0)

            list heap
            queue need

        self.knn_skip = np.ones(self.num_points, dtype=np.intp)
        self.skip_radius = np.zeros(self.num_points, dtype=np.double)

        self.logger.info(f'kNN querying: step expansion %s', self.step_expansion)
        self.knn_dist, self.knn_indices = self.dist_tree.query_init(
                    self.query_X,
                    k=self.step_expansion,
                    dualtree=True,
                    breadth_first=True,
                    )
        self.logger.info('kNN querying: done')

        heap = []
#### Initialization and pure reciprocity (ranks equal)
        self.logger.info(f'MSTree formation: initializing nearest connections. Pure autoconnect.')
        warn, infinitesimal, pure_unreachable = 0, 0, 0
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
            else:
                pure_unreachable += 1

        if self.result_edges >= self.num_points - 1:
            self.logger.info('Two subjects only')
            return
        if warn > 0:
            self.logger.info(
            'A lot of values('+str(warn)+') are the same. Try increasing step_expansion('+str(self.step_expansion)+
            ') parameter.')

        if infinitesimal > 0:
            self.logger.warning('Some distances('+str(infinitesimal)+') are smaller than self.PRECISION ('+str(self.PRECISION)+
                   ') level. Try decreasing double_precision parameter.')

        if pure_unreachable > 0:
            self.logger.warning('A lot of points are unreachable('+str(pure_unreachable)+') from the start. And they are removed from evaluation. Try increasing step_expansion('+str(self.step_expansion)+
            ') parameter.')

        self.logger.info(f'MSTree formation: {self.result_edges:.0f} pure edges {100.*self.result_edges/self.num_points:.2f}%. Continue with complex connections.')
        edge_cases = 0
############
        need = queue.LifoQueue()
        need_expansion = 0
        while heap and self.result_edges < self.num_points - 1:
            rel.reciprocity, i, rel.endpoint, rel.max_rank = heapq.heappop(heap)

            p, op = self.U.mark_up(i), self.U.mark_up(rel.endpoint)
            if p != op:
                if not need:
                    self.result_write(rel.reciprocity, i, rel.endpoint, rel.max_rank)
                    p = self.U.union(i, rel.endpoint, p, op)
                    if rel.max_rank < 0:
                        edge_cases += 1
                else:
                    heapq.heappush(heap, (rel.reciprocity, i, rel.endpoint, rel.max_rank))
                    self._expand_knn(need)
                    need_count = len(need)
                    while need_count:
                        need_count -= 1
                        i = need.pop
                        if self._evaluate_reciprocity(i, self.U.mark_up(i), &rel):
                            heapq.heappush(heap, (rel.reciprocity, i, rel.endpoint, rel.max_rank))
                        elif rel.is_full != 0:
                            need.put(i)
                        else:
                            self.logger.warning('%s point is dropped', i)
                    continue

            if self._evaluate_reciprocity(i, p, &rel):
                heapq.heappush(heap, (rel.reciprocity, i, rel.endpoint, rel.max_rank))
            elif rel.is_full != 0:
                need.put(i)
            else:
                self.logger.warning('%s point is dropped', i)
            continue

###############
        self.logger.info(
            'MSTree formation: %s edges %.2f%%. Done.',
            self.result_edges, 100. * self.result_edges / self.num_points)
        if self.result_edges != self.num_points - 1:
            self.logger.info('%s not connected edges of %s. It is a forest. Try increasing max_ranking(%s) or step_expansion(%s).',
                self.num_points - 1 - self.result_edges, self.num_points - 1, self.max_neighbors_search, self.step_expansion)
            if self.result_pairs_arr is not None:
                self.result_pairs_arr[2 * self.result_edges] = -1
                self.result_pairs_arr[2 * self.result_edges + 1] = -1
            self.result_values_arr[self.result_edges] = -1

        if self.max_neighbors_search < self.num_points - 1 and edge_cases != 0:
            # todo: may be check the actual reachability of indices?
            self.logger.info('%s edges with the max rank. Try increasing max_ranking(%s).',
                edge_cases, self.max_neighbors_search)
