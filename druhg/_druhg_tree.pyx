# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Builds spanning tree for druhg algorithm
# uses dialectics to evaluate reciprocity
# links per-branch heap tops via a FIFO of branches (not a global min-heap)
# Author: Pavel Artamonov
# License: 3-clause BSD


import numpy as np
cimport numpy as np
import sys
import time
import logging

from ._druhg_tree_logging import TreeLogging

cdef extern from "Python.h":
    int PyErr_CheckSignals() except -1

from ._druhg_unionfind import UnionFind
from ._druhg_unionfind cimport UnionFind
from ._druhg_pairwise import PairwiseDistanceTreeSparse, PairwiseDistanceTreeGeneric

import _heapq as heapq
from collections import deque

import bisect

cdef np.double_t INF = sys.float_info.max

def allocate_buffer_values(np.intp_t num_points):
    return np.empty((num_points - 1), dtype=np.double)
def allocate_buffer_edgepairs(np.intp_t num_points):
    return np.empty((num_points*2 - 2), dtype=np.intp)
def allocate_buffer_ranks(np.intp_t num_points):
    return np.empty((num_points - 1), dtype=np.intp)


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
        object log

        np.double_t t0
        np.double_t progress_interval
        np.double_t t_last_progress
        bint interrupted
        object interrupt_reason

        np.ndarray opt_value_arr
        np.ndarray opt_endpoint_arr
        np.ndarray opt_rank_arr
        np.ndarray opt_target_arr
        np.double_t[:] opt_value
        np.intp_t[:] opt_endpoint
        np.intp_t[:] opt_rank
        np.intp_t[:] opt_target
        list branch_heap
        object branch_queue

    def __init__(self, algorithm, tree,
                 buffer_uf, buffer_fast, buffer_values,
                 max_neighbors_search=16, metric='euclidean', leaf_size=20, n_jobs=4,
                 buffer_ranks=None, buffer_edgepairs=None,
                 buffer_clusters=None,
                 progress_interval=None,
                 **kwargs):

        self.log = TreeLogging()
        self.logger = self.log.logger
        self.logger_debug = self.log.debug_enabled

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

        self.progress_interval = 0.
        if progress_interval is not None:
            p = float(progress_interval)
            if p > 0.:
                self.progress_interval = p
        self.interrupted = 0
        self.interrupt_reason = None
        self.t0 = 0.
        self.t_last_progress = 0.

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

    cpdef object get_interrupt_reason(self):
        return self.interrupt_reason

    cpdef tuple get_buffers(self):
        return self.result_values_arr, self.U.parent_arr

    cdef void _note_interrupt(self, reason) except *:
        if self.interrupted or self.result_edges >= self.num_points - 1:
            return
        self.interrupted = 1
        self.interrupt_reason = reason
        self.log.note_interrupt(reason, self.result_edges, self.num_points)

    cdef void _prompt_progress(self) except *:
        self.log.prompt_progress(
            self.result_edges, self.num_points, time.monotonic() - self.t0)
        self.t_last_progress = time.monotonic()

    cdef void _should_stop_mst(self) except *:
        if (self.result_edges < self.num_points - 1
                and self.progress_interval > 0.
                and (time.monotonic() - self.t_last_progress) >= self.progress_interval):
            self._prompt_progress()
        PyErr_CheckSignals()

    cdef void _finalize_incomplete_tree(self):
        if self.result_edges >= self.num_points - 1:
            return
        if self.result_pairs_arr is not None:
            self.result_pairs_arr[2 * self.result_edges] = -1
            self.result_pairs_arr[2 * self.result_edges + 1] = -1
        self.result_values_arr[self.result_edges] = -1

    cdef void _finish_mst(self, np.intp_t edge_cases) except *:
        self.log.finish_mst(
            self.interrupted, self.interrupt_reason,
            self.result_edges, self.num_points,
            edge_cases, self.max_neighbors_search,
            time.monotonic() - self.t0)
        if self.result_edges != self.num_points - 1:
            self._finalize_incomplete_tree()

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
            int rank, orank, r, inter
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

            odistances = knn_dist[j]
            if odistances[r] > dis + self.PRECISION: # outlier part has more information
                continue

            rank = r + 1
            while rank < self.max_neighbors_search and distances[rank] <= dis + self.PRECISION:
                self.ball.add(indices[rank])
                rank += 1

            if odistances[rank-1] > dis + self.PRECISION: # outlier part has more information
                continue

            oindices = knn_indices[j]
            orank = 0
            inter = 0
            while orank < self.max_neighbors_search and odistances[orank] <= dis + self.PRECISION:
                inter += oindices[orank] != i and oindices[orank] in self.ball
                orank += 1

            assert(rank <= orank)

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
                self.logger.debug('  r %s, %s (%s) d %s (%s, %s)', rank+1, orank+1, inter, dis, distances[orank], odistances[rank])

            best = v
            rel.endpoint = j
            rel.max_rank = orank

            res = 1
        rel.reciprocity = best
        return res

    cdef np.intp_t _component_root(self, np.intp_t p):
        cdef np.intp_t parent

        while True:
            parent = self.U.parent_arr[p]
            if parent == 0:
                return p
            p = parent

    cdef void _clear_optimum(self, np.intp_t i):
        self.opt_value[i] = INF
        self.opt_endpoint[i] = -1
        self.opt_rank[i] = 0
        self.opt_target[i] = -1

    cdef void _set_optimum(self, np.intp_t i, Relation* rel) except *:
        cdef np.intp_t p, target

        self._clear_optimum(i)
        self.opt_value[i] = rel.reciprocity
        self.opt_endpoint[i] = rel.endpoint
        self.opt_rank[i] = <np.intp_t> rel.max_rank
        target = self.U.mark_up(rel.endpoint)
        self.opt_target[i] = target
        p = self.U.mark_up(i)
        heapq.heappush(self.branch_heap[p], (rel.reciprocity, i))

    cdef void _refresh_optimum(self, np.intp_t i,
                               np.ndarray[np.intp_t, ndim=2] knn_indices,
                               np.ndarray[np.double_t, ndim=2] knn_dist) except *:
        cdef Relation rel = Relation(0, 0, 0, 0, 0, 0)
        cdef np.intp_t parent

        parent = self.U.mark_up(i)
        self._clear_optimum(i)
        if self._evaluate_reciprocity(i, parent, knn_indices, knn_dist, &rel):
            self._set_optimum(i, &rel)

    cdef void _absorb_heap(self, np.intp_t A, np.intp_t B, np.intp_t C) except *:
        cdef list small, large, ha, hb
        cdef object item

        ha = self.branch_heap[A]
        hb = self.branch_heap[B]
        if len(ha) < len(hb):
            small = ha
            large = hb
        else:
            small = hb
            large = ha
        for item in small:
            heapq.heappush(large, item)
        self.branch_heap[C] = large
        if A != C:
            self.branch_heap[A] = []
        if B != C:
            self.branch_heap[B] = []

    cdef void _reattach_heaps(self) except *:
        cdef np.intp_t lab, root, n
        cdef object item

        n = 2 * self.num_points
        lab = 0
        while lab < n:
            if self.branch_heap[lab]:
                root = self._component_root(lab)
                if root != lab:
                    for item in self.branch_heap[lab]:
                        heapq.heappush(self.branch_heap[root], item)
                    self.branch_heap[lab] = []
            lab += 1

    cdef void _enqueue_live_roots(self) except *:
        cdef np.intp_t i, p
        cdef set seen

        seen = set()
        self.branch_queue = deque()
        i = self.num_points
        while i:
            i -= 1
            p = self.U.mark_up(i)
            if p in seen:
                continue
            seen.add(p)
            self.branch_queue.append(p)

    cdef bint _peek_top(self, np.intp_t A,
                        np.intp_t* out_i, np.intp_t* out_j,
                        np.double_t* out_v, np.intp_t* out_B) except *:
        cdef list heap
        cdef np.intp_t i, j, B, p
        cdef np.double_t v
        cdef object top

        heap = self.branch_heap[A]
        while heap:
            top = heap[0]
            v = top[0]
            i = top[1]
            if i < 0 or i >= self.num_points:
                heapq.heappop(heap)
                continue
            if self.opt_value[i] != v:
                heapq.heappop(heap)
                continue
            p = self.U.mark_up(i)
            if p != A:
                heapq.heappop(heap)
                continue
            j = self.opt_endpoint[i]
            if j < 0:
                heapq.heappop(heap)
                continue
            B = self.U.mark_up(j)
            out_i[0] = i
            out_j[0] = j
            out_v[0] = v
            out_B[0] = B
            return 1
        return 0

    cdef bint _resolve_self_target(self, np.intp_t A,
                                   np.intp_t* out_i, np.intp_t* out_j,
                                   np.double_t* out_v, np.intp_t* out_B,
                                   np.ndarray[np.intp_t, ndim=2] knn_indices,
                                   np.ndarray[np.double_t, ndim=2] knn_dist,
                                   bint* refreshed) except *:
        cdef np.intp_t i, j, B
        cdef np.double_t v

        refreshed[0] = 0
        while self._peek_top(A, &i, &j, &v, &B):
            if B != A:
                out_i[0] = i
                out_j[0] = j
                out_v[0] = v
                out_B[0] = B
                return 1
            refreshed[0] = 1
            heapq.heappop(self.branch_heap[A])
            self._refresh_optimum(i, knn_indices, knn_dist)
        return 0

    cdef np.intp_t _link_branches(self, np.intp_t A, np.intp_t B,
                                  np.intp_t i, np.intp_t j, np.double_t v,
                                  np.intp_t* edge_cases):
        cdef np.intp_t C, rank

        rank = self.opt_rank[i]
        self.result_write(v, i, j, rank)
        C = self.U.union(i, j, A, B)
        if rank == self.max_neighbors_search:
            edge_cases[0] += 1
        self._absorb_heap(A, B, C)
        return C

    cdef void _compute_tree_edges(self) except *:
        cdef np.intp_t edge_cases

        edge_cases = 0
        self.t0 = time.monotonic()
        self.t_last_progress = self.t0
        try:
            edge_cases = self._form_mst()
        except KeyboardInterrupt:
            self._note_interrupt('KeyboardInterrupt')
        try:
            self._finish_mst(edge_cases)
        except KeyboardInterrupt:
            self._note_interrupt('KeyboardInterrupt')

    cdef np.intp_t _form_mst(self) except -1:
        # DRUHG
        # computes DRUHG spanning tree from a FIFO of per-branch heap tops
        cdef:
            np.intp_t i, j, p, op, pp, A, B, C, N, \
                warn, infinitesimal, edge_cases, skips
            np.intp_t bi, bj, bB
            np.double_t v, bv
            bint refreshed

            Relation rel = Relation(0,0,0,0, 0,0)

            np.ndarray[np.double_t, ndim=2] knn_dist
            np.ndarray[np.intp_t, ndim=2] knn_indices

            object qitem

        N = self.num_points
        self.opt_value_arr = np.full(N, INF)
        self.opt_endpoint_arr = np.full(N, -1, dtype=np.intp)
        self.opt_rank_arr = np.zeros(N, dtype=np.intp)
        self.opt_target_arr = np.full(N, -1, dtype=np.intp)
        self.opt_value = self.opt_value_arr
        self.opt_endpoint = self.opt_endpoint_arr
        self.opt_rank = self.opt_rank_arr
        self.opt_target = self.opt_target_arr
        self.branch_heap = [[] for _ in range(2 * N)]
        self.branch_queue = deque()

        edge_cases = 0
        self.log.knn_query_start(
            self.max_neighbors_search, self.num_points,
            self.progress_interval > 0.)
        knn_dist, knn_indices = self.dist_tree.query(
                    self.tree.data,
                    k=self.max_neighbors_search,
                    dualtree=True,
                    breadth_first=True,
                    )
        self.log.knn_query_done()
        self._should_stop_mst()

#### Initialization and pure reciprocity (ranks equal)
        self.logger.info(f'MSTree formation: initializing nearest connections. Pure autoconnect.')
        warn, infinitesimal = 0, 0

        i = self.num_points
        while i:
            self._should_stop_mst()
            i -= 1
            if knn_dist[i][0] < 0.:
                self.log.error('Distances cannot be negative! Exiting. '+str(i)+' '+str(knn_dist[i][0]))
                return edge_cases
            if self._pure_reciprocity(i, knn_indices, knn_dist, &rel, &infinitesimal):
                self.result_write(rel.reciprocity, i, rel.endpoint, rel.max_rank - 1)
                p, op = self.U.mark_up(i), self.U.mark_up(rel.endpoint)
                pp = self.U.union(i, rel.endpoint, p, op)
                self._absorb_heap(p, op, pp)

                if rel.reciprocity == 0.: # values match
                    warn += 1
                    i += 1  # need to relaunch same index
                    continue
                if rel.max_rank > 2:
                    i += 1  # need to relaunch same index
                    continue

            if self._evaluate_reciprocity(i, self.U.mark_up(i), knn_indices, knn_dist, &rel):
                self._set_optimum(i, &rel)

        if self.result_edges >= self.num_points - 1:
            self.log.info('Two subjects only')
            return edge_cases
        if warn > 0:
            self.log.info(
            'A lot of values('+str(warn)+') are the same. Try increasing max_neighbors_search('+str(self.max_neighbors_search)+
            ') parameter.')

        if infinitesimal > 0:
            self.log.warning('Some distances('+str(infinitesimal)+') are smaller than self.PRECISION ('+str(self.PRECISION)+
                   ') level. Try decreasing double_precision parameter.')

        self._reattach_heaps()
        self._enqueue_live_roots()

        self.logger.info(f'MSTree formation: {self.result_edges:.0f} pure edges {100.*self.result_edges/self.num_points:.2f}%. Continue with branch connections.')
############
        skips = 0
        while self.result_edges < self.num_points - 1 and self.branch_queue:
            self._should_stop_mst()
            qitem = self.branch_queue.popleft()
            A = qitem
            # 1. already merged: drop and continue
            if self._component_root(A) != A:
                continue
            # 2. reevaluate while the top still targets this branch
            if not self._resolve_self_target(A, &i, &j, &v, &B, knn_indices, knn_dist, &refreshed):
                continue
            # 3. connect if the opposite branch's outgoing top is worse or equal
            if self._peek_top(B, &bi, &bj, &bv, &bB):
                # if bB != B and bv < v - self.PRECISION:
                if bv < v - self.PRECISION:
                    self.branch_queue.append(A)
                    if refreshed:
                        skips = 0
                    else:
                        skips += 1
                        if skips >= len(self.branch_queue):
                            break
                    continue
            C = self._link_branches(A, B, i, j, v, &edge_cases)
            self.branch_queue.append(C)
            skips = 0

        return edge_cases
