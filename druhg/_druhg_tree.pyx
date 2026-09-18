# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

# Builds spanning tree for druhg algorithm
# uses dialectics to evaluate reciprocity
# links per-branch heap tops via FIFO of targeting bulks (not a global min-heap)
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
        UnionFind B
        np.ndarray ball_stamp_arr
        np.uint32_t[::1] ball_stamp
        np.uint32_t ball_gen

        np.intp_t count_evaluations
        np.intp_t count_inner_evaluations

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

        np.ndarray opt_values_arr
        np.ndarray opt_endpoints_arr
        np.ndarray opt_rank_arr
        np.double_t[:] opt_values
        np.intp_t[:] opt_endpoints
        np.intp_t[:] opt_rank
        list bulk_heap
        object bulk_queue
        np.ndarray bulk_of_arr
        np.intp_t[:] bulk_of
        np.intp_t out_i
        np.intp_t out_j
        np.intp_t out_A
        np.double_t out_v

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
        self.count_evaluations = 0
        self.count_inner_evaluations = 0

        self.PRECISION = kwargs.get('double_precision', 0.0000001)  # relevant if distances are tiny
        self.n_jobs = n_jobs

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

        # Generation-stamped membership: stamp[j] == ball_gen means j is in the ball.
        # Avoids Python set hash traffic on every reciprocity evaluation.
        self.ball_stamp_arr = np.zeros(self.num_points, dtype=np.uint32)
        self.ball_stamp = self.ball_stamp_arr
        self.ball_gen = 0

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

        self.count_evaluations += 1

        indices = knn_indices[i]
        distances = knn_dist[i]

        self.ball_gen += 1
        if self.ball_gen == 0:
            # uint32 wrap: generation 0 collides with the zeroed array, so reset.
            self.ball_stamp_arr.fill(0)
            self.ball_gen = 1
        self.ball_stamp[i] = self.ball_gen
        best = INF
        for r in range(0, self.max_neighbors_search):

            dis = distances[r]
            if dis - self.PRECISION > best: # v всегда >= dis по построению
                break

            j = indices[r]
            self.ball_stamp[j] = self.ball_gen
            if self.U.is_same_parent(parent, j):
                continue
            assert(dis > self.PRECISION)

            self.count_inner_evaluations += 1

            odistances = knn_dist[j]
            # if odistances[r] > dis + self.PRECISION: # outlier part has more information
                # continue

            rank = r + 1
            while rank < self.max_neighbors_search and distances[rank] <= dis + self.PRECISION:
                self.ball_stamp[indices[rank]] = self.ball_gen
                rank += 1

            # if odistances[rank-1] > dis + self.PRECISION: # outlier part has more information
                # continue

            oindices = knn_indices[j]
            orank = 0
            inter = 0
            while orank < self.max_neighbors_search and odistances[orank] <= dis + self.PRECISION:
                inter += oindices[orank] != i and self.ball_stamp[oindices[orank]] == self.ball_gen
                orank += 1

            # assert(rank <= orank)

            # if rank == orank and i < j:
                # continue

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

    cdef void _clear_optimum(self, np.intp_t i):
        self.opt_values[i] = INF
        self.opt_endpoints[i] = -1
        self.opt_rank[i] = 0

    cdef void _set_optimum(self, np.intp_t i, Relation* rel) except *:
        self.opt_values[i] = rel.reciprocity
        self.opt_endpoints[i] = rel.endpoint
        self.opt_rank[i] = <np.intp_t> rel.max_rank

    cdef void _absorb_heap(self, np.intp_t A, np.intp_t B, np.intp_t C) except *:
        cdef list small, large, ha, hb
        cdef object item

        ha = self.bulk_heap[A]
        hb = self.bulk_heap[B]
        assert(ha is not None)
        assert(hb is not None)
        if len(ha) < len(hb):
            small = ha
            large = hb
        else:
            small = hb
            large = ha

        for item in small: # todo: change to merge heaps
            heapq.heappush(large, item)
        self.bulk_heap[C] = large
        assert (A != C)
        assert (B != C)
        self.bulk_heap[A] = None
        self.bulk_heap[B] = None

    cdef void _absorb_heap_init(self, np.intp_t A, np.intp_t B, np.intp_t C) except *:
        cdef list small, large, ha, hb
        cdef object item

        ha = self.bulk_heap[A]
        hb = self.bulk_heap[B]
        if ha is None and hb is None:
            self.bulk_heap[C] = []
        elif ha is None:
            self.bulk_heap[C] = hb
        elif hb is None:
            self.bulk_heap[C] = ha
        else:
            if len(ha) < len(hb):
                small = ha
                large = hb
            else:
                small = hb
                large = ha
            for item in small: # todo: change to merge heaps
                heapq.heappush(large, item)
            self.bulk_heap[C] = large
        self.bulk_heap[A] = None
        self.bulk_heap[B] = None

    cdef bint _refresh(self, np.intp_t A, knn_indices, knn_dist) except *:
        cdef list heap
        cdef np.intp_t i, j, p, op
        cdef np.double_t v
        cdef object top
        cdef Relation rel

        heap = self.bulk_heap[A]
        assert(heap is not None)
        while heap:
            top = heap[0]
            v, i = top[0], top[1]
            assert(i>=0)
            j = self.opt_endpoints[i]
            if j < 0:
                heapq.heappop(heap)
                continue
            p = self.U.mark_up(i)
            op = self.U.mark_up(j)
            if p == op:
                heapq.heappop(heap)
                rel = Relation(0, 0, 0, 0, 0, 0)
                self._clear_optimum(i)
                if self._evaluate_reciprocity(i, p, knn_indices, knn_dist, &rel):
                    self._set_optimum(i, &rel)
                    heapq.heappush(heap, (rel.reciprocity, i))
                continue

            if self.opt_values[i] != v:
                heapq.heappop(heap)
                heapq.heappush(heap, (self.opt_values[i], i))
                continue
            self.out_i = i
            self.out_j = j
            return 1
        if heap is None:
            self.bulk_heap[A] = None
        return 0

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
        # computes DRUHG spanning tree from a FIFO of targeting bulks
        cdef:
            np.intp_t i, j, p, op, pp, A, B, C, D, G, N, \
                warn, infinitesimal, edge_cases
            np.double_t v

            Relation rel = Relation(0,0,0,0, 0,0)

            np.ndarray[np.double_t, ndim=2] knn_dist
            np.ndarray[np.intp_t, ndim=2] knn_indices

        N = self.num_points
        self.opt_values_arr = np.full(N, INF)
        self.opt_endpoints_arr = np.full(N, -1, dtype=np.intp)
        self.opt_rank_arr = np.zeros(N, dtype=np.intp)
        self.opt_values = self.opt_values_arr
        self.opt_endpoints = self.opt_endpoints_arr
        self.opt_rank = self.opt_rank_arr
        self.bulk_heap = [None for _ in range(2 * N)]
        self.bulk_of_arr = np.zeros(2 * N, dtype=np.intp)
        self.bulk_of = self.bulk_of_arr
        self.B = UnionFind(
            self.num_points,
            np.zeros(2 * N, dtype=np.intp),
            np.zeros(N, dtype=np.intp),
        )
        self.B.nullify()
        self.bulk_queue = deque()
        self.out_i = -1
        self.out_j = -1
        self.out_A = -1
        self.out_v = INF

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

        self.bulk_queue = deque()
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
                j = rel.endpoint
                self.result_write(rel.reciprocity, i, j, rel.max_rank - 1)
                p, op = self.U.mark_up(i), self.U.mark_up(j)
                pp = self.U.union(i, rel.endpoint, p, op)

                p, op = self.B.mark_up(i), self.B.mark_up(j)
                if p != op:
                    pp = self.B.union(i, j, p, op)
                    self._absorb_heap_init(p, op, pp)

                if rel.reciprocity == 0.: # values match
                    warn += 1
                    i += 1  # need to relaunch same index
                    continue
                if rel.max_rank > 2:
                    i += 1  # need to relaunch same index
                    continue

            if self._evaluate_reciprocity(i, self.U.mark_up(i), knn_indices, knn_dist, &rel):
                self._set_optimum(i, &rel)
                j = rel.endpoint
                p, op = self.B.mark_up(i), self.B.mark_up(j)
                pp = p
                if p != op:
                    pp = self.B.union(i, j, p, op)
                    self._absorb_heap_init(p, op, pp)
                    self.bulk_queue.append(pp)
                heapq.heappush(self.bulk_heap[pp], (rel.reciprocity, i))


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

        self.logger.info(f'MSTree formation: {self.result_edges:.0f} pure edges {100.*self.result_edges/self.num_points:.2f}%. Continue with branch connections.')
############
        self.logger.info(f'bulks {len(self.bulk_queue):.0f} edges {self.result_edges:.0f} evals {self.count_evaluations:.0f}/{self.count_inner_evaluations:.0f}')
        run = 0
        stuck = 0
        while self.result_edges < self.num_points - 1 and self.bulk_queue:
            self._should_stop_mst()

            if run == 0:
                if stuck:
                    break
                run = len(self.bulk_queue)
                stuck = 1
                self.logger.info(f'bulks {len(self.bulk_queue):.0f} edges {self.result_edges:.0f} evals {self.count_evaluations:.0f}/{self.count_inner_evaluations:.0f}')

            run -= 1

            A = self.bulk_queue.popleft()
            heap = self.bulk_heap[A]
            if heap is None:
                continue

            top = heap[0]
            v, i = top[0], top[1]
            j = self.opt_endpoints[i]

            self.result_write(v, i, j, self.opt_rank[i])
            self.U.union(i, j, self.U.mark_up(i), self.U.mark_up(j))
            stuck = 0

            if self._refresh(A, knn_indices, knn_dist):
                B = self.B.mark_up(self.out_j)
                if A == B:
                    self.bulk_queue.append(A)
                else:
                    C = self.B.union(self.out_i, self.out_j, A, B)
                    self._absorb_heap(A, B, C)
                    self.bulk_queue.append(C)

        self.logger.info(f'bulks {len(self.bulk_queue):.0f} edges {self.result_edges:.0f} evals {self.count_evaluations:.0f}/{self.count_inner_evaluations:.0f}')

        return edge_cases
