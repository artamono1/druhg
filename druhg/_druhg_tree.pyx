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

from ._druhg_unionfind import UnionFind
from ._druhg_unionfind cimport UnionFind

import _heapq as heapq
# from ._cyheapq import merge as heapq_merge

from libc.math cimport fabs, pow
import bisect

cdef np.double_t INF = sys.float_info.max

from sklearn.neighbors import KDTree, BallTree
# from sklearn import preprocessing
from joblib import Parallel, delayed

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

        knn_dist = INF*np.ones((self.data_size, k+1))
        knn_indices = np.zeros((self.data_size, k+1), dtype=np.intp)

        warning = 0

        i = self.data_size
        while i:
            i -= 1
            row = self.data_arr.getrow(i)
            idx, data = row.indices, row.data
            sorted = np.argsort(data)
            j = min(k,len(idx))
            if idx[sorted[0]] != i:
                while j:
                    j -= 1
                    knn_dist[i][j+1] = data[sorted[j]]
                    knn_indices[i][j+1] = idx[sorted[j]]
            else:
                # edge loops itself
                warning += 1
                while j:
                    j -= 1
                    knn_dist[i][j] = data[sorted[j]]
                    knn_indices[i][j] = idx[sorted[j]]

            knn_dist[i][0], knn_indices[i][0] = 0., i # have to add itself. Edge to itself have to be zero!

        if warning:
            print ('Attention!: Sparse matrix has an edge that forms a loop! They were zeroed.', warning)

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
            j = k
            while j:
                j -= 1
                knn_dist[i][j] = row[sorted[j]]
                knn_indices[i][j] = sorted[j]

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
        sklearn K-NearestNeighbor uses it.
        Used only with KDTree/BallTree option.

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

        np.intp_t result_edges
        np.ndarray result_values_arr
        np.ndarray result_pairs_arr
        np.ndarray result_rank_arr

        public np.ndarray knn_d
        public np.ndarray knn_i

    def __init__(self, algorithm, tree,
                 buffer_uf, buffer_fast, buffer_values,
                 max_neighbors_search=16, metric='euclidean', leaf_size=20, n_jobs=4,
                 buffer_ranks=None, buffer_edgepairs=None,
                 buffer_clusters=None,
                 **kwargs):

        self.PRECISION = kwargs.get('double_precision', 0.0000001) # this is only relevant if distances between datapoints are super small
        self.n_jobs = n_jobs

        if algorithm == 0:
            self.dist_tree = tree
            self.tree = KDTree(tree.data, metric=metric, leaf_size=leaf_size, **kwargs)
            self.num_points = self.tree.data.shape[0]
        elif algorithm == 1:
            self.dist_tree = tree
            self.tree = BallTree(tree.data, metric=metric, leaf_size=leaf_size, **kwargs)
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

        # self.num_features = self.tree.data.shape[1]

        self.U = UnionFind(self.num_points, buffer_uf, buffer_fast)
        self.U.nullify()

        self.result_edges = 0

        self.result_values_arr = buffer_values
        if len(self.result_values_arr) < self.num_points - 1:
            print('ERROR: values buffer is too small', len(self.result_values_arr), self.num_points - 1)
            return

        self.result_pairs_arr = buffer_edgepairs # np.empty((self.num_points*2 - 2))
        if self.result_pairs_arr is not None and len(self.result_pairs_arr) < self.num_points*2 - 2:
            print('ERROR: edgepairs buffer is too small', len(self.result_pairs_arr), self.num_points*2 - 2)
            return

        self.result_rank_arr = buffer_ranks # np.empty((self.num_points - 1))
        if self.result_rank_arr is not None and len(self.result_rank_arr) < self.num_points - 1:
            print('ERROR: ranks buffer is too small', len(self.result_rank_arr), self.num_points - 1)
            return

        self._compute_tree_edges()

    cpdef tuple get_tree(self):
        return self.result_values_arr[:self.result_edges * 2], self.result_pairs_arr[:self.result_edges*2].astype(int),\
               self.result_rank_arr[:self.result_edges*2].astype(int)

    cpdef tuple get_buffers(self):
        return self.result_values_arr, self.U.parent_arr

    cdef void result_write(self, np.double_t v, np.intp_t a, np.intp_t b, np.double_t r):
        cdef np.intp_t i

        i = self.result_edges
        self.result_edges += 1
        self.result_values_arr[i] = v
        # self.result_values_arr[i] = pow(v, 2.0)

        if self.result_pairs_arr is not None:
            self.result_pairs_arr[2 * i] = a
            self.result_pairs_arr[2 * i + 1] = b
        if self.result_rank_arr is not None:
            self.result_rank_arr[i] = r
        # print ('result_write', a,b, v, r)


    cdef bint _pure_reciprocity(self, np.intp_t i, np.ndarray[np.intp_t, ndim=2] knn_indices, np.ndarray[np.double_t, ndim=2] knn_dist,
                                       Relation* rel, np.intp_t* infinitesimal):
        cdef:
            np.intp_t ranki, j, \
                parent, \
                rank

            np.double_t dis, core_dis

            np.ndarray indices, ind_opp
            np.ndarray distances, dis_opp

        parent = self.U.mark_up(i)
        indices, distances = knn_indices[i], knn_dist[i]

        rel.reciprocity = INF
        core_dis = distances[1]
        for ranki in range(0, self.max_neighbors_search + 1):
            j = indices[ranki]
            if parent == self.U.mark_up(j):
                continue

            dis = distances[ranki]
            if dis > core_dis + self.PRECISION:
                break

            if dis == 0.: # degenerate case.
                rel.reciprocity = 0.
                rel.endpoint = j
                rel.max_rank = bisect.bisect(distances, 0. + self.PRECISION)
                return 1
            infinitesimal += dis <= self.PRECISION

            if knn_dist[j][1] + self.PRECISION < dis:
                return 0

            # только для 2-2
            # if bisect.bisect(distances, dis + self.PRECISION) > 2 \
            # or bisect.bisect(knn_dist[j], dis + self.PRECISION) > 2:
            #     return 0
            rank = bisect.bisect(distances, dis + self.PRECISION)

            # print ('core', core_dis, 'dis', dis)
            # print('i', i, 'rank_i', rank,
            #       'j', j, 'rank_j', bisect.bisect(knn_dist[j], dis + self.PRECISION))

            if rank != bisect.bisect(knn_dist[j], dis + self.PRECISION):
                continue
            # print ('pure')

            rel.reciprocity = dis
            rel.endpoint = j
            rel.max_rank = rank
            return 1
        return 0

    cdef bint _evaluate_reciprocity(self, np.intp_t i, np.intp_t parent, np.ndarray[np.intp_t, ndim=2] knn_indices, np.ndarray[np.double_t, ndim=2] knn_dist, Relation* rel):
        cdef:
            int ranki, rank, orank
            np.intp_t j, \
                res = 0

            np.double_t best, v, \
                dis, odis

            np.intp_t[:] indices
            np.double_t[:] distances

        indices: np.intp_t[:] = knn_indices[i]
        distances: np.double_t[:] = knn_dist[i]

        best = INF
        for ranki in range(1, self.max_neighbors_search + 1):
            dis = distances[ranki]
            if dis - self.PRECISION > best:
                break

            j = indices[ranki]
            if self.U.is_same_parent(parent, j):
                continue
            rank = bisect.bisect(distances, dis + self.PRECISION)
            orank = bisect.bisect(knn_dist[j], dis + self.PRECISION)  # !reminder! bisect.bisect(odis, dis) >= bisect.bisect_left(odis, dis)
            if rank > orank:
                continue

            odis = distances[orank-1]
            v = min(odis, dis * orank / (1.*rank)) # evaluates from POV of the i and the opp

            if v >= best:
                continue

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

            Relation rel = Relation(0,0,0,0)

            np.ndarray[np.double_t, ndim=2] knn_dist
            np.ndarray[np.intp_t, ndim=2] knn_indices

            list heap

        if self.tree.data.shape[0] > 16384 and self.n_jobs > 1: # multicore 2-3x speed up for big datasets
        # if self.n_jobs > 1:
            split_cnt = self.num_points // self.n_jobs
            datasets = []
            for i in range(self.n_jobs):
                if i == self.n_jobs - 1:
                    datasets.append(np.asarray(self.tree.data[i*split_cnt:]))
                else:
                    datasets.append(np.asarray(self.tree.data[i*split_cnt:(i+1)*split_cnt]))

            knn_data = Parallel(n_jobs=self.n_jobs)(
                delayed(self.tree.query)
                (points,
                 self.max_neighbors_search + 1,
                 dualtree=True,
                 breadth_first=True
                 )
                for points in datasets)
            knn_dist = np.vstack([x[0] for x in knn_data])
            knn_indices = np.vstack([x[1] for x in knn_data])
        else:
            knn_dist, knn_indices = self.dist_tree.query(
                        self.tree.data,
                        k=self.max_neighbors_search + 1,
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
                print ('Distances cannot be negative! Exiting. ', i, knn_dist[i][0])
                return
            if self._pure_reciprocity(i, knn_indices, knn_dist, &rel, &infinitesimal):
                self.result_write(rel.reciprocity, i, rel.endpoint, rel.max_rank)
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
            print ('Two subjects only')
            return
        if warn > 0:
            print (
            'A lot of values(', warn, ') are the same. Try increasing max_neighbors_search(', self.max_neighbors_search,
            ') parameter.')

        if infinitesimal > 0:
            print ('Some distances(', infinitesimal, ') are smaller than self.PRECISION (', self.PRECISION,
                   ') level. Try decreasing double_precision parameter.')

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
            print (str(
                self.num_points - 1 - self.result_edges) + ' not connected edges of', self.num_points - 1,'. It is a forest. Try increasing max_neighbors(max_ranking) value ' + str(
                self.max_neighbors_search) + ' for a better result.')
        if self.max_neighbors_search < self.num_points - 1 and edge_cases != 0:
            # todo: may be check the actual reachability of indices?
            print (str(edge_cases) + ' edges with the max rank. Try increasing max_neighbors(max_ranking) value '+ str(
                self.max_neighbors_search) + 'or pick the square mode (not available).')
