# -*- coding: utf-8 -*-
"""
DRUHG: Dialectical Reflection Universal Hierarchical Grouping
Clustering made by self-unrolling the relationships between the objects.
It is most natural clusterization and requires ZERO parameters.
"""

# Author: Pavel Artamonov
# druhg.p@gmail.com
# License: 3-clause BSD

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from scipy.sparse import issparse
from sklearn.neighbors import KDTree, BallTree
import copy
# from sklearn.externals.joblib import Memory
# from sklearn.externals import six
from warnings import warn
# from sklearn.utils import check_array
from joblib.parallel import cpu_count

from ._druhg_tree import UniversalReciprocity

from ._druhg_label import Clusterizer

from .plots import ClusterTree
import time

# memory allocations
from ._druhg_unionfind import allocate_unionfind_pair
from ._druhg_tree import allocate_buffer_values, allocate_buffer_edgepairs, allocate_buffer_ranks
from ._druhg_group import allocate_buffer_groups, allocate_buffer_clusters, allocate_buffer_sizes
from ._druhg_label import allocate_buffer_labels # can be used with relabels

KDTREE_VALID_METRICS = ["euclidean", "l2", "minkowski", "p", "manhattan", "cityblock", "l1", "chebyshev", "infinity"]
BALLTREE_VALID_METRICS = KDTREE_VALID_METRICS + ["braycurtis", "canberra", "dice", "hamming", "haversine", "jaccard",
                                                 "mahalanobis", "rogerstanimoto", "russellrao", "seuclidean",
                                                 "sokalmichener", "sokalsneath",]
FAST_METRICS = KDTREE_VALID_METRICS + BALLTREE_VALID_METRICS + ["cosine", "arccos"]



def druhg(X, max_ranking=16,
          do_labeling=True,
          size_range=None,
          limitL=None, limitH=None,
          exclude=None, fix_outliers=False,
          metric='minkowski', p=2,
          algorithm='best', leaf_size=40,
          core_n_jobs=None,

          buffer_values=None, buffer_ranks=None, buffer_uf=None, buffer_uf_fast=None, buffer_out=None, buffer_groups=None,
          do_edges=None, buffer_mst=None,
          do_ranks=False,
          buffer_labels=None,
          buffer_clusters=None,
          buffer_sizes=None,
          verbose=False, **kwargs):
    """Perform DRUHG clustering from a vector array or distance matrix.

    Parameters
    ----------
    X : array matrix of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.

    max_ranking : int, optional (default=15)
        The maximum number of neighbors to search.
        Affects performance vs precision.

    do_labeling : bool (default=True)
        It returns labels, otherwise new data point.

    size_range : [float, float], optional (default=[sqrt(size), size/2])
        Clusters that are smaller or bigger than this limit treated as noise.
        Use [1,1] to find True outliers.
        Numbers under 1 treated as percentage of the dataset size

    exclude: list, optional (default=None)
        Clusters with these indexes would not be formed.
        Use it for surgical cluster removal.

    fix_outliers: bool, optional (default=False)
        In case of True - forces `do_edges` and all outliers will be assigned to the nearest cluster

    do_edges: bool, optional (default=None)
        In case of True - extracts edge pairs

    metric : string or callable, optional (default='minkowski')
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.

    p : int, optional (default=2)
        p value to use if using the minkowski metric.

    leaf_size : int, optional (default=40)
        Leaf size for trees responsible for fast nearest
        neighbour queries.

    algorithm : string, optional (default='best')
        Exactly, which algorithm to use; DRUHG has variants specialized
        for different characteristics of the data. By default, this is set
        to ``best`` which chooses the "best" algorithm given the nature of
        the data. You can force other options if you believe you know
        better. Options are:
            * ``best``
            * ``kdtree``
            * ``balltree``
        If you want it to be accurate add:
            * ``slow`` (todo)

    core_n_jobs : int, optional (default=None)
        Number of parallel jobs to run in neighbors distance computations (if
        supported by the specific algorithm).
        For default, (n_cpus + 1 + core_dist_n_jobs) is used.

    **kwargs : optional
        Arguments passed to the distance metric

    Returns
    -------
    labels : ndarray, shape (n_samples)
        Cluster labels for each point. Noisy samples are given the label -1.

    min_spanning_tree : ndarray, shape (2*n_samples - 2)
        The minimum spanning tree as edgepairs.

    values_edges : ndarray, shape (n_samples - 1)
        Values of the edges.


    References
    ----------

    None

    """
    if type(X) is list:
        raise ValueError('X must be array! Not a list!')

    size = X.shape[0]

    if core_n_jobs is None:
        core_n_jobs = max(cpu_count(), 1)
    elif core_n_jobs < 0:
        core_n_jobs = max(cpu_count() + 1 + core_n_jobs, 1)

    if max_ranking is not None:
        if type(max_ranking) is not int:
            raise ValueError('Max ranking must be integer!')
        if max_ranking < 0:
            raise ValueError('Max ranking must be non-negative integer!')

    if leaf_size < 1:
        raise ValueError('Leaf size must be greater than 0!')

    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not'
                             ' defined!')
    printout = ''
    if max_ranking is None:
        max_ranking = 16
        printout += 'max_ranking is set to '+str(max_ranking)+', '

    max_ranking = min(size - 1, max_ranking)

    if size_range is not None:
        limitL, limitH = size_range[0], size_range[1]

    if limitL is None:
        limitL = int(np.sqrt(size))
        printout += 'Size_range`s lower bound is set to '+str(limitL)+', '
    else:
        if limitL < 0:
            raise ValueError('Size_range must be non-negative!')
        if limitL < 1:
            limitL = int(limitL*size)

    if limitH is None:
        limitH = int(size/2 + 1)
        printout += 'Size_range`s higher bound is set to '+str(limitH)+', '
    else:
        if limitH < 0:
            raise ValueError('Size_range must be non-negative!')
        if limitH <= 1:
            limitH = int(limitH*size + 1)

    if algorithm == 'best':
        algorithm = 'kd_tree'

    if algorithm == 'slow': # todo: add XbyX matrix and forced precomputed
        algorithm = 'kd_tree'

    if X.dtype != np.float64:
        print ('Converting data to numpy float64')
        X = X.astype(np.float64)

    algo_code = 0
    if "precomputed" in algorithm.lower() or "precomputed" in metric.lower() or issparse(X):
        algo_code = 2
        if issparse(X):
            algo_code = 3
        elif len(X.shape)==2 and X.shape[0] != X.shape[1]:
            raise ValueError('Precomputed matrix is not a square.')
        tree = X
    else:
        # The Cython routines used require contiguous arrays
        if not X.flags['C_CONTIGUOUS']:
            X = np.array(X, dtype=np.double, order='C')

        if "kd" in algorithm.lower() and "tree" in algorithm.lower():
            algo_code = 0
            if metric not in KDTREE_VALID_METRICS: #KDTree.valid_metrics:
                raise ValueError('Metric: %s\n'
                                 'Cannot be used with KDTree' % metric)
            tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
        elif "ball" in algorithm.lower() and "tree" in algorithm.lower():
            algo_code = 1
            tree = BallTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
        else:
            algo_code = 0
            if metric not in KDTREE_VALID_METRICS:
                raise ValueError('Metric: %s\n'
                                 'Cannot be used with KDTree' % metric)
            tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
            # raise TypeError('Unknown algorithm type %s specified' % algorithm)

    if printout:
        print('Druhg is using defaults for: ' + printout)

    if fix_outliers and do_edges is not False:
        do_edges = True

    if do_edges and buffer_mst is None:
        buffer_mst = allocate_buffer_edgepairs(size)
    else:
        buffer_mst = None

    if buffer_ranks is None: # and do_ranks
        buffer_ranks = allocate_buffer_ranks(size)

    if buffer_values is None:
        buffer_values = allocate_buffer_values(size)

    if buffer_uf is None:
        buffer_uf, buffer_uf_fast = allocate_unionfind_pair(size)
    if buffer_groups is None:
        ndim = 0
        if not do_labeling:
            ndim = X.shape[1] # TODO: precomputed won't work
        buffer_groups = allocate_buffer_groups(size, ndim)
    if buffer_clusters is None:
        buffer_clusters = allocate_buffer_clusters(size)
    if buffer_sizes is None:
        buffer_sizes = allocate_buffer_sizes(size)

    if do_labeling and buffer_labels is None:
        buffer_labels = allocate_buffer_labels(size)

    ur = UniversalReciprocity(algo_code, tree,
                              buffer_uf, buffer_uf_fast,
                              buffer_values,
                              max_neighbors_search=max_ranking, metric=metric,
                              leaf_size=leaf_size // 3, n_jobs=core_n_jobs,
                              buffer_ranks=buffer_ranks,
                              buffer_edgepairs=buffer_mst,
                              **kwargs)
    buffer_values, buffer_uf = ur.get_buffers() # no need in getting it
    if do_labeling:
        clusterizer = Clusterizer(buffer_uf, size, buffer_values, X, buffer_clusters, buffer_sizes, buffer_groups)
        buffer_clusters, buffer_sizes = clusterizer.emerge()
        buffer_labels = clusterizer.label(buffer_labels,
                       exclude=exclude, size_range=[int(limitL), int(limitH)],
                       fix_outliers=fix_outliers, edgepairs_arr=buffer_mst, **kwargs)
        return (buffer_labels,
                buffer_values, buffer_ranks, buffer_uf,
                buffer_groups, buffer_mst, buffer_sizes, buffer_clusters,
                )
    out_data = develop(buffer_values, buffer_ranks, buffer_uf, size, buffer_groups, X, buffer_out,  **kwargs)

    return (out_data,
            buffer_values, buffer_ranks, buffer_uf,
            buffer_groups, buffer_mst, buffer_sizes, buffer_clusters,
            )

class DRUHG(BaseEstimator, ClusterMixin):
    def __init__(self, metric='euclidean',
                 algorithm='best',
                 max_ranking=24,
                 limitL=None,
                 limitH=None,
                 exclude=None,
                 fix_outliers=0,
                 leaf_size=40,
                 verbose=False,
                 core_n_jobs=None,
                 **kwargs):
        self.max_ranking = max_ranking
        self.limitL = limitL
        self.limitH = limitH
        self.exclude = exclude
        self.fix_outliers = fix_outliers
        self.metric = metric
        self.algorithm = algorithm
        self.verbose = verbose
        self.leaf_size = leaf_size
        self.core_n_jobs = core_n_jobs
        self._metric_kwargs = kwargs

        # self._outlier_scores = None
        # self._prediction_data = None
        self._size = 0
        self._raw_data = None
        self.uf_ = None
        self.labels_ = None
        self.sizes_ = None
        self.clusters_ = None # negative if parent didnt cluster, else clusterized #clusters
        self.values_ = None
        self.groups_ = None
        self.mst_ = None
        self.ranks_ = None

        self.new_data_ = None

        # for reusal in all three parts
        self._buffer_data1 = None
        self._buffer_data2 = None
        self._buffer_uf = None
        self._buffer_clusters = None
        self._buffer_sizes = None
        self._buffer_uf_fast = None
        self._buffer_values = None
        self._buffer_ranks = None
        self._buffer_mst = None
        self._buffer_groups = None

    def fit(self, X, y=None):
        """Perform DRUHG clustering.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.

        Returns
        -------
        self : object
            Returns self
        """
        kwargs = self.get_params()
        kwargs.update(self._metric_kwargs)

        self._size = X.shape[0]
        self._raw_data = X

        (self.labels_,
         self.values_,
         self.ranks_,
         self.uf_,
         self.groups_,
         self.mst_,
         self.sizes_,
         self.clusters_
         ) = druhg(X, **kwargs)

        return self

    def fit_predict(self, X, y=None):
        """Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.

        Returns
        -------
        y : ndarray, shape (n_samples, )
            cluster labels
        """
        self.fit(X)
        return self.labels_

    def hierarchy(self):
        # converts to standard hierarchical tree format + errors
        # https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

        print ('todo: not done yet')
        return None

    def relabel(self, exclude=None, size_range=None, limitL=None, limitH=None, fix_outliers=None, **kwargs):
        """Relabeling with the limits on cluster size.

        Parameters
        ----------

        exclude : list of cluster-indexes, for surgical removal of certain clusters,
            could be omitted.

        size_range : [float, float], optional (default=[sqrt(size), size/2])
            Clusters that are smaller or bigger than this limit treated as noise.
            Use [1,1] to find True outliers.
            Numbers under 1 treated as percentage of the dataset size

        fix_outliers : glues outliers to the nearest clusters

        Returns
        -------
        y : ndarray, shape (n_samples, )
            cluster labels,
            -1 are outliers
        """
        printout = ''
        size = self._size

        if size_range is not None:
            limitL, limitH = size_range[0], size_range[1]

        if limitL is None:
            limitL = int(np.sqrt(size))
            printout += 'Size_range`s lower bound is set to ' + str(limitL) + ', '
        else:
            if limitL < 0:
                raise ValueError('Size_range must be non-negative!')
            if limitL < 1:
                limitL = int(limitL * size)

        if limitH is None:
            limitH = int(size / 2 + 1)
            printout += 'Size_range`s higher bound is set to ' + str(limitH) + ', '
        else:
            if limitH < 0:
                raise ValueError('Size_range must be non-negative!')
            if limitH <= 1:
                limitH = int(limitH * size + 1)

        if fix_outliers is None:
            fix_outliers = 0
            printout += 'fix_outliers is set to ' + str(fix_outliers) + ', '

        # this is only relevant if distances between datapoints are super small
        precision = kwargs.get('double_precision2', kwargs.get('double_precision', None))

        if printout:
            print('Relabeling using defaults for: ' + printout)

        clusterizer = Clusterizer(self.uf_, self._size, self.values_, self._raw_data,
                                  self.clusters_, self.sizes_, self.groups_)

        self.labels_ = clusterizer.label(self.labels_,
                       exclude=exclude, size_range=[int(limitL), int(limitH)],
                       fix_outliers=fix_outliers, edgepairs_arr=self._buffer_mst,
                       precision=precision, **kwargs)

        return self.labels_

    def buffer_develop(self, **kwargs):

        kwargs = self.get_params()
        kwargs.update(self._metric_kwargs)

        (self.new_data_,
         self.values_, self.ranks_, self._buffer_uf,
         self._buffer_uf, _, self.mst_, self.sizes_, self.clusters_) = druhg(self._buffer_data1, do_labeling=False,
                                  buffer_values=self._buffer_values, buffer_ranks=self._buffer_ranks,
                                  buffer_uf=self._buffer_uf, buffer_uf_fast=self._buffer_uf_fast,
                                  buffer_out=self._buffer_data2,
                                  buffer_groups=self._buffer_groups,
                                  buffer_sizes=self._buffer_sizes,
                                  buffer_clusters=self._buffer_clusters,
                                  **kwargs)

        self._buffer_data2 = self.new_data_
        self._buffer_data1, self._buffer_data2 = self._buffer_data2, self._buffer_data1
        return self

    def develop(self, XX, **kwargs):
        kwargs = self.get_params()
        kwargs.update(self._metric_kwargs)

        (self.new_data_,
         self.values_, self.ranks_,self._buffer_uf,
         _, self.mst_, self.sizes_, self.clusters_) = druhg(XX, do_labeling=False, **kwargs)

        return self

    def allocate_buffers(self, XX):
        self._size = XX.shape[0]
        self._raw_data = XX
        # TODO: reuse if buffers are present
        if self._buffer_data1 is None:
             self._buffer_data1 = copy.deepcopy(self._raw_data)
        if self._buffer_data2 is None:
             self._buffer_data2 = np.copy(self._buffer_data1)

        if self._buffer_uf is None:
            self._buffer_uf, self._buffer_uf_fast = allocate_unionfind_pair(self._size)

        if self._buffer_values is None:
            self._buffer_values = allocate_buffer_values(self._size)
        if self._buffer_ranks is None:
            self._buffer_ranks = allocate_buffer_ranks(self._size)

        if self._buffer_groups is None:
            self._buffer_groups = allocate_buffer_groups(self._size, self._raw_data.ndim)
        if self._buffer_clusters is None:
            self._buffer_clusters = allocate_buffer_clusters(self._size)

        if self._buffer_sizes is None:
            self._buffer_sizes = allocate_buffer_sizes(self._size)

        return self

    @property
    def single_linkage_(self):
        if self.mst_ is not None:
            if self._raw_data is not None:
                return ClusterTree(self.uf_,
                               self._raw_data,
                               self.values_,
                               self.sizes_,
                               self.clusters_,
                               self.mst_)
            else:
                warn('No raw data is available.')
                return None
        else:
            raise AttributeError('No minimum spanning tree was generated. Need ``do_edges=True``.')

    def plot(self, static_labels=None, axis=None, **kwargs):
        if self._raw_data is not None:
            return ClusterTree(self.uf_,
                               self._raw_data,
                               self.values_,
                               self.sizes_,
                               self.clusters_,
                               self.mst_).plot(static_labels=static_labels, axis=axis, **kwargs)
        else:
            warn('No raw data is available.')
            return None
