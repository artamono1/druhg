# -*- coding: utf-8 -*-
"""
DRUHG: Dialectical Reflection Universal Hierarchical Grouping
Clustering made by self-unrolling the relationships between the objects.
It is most natural clusterization and requires ZERO parameters.
"""

# Author: Pavel Artamonov
# druhg.p@gmail.com
# License: 3-clause BSD

import copy
import logging
import argparse
from enum import Enum

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from scipy.sparse import issparse
from sklearn.neighbors import KDTree, BallTree
from joblib.parallel import cpu_count

from ._druhg_tree import UniversalReciprocity
from ._druhg_label import Clusterizer
from ._druhg_displacement import develop
from .plots import ClusterTree
from .animation import Frames

from ._druhg_unionfind import allocate_unionfind_pair
from ._druhg_tree import allocate_buffer_values, allocate_buffer_edgepairs, allocate_buffer_ranks
from ._druhg_group import allocate_buffer_clusters, allocate_buffer_sizes
from ._druhg_label import allocate_buffer_labels

KDTREE_VALID_METRICS = [
    "euclidean", "l2", "minkowski", "p", "manhattan", "cityblock", "l1", "chebyshev", "infinity",
]
BALLTREE_VALID_METRICS = KDTREE_VALID_METRICS + [
    "braycurtis", "canberra", "dice", "hamming", "haversine", "jaccard",
    "mahalanobis", "rogerstanimoto", "russellrao", "seuclidean",
    "sokalmichener", "sokalsneath",
]
FAST_METRICS = KDTREE_VALID_METRICS + BALLTREE_VALID_METRICS + ["cosine", "arccos"]

Buffer = Enum('Buffer', [
    ('UNIONFIND', 10), ('UNIONFIND_FAST', 11),
    ('VALUES', 20), ('GROUPS', 21),
    ('LABELS', 30), ('CLUSTERS', 31), ('SIZES', 32),
    ('DATA0', 50), ('DATA1', 51),
    ('OUTPUT', 100), ('MST', 101), ('RANKS', 102),
])


def _allocate_if_needed(buffers, size, do_edges, do_labeling):
    if buffers is None:
        buffers = {}

    if do_edges and Buffer.MST.value not in buffers:
        buffers[Buffer.MST.value] = allocate_buffer_edgepairs(size)

    if Buffer.RANKS.value not in buffers:
        buffers[Buffer.RANKS.value] = allocate_buffer_ranks(size)

    if Buffer.VALUES.value not in buffers:
        buffers[Buffer.VALUES.value] = allocate_buffer_values(size)

    if Buffer.UNIONFIND.value not in buffers:
        buffers[Buffer.UNIONFIND.value], buffers[Buffer.UNIONFIND_FAST.value] = allocate_unionfind_pair(size)

    # TODO: precomputed won't work — allocate GROUPS with ndim from X.shape[1]
    # if Buffer.GROUPS.value not in buffers:
    #     buffers[Buffer.GROUPS.value] = allocate_buffer_groups(size, ndim)

    if Buffer.CLUSTERS.value not in buffers:
        buffers[Buffer.CLUSTERS.value] = allocate_buffer_clusters(size)

    if Buffer.SIZES.value not in buffers:
        buffers[Buffer.SIZES.value] = allocate_buffer_sizes(size)

    if do_labeling and Buffer.LABELS.value not in buffers:
        buffers[Buffer.LABELS.value] = allocate_buffer_labels(size)

    for bk in Buffer:
        if bk.value not in buffers:
            buffers[bk.value] = None

    return buffers


def _resolve_size_range(size, size_range, limitL, limitH):
    printout = ''

    if size_range is not None:
        limitL, limitH = size_range[0], size_range[1]

    if limitL is None:
        limitL = int(np.sqrt(size))
        printout += "Size_range's lower bound is set to " + str(limitL) + ', '
    else:
        if limitL < 0:
            raise ValueError('Size_range must be non-negative!')
        if limitL < 1:
            limitL = int(limitL * size)

    if limitH is None:
        limitH = int(size / 2 + 1)
        printout += "Size_range's higher bound is set to " + str(limitH) + ', '
    else:
        if limitH < 0:
            raise ValueError('Size_range must be non-negative!')
        if limitH <= 1:
            limitH = int(limitH * size + 1)

    return printout, limitL, limitH


def _check_input(X, core_n_jobs, max_ranking, leaf_size, metric, p,
                 size_range, limitL, limitH):
    printout = ''
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
            raise ValueError('Minkowski metric with negative p value is not defined!')

    if max_ranking is None:
        max_ranking = 16
        printout += 'max_ranking is set to ' + str(max_ranking) + ', '

    max_ranking = min(size - 1, max_ranking)

    extra, limitL, limitH = _resolve_size_range(size, size_range, limitL, limitH)
    printout += extra

    return printout, core_n_jobs, max_ranking, limitL, limitH


def _tune_treealgo(X, metric, algorithm, leaf_size, **kwargs):
    algo_code = 0
    tree = None

    if algorithm == 'best':
        algorithm = 'kd_tree'

    if algorithm == 'slow':  # todo: add XbyX matrix and forced precomputed
        algorithm = 'kd_tree'

    if "precomputed" in algorithm.lower() or "precomputed" in metric.lower() or issparse(X):
        algo_code = 2
        if issparse(X):
            algo_code = 3
        elif len(X.shape) == 2 and X.shape[0] != X.shape[1]:
            raise ValueError('Precomputed matrix is not a square.')
        tree = X
    else:
        if not X.flags['C_CONTIGUOUS']:
            raise ValueError('Array has to be C_CONTIGUOUS')

        if "kd" in algorithm.lower() and "tree" in algorithm.lower():
            algo_code = 0
            if metric not in KDTREE_VALID_METRICS:
                raise ValueError('Metric: %s\nCannot be used with KDTree' % metric)
            tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
        elif "ball" in algorithm.lower() and "tree" in algorithm.lower():
            algo_code = 1
            tree = BallTree(X, metric=metric, leaf_size=leaf_size, **kwargs)
        else:
            algo_code = 0
            if metric not in KDTREE_VALID_METRICS:
                raise ValueError('Metric: %s\nCannot be used with KDTree' % metric)
            tree = KDTree(X, metric=metric, leaf_size=leaf_size, **kwargs)

    return tree, algo_code


def _parsing_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
    )
    parser.add_argument(
        '-log', '--loglevel',
        help='Set the logging verbosity level.',
    )
    parser.add_argument(
        '-log_cli', '--log-cli-level',
        help='Set the logging verbosity level.',
    )

    args, unknown = parser.parse_known_args()
    logging.basicConfig()
    return args.loglevel


def druhg(X, max_ranking=16,
          do_labeling=True,
          size_range=None,
          limitL=None, limitH=None,
          exclude=None, fix_outliers=False,
          metric='minkowski', p=2,
          algorithm='best', leaf_size=40,
          core_n_jobs=None,
          buffers=None,
          do_edges=None, do_ranks=False,
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
    buffers : dict, enum (Buffer.__members__)
        Buffer's dictionary with every result including intermediate. Labels(30) for each point where outliers labeled -1.

    num_edges : int
        The amount of connected edges. The tree might be disconnected due to low `max_ranking` near-neighbors.

    References
    ----------

    None

    """
    logger = logging.getLogger(__package__)
    _loglevel = _parsing_setup()
    if _loglevel:
        logger.setLevel(_loglevel)
    if verbose is True:
        logger.setLevel(logging.INFO)
    elif str(verbose).lower() == 'debug':
        logger.setLevel(logging.DEBUG)

    printout, core_n_jobs, max_ranking, limitL, limitH = _check_input(
        X, core_n_jobs, max_ranking, leaf_size, metric, p, size_range, limitL, limitH)
    if printout:
        logger.info('Druhg is using defaults for: ' + printout)

    if type(X) is list:
        raise ValueError('X must be array! Not a list!')
    if not ("precomputed" in algorithm.lower() or "precomputed" in metric.lower() or issparse(X)):
        if not X.flags['C_CONTIGUOUS']:
            logger.info('Converting data array to c-contiguous')
            X = np.array(X, dtype=np.float64, order='C')
    if X.dtype != np.float64:
        logger.info('Converting data array to numpy float64')
        X = X.astype(np.float64)

    tree, algo_code = _tune_treealgo(X, metric, algorithm, leaf_size, **kwargs)

    if fix_outliers and do_edges is not False:
        do_edges = True

    size = X.shape[0]
    buffers = _allocate_if_needed(buffers, size, do_edges, do_labeling)

    ur = UniversalReciprocity(algo_code, tree,
                              buffers[Buffer.UNIONFIND.value], buffers[Buffer.UNIONFIND_FAST.value],
                              buffers[Buffer.VALUES.value],
                              max_neighbors_search=max_ranking, metric=metric,
                              leaf_size=leaf_size // 3, n_jobs=core_n_jobs,
                              buffer_ranks=buffers[Buffer.RANKS.value],
                              buffer_edgepairs=buffers[Buffer.MST.value],
                              **kwargs)

    num_edges = ur.get_num_edges()

    clusterizer = Clusterizer(buffers[Buffer.UNIONFIND.value], size, buffers[Buffer.VALUES.value], X,
                              buffers[Buffer.CLUSTERS.value], buffers[Buffer.SIZES.value], buffers[Buffer.GROUPS.value])
    precision = kwargs.get('double_precision2', kwargs.get('double_precision', 0))
    buffers[Buffer.CLUSTERS.value], buffers[Buffer.SIZES.value], buffers[Buffer.GROUPS.value] = clusterizer.emerge(
        precision=precision, run_motion=not do_labeling)

    if do_labeling:
        buffers[Buffer.LABELS.value] = clusterizer.label(
            buffers[Buffer.LABELS.value],
            exclude=exclude, size_range=[int(limitL), int(limitH)],
            fix_outliers=fix_outliers, edgepairs_arr=buffers[Buffer.MST.value], **kwargs)
        return buffers, num_edges

    buffers[Buffer.OUTPUT.value] = develop(
        buffers[Buffer.VALUES.value], buffers[Buffer.UNIONFIND.value], size,
        buffers[Buffer.GROUPS.value], X, buffers[Buffer.SIZES.value], buffers[Buffer.CLUSTERS.value],
        buffers[Buffer.OUTPUT.value], **kwargs)

    return buffers, num_edges


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

        self._size = 0
        self.num_edges_ = 0
        self._raw_data = None
        self.labels_ = None
        self.values_ = None
        self.mst_ = None
        self.ranks_ = None
        self.new_data_ = None
        self.buffers_ = None

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

        self.buffers_, self.num_edges_ = druhg(X, **kwargs)

        self.labels_ = self.buffers_[Buffer.LABELS.value]
        self.values_ = self.buffers_[Buffer.VALUES.value]
        self.ranks_ = self.buffers_[Buffer.RANKS.value]
        self.mst_ = self.buffers_[Buffer.MST.value]
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
        # converts to standard hierarchical tree format
        # https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
        # TODO: not done yet
        logging.getLogger(__package__).info('hierarchy() is not implemented yet')
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
        printout, limitL, limitH = _resolve_size_range(self._size, size_range, limitL, limitH)

        if fix_outliers is None:
            fix_outliers = 0
            printout += 'fix_outliers is set to ' + str(fix_outliers) + ', '

        precision = kwargs.get('double_precision2', kwargs.get('double_precision', None))

        if printout:
            logging.getLogger(__package__).info('Relabeling using defaults for: ' + printout)

        clusterizer = Clusterizer(
            self.buffers_[Buffer.UNIONFIND.value], self._size,
            self.buffers_[Buffer.VALUES.value], self._raw_data,
            self.buffers_[Buffer.CLUSTERS.value], self.buffers_[Buffer.SIZES.value],
            self.buffers_[Buffer.GROUPS.value])

        self.labels_ = clusterizer.label(
            self.labels_,
            exclude=exclude, size_range=[int(limitL), int(limitH)],
            fix_outliers=fix_outliers, edgepairs_arr=self.buffers_[Buffer.MST.value],
            precision=precision, **kwargs)

        return self.labels_

    def buffer_develop(self, **kwargs):
        params = self.get_params()
        params.update(self._metric_kwargs)
        params.update(kwargs)

        self.buffers_, self.num_edges_ = druhg(
            self.buffers_[Buffer.DATA0.value],
            do_labeling=False,
            buffers=self.buffers_,
            **params)

        self.new_data_ = self.buffers_[Buffer.OUTPUT.value]
        self.buffers_[Buffer.DATA1.value] = self.buffers_[Buffer.DATA0.value]
        self.buffers_[Buffer.DATA0.value] = self.new_data_
        self.buffers_[Buffer.OUTPUT.value] = self.buffers_[Buffer.DATA1.value]
        return self

    def develop(self, XX, **kwargs):
        params = self.get_params()
        params.update(self._metric_kwargs)
        params.update(kwargs)

        self.buffers_, self.num_edges_ = druhg(XX, do_labeling=False, **params)
        self.new_data_ = self.buffers_[Buffer.OUTPUT.value]
        return self

    def allocate_buffers(self, XX):
        self._size = XX.shape[0]
        self._raw_data = XX
        self.buffers_ = _allocate_if_needed(self.buffers_, self._size, False, False)

        if self.buffers_[Buffer.DATA0.value] is None:
            self.buffers_[Buffer.DATA0.value] = copy.deepcopy(XX)
        if self.buffers_[Buffer.DATA1.value] is None:
            self.buffers_[Buffer.DATA1.value] = copy.deepcopy(XX)
        self.new_data_ = self.buffers_[Buffer.DATA0.value]
        return self

    @property
    def frames_(self):
        return Frames(self)

    @property
    def single_linkage_(self):
        if self.mst_ is None:
            raise AttributeError('No minimum spanning tree was generated. Need ``do_edges=True``.')
        if self._raw_data is None:
            logging.getLogger(__package__).warning('No raw data is available.')
            return None
        return ClusterTree(
            self.buffers_[Buffer.UNIONFIND.value],
            self._raw_data,
            self.values_,
            self.buffers_[Buffer.SIZES.value],
            self.buffers_[Buffer.CLUSTERS.value],
            self.mst_,
            self.num_edges_,
        )

    def plot(self, static_labels=None, axis=None, **kwargs):
        if self._raw_data is None:
            logging.getLogger(__package__).warning('No raw data is available.')
            return None

        return ClusterTree(
            self.buffers_[Buffer.UNIONFIND.value],
            self._raw_data,
            self.values_,
            self.buffers_[Buffer.SIZES.value],
            self.buffers_[Buffer.CLUSTERS.value],
            self.mst_,
            self.num_edges_,
        ).plot(static_labels=static_labels, axis=axis, **kwargs)
