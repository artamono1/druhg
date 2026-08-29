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
from joblib.parallel import cpu_count

from ._druhg_neighbors_k_extend import KDTree, BallTree, KDTREE_VALID_METRICS, BALLTREE_VALID_METRICS, _as_sample_matrix

from ._druhg_tree import UniversalReciprocity
from ._druhg_label import Clusterizer
from ._druhg_displacement import develop
from .plots import ClusterTree
from .animation import Frames

from ._druhg_unionfind import allocate_unionfind_pair
from ._druhg_tree import allocate_buffer_values, allocate_buffer_edgepairs, allocate_buffer_ranks
from ._druhg_group import allocate_buffer_clusters, allocate_buffer_sizes
from ._druhg_label import allocate_buffer_labels

FAST_METRICS = list(dict.fromkeys(KDTREE_VALID_METRICS + BALLTREE_VALID_METRICS))

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


def _check_input(X, core_n_jobs, max_ranking, step_expansion, leaf_size, metric, p,
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
        if max_ranking < 1:
            raise ValueError('Max ranking must be a positive integer!')

    if step_expansion is None:
        step_expansion = 16
        printout += 'step_expansion is set to ' + str(step_expansion) + ', '
    elif type(step_expansion) is not int:
        raise ValueError('step_expansion must be integer!')
    elif step_expansion < 1:
        raise ValueError('step_expansion must be a positive integer!')

    if leaf_size < 1:
        raise ValueError('Leaf size must be greater than 0!')

    if metric == 'minkowski':
        if p is None:
            raise TypeError('Minkowski metric given but no p value supplied!')
        if p < 0:
            raise ValueError('Minkowski metric with negative p value is not defined!')

    if max_ranking is None:
        max_ranking = size - 1
    else:
        max_ranking = min(size - 1, max_ranking)

    step_expansion = min(size - 1, max_ranking, step_expansion)

    extra, limitL, limitH = _resolve_size_range(size, size_range, limitL, limitH)
    printout += extra

    return printout, core_n_jobs, max_ranking, step_expansion, limitL, limitH


def _coerce_feature_array(X, algorithm, metric):
    """Interpret a 1-d vector as n samples with one feature."""
    if type(X) is list:
        raise ValueError('X must be array! Not a list!')
    if "precomputed" in str(algorithm).lower() or "precomputed" in str(metric).lower() or issparse(X):
        return X
    return _as_sample_matrix(X)


def _tree_constructor_kwargs(metric, p, kwargs):
    tree_kwargs = {}
    metric_l = metric.lower()
    if metric_l in ('minkowski', 'p') and p is not None:
        tree_kwargs['p'] = p
    for key in ('p', 'w', 'V', 'VI'):
        if key in kwargs:
            tree_kwargs[key] = kwargs[key]
    return tree_kwargs


def _tune_treealgo(X, metric, algorithm, leaf_size, p=2, **kwargs):
    algo_code = 0
    tree = None
    algorithm_l = algorithm.lower()
    metric_l = metric.lower()
    tree_kwargs = _tree_constructor_kwargs(metric, p, kwargs)

    if algorithm_l == 'best':
        if metric_l in KDTREE_VALID_METRICS:
            algorithm_l = 'kd_tree'
        elif metric_l in BALLTREE_VALID_METRICS:
            algorithm_l = 'ball_tree'
        else:
            algorithm_l = 'kd_tree'

    if algorithm_l == 'slow':  # todo: add XbyX matrix and forced precomputed
        algorithm_l = 'kd_tree'

    if "precomputed" in algorithm_l or "precomputed" in metric_l or issparse(X):
        algo_code = 2
        if issparse(X):
            algo_code = 3
        elif len(X.shape) == 2 and X.shape[0] != X.shape[1]:
            raise ValueError('Precomputed matrix is not a square.')
        tree = X
    else:
        if not X.flags['C_CONTIGUOUS']:
            raise ValueError('Array has to be C_CONTIGUOUS')

        if "kd" in algorithm_l and "tree" in algorithm_l:
            algo_code = 0
            if metric_l not in KDTREE_VALID_METRICS:
                raise ValueError('Metric: %s\nCannot be used with KDTree' % metric)
            tree = KDTree(X, metric=metric_l, leaf_size=leaf_size, **tree_kwargs)
        elif "ball" in algorithm_l and "tree" in algorithm_l:
            algo_code = 1
            if metric_l not in BALLTREE_VALID_METRICS:
                raise ValueError('Metric: %s\nCannot be used with BallTree' % metric)
            tree = BallTree(X, metric=metric_l, leaf_size=leaf_size, **tree_kwargs)
        else:
            algo_code = 0
            if metric_l not in KDTREE_VALID_METRICS:
                raise ValueError('Metric: %s\nCannot be used with KDTree' % metric)
            tree = KDTree(X, metric=metric_l, leaf_size=leaf_size, **tree_kwargs)

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


def druhg(X, max_ranking=None, step_expansion=16,
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
    X : array matrix of shape (n_samples, n_features), \
            (n_samples,), or (n_samples, n_samples)
        A feature array (a 1-d vector is n samples with one feature), or
        array of distances between samples if ``metric='precomputed'``.

    max_ranking : int, optional (default=None)
        Hard cap on how many nearest neighbors are stored per point.
        ``None`` allows up to ``n - 1``. The spanning tree may be a forest
        if this cap is too small to connect all points.

    step_expansion : int, optional (default=16)
        Neighbor-query batch size. The spanning tree starts with this many
        nearest neighbors and fetches another batch on demand (via
        ``skip_radius``) until ``max_ranking`` is reached.

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
        Distance used for neighbor search. KD-tree supports Minkowski-family
        metrics (including cosine via L2-normalized Euclidean). Ball-tree
        additionally supports metrics such as canberra, braycurtis, haversine,
        and mahalanobis. If metric is "precomputed", X is a square distance matrix.

    p : int, optional (default=2)
        p value to use if using the minkowski metric.

    leaf_size : int, optional (default=40)
        Leaf size for the KD-tree / Ball-tree used in nearest-neighbour queries.

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

    printout, core_n_jobs, max_ranking, step_expansion, limitL, limitH = _check_input(
        X, core_n_jobs, max_ranking, step_expansion, leaf_size, metric, p, size_range, limitL, limitH)
    if printout:
        logger.info('Druhg is using defaults for: ' + printout)

    X = _coerce_feature_array(X, algorithm, metric)
    if not ("precomputed" in algorithm.lower() or "precomputed" in metric.lower() or issparse(X)):
        if not X.flags['C_CONTIGUOUS']:
            logger.info('Converting data array to c-contiguous')
            X = np.array(X, dtype=np.float64, order='C')
        if X.dtype != np.float64:
            logger.info('Converting data array to numpy float64')
            X = X.astype(np.float64)

    tree, algo_code = _tune_treealgo(X, metric, algorithm, leaf_size, p=p, **kwargs)

    if fix_outliers and do_edges is not False:
        do_edges = True

    size = X.shape[0]
    buffers = _allocate_if_needed(buffers, size, do_edges, do_labeling)

    ur = UniversalReciprocity(algo_code, tree,
                              buffers[Buffer.UNIONFIND.value], buffers[Buffer.UNIONFIND_FAST.value],
                              buffers[Buffer.VALUES.value],
                              max_neighbors_search=max_ranking, step_expansion=step_expansion,
                              metric=metric,
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


def _scipy_id(node, n):
    """Map a DRUHG union-find node to a SciPy linkage index.

    Leaves ``0 .. n-1`` are unchanged. DRUHG skips label ``n``
    (``next_label`` starts at ``n + 1``), so internal node ``n + 1 + k``
    becomes SciPy cluster ``n + k``.
    """
    return node if node < n else node - 1


def unionfind_to_linkage(parent, n, values, num_edges):
    """Convert a DRUHG union-find tree to a SciPy linkage matrix.

    Parameters
    ----------
    parent : array, shape (2 * n,)
        Union-find parent buffer. Leaves are ``0 .. n-1``; created clusters
        are ``n + 1, n + 2, ...``. Parent ``0`` means a root (or unused).

    n : int
        Number of samples.

    values : array, shape (n - 1,)
        Dialectical distance of each merge, in merge order.

    num_edges : int
        Number of real merges. May be ``< n - 1`` when the tree is a forest.

    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        SciPy linkage matrix. Each row is ``[idx1, idx2, dist, sample_count]``.
        Remaining forest components are joined with distances slightly above
        the last real merge so the matrix is a complete binary tree.
    """
    logger = logging.getLogger(__package__)
    parent = np.asarray(parent)
    values = np.asarray(values, dtype=np.float64)
    offset = n + 1
    n_merges = n - 1
    logger.info('Hierarchy: converting %s samples, %s edges', n, num_edges)
    if n < 2:
        logger.info('Hierarchy: fewer than 2 samples, empty linkage')
        return np.empty((0, 4), dtype=np.float64)
    if num_edges < 0:
        num_edges = 0
    if num_edges > n_merges:
        num_edges = n_merges

    children = np.full((n_merges, 2), -1, dtype=np.intp)
    filled = np.zeros(n_merges, dtype=np.intp)

    def add_child(node, p):
        if p == 0:
            return
        i = int(p) - offset
        if i < 0 or i >= num_edges:
            return
        slot = filled[i]
        if slot >= 2:
            return
        children[i, slot] = _scipy_id(int(node), n)
        filled[i] = slot + 1

    for node in range(n):
        add_child(node, int(parent[node]))
    for k in range(num_edges):
        add_child(offset + k, int(parent[offset + k]))

    Z = np.empty((n_merges, 4), dtype=np.float64)
    for i in range(num_edges):
        a, b = int(children[i, 0]), int(children[i, 1])
        if a < 0 or b < 0:
            raise ValueError(
                'Union-find merge %s does not have two children (got %s, %s).'
                % (i, a, b))
        if a > b:
            a, b = b, a
        size_a = 1 if a < n else Z[a - n, 3]
        size_b = 1 if b < n else Z[b - n, 3]
        Z[i, 0] = a
        Z[i, 1] = b
        Z[i, 2] = values[i]
        Z[i, 3] = size_a + size_b
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                'Hierarchy: merge %s [%s, %s] dist %s size %s',
                i, a, b, values[i], Z[i, 3])

    if num_edges == n_merges:
        logger.info(
            'Hierarchy: linkage %s x 4, last dist %s',
            n_merges, Z[-1, 2] if n_merges else 0)
        return Z

    comp_ids = []
    comp_sizes = []
    for node in range(n):
        if parent[node] == 0:
            comp_ids.append(node)
            comp_sizes.append(1)
    for k in range(num_edges):
        if parent[offset + k] == 0:
            comp_ids.append(n + k)
            comp_sizes.append(int(Z[k, 3]))

    max_d = 0.0
    if num_edges > 0:
        max_d = float(np.max(values[:num_edges]))
        if not np.isfinite(max_d) or max_d < 0.0:
            max_d = 0.0
    gap = max_d * 0.05 if max_d > 0.0 else 1.0

    n_components = len(comp_ids)
    logger.info(
        'Hierarchy: forest with %s components, joining %s dummy merges',
        n_components, n_merges - num_edges)

    cur_id = comp_ids[0]
    cur_sz = comp_sizes[0]
    for t, i in enumerate(range(num_edges, n_merges)):
        other_id = comp_ids[t + 1]
        other_sz = comp_sizes[t + 1]
        a, b = cur_id, other_id
        if a > b:
            a, b = b, a
        Z[i, 0] = a
        Z[i, 1] = b
        Z[i, 2] = max_d + gap * (t + 1)
        Z[i, 3] = cur_sz + other_sz
        cur_id = n + i
        cur_sz = int(Z[i, 3])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                'Hierarchy: dummy merge %s [%s, %s] dist %s size %s',
                i, a, b, Z[i, 2], Z[i, 3])

    logger.info(
        'Hierarchy: linkage %s x 4, last dist %s',
        n_merges, Z[-1, 2] if n_merges else 0)
    return Z


def labels_to_link_color_func(Z, labels, palette=None, mixed_color='#BFBFBF'):
    """Build a ``dendrogram(..., link_color_func=...)`` callable from flat labels.

    SciPy calls ``link_color_func(k)`` with the cluster id of each U-link
    (``k`` in ``n .. 2n-2``), **not** a sample index. You cannot write
    ``lambda k: palette[labels[k]]``. This walks ``Z`` instead and paints a
    link with a cluster color only when both children share that color.
    Mixed joins and outliers (``label < 0``) use ``mixed_color``.

    The callable returns a matplotlib color **string** (hex), as SciPy requires.

    Parameters
    ----------
    Z : ndarray, shape (n - 1, 4)
        SciPy linkage matrix.

    labels : array, shape (n,)
        Flat cluster labels, e.g. ``DRUHG.labels_``. Negative values are noise.

    palette : sequence of colors, optional
        Cycled with ``label % len(palette)``, matching MST plots. Defaults to
        the DRUHG plot palette.

    mixed_color : str, optional (default ``'#BFBFBF'``)
        Color for mixed joins and outliers.

    Returns
    -------
    link_color_func : callable
        Pass to ``scipy.cluster.hierarchy.dendrogram``.
    """
    from matplotlib.colors import to_hex

    if palette is None:
        from .plots import _palette
        palette = _palette

    labels = np.asarray(labels)
    n = labels.shape[0]
    if Z.shape[0] != n - 1:
        raise ValueError(
            'Z has %s rows, expected %s for %s labels.'
            % (Z.shape[0], n - 1, n))

    hex_palette = []
    for c in palette:
        if isinstance(c, str):
            hex_palette.append(c)
        else:
            hex_palette.append(to_hex(c[:3]))

    colors = [mixed_color] * (2 * n - 1)
    for i, lab in enumerate(labels):
        if lab >= 0:
            colors[i] = hex_palette[int(lab) % len(hex_palette)]

    for i, (a, b) in enumerate(np.asarray(Z[:, :2], dtype=np.intp)):
        c1, c2 = colors[int(a)], colors[int(b)]
        colors[n + i] = c1 if c1 == c2 else mixed_color

    def link_color_func(k):
        return colors[int(k)]

    return link_color_func


class DRUHG(BaseEstimator, ClusterMixin):
    def __init__(self, metric='euclidean',
                 algorithm='best',
                 max_ranking=None,
                 step_expansion=16,
                 limitL=None,
                 limitH=None,
                 exclude=None,
                 fix_outliers=0,
                 leaf_size=40,
                 verbose=False,
                 core_n_jobs=None,
                 **kwargs):
        self.max_ranking = max_ranking
        self.step_expansion = step_expansion
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
        self.linkage_ = None

    def fit(self, X, y=None):
        """Perform DRUHG clustering.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), \
                (n_samples,), or (n_samples, n_samples)
            A feature array (a 1-d vector is n samples with one feature), or
            array of distances between samples if ``metric='precomputed'``.

        Returns
        -------
        self : object
            Returns self
        """
        kwargs = self.get_params()
        kwargs.update(self._metric_kwargs)

        X = _coerce_feature_array(X, self.algorithm, self.metric)
        self._size = X.shape[0]
        self._raw_data = X

        self.buffers_, self.num_edges_ = druhg(X, **kwargs)

        self.labels_ = self.buffers_[Buffer.LABELS.value]
        self.values_ = self.buffers_[Buffer.VALUES.value]
        self.ranks_ = self.buffers_[Buffer.RANKS.value]
        self.mst_ = self.buffers_[Buffer.MST.value]
        self.linkage_ = None
        return self

    def fit_predict(self, X, y=None):
        """Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), \
                (n_samples,), or (n_samples, n_samples)
            A feature array (a 1-d vector is n samples with one feature), or
            array of distances between samples if ``metric='precomputed'``.

        Returns
        -------
        y : ndarray, shape (n_samples, )
            cluster labels
        """
        self.fit(X)
        return self.labels_

    def hierarchy(self, plot=True, axis=None, labels=None, **kwargs):
        """Convert the DRUHG tree to SciPy linkage format.

        The matrix matches ``scipy.cluster.hierarchy.linkage`` and the
        tutorial at
        https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

        Each row is ``[idx1, idx2, dist, sample_count]``. Indices ``0 .. n-1``
        are the original samples; index ``n + i`` is the cluster formed at
        row ``i``. ``dist`` is the DRUHG dialectical distance of that merge.

        The result is also stored on ``self.linkage_``. Pass it to
        ``scipy.cluster.hierarchy.dendrogram``, ``fcluster``, or ``cophenet``.
        Distances follow DRUHG merge order and are not always monotonic.

        If the spanning tree is a forest, remaining
        components are joined with distances slightly above the last real merge.

        Parameters
        ----------
        plot : bool, optional (default=True)
            Draw a dendrogram with ``scipy.cluster.hierarchy.dendrogram``.

        axis : matplotlib axis, optional
            Axis to draw on. Created if omitted and ``plot=True``.

        labels : array, optional
            Flat cluster labels (``labels_``) used to color U-links.
            SciPy's ``link_color_func`` is called with cluster ids
            ``n .. 2n-2``, not sample indices, so this builds that mapping.
            A link keeps a cluster color only while both children share it;
            mixed joins and outliers are gray. This is **not** SciPy's
            ``labels`` argument (that one is leaf *text*).

        **kwargs :
            Passed to ``scipy.cluster.hierarchy.dendrogram``.

        Returns
        -------
        Z : ndarray, shape (n_samples - 1, 4)
            SciPy hierarchical clustering linkage matrix.
        """
        logger = logging.getLogger(__package__)
        if self.buffers_ is None or self.buffers_[Buffer.UNIONFIND.value] is None:
            raise AttributeError('Call fit() before hierarchy().')
        if self._size < 2:
            logger.info('Hierarchy: fewer than 2 samples, empty linkage')
            self.linkage_ = np.empty((0, 4), dtype=np.float64)
            return self.linkage_

        Z = unionfind_to_linkage(
            self.buffers_[Buffer.UNIONFIND.value],
            self._size,
            self.values_,
            self.num_edges_,
        )
        self.linkage_ = Z

        if plot:
            try:
                from matplotlib import pyplot as plt
                from scipy.cluster.hierarchy import dendrogram
            except ImportError:
                raise ImportError(
                    'You must install matplotlib and scipy to plot a dendrogram.')

            dendrogram_kwargs = {
                'leaf_rotation': 90.,
                'leaf_font_size': 8.,
            }
            dendrogram_kwargs.update(kwargs)
            if 'link_color_func' not in dendrogram_kwargs and (labels is not None or self.buffers_[Buffer.LABELS.value] is not None):
                if labels is not None:
                    labels_arr = np.asarray(labels)
                elif self.buffers_[Buffer.LABELS.value] is not None:
                    labels_arr = np.asarray(self.buffers_[Buffer.LABELS.value])
                if labels_arr.shape[0] != self._size:
                    raise ValueError(
                        'labels length %s != n_samples %s'
                        % (labels_arr.shape[0], self._size))
                n_clusters = len(set(int(x) for x in labels_arr if x >= 0))
                logger.info(
                    'Hierarchy: coloring dendrogram by labels_ (%s clusters)',
                    n_clusters)
                dendrogram_kwargs['link_color_func'] = labels_to_link_color_func(
                    Z, labels_arr)
                dendrogram_kwargs.setdefault('color_threshold', 0)
            else:
                logger.info('Hierarchy: plotting dendrogram')
            if axis is None:
                plt.figure(figsize=(25, 10))
                axis = plt.gca()
            axis.set_title('Hierarchical Clustering Dendrogram')
            axis.set_xlabel('sample index')
            axis.set_ylabel('distance')
            dendrogram(Z, ax=axis, **dendrogram_kwargs)

        return Z

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
        XX = _coerce_feature_array(XX, self.algorithm, self.metric)
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
