"""Tests for neighbors-k-extend: query_init and query_extend with per-index knn_scope."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from druhg._druhg_neighbors_k_extend import KDTree, BallTree


def _brute_knn(X, k, metric, **kwargs):
    scipy_metric = 'cityblock' if metric == 'manhattan' else metric
    D = cdist(X, X, metric=scipy_metric, **kwargs)
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    dist = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(dist, axis=1)
    dist = np.take_along_axis(dist, order, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    return dist, idx


def _brute_k_extend(X, skip_radii, knn_scope, metric, indices=None, **kwargs):
    scipy_metric = 'cityblock' if metric == 'manhattan' else metric
    D = cdist(X, X, metric=scipy_metric, **kwargs)
    np.fill_diagonal(D, np.inf)
    skip = np.asarray(skip_radii, dtype=np.float64)
    scope = np.asarray(knn_scope, dtype=np.intp)
    if scope.ndim == 0:
        scope = np.full(len(X), int(scope), dtype=np.intp)
    if indices is None:
        indices = np.arange(len(X))
    indices = np.asarray(indices, dtype=np.intp).reshape(-1)
    if scope.shape[0] == len(X):
        k_use = scope[indices]
    else:
        k_use = scope

    knn_skip = np.empty(len(indices), dtype=np.intp)
    rows_d = []
    rows_i = []
    width = int(k_use.max()) if len(k_use) else 1
    for q, i in enumerate(indices):
        row = D[i]
        knn_skip[q] = np.sum(row < skip[i])
        k = int(k_use[q])
        beyond = row.copy()
        beyond[row < skip[i]] = np.inf
        finite = np.isfinite(beyond)
        idx_all = np.flatnonzero(finite)
        if idx_all.size == 0:
            rows_d.append(np.empty(0, dtype=np.float64))
            rows_i.append(np.empty(0, dtype=np.intp))
            continue
        dist_all = beyond[idx_all]
        order = np.argsort(dist_all, kind='mergesort')
        dist_all = dist_all[order]
        idx_all = idx_all[order]
        if dist_all.size <= k:
            keep_d, keep_i = dist_all, idx_all
        else:
            tau = dist_all[k - 1]
            tol = 1e-12 * max(1.0, abs(tau)) + 1e-15
            mask = dist_all <= tau + tol
            keep_d, keep_i = dist_all[mask], idx_all[mask]
        rows_d.append(keep_d)
        rows_i.append(keep_i)
        width = max(width, len(keep_d))

    dist_arr = np.zeros((len(indices), width), dtype=np.float64)
    ind_arr = np.zeros((len(indices), width), dtype=np.intp)
    for q in range(len(indices)):
        n = len(rows_d[q])
        dist_arr[q, :n] = rows_d[q]
        ind_arr[q, :n] = rows_i[q]
    return knn_skip, dist_arr, ind_arr


@pytest.mark.parametrize('Tree', [KDTree, BallTree])
@pytest.mark.parametrize('metric,kwargs', [
    ('euclidean', {}),
    ('manhattan', {}),
    ('chebyshev', {}),
    ('minkowski', {'p': 3}),
    ('cosine', {}),
])
def test_query_init_matches_brute(Tree, metric, kwargs):
    rng = np.random.RandomState(0)
    X = np.ascontiguousarray(rng.randn(120, 4))
    k = 7
    tree = Tree(X, leaf_size=8, metric=metric, **kwargs)
    dist, ind = tree.query_init(k=k)
    brute_dist, _ = _brute_knn(X, k, metric, **kwargs)
    np.testing.assert_allclose(dist, brute_dist, rtol=1e-7, atol=1e-9)
    assert not np.any(ind == np.arange(len(X))[:, None])
    assert np.all(dist[:, 0] >= 0)


def test_balltree_extra_metrics_initial():
    rng = np.random.RandomState(2)
    X = np.ascontiguousarray(np.abs(rng.randn(40, 3)) + 0.1)
    k = 3
    for metric in ('canberra', 'braycurtis', 'cosine'):
        tree = BallTree(X, leaf_size=6, metric=metric)
        dist, _ = tree.query_init(k=k)
        brute_dist, _ = _brute_knn(X, k, metric)
        np.testing.assert_allclose(dist, brute_dist, rtol=1e-6, atol=1e-8)


def test_kdtree_rejects_ball_metric():
    X = np.ascontiguousarray([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match='Cannot be used with KDTree'):
        KDTree(X, metric='canberra')


def test_query_k_bounds():
    X = np.ascontiguousarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    tree = KDTree(X, leaf_size=1)
    with pytest.raises(ValueError):
        tree.query_init(k=3)
    with pytest.raises(ValueError):
        tree.query_init(k=0)
    with pytest.raises(ValueError):
        tree.query_extend([0], np.zeros(3), np.array([3, 1, 1]))
    with pytest.raises(ValueError):
        tree.query_extend([0], np.zeros(3), 0)


def test_sklearn_agreement_if_present():
    pytest.importorskip('sklearn')
    from sklearn.neighbors import KDTree as SKKD, BallTree as SKBall

    rng = np.random.RandomState(3)
    X = np.ascontiguousarray(rng.randn(90, 5))
    k = 6
    for Ours, Theirs, metric in (
        (KDTree, SKKD, 'euclidean'),
        (BallTree, SKBall, 'manhattan'),
    ):
        ours = Ours(X, leaf_size=10, metric=metric)
        theirs = Theirs(X, leaf_size=10, metric=metric)
        d0, i0 = ours.query_init(k=k)
        d1, i1 = theirs.query(X, k=k + 1)
        np.testing.assert_allclose(d0, d1[:, 1:], rtol=1e-7, atol=1e-9)
        np.testing.assert_array_equal(i0, i1[:, 1:])


@pytest.mark.parametrize('Tree', [KDTree, BallTree])
@pytest.mark.parametrize('metric,kwargs', [
    ('euclidean', {}),
    ('manhattan', {}),
    ('chebyshev', {}),
    ('minkowski', {'p': 3}),
    ('cosine', {}),
])
def test_query_extend_matches_brute(Tree, metric, kwargs):
    rng = np.random.RandomState(5)
    X = np.ascontiguousarray(rng.randn(90, 3))
    tree = Tree(X, leaf_size=6, metric=metric, **kwargs)
    full_dist, _ = tree.query_init(k=8)
    skip_radii = 0.5 * (full_dist[:, 3] + full_dist[:, 4])
    knn_scope = np.full(len(X), 5, dtype=np.intp)
    knn_skip, dist, ind = tree.query_extend(np.arange(len(X)), skip_radii, knn_scope)
    brute_y, brute_dist, brute_ind = _brute_k_extend(
        X, skip_radii, knn_scope, metric, **kwargs)
    np.testing.assert_array_equal(knn_skip, brute_y)
    assert dist.shape[1] >= 5
    np.testing.assert_allclose(dist, brute_dist, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(ind, brute_ind)
    assert not np.any((dist > 0) & (ind == np.arange(len(X))[:, None]))


def test_query_extend_zero_skip_matches_init_prefix():
    rng = np.random.RandomState(6)
    X = np.ascontiguousarray(rng.randn(40, 2))
    tree = KDTree(X, leaf_size=4, metric='euclidean')
    k = 6
    d0, i0 = tree.query_init(k=k)
    knn_skip, dist, ind = tree.query_extend(
        np.arange(len(X)), np.zeros(len(X)), np.full(len(X), k, dtype=np.intp))
    np.testing.assert_array_equal(knn_skip, 0)
    assert dist.shape[1] >= k
    np.testing.assert_allclose(dist[:, :k], d0, rtol=1e-7, atol=1e-9)
    np.testing.assert_array_equal(ind[:, :k], i0)
    assert not np.any(np.isinf(dist))


def test_query_extend_delimiter_is_zero_not_inf():
    rng = np.random.RandomState(7)
    X = np.ascontiguousarray(rng.randn(40, 2))
    tree = KDTree(X, leaf_size=5, metric='euclidean')
    knn_scope = np.full(len(X), 3, dtype=np.intp)
    knn_scope[:10] = 8
    knn_skip, dist, ind = tree.query_extend(
        np.arange(len(X)), np.zeros(len(X)), knn_scope)
    assert dist.shape[1] >= 8
    assert not np.any(np.isinf(dist))
    assert not np.any(np.isinf(ind.astype(np.float64)))
    for i in range(10, len(X)):
        n = int(np.sum(dist[i] > 0))
        assert n >= 3
        np.testing.assert_array_equal(dist[i, n:], 0)
        np.testing.assert_array_equal(ind[i, n:], 0)


def test_query_extend_grows_past_knn_scope_on_ties():
    # Center plus five equidistant neighbors; k=3 must keep all five.
    X = np.ascontiguousarray([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [np.sqrt(0.5), np.sqrt(0.5)],
        [8.0, 8.0],
        [-8.0, 8.0],
        [8.0, -8.0],
    ])
    tree = KDTree(X, leaf_size=2, metric='euclidean')
    skip = np.zeros(len(X))
    scope = np.full(len(X), 3, dtype=np.intp)
    knn_skip, dist, ind = tree.query_extend([0], skip, scope)
    assert knn_skip[0] == 0
    n = int(np.sum(dist[0] > 0))
    assert n >= 5
    assert dist.shape[1] >= 5
    np.testing.assert_allclose(dist[0, :5], 1.0, rtol=1e-7, atol=1e-9)
    assert set(ind[0, :5].tolist()) == {1, 2, 3, 4, 5}
    np.testing.assert_array_equal(dist[0, n:], 0)


def test_query_extend_open_skip_ball():
    rng = np.random.RandomState(9)
    X = np.ascontiguousarray(rng.randn(70, 2))
    tree = KDTree(X, leaf_size=5, metric='euclidean')
    full_dist, full_ind = tree.query_init(k=10)
    skip = full_dist[:, 5]
    scope = np.full(len(X), 4, dtype=np.intp)
    knn_skip, dist, ind = tree.query_extend(np.arange(len(X)), skip, scope)
    np.testing.assert_array_equal(knn_skip, 5)
    np.testing.assert_allclose(dist[:, :4], full_dist[:, 5:9], rtol=1e-7, atol=1e-9)
    np.testing.assert_array_equal(ind[:, :4], full_ind[:, 5:9])


def test_query_extend_per_index_scope():
    rng = np.random.RandomState(10)
    X = np.ascontiguousarray(rng.randn(50, 2))
    tree = KDTree(X, leaf_size=5, metric='euclidean')
    skip = np.zeros(len(X))
    scope = np.full(len(X), 2, dtype=np.intp)
    scope[3] = 7
    scope[10] = 4
    idx = np.array([3, 10, 21])
    knn_skip, dist, ind = tree.query_extend(idx, skip, scope)
    brute_y, brute_dist, brute_ind = _brute_k_extend(X, skip, scope, 'euclidean', indices=idx)
    np.testing.assert_array_equal(knn_skip, brute_y)
    assert dist.shape[1] >= 7
    np.testing.assert_allclose(dist, brute_dist, rtol=1e-7, atol=1e-9)
    np.testing.assert_array_equal(ind, brute_ind)
    assert not np.any(ind[:, :2] == idx[:, None])


def test_query_extend_scope_aligned_with_indices():
    rng = np.random.RandomState(11)
    X = np.ascontiguousarray(rng.randn(40, 2))
    tree = KDTree(X, leaf_size=4, metric='euclidean')
    skip = np.zeros(len(X))
    idx = np.array([2, 8, 15, 30])
    scope_q = np.array([2, 5, 3, 6])
    knn_skip, dist, ind = tree.query_extend(idx, skip, scope_q)
    scope_full = np.ones(len(X), dtype=np.intp)
    scope_full[idx] = scope_q
    brute_y, brute_dist, brute_ind = _brute_k_extend(
        X, skip, scope_full, 'euclidean', indices=idx)
    np.testing.assert_array_equal(knn_skip, brute_y)
    np.testing.assert_allclose(dist, brute_dist, rtol=1e-7, atol=1e-9)
    np.testing.assert_array_equal(ind, brute_ind)


def test_query_extend_subset_matches_full_rows():
    rng = np.random.RandomState(12)
    X = np.ascontiguousarray(rng.randn(80, 3))
    idx = np.array([3, 10, 21, 55, 70])
    skip = np.zeros(len(X))
    scope = np.full(len(X), 5, dtype=np.intp)
    for Tree, metric in ((KDTree, 'euclidean'), (BallTree, 'manhattan')):
        tree = Tree(X, leaf_size=6, metric=metric)
        dist_all, ind_all = tree.query_init(k=5)
        knn_skip, dist, ind = tree.query_extend(idx, skip, scope)
        np.testing.assert_array_equal(knn_skip, 0)
        assert not np.any(ind[:, :5] == idx[:, None])
        np.testing.assert_allclose(dist[:, :5], dist_all[idx], rtol=1e-7, atol=1e-9)
        np.testing.assert_array_equal(ind[:, :5], ind_all[idx])


def test_query_extend_rejects_short_arrays():
    X = np.ascontiguousarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    tree = KDTree(X, leaf_size=1)
    with pytest.raises(ValueError, match='skip_radii must have length n_samples'):
        tree.query_extend([0, 1], np.array([0.1, 0.2]), np.array([1, 1, 1]))
    with pytest.raises(ValueError, match='knn_scope must have length n_samples or match indices'):
        tree.query_extend([0, 1], np.zeros(3), np.array([1]))


def test_query_extend_empty_indices():
    X = np.ascontiguousarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    tree = KDTree(X, leaf_size=1)
    knn_skip, dist, ind = tree.query_extend([], np.zeros(3), np.array([1, 2, 1]))
    assert knn_skip.shape == (0,)
    assert dist.shape == (0, 2)
    assert ind.shape == (0, 2)


def test_query_extend_scalar_scope():
    rng = np.random.RandomState(13)
    X = np.ascontiguousarray(rng.randn(30, 2))
    tree = KDTree(X, leaf_size=4, metric='euclidean')
    skip = np.zeros(len(X))
    knn_skip, dist, ind = tree.query_extend(np.arange(len(X)), skip, 4)
    brute_y, brute_dist, brute_ind = _brute_k_extend(X, skip, 4, 'euclidean')
    np.testing.assert_array_equal(knn_skip, brute_y)
    np.testing.assert_allclose(dist, brute_dist, rtol=1e-7, atol=1e-9)
    np.testing.assert_array_equal(ind, brute_ind)


def test_query_extend_parallel_matches_brute():
    rng = np.random.RandomState(14)
    X = np.ascontiguousarray(rng.randn(140, 3))
    tree = KDTree(X, leaf_size=8, metric='euclidean')
    full_dist, _ = tree.query_init(k=6)
    skip = 0.5 * (full_dist[:, 1] + full_dist[:, 2])
    scope = np.full(len(X), 3, dtype=np.intp)
    scope[::7] = 5
    knn_skip, dist, ind = tree.query_extend(np.arange(len(X)), skip, scope)
    brute_y, brute_dist, brute_ind = _brute_k_extend(X, skip, scope, 'euclidean')
    np.testing.assert_array_equal(knn_skip, brute_y)
    np.testing.assert_allclose(dist, brute_dist, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(ind, brute_ind)
    assert not np.any(np.isinf(dist))
