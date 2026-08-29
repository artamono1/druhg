"""Tests for neighbors-skip: query_init and query_skip."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from druhg._druhg_neighbors_skip import KDTree, BallTree


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


def _brute_skip_annulus(X, skip_radius, k, metric, **kwargs):
    scipy_metric = 'cityblock' if metric == 'manhattan' else metric
    D = cdist(X, X, metric=scipy_metric, **kwargs)
    np.fill_diagonal(D, np.inf)
    r = np.asarray(skip_radius, dtype=np.float64)
    knn_skip = np.sum(D < r[:, None], axis=1).astype(np.intp)
    D = np.where(D < r[:, None], np.inf, D)
    idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    dist = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(dist, axis=1)
    dist = np.take_along_axis(dist, order, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    return knn_skip, dist, idx


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
        tree.query_skip([0], np.zeros(3), k=3)


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


def test_one_dimensional_query_init():
    rng = np.random.RandomState(4)
    x1 = np.ascontiguousarray(rng.randn(60))
    x2 = x1.reshape(-1, 1)
    k = 5
    for Tree in (KDTree, BallTree):
        t1 = Tree(x1, leaf_size=4, metric='euclidean')
        t2 = Tree(x2, leaf_size=4, metric='euclidean')
        assert t1.data.shape == (60, 1)
        d1, _ = t1.query_init(k=k)
        d2, _ = t2.query_init(k=k)
        np.testing.assert_allclose(d1, d2, rtol=1e-9, atol=1e-12)
        brute, _ = _brute_knn(x2, k, 'euclidean')
        np.testing.assert_allclose(d1, brute, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize('Tree', [KDTree, BallTree])
@pytest.mark.parametrize('metric,kwargs', [
    ('euclidean', {}),
    ('manhattan', {}),
    ('chebyshev', {}),
    ('minkowski', {'p': 3}),
    ('cosine', {}),
])
def test_query_skip_matches_brute(Tree, metric, kwargs):
    rng = np.random.RandomState(5)
    X = np.ascontiguousarray(rng.randn(90, 3))
    tree = Tree(X, leaf_size=6, metric=metric, **kwargs)
    full_dist, _ = tree.query_init(k=8)
    skip_radius = 0.5 * (full_dist[:, 3] + full_dist[:, 4])
    n = 5
    knn_skip, dist, ind = tree.query_skip(np.arange(len(X)), skip_radius, k=n)
    brute_y, brute_dist, _ = _brute_skip_annulus(X, skip_radius, n, metric, **kwargs)
    np.testing.assert_array_equal(knn_skip, brute_y)
    np.testing.assert_allclose(dist, brute_dist, rtol=1e-6, atol=1e-8)
    assert not np.any(ind == np.arange(len(X))[:, None])
    assert np.all(dist[:, 0] + 1e-12 >= skip_radius)


def test_query_skip_zero_matches_initial():
    rng = np.random.RandomState(6)
    X = np.ascontiguousarray(rng.randn(40, 2))
    tree = KDTree(X, leaf_size=4, metric='euclidean')
    d0, i0 = tree.query_init(k=6)
    knn_skip, d1, i1 = tree.query_skip(np.arange(len(X)), np.zeros(len(X)), k=6)
    np.testing.assert_array_equal(knn_skip, 0)
    np.testing.assert_allclose(d0, d1)
    np.testing.assert_array_equal(i0, i1)


def test_query_skip_on_knn_distance_is_open_ball():
    rng = np.random.RandomState(8)
    X = np.ascontiguousarray(rng.randn(70, 2))
    tree = KDTree(X, leaf_size=5, metric='euclidean')
    full_dist, full_ind = tree.query_init(k=10)
    r = full_dist[:, 5]
    knn_skip, dist, ind = tree.query_skip(np.arange(len(X)), r, k=4)
    np.testing.assert_array_equal(knn_skip, 5)
    np.testing.assert_allclose(dist, full_dist[:, 5:9], rtol=1e-7, atol=1e-9)
    np.testing.assert_array_equal(ind, full_ind[:, 5:9])


def test_query_skip_subset_matches_full_rows():
    rng = np.random.RandomState(10)
    X = np.ascontiguousarray(rng.randn(80, 3))
    idx = np.array([3, 10, 21, 55, 70])
    skip_zero = np.zeros(len(X))
    for Tree, metric in ((KDTree, 'euclidean'), (BallTree, 'manhattan')):
        tree = Tree(X, leaf_size=6, metric=metric)
        dist_all, ind_all = tree.query_init(k=5)
        knn_skip, dist, ind = tree.query_skip(idx, skip_zero, k=5)
        np.testing.assert_array_equal(knn_skip, 0)
        np.testing.assert_allclose(dist, dist_all[idx], rtol=1e-7, atol=1e-9)
        np.testing.assert_array_equal(ind, ind_all[idx])
        assert not np.any(ind == idx[:, None])


def test_query_skip_subset_with_full_skip_radius():
    rng = np.random.RandomState(11)
    X = np.ascontiguousarray(rng.randn(60, 2))
    tree = KDTree(X, leaf_size=5, metric='euclidean')
    full_dist, full_ind = tree.query_init(k=8)
    idx = np.array([1, 7, 22, 40])
    r_full = full_dist[:, 3]
    knn_skip, dist, ind = tree.query_skip(idx, r_full, k=3)
    np.testing.assert_array_equal(knn_skip, 3)
    np.testing.assert_allclose(dist, full_dist[idx][:, 3:6], rtol=1e-7, atol=1e-9)
    np.testing.assert_array_equal(ind, full_ind[idx][:, 3:6])


def test_query_skip_rejects_short_skip_radius():
    X = np.ascontiguousarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    tree = KDTree(X, leaf_size=1)
    with pytest.raises(ValueError, match='skip_radius must have length n_samples'):
        tree.query_skip([0, 1], np.array([0.1, 0.2]), k=1)


def test_query_skip_empty_indices():
    X = np.ascontiguousarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    tree = KDTree(X, leaf_size=1)
    knn_skip, dist, ind = tree.query_skip([], np.zeros(3), k=1)
    assert knn_skip.shape == (0,)
    assert dist.shape == (0, 1)
    assert ind.shape == (0, 1)
