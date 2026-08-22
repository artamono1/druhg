"""Tests for DRUHG KD-tree and Ball-tree kNN queries."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from druhg._druhg_neighbors import KDTree, BallTree


def _brute_knn(X, Y, k, metric, exclude_self=True, **kwargs):
    scipy_metric = 'cityblock' if metric == 'manhattan' else metric
    D = cdist(Y, X, metric=scipy_metric, **kwargs)
    if exclude_self:
        if Y.shape == X.shape and np.array_equal(Y, X):
            np.fill_diagonal(D, np.inf)
        elif np.shares_memory(Y, X):
            x0 = X.ctypes.data
            step = X.strides[0]
            y0 = Y.ctypes.data
            ystep = Y.strides[0]
            for i in range(len(Y)):
                j = (y0 + i * ystep - x0) // step
                if 0 <= j < len(X):
                    D[i, j] = np.inf
    idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    dist = np.take_along_axis(D, idx, axis=1)
    order = np.argsort(dist, axis=1)
    dist = np.take_along_axis(dist, order, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    return dist, idx


@pytest.mark.parametrize('Tree', [KDTree, BallTree])
@pytest.mark.parametrize('metric,kwargs', [
    ('euclidean', {}),
    ('manhattan', {}),
    ('chebyshev', {}),
    ('minkowski', {'p': 3}),
    ('cosine', {}),
])
def test_knn_matches_brute(Tree, metric, kwargs):
    rng = np.random.RandomState(0)
    X = np.ascontiguousarray(rng.randn(120, 4))
    k = 7
    tree = Tree(X, leaf_size=8, metric=metric, **kwargs)
    dist, ind = tree.query(X, k=k)
    brute_dist, _ = _brute_knn(X, X, k, metric, **kwargs)
    np.testing.assert_allclose(dist, brute_dist, rtol=1e-7, atol=1e-9)
    assert not np.any(ind == np.arange(len(X))[:, None])
    assert np.all(dist[:, 0] >= 0)


@pytest.mark.parametrize('Tree', [KDTree, BallTree])
def test_query_subset_matches_brute(Tree):
    rng = np.random.RandomState(1)
    X = np.ascontiguousarray(rng.randn(80, 3))
    Y = np.ascontiguousarray(X[10:25])
    tree = Tree(X, leaf_size=5, metric='euclidean')
    dist, ind = tree.query(Y, k=4, dualtree=True, breadth_first=True)
    brute_dist, _ = _brute_knn(X, Y, 4, 'euclidean')
    np.testing.assert_allclose(dist, brute_dist, rtol=1e-7, atol=1e-9)
    assert not np.any(ind == np.arange(10, 25)[:, None])


def test_balltree_extra_metrics():
    rng = np.random.RandomState(2)
    X = np.ascontiguousarray(np.abs(rng.randn(40, 3)) + 0.1)
    k = 3
    for metric in ('canberra', 'braycurtis', 'cosine'):
        tree = BallTree(X, leaf_size=6, metric=metric)
        dist, _ = tree.query(X, k=k)
        brute_dist, _ = _brute_knn(X, X, k, metric)
        np.testing.assert_allclose(dist, brute_dist, rtol=1e-6, atol=1e-8)


def test_kdtree_rejects_ball_metric():
    X = np.ascontiguousarray([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match='Cannot be used with KDTree'):
        KDTree(X, metric='canberra')


def test_query_k_bounds():
    X = np.ascontiguousarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    tree = KDTree(X, leaf_size=1)
    with pytest.raises(ValueError):
        tree.query(X, k=3)
    with pytest.raises(ValueError):
        tree.query(X, k=0)


def test_druhg_kd_and_ball_trees():
    from druhg import DRUHG
    rng = np.random.RandomState(0)
    X = np.ascontiguousarray(rng.randn(40, 2))
    kd = DRUHG(max_ranking=10, algorithm='kd_tree', verbose=False).fit(X)
    ball = DRUHG(max_ranking=10, algorithm='balltree', verbose=False).fit(X)
    assert kd.num_edges_ == 39
    assert ball.num_edges_ == 39
    np.testing.assert_array_equal(kd.labels_, ball.labels_)


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
        d0, i0 = ours.query(X, k=k)
        d1, i1 = theirs.query(X, k=k + 1)
        np.testing.assert_allclose(d0, d1[:, 1:], rtol=1e-7, atol=1e-9)
        np.testing.assert_array_equal(i0, i1[:, 1:])


def test_one_dimensional_knn_and_druhg():
    rng = np.random.RandomState(4)
    x1 = np.ascontiguousarray(rng.randn(60))
    x2 = x1.reshape(-1, 1)
    k = 5
    for Tree in (KDTree, BallTree):
        t1 = Tree(x1, leaf_size=4, metric='euclidean')
        t2 = Tree(x2, leaf_size=4, metric='euclidean')
        assert t1.data.shape == (60, 1)
        d1, _ = t1.query(x1, k=k)
        d2, _ = t2.query(x2, k=k)
        np.testing.assert_allclose(d1, d2, rtol=1e-9, atol=1e-12)
        brute, _ = _brute_knn(x2, x2, k, 'euclidean')
        np.testing.assert_allclose(d1, brute, rtol=1e-7, atol=1e-9)

    from druhg import DRUHG
    dr = DRUHG(max_ranking=12, verbose=False).fit(x1)
    assert dr._raw_data.shape == (60, 1)
    assert dr.num_edges_ == 59
    assert dr.labels_.shape == (60,)
