"""Tests for per-branch MST linking (heap-branching2)."""
import numpy as np

from druhg import DRUHG


def test_random_blob_is_a_tree():
    rng = np.random.RandomState(0)
    X = np.ascontiguousarray(rng.randn(40, 2), dtype=np.float64)
    dr = DRUHG(limitL=1, limitH=40, verbose=False)
    dr.fit(X)
    assert dr.num_edges_ == 39
    assert dr.interrupted_ is False


def test_two_blobs_still_connect():
    rng = np.random.RandomState(1)
    a = rng.randn(20, 2) + np.array([0., 0.])
    b = rng.randn(20, 2) + np.array([8., 8.])
    X = np.ascontiguousarray(np.vstack([a, b]), dtype=np.float64)
    dr = DRUHG(max_ranking=24, limitL=1, limitH=40, verbose=False)
    dr.fit(X)
    assert dr.num_edges_ == 39
    labels = dr.labels_
    assert labels is not None
    assert labels.shape == (40,)
