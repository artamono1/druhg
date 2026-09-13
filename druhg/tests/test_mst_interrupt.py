"""Tests for interrupting MST construction and labeling the partial forest."""
import _thread
import threading
import time

import numpy as np
import pytest
from scipy.cluster.hierarchy import is_valid_linkage

from druhg import Buffer, DRUHG, druhg


def _blob(n=60, seed=0):
    rng = np.random.RandomState(seed)
    return np.ascontiguousarray(rng.randn(n, 2), dtype=np.float64)


def test_max_edges_stops_and_labels():
    X = _blob()
    n = X.shape[0]
    dr = DRUHG(max_edges=8, limitL=1, limitH=n, verbose=False, do_edges=True)
    dr.fit(X)

    assert dr.interrupted_ is True
    assert dr.interrupt_reason_ == 'max_edges'
    assert dr.num_edges_ == 8
    assert dr.labels_ is not None
    assert dr.labels_.shape == (n,)
    assert dr.values_[8] == -1
    assert dr.mst_[16] == -1
    assert dr.mst_[17] == -1


def test_max_edges_complete_tree_is_not_interrupted():
    X = _blob(n=20)
    dr = DRUHG(max_edges=1000, limitL=1, limitH=1000, verbose=False)
    dr.fit(X)

    assert dr.interrupted_ is False
    assert dr.interrupt_reason_ is None
    assert dr.num_edges_ == 19


def test_fit_tree_then_label():
    X = _blob(n=40)
    dr = DRUHG(max_edges=6, limitL=1, limitH=40, verbose=False)
    dr.fit_tree(X)

    assert dr.labels_ is None
    assert dr.interrupted_ is True
    assert dr.num_edges_ == 6

    labels = dr.label()
    assert labels is not None
    assert labels.shape == (40,)
    assert dr.labels_ is labels


def test_timeout_stops_then_labels():
    X = _blob(n=50)
    dr = DRUHG(timeout=1e-15, limitL=1, limitH=50, verbose=False)
    dr.fit(X)

    assert dr.interrupted_ is True
    assert dr.interrupt_reason_ == 'timeout'
    assert dr.num_edges_ < 49
    assert dr.labels_.shape == (50,)


def test_partial_tree_hierarchy_and_relabel():
    X = _blob(n=30)
    dr = DRUHG(max_edges=10, limitL=1, limitH=30, verbose=False, do_edges=True)
    dr.fit(X)

    Z = dr.hierarchy(plot=False)
    assert Z.shape == (29, 4)
    assert is_valid_linkage(Z, throw=True)

    labels = dr.relabel(limitL=1, limitH=30)
    assert labels.shape == (30,)


def test_druhg_function_max_edges():
    X = _blob(n=25)
    buffers, num_edges = druhg(
        X, max_edges=4, limitL=1, limitH=25, verbose=False, do_edges=True)
    assert num_edges == 4
    assert buffers[Buffer.INTERRUPTED.value] == 'max_edges'
    assert buffers[Buffer.LABELS.value].shape == (25,)


def test_label_before_fit_tree_raises():
    dr = DRUHG()
    with pytest.raises(AttributeError, match='fit_tree'):
        dr.label()


def test_max_edges_validation():
    X = _blob(n=10)
    with pytest.raises(ValueError, match='max_edges'):
        DRUHG(max_edges=0).fit(X)
    with pytest.raises(ValueError, match='timeout'):
        DRUHG(timeout=-1).fit(X)


def test_keyboard_interrupt_still_labels():
    # Warm neighbor-tree so Ctrl+C lands in MST, not first compile.
    DRUHG(verbose=False, limitL=1, limitH=20).fit(_blob(n=20))
    X = _blob(n=400, seed=1)
    dr = DRUHG(verbose=False, limitL=1, limitH=400)

    def boom():
        time.sleep(0.02)
        _thread.interrupt_main()

    threading.Thread(target=boom, daemon=True).start()
    dr.fit(X)

    assert dr.interrupted_ is True
    assert dr.interrupt_reason_ == 'KeyboardInterrupt'
    assert dr.num_edges_ < 399
    assert dr.labels_ is not None
    assert dr.labels_.shape == (400,)
