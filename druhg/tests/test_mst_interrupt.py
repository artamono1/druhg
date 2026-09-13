"""Tests for interrupting MST construction and labeling the partial forest."""
import logging
import _thread
import threading

import numpy as np
import pytest
from scipy.cluster.hierarchy import is_valid_linkage

from druhg import Buffer, DRUHG, druhg


def _blob(n=60, seed=0):
    rng = np.random.RandomState(seed)
    return np.ascontiguousarray(rng.randn(n, 2), dtype=np.float64)


def _fit_with_interrupt(dr, X):
    started = threading.Event()
    finished = threading.Event()

    def boom():
        started.wait(timeout=5)
        if not finished.is_set():
            _thread.interrupt_main()

    threading.Thread(target=boom, daemon=True).start()
    started.set()
    try:
        try:
            dr.fit(X)
        except KeyboardInterrupt:
            # Ctrl+C landed in post-MST logging/labeling; MST catch already ran or tree is usable.
            pass
    finally:
        finished.set()
    return dr


def test_max_edges_prompts_and_continues(caplog):
    X = _blob()
    n = X.shape[0]
    caplog.set_level(logging.WARNING, logger='druhg')
    dr = DRUHG(max_edges=8, limitL=1, limitH=n, verbose=False, do_edges=True)
    dr.fit(X)

    assert dr.interrupted_ is False
    assert dr.interrupt_reason_ is None
    assert dr.num_edges_ == n - 1
    assert dr.labels_ is not None
    assert dr.labels_.shape == (n,)
    assert 'max_edges limit reached after 8 edges' in caplog.text
    assert '13.56% of 59' in caplog.text
    assert 'Ctrl+C' in caplog.text


def test_max_edges_complete_tree_is_not_interrupted():
    X = _blob(n=20)
    dr = DRUHG(max_edges=1000, limitL=1, limitH=1000, verbose=False)
    dr.fit(X)

    assert dr.interrupted_ is False
    assert dr.interrupt_reason_ is None
    assert dr.num_edges_ == 19


def test_fit_tree_then_label():
    X = _blob(n=40)
    dr = DRUHG(limitL=1, limitH=40, verbose=False)
    dr.fit_tree(X)

    assert dr.labels_ is None
    assert dr.interrupted_ is False
    assert dr.num_edges_ == 39

    labels = dr.label()
    assert labels is not None
    assert labels.shape == (40,)
    assert dr.labels_ is labels


def test_timeout_prompts_and_continues(caplog):
    X = _blob(n=50)
    caplog.set_level(logging.WARNING, logger='druhg')
    dr = DRUHG(timeout=1e-15, limitL=1, limitH=50, verbose=False)
    dr.fit(X)

    assert dr.interrupted_ is False
    assert dr.interrupt_reason_ is None
    assert dr.num_edges_ == 49
    assert dr.labels_.shape == (50,)
    assert 'timeout limit reached after' in caplog.text
    assert '% of 49' in caplog.text
    assert 'Ctrl+C' in caplog.text


def test_partial_tree_hierarchy_and_relabel():
    DRUHG(verbose=False, limitL=1, limitH=20).fit(_blob(n=20))
    X = _blob(n=800, seed=2)
    dr = DRUHG(verbose=False, limitL=1, limitH=800, do_edges=True)
    _fit_with_interrupt(dr, X)

    assert dr.interrupted_ is True
    assert dr.interrupt_reason_ == 'KeyboardInterrupt'
    assert dr.num_edges_ < 799

    Z = dr.hierarchy(plot=False)
    assert Z.shape == (799, 4)
    assert is_valid_linkage(Z, throw=True)

    labels = dr.relabel(limitL=1, limitH=800)
    assert labels.shape == (800,)


def test_druhg_function_max_edges_keeps_building(caplog):
    X = _blob(n=25)
    caplog.set_level(logging.WARNING, logger='druhg')
    buffers, num_edges = druhg(
        X, max_edges=4, limitL=1, limitH=25, verbose=False, do_edges=True)
    assert num_edges == 24
    assert buffers[Buffer.INTERRUPTED.value] is None
    assert buffers[Buffer.LABELS.value].shape == (25,)
    assert 'max_edges limit reached after 4 edges' in caplog.text
    assert '16.67% of 24' in caplog.text


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
    X = _blob(n=800, seed=1)
    dr = DRUHG(verbose=False, limitL=1, limitH=800)
    _fit_with_interrupt(dr, X)

    assert dr.interrupted_ is True
    assert dr.interrupt_reason_ == 'KeyboardInterrupt'
    assert dr.num_edges_ < 799
    assert dr.labels_ is not None
    assert dr.labels_.shape == (800,)
