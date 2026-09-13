"""Tests for interrupting MST construction and labeling the partial forest."""
import logging
import sys

import numpy as np
import pytest

from druhg import Buffer, DRUHG, druhg


def _blob(n=60, seed=0):
    rng = np.random.RandomState(seed)
    return np.ascontiguousarray(rng.randn(n, 2), dtype=np.float64)


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


def test_progress_interval_prompts_and_continues(caplog):
    X = _blob(n=50)
    caplog.set_level(logging.WARNING, logger='druhg')
    dr = DRUHG(progress_interval=1e-15, limitL=1, limitH=50, verbose=False)
    dr.fit(X)

    assert dr.interrupted_ is False
    assert dr.num_edges_ == 49
    assert 'Still working' in caplog.text
    assert '% of 49' in caplog.text
    assert 'Ctrl+C' in caplog.text


def test_progress_inplace_on_tty(capsys, monkeypatch, caplog):
    monkeypatch.setattr(sys.stderr, 'isatty', lambda: True)
    X = _blob(n=50)
    caplog.set_level(logging.WARNING, logger='druhg')
    dr = DRUHG(progress_interval=1e-15, limitL=1, limitH=50, verbose=False)
    dr.fit(X)

    err = capsys.readouterr().err
    assert '\r' in err
    assert '\033[2K' in err
    assert 'Ctrl+C stops MST' in err
    assert 'Still working' not in err
    assert 'Still working' not in caplog.text
    assert dr.num_edges_ == 49


class _FakeJupyterProgress(object):
    def __init__(self):
        self.msgs = []
        self.closed = 0

    def write(self, msg):
        self.msgs.append(msg)

    def close(self):
        self.closed += 1


def test_progress_inplace_in_jupyter(monkeypatch, caplog):
    fake = _FakeJupyterProgress()
    monkeypatch.setattr('druhg._druhg_tree_logging.try_jupyter_progress', lambda: fake)
    X = _blob(n=50)
    caplog.set_level(logging.WARNING, logger='druhg')
    dr = DRUHG(progress_interval=1e-15, limitL=1, limitH=50, verbose=False)
    dr.fit(X)

    assert fake.msgs
    assert any('Interrupt kernel stops MST' in msg for msg in fake.msgs)
    assert fake.closed >= 1
    assert 'Still working' not in caplog.text
    assert dr.num_edges_ == 49


def test_jupyter_progress_skipped_on_tty(monkeypatch, capsys, caplog):
    fake = _FakeJupyterProgress()
    monkeypatch.setattr('druhg._druhg_tree_logging.try_jupyter_progress', lambda: fake)
    monkeypatch.setattr(sys.stderr, 'isatty', lambda: True)
    X = _blob(n=50)
    caplog.set_level(logging.WARNING, logger='druhg')
    DRUHG(progress_interval=1e-15, limitL=1, limitH=50, verbose=False).fit(X)

    err = capsys.readouterr().err
    assert '\r' in err
    assert 'Ctrl+C stops MST' in err
    assert not any('Interrupt kernel stops MST' in msg for msg in fake.msgs)


def test_try_jupyter_progress_requires_notebook_shell(monkeypatch):
    from druhg._druhg_tree_logging import try_jupyter_progress

    class TerminalInteractiveShell(object):
        pass

    class ZMQInteractiveShell(object):
        pass

    monkeypatch.setattr('IPython.get_ipython', lambda: TerminalInteractiveShell())
    assert try_jupyter_progress() is None

    captured = []

    class Handle(object):
        def update(self, payload):
            captured.append(payload)

    def fake_display(payload, display_id=True):
        captured.append(payload)
        return Handle()

    monkeypatch.setattr('IPython.get_ipython', lambda: ZMQInteractiveShell())
    monkeypatch.setattr('IPython.display.display', fake_display)
    prog = try_jupyter_progress()
    assert prog is not None
    prog.write('hello')
    prog.write('world')
    assert len(captured) == 2
    assert 'hello' in captured[0].data
    assert 'world' in captured[1].data


def test_knn_start_warns_on_large_input(caplog):
    X = _blob(n=1000)
    caplog.set_level(logging.WARNING, logger='druhg')
    dr = DRUHG(progress_interval=0, limitL=1, limitH=1000, verbose=False)
    dr.fit(X)

    assert 'kNN querying: 24 neighbors for 1000 points' in caplog.text
    assert 'Ctrl+C' in caplog.text


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
    with pytest.raises(ValueError, match='progress_interval'):
        DRUHG(progress_interval=-1).fit(X)
