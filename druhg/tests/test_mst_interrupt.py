"""Tests for interrupting MST construction and labeling the partial forest."""
import logging
import sys

import numpy as np
import pytest

from druhg import DRUHG


def _blob(n=60, seed=0):
    rng = np.random.RandomState(seed)
    return np.ascontiguousarray(rng.randn(n, 2), dtype=np.float64)


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
    assert 'Done. Continue labeling' in caplog.text


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
    assert 'Done. Continue labeling' in err
    assert err.rfind('Done. Continue labeling') > err.rfind('Ctrl+C stops MST')
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
    assert 'Done. Continue labeling' in fake.msgs[-1]
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
    assert 'Done. Continue labeling' in err
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

    assert 'kNNeighbors: 24 neighbors for 1000 points' in caplog.text
    assert 'Ctrl+C' in caplog.text


def test_label_before_fit_tree_raises():
    dr = DRUHG()
    with pytest.raises(AttributeError, match='fit_tree'):
        dr.label()


def test_progress_interval_validation():
    X = _blob(n=10)
    with pytest.raises(ValueError, match='progress_interval'):
        DRUHG(progress_interval=-1).fit(X)
