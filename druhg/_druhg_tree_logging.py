# -*- coding: utf-8 -*-
"""MST progress and interrupt logging for spanning-tree construction."""

# Author: Pavel Artamonov
# License: 3-clause BSD

import html
import logging
import os
import shutil
import sys

_WAIT_HINT = (
    'Ctrl+C to stop MST and continue labeling, or wait to keep building.')


def in_jupyter_shell():
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    name = ip.__class__.__name__
    if name == 'TerminalInteractiveShell':
        return False
    if name == 'ZMQInteractiveShell':
        return True
    module = type(ip).__module__
    if 'colab' in module or 'ipykernel' in module:
        return True
    config = getattr(ip, 'config', None)
    try:
        return bool(config and 'IPKernelApp' in config)
    except Exception:
        return False


class JupyterProgress(object):
    """One updatable output in a notebook. IPython is imported lazily."""

    def __init__(self):
        from IPython.display import HTML, display
        self._handle = None
        self._HTML = HTML
        self._display = display

    def write(self, msg):
        payload = self._HTML(
            '<pre style="margin:0">%s</pre>' % html.escape(str(msg), quote=False))
        if self._handle is None:
            self._handle = self._display(payload, display_id=True)
        else:
            self._handle.update(payload)

    def close(self):
        self._handle = None


def try_jupyter_progress():
    if not in_jupyter_shell():
        return None
    return JupyterProgress()


def _edge_progress(result_edges, num_points):
    total = num_points - 1
    if total < 1:
        total = 1
    return total, 100. * result_edges / total


def _detect_tty():
    try:
        if sys.stderr.isatty():
            return True
    except Exception:
        pass
    try:
        return bool(os.isatty(sys.stderr.fileno()))
    except Exception:
        return False


class TreeLogging(object):
    """TTY / Jupyter / logger heartbeats for MST formation."""

    def __init__(self):
        self.logger = logging.getLogger(__package__)
        self.debug_enabled = self.logger.isEnabledFor(logging.DEBUG)
        self.jupyter_progress = try_jupyter_progress()
        self.is_tty = _detect_tty()
        self.progress_line_open = False
        self.saw_heartbeat = False

    def _tty_columns(self):
        try:
            cols = os.get_terminal_size(sys.stderr.fileno()).columns
        except Exception:
            cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        if cols < 20:
            cols = 20
        return cols

    def end_progress_line(self):
        if self.jupyter_progress is not None:
            self.jupyter_progress.close()
        if not self.progress_line_open:
            return
        if self.is_tty:
            sys.stderr.write('\n')
            sys.stderr.flush()
        self.progress_line_open = False

    def _write_tty_status(self, msg):
        cols = self._tty_columns()
        if len(msg) >= cols:
            msg = msg[:cols - 1]
        # One visual row: disable wrap, CR, erase line. A wrapped
        # status makes \\r look like spam in Alacritty.
        sys.stderr.write('\033[?7l\r\033[2K' + msg + '\033[?7h')
        sys.stderr.flush()
        self.progress_line_open = True

    def write_inplace_status(self, msg):
        if self.is_tty:
            self._write_tty_status(msg)
            return
        if self.jupyter_progress is not None:
            try:
                self.jupyter_progress.write(msg)
            except Exception:
                self.logger.warning('%s', msg)
                return
            self.progress_line_open = True

    def info(self, msg, *args):
        self.end_progress_line()
        self.logger.info(msg, *args)

    def warning(self, msg, *args):
        self.end_progress_line()
        self.logger.warning(msg, *args)

    def error(self, msg, *args):
        self.end_progress_line()
        self.logger.error(msg, *args)

    def note_interrupt(self, reason, result_edges, num_points):
        total, pct = _edge_progress(result_edges, num_points)
        self.warning(
            'MSTree formation: interruption started (%s) after %s edges %.2f%% of %s.',
            reason, result_edges, pct, total)

    def _inplace_status(self, result_edges, total, pct, elapsed, hint):
        return 'MSTree formation: %s/%s edges (%.1f%%) %.1fs  %s' % (
            result_edges, total, pct, elapsed, hint)

    def prompt_progress(self, result_edges, num_points, elapsed):
        total, pct = _edge_progress(result_edges, num_points)
        self.saw_heartbeat = True
        if self.is_tty or self.jupyter_progress is not None:
            hint = 'Ctrl+C stops MST' if self.is_tty else 'Interrupt kernel stops MST'
            self.write_inplace_status(
                self._inplace_status(result_edges, total, pct, elapsed, hint))
            return
        self.logger.warning(
            'MSTree formation: %s edges %.2f%% of %s after %.1fs. Still working. %s',
            result_edges, pct, total, elapsed, _WAIT_HINT)

    def knn_query_start(self, max_neighbors_search, num_points, show_progress):
        head = 'kNN querying: %s neighbors for %s points.' % (
            max_neighbors_search, num_points)
        if num_points >= 1000:
            self.logger.warning(
                '%s Ctrl+C after this step stops MST and continues labeling.',
                head)
            if show_progress:
                self.write_inplace_status(
                    'kNN querying: blocking, no ticks until neighbors return')
        else:
            self.logger.info('%s', head)

    def knn_query_done(self):
        self.info('kNN querying: done')

    def finish_mst(self, interrupted, interrupt_reason, result_edges, num_points,
                   edge_cases, max_neighbors_search, elapsed=0.):
        total, pct = _edge_progress(result_edges, num_points)
        if interrupted:
            if result_edges >= total:
                suffix = 'Tree complete.'
            else:
                suffix = 'Partial forest. Continuing to labeling.'
            self.warning(
                'MSTree formation: interruption result: %s edges %.2f%% of %s (%s). %s',
                result_edges, pct, total, interrupt_reason, suffix)
        elif self.saw_heartbeat:
            if self.is_tty or self.jupyter_progress is not None:
                self.write_inplace_status(
                    self._inplace_status(
                        result_edges, total, pct, elapsed,
                        'Done. Continue labeling'))
                self.end_progress_line()
            else:
                self.logger.warning(
                    'MSTree formation: %s edges %.2f%% of %s after %.1fs. Done. Continue labeling.',
                    result_edges, pct, total, elapsed)
        else:
            self.info(
                'MSTree formation: %s edges %.2f%%. Done.',
                result_edges, 100. * result_edges / num_points)
        if result_edges != num_points - 1:
            self.info(
                '%s not connected edges of %s. It is a forest. Try increasing max_neighbors(max_ranking) value %s for a better result.',
                num_points - 1 - result_edges, num_points - 1, max_neighbors_search)
        if max_neighbors_search < num_points - 1 and edge_cases != 0:
            self.info(
                '%s edges with the max rank. Try increasing max_neighbors(max_ranking) value %s or pick the square mode (not available yet).',
                edge_cases, max_neighbors_search)
