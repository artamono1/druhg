# -*- coding: utf-8 -*-
"""MST progress and interrupt logging for spanning-tree construction."""

# Author: Pavel Artamonov
# License: 3-clause BSD

import html
import logging
import os
import shutil
import sys


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
        self._handle = None

    def write(self, msg):
        from IPython.display import HTML, display
        payload = HTML(
            '<pre style="margin:0">%s</pre>' % html.escape(str(msg), quote=False))
        if self._handle is None:
            self._handle = display(payload, display_id=True)
        else:
            self._handle.update(payload)

    def close(self):
        self._handle = None


def try_jupyter_progress():
    if not in_jupyter_shell():
        return None
    return JupyterProgress()


def _total_edges(num_points):
    total = num_points - 1
    if total < 1:
        return 1
    return total


class TreeLogging(object):
    """TTY / Jupyter / logger heartbeats for MST formation."""

    def __init__(self, jupyter_progress=None):
        self.logger = logging.getLogger(__package__)
        self.debug_enabled = self.logger.isEnabledFor(logging.DEBUG)
        self.jupyter_progress = jupyter_progress
        self.progress_line_open = False
        self.progress_line_width = 0

    def stderr_is_tty(self):
        try:
            if sys.stderr.isatty():
                return True
        except Exception:
            pass
        try:
            if os.isatty(sys.stderr.fileno()):
                return True
        except Exception:
            pass
        return False

    def _tty_columns(self):
        try:
            cols = os.get_terminal_size(sys.stderr.fileno()).columns
        except Exception:
            try:
                cols = shutil.get_terminal_size(fallback=(80, 24)).columns
            except Exception:
                cols = 80
        if cols < 20:
            cols = 20
        return cols

    def end_progress_line(self):
        if self.jupyter_progress is not None:
            try:
                self.jupyter_progress.close()
            except Exception:
                pass
        if not self.progress_line_open:
            return
        if self.stderr_is_tty():
            sys.stderr.write('\n')
            sys.stderr.flush()
        self.progress_line_open = False
        self.progress_line_width = 0

    def _write_tty_status(self, msg):
        cols = self._tty_columns()
        if len(msg) >= cols:
            msg = msg[:cols - 1]
        # One visual row: disable wrap, CR, erase line. A wrapped
        # status makes \\r look like spam in Alacritty.
        sys.stderr.write('\033[?7l\r\033[2K' + msg + '\033[?7h')
        sys.stderr.flush()
        self.progress_line_open = True
        self.progress_line_width = len(msg)

    def _write_jupyter_status(self, msg):
        try:
            self.jupyter_progress.write(msg)
        except Exception:
            self.logger.warning('%s', msg)
            return
        self.progress_line_open = True

    def write_inplace_status(self, msg):
        if self.stderr_is_tty():
            self._write_tty_status(msg)
            return
        if self.jupyter_progress is not None:
            self._write_jupyter_status(msg)

    def note_interrupt(self, reason, result_edges, num_points):
        self.end_progress_line()
        total = _total_edges(num_points)
        self.logger.warning(
            'MSTree formation: interruption started (%s) after %s edges %.2f%% of %s.',
            reason,
            result_edges,
            100. * result_edges / total,
            total)

    def prompt_limit(self, reason, result_edges, num_points):
        self.end_progress_line()
        total = _total_edges(num_points)
        self.logger.warning(
            'MSTree formation: %s limit reached after %s edges %.2f%% of %s. '
            'Ctrl+C to stop MST and continue labeling, or wait to keep building.',
            reason,
            result_edges,
            100. * result_edges / total,
            total)

    def prompt_progress(self, result_edges, num_points, elapsed):
        total = _total_edges(num_points)
        log_msg = (
            'MSTree formation: %s edges %.2f%% of %s after %.1fs. Still working. '
            'Ctrl+C to stop MST and continue labeling, or wait to keep building.'
            % (result_edges, 100. * result_edges / total, total, elapsed)
        )
        if self.stderr_is_tty():
            msg = 'MSTree formation: %s/%s edges (%.1f%%) %.1fs  Ctrl+C stops MST' % (
                result_edges, total, 100. * result_edges / total, elapsed)
            self.write_inplace_status(msg)
            return
        if self.jupyter_progress is not None:
            msg = (
                'MSTree formation: %s/%s edges (%.1f%%) %.1fs  Interrupt kernel stops MST'
                % (result_edges, total, 100. * result_edges / total, elapsed)
            )
            self.write_inplace_status(msg)
            return
        self.logger.warning('%s', log_msg)

    def knn_query_start(self, max_neighbors_search, num_points, show_progress):
        if num_points >= 1000:
            self.logger.warning(
                'kNN querying: %s neighbors for %s points. '
                'Ctrl+C after this step stops MST and continues labeling.',
                max_neighbors_search, num_points)
            if show_progress:
                self.write_inplace_status(
                    'kNN querying: blocking, no ticks until neighbors return')
        else:
            self.logger.info(
                'kNN querying: %s neighbors for %s points.',
                max_neighbors_search, num_points)

    def knn_query_done(self):
        self.end_progress_line()
        self.logger.info('kNN querying: done')

    def finish_mst(self, interrupted, interrupt_reason, result_edges, num_points,
                   edge_cases, max_neighbors_search):
        total = _total_edges(num_points)
        self.end_progress_line()
        if interrupted:
            if result_edges >= total:
                self.logger.warning(
                    'MSTree formation: interruption result: %s edges %.2f%% of %s (%s). Tree complete.',
                    result_edges,
                    100. * result_edges / total,
                    total,
                    interrupt_reason)
            else:
                self.logger.warning(
                    'MSTree formation: interruption result: %s edges %.2f%% of %s (%s). '
                    'Partial forest. Continuing to labeling.',
                    result_edges,
                    100. * result_edges / total,
                    total,
                    interrupt_reason)
        else:
            self.logger.info(
                'MSTree formation: %s edges %.2f%%. Done.',
                result_edges, 100. * result_edges / num_points)
        if result_edges != num_points - 1:
            self.logger.info(
                '%s not connected edges of %s. It is a forest. Try increasing max_neighbors(max_ranking) value %s for a better result.',
                num_points - 1 - result_edges, num_points - 1, max_neighbors_search)
        if max_neighbors_search < num_points - 1 and edge_cases != 0:
            self.logger.info(
                '%s edges with the max rank. Try increasing max_neighbors(max_ranking) value %s or pick the square mode (not available yet).',
                edge_cases, max_neighbors_search)
