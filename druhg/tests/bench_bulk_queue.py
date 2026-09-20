"""Sweep bulk_queue append vs appendleft: MST-loop time and refresh work."""
import os
import urllib.request

import numpy as np
import pandas as pd
import sklearn.datasets as datasets
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from druhg import DRUHG

_HERE = os.path.dirname(os.path.abspath(__file__))
_CLUSTERABLE = os.path.join(_HERE, 'clusterable_data.npy')
_CLUSTERABLE_URL = (
    'https://github.com/scikit-learn-contrib/hdbscan/raw/master/'
    'notebooks/clusterable_data.npy'
)
_WARMUP = 1
_REPEATS = 50


def _ensure_clusterable():
    if os.path.isfile(_CLUSTERABLE):
        return _CLUSTERABLE
    urllib.request.urlretrieve(_CLUSTERABLE_URL, _CLUSTERABLE)
    return _CLUSTERABLE


def _datasets():
    moons, _ = datasets.make_moons(n_samples=50, noise=0.05, random_state=0)
    blobs, _ = datasets.make_blobs(
        n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25,
        random_state=0)
    noisy_moons, _ = datasets.make_moons(n_samples=1500, noise=0.05, random_state=0)
    compound = np.array(
        pd.read_csv(os.path.join(_HERE, 'Compound.csv'), sep=',', header=None)
        .drop(2, axis=1)
    )
    return [
        ('iris', datasets.load_iris()['data'], dict(max_ranking=50)),
        ('moons_and_blobs', np.vstack([moons, blobs]), dict(max_ranking=50)),
        ('two_moons', noisy_moons, dict(max_ranking=1000, size_range=[1, 1])),
        (
            'hdbscan_clusterable',
            np.load(_ensure_clusterable()),
            dict(max_ranking=1000, size_range=[0.05, 0.25]),
        ),
        ('compound', compound, dict(max_ranking=1550, limitL=3)),
    ]


def _combo_label(push):
    letters = ''.join('L' if p else 'A' for p in push)
    extra = ' (default)' if push == (0, 1, 1) else ''
    return '%s%s%s%s' % (letters[0], letters[1], letters[2], extra)


def _fmt(x, digits=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '-'
    return ('%%.%df' % digits) % x


def _fit_once(X, params, push, scores=None, perf=None):
    if scores is None:
        scores = []
    if perf is None:
        perf = {}
    dr = DRUHG(
        verbose=False,
        do_tree_only=True,
        progress_interval=0,
        bulk_queue_push=push,
        heap_scores=scores,
        mst_perf=perf,
        **params
    )
    dr.fit(X)
    return dr, scores, perf


def plot_dataset(name, series, rows, out_dir):
    labels = [r['label'] for r in rows]
    means = np.array([r['t_mean'] for r in rows]) * 1000.0
    stds = np.array([r['t_std'] for r in rows]) * 1000.0
    steps = np.array([r['refresh_steps'] for r in rows], dtype=np.float64)
    x = np.arange(len(labels))

    fig = plt.figure(figsize=(11.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.28)
    ax_score = fig.add_subplot(gs[0, :])
    ax_time = fig.add_subplot(gs[1, 0])
    ax_work = fig.add_subplot(gs[1, 1])

    for label, scores in series:
        y = np.asarray(scores, dtype=np.float64)
        ev = np.arange(1, y.size + 1)
        kw = dict(linewidth=2.2) if 'default' in label else dict(linewidth=1.2, alpha=0.9)
        ax_score.plot(ev, y, label=label, **kw)
    ax_score.set_title('%s  1/size' % name)
    ax_score.set_xlabel('event')
    ax_score.set_ylabel('1 / heap.size')
    ax_score.legend(loc='upper right', fontsize=8, frameon=False)
    ax_score.grid(True, alpha=0.3)

    ax_time.bar(x, means, yerr=stds, capsize=3, color='#4c78a8')
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax_time.set_ylabel('main-loop time (ms)')
    ax_time.set_title('wall time')
    ax_time.grid(True, axis='y', alpha=0.3)

    ax_work.bar(x, steps, color='#f58518')
    ax_work.set_xticks(x)
    ax_work.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax_work.set_ylabel('refresh steps')
    ax_work.set_title('heap-refresh work')
    ax_work.grid(True, axis='y', alpha=0.3)

    path = os.path.join(out_dir, 'bench_bulk_queue_%s.png' % name)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print('wrote %s' % path, flush=True)
    return path


def run():
    combos = [(p1, p2, p3) for p1 in (0, 1) for p2 in (0, 1) for p3 in (0, 1)]
    tables = []
    plot_paths = []
    for name, X, params in _datasets():
        rows = []
        series = []
        print('dataset %s  n=%s' % (name, len(X)), flush=True)
        for push in combos:
            label = _combo_label(push)
            for _ in range(_WARMUP):
                _fit_once(X, params, push)
            times = []
            last_perf = None
            last_scores = None
            last_dr = None
            for _ in range(_REPEATS):
                scores = []
                perf = {}
                dr, scores, perf = _fit_once(X, params, push, scores, perf)
                times.append(perf.get('t_loop', np.nan))
                last_perf = perf
                last_scores = scores
                last_dr = dr
            times = np.asarray(times, dtype=np.float64)
            n_events = len(last_scores)
            refresh = int(last_perf.get('refresh_steps', 0))
            recip = int(last_perf.get('reciprocity', 0))
            pops = int(last_perf.get('queue_pops', 0))
            merges = int(last_perf.get('merges', 0))
            finished = last_dr.num_edges_ >= (len(X) - 1)
            row = dict(
                label=label,
                n_events=n_events,
                t_mean=float(np.mean(times)),
                t_std=float(np.std(times, ddof=1) if times.size > 1 else 0.0),
                refresh_steps=refresh,
                reciprocity=recip,
                queue_pops=pops,
                merges=merges,
                steps_per_event=(refresh / n_events) if n_events else np.nan,
                net_score=float(np.sum(last_scores)) if n_events else 0.0,
                finished=finished,
            )
            rows.append(row)
            series.append((label, list(last_scores)))
            print('  %s  t=%.3f±%.3f ms  refresh=%d  recip=%d' % (
                label, row['t_mean'] * 1000.0, row['t_std'] * 1000.0,
                refresh, recip), flush=True)
            if not finished:
                print('  %s did not finish MST (%s edges)' % (
                    label, last_dr.num_edges_), flush=True)
        tables.append((name, rows))
        plot_paths.append(plot_dataset(name, series, rows, _HERE))
    return tables, plot_paths


def render(tables):
    lines = []
    for name, rows in tables:
        lines.append('## %s' % name)
        lines.append('')
        lines.append(
            '| combo | n | t_loop ms | ± | refresh | recip | '
            'q_pops | merges | steps/ev | net 1/size |'
        )
        lines.append(
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
        )
        for r in rows:
            lines.append(
                '| %s | %d | %s | %s | %d | %d | %d | %d | %s | %s |' % (
                    r['label'], r['n_events'],
                    _fmt(r['t_mean'] * 1000.0, 3),
                    _fmt(r['t_std'] * 1000.0, 3),
                    r['refresh_steps'], r['reciprocity'],
                    r['queue_pops'], r['merges'],
                    _fmt(r['steps_per_event'], 2),
                    _fmt(r['net_score'], 4),
                ))
        lines.append('')
        lines.append(
            't_loop is main-loop wall time only (kNN excluded), '
            'mean±sample std over %d runs after %d warmup. '
            'refresh = `_refresh` inner-loop trips; recip = '
            '`_evaluate_reciprocity` calls from `_refresh`.' % (
                _REPEATS, _WARMUP)
        )
        lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    tables, _paths = run()
    print(render(tables))
