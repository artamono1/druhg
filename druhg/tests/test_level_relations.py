"""Point-to-point relations organized at each union-find level."""
import numpy as np

from druhg._druhg_displacement import (
    point_deltas_by_level,
    relation_magnitudes_by_level,
    relations_by_level,
)


def _pairs(level):
    return set(map(tuple, level.tolist()))


def test_every_point_relates_inside_its_level():
    # 0+1 -> label 5,  2+3 -> label 6,  then those two components -> label 7.
    n = 4
    parent = np.zeros(2 * n, dtype=np.intp)
    parent[0] = parent[1] = 5
    parent[2] = parent[3] = 6
    parent[5] = parent[6] = 7

    levels = relations_by_level(parent, n)

    assert _pairs(levels[0]) == {(0, 1), (1, 0)}
    assert _pairs(levels[1]) == {(2, 3), (3, 2)}
    # Opposite side of {0, 1} is represented by 2; opposite side of {2, 3} by 0.
    assert _pairs(levels[2]) == {(0, 2), (1, 2), (2, 0), (3, 0)}

    for level in levels:
        if len(level) == 0:
            continue
        sources = level[:, 0]
        members = set(sources.tolist())
        assert len(sources) == len(members)
        assert set(level[:, 1].tolist()) <= members
        assert np.all(level[:, 0] != level[:, 1])


def _mags(level):
    return {(int(row[0]), int(row[1])): row[2] for row in level}


def test_relation_is_clustered_edges_over_other_edges():
    # Two clustered pairs, then those pairs merge without clusterizing.
    # Pair {0, 1} has edge 1. Pair {2, 3} has edge 4.
    n = 4
    parent = np.zeros(2 * n, dtype=np.intp)
    parent[0] = parent[1] = 5
    parent[2] = parent[3] = 6
    parent[5] = parent[6] = 7
    values = np.array([1.0, 4.0, 10.0])
    clusters = np.array([-1, -1, 1], dtype=np.intp)

    levels = relation_magnitudes_by_level(parent, values, clusters, n)

    assert _mags(levels[0]) == {(0, 1): 0.0, (1, 0): 0.0}
    assert _mags(levels[1]) == {(2, 3): 0.0, (3, 2): 0.0}
    # {0, 1} -> {2, 3}: clustered 1 / edges 4. Opposite: 4 / 1.
    top = _mags(levels[2])
    assert top[(0, 2)] == np.float64(1.0 / 4.0)
    assert top[(1, 2)] == np.float64(1.0 / 4.0)
    assert top[(2, 0)] == np.float64(4.0)
    assert top[(3, 0)] == np.float64(4.0)


def test_unclustered_link_is_not_a_clustered_edge():
    # {0, 1} clusterizes on edge 2. That pair then links to {3, 4}
    # on edge 10 without clusterizing. {3, 4} clusterizes on edge 3.
    n = 5
    parent = np.zeros(2 * n, dtype=np.intp)
    parent[0] = parent[1] = 6          # level 0
    parent[3] = parent[4] = 7          # level 1
    parent[6] = parent[2] = 8          # level 2: pair {0, 1} + point 2
    parent[8] = parent[7] = 9          # level 3: that component + pair {3, 4}
    values = np.array([2.0, 3.0, 10.0, 5.0])
    clusters = np.array([-1, -1, 1, 1], dtype=np.intp)

    levels = relation_magnitudes_by_level(parent, values, clusters, n)
    # Component {0, 1, 2}: edges 2+10 = 12, clustered edges only the inner 2.
    # Pair {3, 4}: edges 3, all clustered.
    top = _mags(levels[3])
    assert np.isclose(top[(0, 3)], 2.0 / 3.0)
    assert np.isclose(top[(1, 3)], 2.0 / 3.0)
    assert np.isclose(top[(2, 3)], 2.0 / 3.0)
    assert np.isclose(top[(3, 0)], 3.0 / 12.0)
    assert np.isclose(top[(4, 0)], 3.0 / 12.0)


def _deltas(level):
    return {int(row[0]): row[1:] for row in level}


def test_point_delta_is_center_shift_plus_vectors_to_the_rest():
    # Pair {0, 1} edge 1, pair {2, 3} edge 4. Both clusterized.
    n = 4
    parent = np.zeros(2 * n, dtype=np.intp)
    parent[0] = parent[1] = 5
    parent[2] = parent[3] = 6
    parent[5] = parent[6] = 7
    values = np.array([1.0, 4.0, 10.0])
    clusters = np.array([-1, -1, 1], dtype=np.intp)
    coords = np.array([[0., 0.], [2., 0.], [0., 10.], [2., 10.]])

    levels = point_deltas_by_level(parent, values, clusters, coords, n)

    # No clustered edges on a lone point, so the center term is 0.
    # The connection is the two points, so the rest is the other point.
    leaf = _deltas(levels[0])
    assert np.allclose(leaf[0], coords[1] - coords[0])
    assert np.allclose(leaf[1], coords[0] - coords[1])

    c_own = coords[:2].mean(axis=0)
    c_other = coords[2:].mean(axis=0)
    # Relations toward {0, 1}: two points times 4/1. Toward {2, 3}: two points times 1/4.
    scale_own = (2 * 4.0) / (2 * 0.25)
    scale_other = (2 * 0.25) / (2 * 4.0)
    top = _deltas(levels[2])
    for i in (0, 1):
        center_delta = (c_other - c_own) * scale_own
        to_rest = sum(coords[j] - coords[i] for j in range(n) if j != i)
        assert np.allclose(top[i], center_delta + to_rest)
    for i in (2, 3):
        center_delta = (c_own - c_other) * scale_other
        to_rest = sum(coords[j] - coords[i] for j in range(n) if j != i)
        assert np.allclose(top[i], center_delta + to_rest)
