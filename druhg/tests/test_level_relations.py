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

    # Two points: each clusterizes, so both parts of the relation are the connection edge.
    assert _mags(levels[0]) == {(0, 1): 1.0, (1, 0): 1.0}
    assert _mags(levels[1]) == {(2, 3): 1.0, (3, 2): 1.0}
    # Connection 10 joins clustered edges 1 and 4.
    # {0, 1} -> {2, 3}: (1+10) / (4+10). Opposite: (4+10) / (1+10).
    top = _mags(levels[2])
    assert np.isclose(top[(0, 2)], 11.0 / 14.0)
    assert np.isclose(top[(1, 2)], 11.0 / 14.0)
    assert np.isclose(top[(2, 0)], 14.0 / 11.0)
    assert np.isclose(top[(3, 0)], 14.0 / 11.0)


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
    # Connection 5. {0, 1, 2} does not clusterize, so 5 stays out of its numerator.
    # {3, 4} does, so 5 is added there. Both denominators receive 5.
    top = _mags(levels[3])
    assert np.isclose(top[(0, 3)], 2.0 / 8.0)
    assert np.isclose(top[(1, 3)], 2.0 / 8.0)
    assert np.isclose(top[(2, 3)], 2.0 / 8.0)
    assert np.isclose(top[(3, 0)], 8.0 / 17.0)
    assert np.isclose(top[(4, 0)], 8.0 / 17.0)


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

    # Both points clusterize onto edge 1, so each relation is 1.
    # Center shift and the vector to the other point are the same vector.
    leaf = _deltas(levels[0])
    assert np.allclose(leaf[0], 2 * (coords[1] - coords[0]))
    assert np.allclose(leaf[1], 2 * (coords[0] - coords[1]))

    c_own = coords[:2].mean(axis=0)
    c_other = coords[2:].mean(axis=0)
    # Connection 10. Relations (1+10)/(4+10) and (4+10)/(1+10).
    rel_own = 11.0 / 14.0
    rel_other = 14.0 / 11.0
    scale_own = (2 * rel_other) / (2 * rel_own)
    scale_other = (2 * rel_own) / (2 * rel_other)
    top = _deltas(levels[2])
    for i in (0, 1):
        center_delta = (c_other - c_own) * scale_own
        to_rest = sum(coords[j] - coords[i] for j in range(n) if j != i)
        assert np.allclose(top[i], center_delta + to_rest)
    for i in (2, 3):
        center_delta = (c_own - c_other) * scale_other
        to_rest = sum(coords[j] - coords[i] for j in range(n) if j != i)
        assert np.allclose(top[i], center_delta + to_rest)
