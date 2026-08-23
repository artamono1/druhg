"""Tests for DRUHG SciPy linkage and dendrogram conversion."""
import numpy as np
import sklearn.datasets as datasets
from scipy.cluster.hierarchy import dendrogram, fcluster, is_valid_linkage, to_tree

from druhg import Buffer, DRUHG, labels_to_link_color_func, unionfind_to_linkage

_plot_graph = False
_not_fail_all = True


def test_unionfind_to_linkage_synthetic():
    # n=4: merge 0+1 -> uf 5, merge 2+3 -> uf 6, merge those -> uf 7
    n = 4
    parent = np.zeros(2 * n, dtype=np.intp)
    parent[0] = parent[1] = 5
    parent[2] = parent[3] = 6
    parent[5] = parent[6] = 7
    values = np.array([0.1, 0.2, 0.5])
    Z = unionfind_to_linkage(parent, n, values, 3)
    expected = np.array([
        [0., 1., 0.1, 2.],
        [2., 3., 0.2, 2.],
        [4., 5., 0.5, 4.],
    ])
    np.testing.assert_allclose(Z, expected)
    assert is_valid_linkage(Z, throw=True)


def test_unionfind_to_linkage_forest():
    # Two components: (0,1) and (2,3). One dummy merge should join them.
    n = 4
    parent = np.zeros(2 * n, dtype=np.intp)
    parent[0] = parent[1] = 5
    parent[2] = parent[3] = 6
    values = np.array([0.1, 0.2, -1.])
    Z = unionfind_to_linkage(parent, n, values, 2)
    assert is_valid_linkage(Z, throw=True)
    assert Z[2, 3] == 4
    assert Z[2, 2] > Z[1, 2]
    # Dummy join of scipy clusters 4 and 5 (the two pairs).
    assert set(Z[2, :2].astype(int)) == {4, 5}


def test_hierarchy_linkage():
    XX = np.array([[0., 0.], [1., 1.], [13., 2.], [14., 1.], [15., 2.]])
    dr = DRUHG(max_ranking=200, limitL=1, limitH=1000, verbose=False)
    dr.fit(XX)
    Z = dr.hierarchy(plot=False)

    assert Z.shape == (len(XX) - 1, 4)
    assert dr.linkage_ is Z
    assert is_valid_linkage(Z, throw=True)
    assert Z[-1, 3] == len(XX)
    assert np.all(Z[:, 2] >= 0)
    labels = fcluster(Z, t=2, criterion='maxclust')
    assert len(np.unique(labels)) == 2
    root = to_tree(Z)
    assert root.count == len(XX)


def test_hierarchy_iris_linkage():
    iris = datasets.load_iris()
    XX = iris['data']
    dr = DRUHG(max_ranking=50, verbose=False)
    dr.fit(XX)
    Z = dr.hierarchy(plot=False)
    assert Z.shape == (len(XX) - 1, 4)
    assert is_valid_linkage(Z, throw=True)
    assert Z[-1, 3] == len(XX)
    sizes = dr.buffers_[Buffer.SIZES.value]
    for i in range(dr.num_edges_):
        assert int(Z[i, 3]) == int(sizes[i])


def test_labels_to_link_color_func():
    XX = np.array([[0., 0.], [1., 1.], [13., 2.], [14., 1.], [15., 2.]])
    dr = DRUHG(max_ranking=200, limitL=1, limitH=1000, verbose=False)
    dr.fit(XX)
    Z = dr.hierarchy(plot=False)
    n = len(XX)
    color_of = labels_to_link_color_func(Z, dr.labels_)

    assert isinstance(color_of(n), str)
    # 0 and 1 merge first and share a label — that U-link is a cluster color.
    assert dr.labels_[0] == dr.labels_[1]
    assert color_of(n) != '#BFBFBF'
    # The final merge joins the two groups, so it is mixed/gray.
    assert color_of(n + n - 2) == '#BFBFBF'

    ddata = dendrogram(Z, no_plot=True, link_color_func=color_of, color_threshold=0)
    assert all(isinstance(c, str) for c in ddata['color_list'])


def test_plot_dendrogram(filename=None):
    if filename is None:
        filename = test_plot_dendrogram.__name__
    iris = datasets.load_iris()
    XX = iris['data']
    dr = DRUHG(max_ranking=50, limitH=int(len(XX) / 2), fix_outliers=1)
    dr.fit(XX)
    Z = dr.hierarchy(plot=False)
    assert Z is not None and Z.shape[0] == len(XX) - 1
    if _plot_graph:
        from matplotlib import pyplot as plt
        plt.close('all')
        dr.hierarchy(plot=True, labels=dr.labels_)
        plt.savefig(filename + '.png')
    assert _not_fail_all
