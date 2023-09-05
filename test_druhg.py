"""
Tests for DRUHG clustering algorithm
"""
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import sparse
from scipy import stats
import pytest  #delete __init__.py or it wont work


from druhg import (DRUHG,
                   druhg)

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from sklearn.cluster.tests.common import generate_clustered_data
import sklearn.datasets as datasets
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
from sklearn.metrics import adjusted_rand_score

from tempfile import mkdtemp
from functools import wraps

import warnings

moons, _ = datasets.make_moons(n_samples=50, noise=0.05)
blobs, _ = datasets.make_blobs(n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
X = np.vstack([moons, blobs])

_plot_graph = 0


def test_iris(filename=None):
    if filename is None:
        filename = test_iris.__name__
    iris = datasets.load_iris()
    XX = iris['data']
    # print(XX, type(XX))
    dr = DRUHG(max_ranking=50, verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'1'+'.png')

    ari = adjusted_rand_score(iris['target'], labels)
    print('iris ari', ari)
    assert (ari >= 0.50)
    # breaking biggest cluster
    labels = dr.relabel(limit1=0, limit2=int(len(XX)/2), fix_outliers=1)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename + '2' + '.png')

    ari = adjusted_rand_score(iris['target'], labels)
    print('iris ari', ari)
    assert (ari >= 0.85)

def test_plot_mst():
    iris = datasets.load_iris()
    XX = iris['data']
    dr = DRUHG(max_ranking=50)
    dr.fit(XX)
    dr.minimum_spanning_tree_.plot()

def test_plot_dendrogram(filename=None):
    if filename is None:
        filename = test_plot_dendrogram.__name__
    iris = datasets.load_iris()
    XX = iris['data']
    dr = DRUHG(max_ranking=50, limit2=int(len(XX)/2), fix_outliers=1)  #, limit1=0, limit2=int(len(XX)/2), fix_outliers=1)
    dr.fit(XX)
    if _plot_graph:
        plt.close('all')
        dr.single_linkage_.plot()
        plt.savefig(filename+'.png')
    assert (False)

def test_plot_one_dimension():
    iris = datasets.load_iris()
    XX = iris['data']
    XX = XX.reshape(XX.size,1)
    XX = np.array(sorted(XX))
    dr = DRUHG(max_ranking=50)
    dr.fit(XX)
    dr.minimum_spanning_tree_.plot()


def test_2and3():
    cons = 10.
    XX = [[0.,0.],[1.,1.],[cons+3.,2.],[cons+4.,1.],[cons+5.,2.]]
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limit1 = 1, limit2 = 1000, verbose=False)
    dr.fit(XX)
    # two clusters
    # assert (len(dr.parents_) == 2)
    print(dr.mst_)
    print(dr.mst_[6]*dr.mst_[7])

    labels = dr.labels_
    print('pairs', dr.mst_)
    print('labels', dr.labels_)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)

    # proper connection between two groups
    # assert (dr.mst_[6]*dr.mst_[7] == 2) # this is not working anymore

    assert (labels[0]==labels[1])
    assert (not all(x == labels[0] for x in labels))
    assert (labels[2] == labels[3] == labels[4])
    assert (labels[0] != labels[2])
    assert (n_clusters == 2)
    # assert (False)

def test_scaled_flatrangle(filename=None):
    if filename is None:
        filename = test_scaled_flatrangle.__name__
    for i in range(-5, 4):
        test_line(size=3, scale=10 ** i, filename=filename, all_clusters=True)

def test_rightangle(scale = 1., filename = None, all_clusters = True):
    if filename is None:
        filename = test_rightangle.__name__
    XX = [[0., 0.], [scale, 0], [0, scale]]
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limit1=1, limit2=1000, verbose=False)
    dr.fit(XX)
    # two clusters
    # assert (len(dr.parents_) == 2)

    labels = dr.labels_
    print('pairs', dr.mst_)
    print('labels', dr.labels_)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)

    assert (all(x == labels[0] for x in labels))

def test_scaled_rightangle(scale = 1., filename = None, all_clusters = True):
    if filename is None:
        filename = test_scaled_rightangle.__name__
    for i in range(-5, 4):
        test_rightangle(scale=10 ** i, filename=filename, all_clusters=True)

def test_equilateral(scale = 1., filename = None, all_clusters = True):
    if filename is None:
        filename = test_equilateral.__name__
    XX = [[scale, 0., 0.], [0., scale, 0], [0., 0., scale]]
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limit1=1, limit2=1000, verbose=False)
    dr.fit(XX)
    # two clusters
    # assert (len(dr.parents_) == 2)

    labels = dr.labels_
    print('pairs', dr.mst_)
    print('labels', dr.labels_)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)

    assert (all(x == labels[0] for x in labels))


def test_scaled_equilateral(scale = 1., filename = None, all_clusters = True):
    if filename is None:
        filename = test_scaled_equilateral.__name__
    for i in range(-5, 4):
        test_equilateral(scale=10 ** i, filename=filename, all_clusters=True)

def test_line(size=4, scale = 1., filename = None, all_clusters = False):
    if filename is None:
        filename = test_line.__name__
    XX = []
    for i in range(0,size):
        XX.append([0.,i*scale])
    XX = np.array(XX)

    dr = DRUHG(max_ranking=50, limit1=1, limit2=len(XX), verbose=False)
    dr.fit(XX)
    # s = 2*len(XX) - 2
    # starts somewhere in the middle
    # and grows one by one
    # that's why there are no clusters
    print('pairs', dr.mst_)
    print('labels', dr.labels_)
    np.set_printoptions(precision=2)
    print('values', dr.values_)
    # assert (len(dr.parents_)==0)
    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot(vary_line_width=None)
        plt.savefig(filename+'.png')
    labels = dr.labels_

    if all_clusters:
        assert (all(x == labels[0] for x in labels))
        return

    assert (not all(x == labels[0] for x in labels))
    assert (labels[0] == labels[len(labels)-1])
    assert (labels[1] == labels[len(labels)-2])
    assert (labels[0] != labels[1])
    # assert 0

def test_longline(filename=None):
    if filename is None:
        filename = test_longline.__name__
    test_line(size=100, filename=filename)

def test_fiveline(filename=None):
    if filename is None:
        filename = test_fiveline.__name__
    test_line(size=5, filename=filename)

# def test_hypersquare():
#     XX = []
#     size, scale = 6, 1.
#     for i1 in range(0, size):
#         for i2 in range(0, size):
#             for i3 in range(0, size):
#                 for i4 in range(0, size):
#                     for i5 in range(0, size):
#                         XX.append([i1*scale,i2*scale,i3*scale,i4*scale,i5*scale])
#     XX = np.array(XX)
#     dr = DRUHG(max_ranking=10, limit1=1, limit2=len(XX), verbose=False)
#     dr.fit(XX)
#     s = 2*len(XX) - 2
#     print(dr.mst_)
#     print(dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
#     labels = dr.labels_
#     n_clusters = len(set(labels)) - int(-1 in labels)
#     print('n_clusters', n_clusters)
#     print(dr.mst_)
#     print(XX)
#     print(dr.labels_)
#     assert (n_clusters==1)
#     labels = dr.relabel(limit1=1)
#     n_clusters = len(set(labels)) - int(-1 in labels)
#     print('n_clusters', n_clusters)
#     assert (n_clusters == 5)
#     assert (0==1)

def test_square(showplot=True, size=10, scale=5., filename=None):
    if filename is None:
        filename = test_square.__name__
    XX = []
    for i in range(0, size):
        for j in range(0, size):
            XX.append([scale*i,scale*j])

    XX = np.array(XX)
    np.random.shuffle(XX)

    dr = DRUHG(max_ranking=2000, algorithm='slow', limit1=1, limit2=len(XX), verbose=False)
    dr.fit(XX)

    if False and showplot and _plot_graph:
        plt.close('all')
        dr.single_linkage_.plot()
        plt.savefig(filename + 'plot' + '.png')

    s = 2*len(XX) - 2
    # print(dr.mst_)
    # print(dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)
    print(dr.mst_)
    # print(XX)
    print(dr.labels_)
    # assert (n_clusters==1)
    # labels = dr.relabel(limit1=1, limit2=size*2)
    n_clusters = len(set(labels)) - int(-1 in labels)
    # print('n_clusters', n_clusters)
    # print('pairs', dr.mst_)
    print('labels', dr.labels_)

    un, cn = np.unique(labels, return_counts=True)
    for i in range(0, len(un)):
        print('square', un[i], cn[i] )

    if showplot and _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot(vary_line_width = False)
        plt.savefig(filename+'.png')
    labels = dr.labels_
    un, cn = np.unique(labels, return_counts=True)
    print('uniques', un, cn)

    sorteds = np.argsort(cn)
    print('hello')
    print('hello2', un[sorteds[0]],un[sorteds[1]])
    # assert (False)
    dr.relabel(limit1=1, limit2=size*2)
    if showplot and _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'2'+'.png')
    assert (n_clusters >= 5)
    assert (cn[sorteds[-1]] == (size-2)**2)
    assert (un[sorteds[0]] == -1 and cn[sorteds[0]] == 4)
    for i in range(1,5):
        assert (cn[sorteds[i]] == size - 2)
    assert (False)

def test_scaled_square(filename=None):
    if filename is None:
        filename = test_scaled_square.__name__
    for i in range(-5, 4):
        test_square(scale=10**i, filename=filename)



def test_two_squares(filename=None):
    if filename is None:
        filename = test_two_squares.__name__
    XX = []
    size, scale = 6, 1
    for i in range(0, size):
        for j in range(0, size):
            XX.append([scale*i, scale*j])
            XX.append([2*size + scale*i, scale*j])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, verbose=False)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print(dr.mst_)
    print(dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'.png')
    assert (n_clusters==2)


def test_particles(filename=None):
    if filename is None:
        filename = test_particles.__name__
    XX = [[-0.51,1.5], [1.51,1.5]]
    for i in range(-3, 5):
        for j in range(-6, 1):
            XX.append([i,j])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, algorithm='slow', limit1=1, limit2=len(XX), verbose=False)
    dr.fit(XX)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'.png')

    s = 2*len(XX) - 2
    print(dr.mst_)
    print(dr.labels_)
    print(dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    # two points are further metrically but close reciprocally
    assert (dr.mst_[s-4]*dr.mst_[s-3] == 0)
    assert (dr.mst_[s-4] + dr.mst_[s-3] == 1)

def test_particles_big(filename=None):
    if filename is None:
        filename = test_particles_big.__name__
    XX = [[-0.51,1.5], [1.51,1.5]]
    for i in range(-5, 7):
        for j in range(-8, 1):
            XX.append([i,j])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, algorithm='slow', limit1=1, limit2=len(XX), verbose=False)
    dr.fit(XX)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'.png')

    s = 2*len(XX) - 2
    print(dr.mst_)
    print(dr.labels_)
    print(dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    # two points are further metrically but close reciprocally
    assert (dr.mst_[s-4]*dr.mst_[s-3] == 0)
    assert (dr.mst_[s-4] + dr.mst_[s-3] == 1)
    assert (False)

#
def test_bomb():
    XX = [[0.,1.],[0.,2.],[0.,3.],[0.,4.],[0.,5.]]
    for i in range(-3, 4):
        for j in range(-6, 1):
            XX.append([i,j])
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limit1=1, limit2=len(XX), verbose=False)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print(dr.mst_)
    x = 12
    labs = dr.labels_
    # fuse is separate
    print(labs)
    assert (labs[0]==labs[1]==labs[2]==labs[3])
    assert (np.count_nonzero(labs == labs[0]) == 4)

def test_t(filename=None):
    if filename is None:
        filename = test_t.__name__
    XX = []
    for i in range(1, 10):
        XX.append([0.,i])
    for j in range(-9, 10):
        XX.append([j,0.])
    XX = np.array(XX)
    # np.random.shuffle(XX)
    dr = DRUHG(max_ranking=200, algorithm='slow', verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)
    print('labels', len(labels), labels)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'.png')
    # t-center is an outlier too
    assert (n_clusters == 3)
    assert (np.count_nonzero(labels == -1) == 4)
    # assert (False)

#
def test_cross(filename=None):
    if filename is None:
        filename = test_cross.__name__
    XX = []
    for j in range(-10, 10):
        XX.append([j,0.])
    for i in range(1, 10):
        XX.append([0., i])
        XX.append([0., i - 10])
    XX = np.array(XX)
    # np.random.shuffle(XX)
    dr = DRUHG(max_ranking=200, verbose=False)
    dr.fit(XX)
    print(XX)
    # center is an outlier
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)
    print('labels', len(labels))

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'.png')

    assert (n_clusters == 4)
    assert (np.count_nonzero(labels == -1) == 5)
    # assert (False)


def test_cube(size=5, scale=1., showplot=True, filename=None):
    if filename is None:
        filename = test_cube.__name__

    XX = []
    for i in range(0, size):
        for j in range(0, size):
            for k in range(0, size):
                XX.append([i*scale,j*scale,k*scale])
    XX = np.array(XX)
    np.random.shuffle(XX)
    print(XX)
    # for i, x in enumerate(XX):
    #     print(i, x)
    dr = DRUHG(algorithm='slow', max_ranking=200, limit1=1, limit2=int(len(XX)/2), verbose=False)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print(dr.mst_)
    # assert (0==1)
    labels = dr.labels_
    unique, counts = np.unique(labels, return_counts=True)
    print(unique, counts)
    # labels = dr.relabel(limit1=1, limit2=len(XX)/2)

    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters, set(labels))
    # print('labels', labels)

    # sorteds = np.argsort(counts)
    # dr.relabel(limit1=1, limit2=len(XX), exclude=[unique[sorteds[0]], unique[sorteds[1]]])
    # labels = dr.labels_

    if showplot and _plot_graph:
        import seaborn as sns

        plt.close('all')
        fig = plt.figure()
        ax = Axes3D(fig)

        unique, counts = np.unique(labels, return_counts=True)
        sorteds = np.argsort(counts)
        s = len(sorteds)

        i = sorteds[s - 1]
        max_size = counts[i]
        if unique[i] == 0:
            max_size = counts[sorteds[s - 2]]

        color_map = {}
        palette = sns.color_palette('bright', s + 1)
        col = 0
        a = (1. - 0.3) / (max_size - 1)
        b = 0.3 - a
        while s:
            s -= 1
            i = sorteds[s]
            if unique[i] == 0:
                continue
            alpha = a * counts[i] + b
            color_map[unique[i]] = palette[col] + (alpha,)
            col += 1

        color_map[0] = (0., 0., 0., 0.15)
        colors = [color_map[x] for x in labels]
        s = [50 for x in labels]
        # ax = fig.add_subplot(111, projection='3d')
        ax.scatter(XX[:, 0:1], XX[:, 1:2], XX[:, 2:3], c=colors, s=s)

        for i in range(0,len(labels)-1):
            print('jajjajaja')
            print(i, XX[dr.mst_[2*i]])
            x1, y1, z1 = list(XX[dr.mst_[2*i]])
            x2, y2, z2 = list(XX[dr.mst_[2 * i + 1]])

            ax.plot([x1,x2], [y1,y2], [z1,z2], c=colors[dr.mst_[2*i]])

        plt.show()
        plt.savefig(filename+'1'+'.png')

    if showplot and _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'.png')

    assert (n_clusters == 1+6+12)
    unique, counts = np.unique(labels, return_counts=True)

    assert (-1 in unique)
    i = np.where(unique == -1)[0][0]

    assert(counts[i] == 8)
    counts = np.delete(counts,[i])

    counts = np.sort(counts)[::-1]
    print(counts, unique)

    assert ((size-2)**3==counts[0])

    for i in range(1, 1+6):
       assert (counts[i]==(size-2)**2)


    for i in range(7, 1+6+12):
        assert (counts[i] == size-2)

    # # assert (False)
    # # labels = dr.relabel(limit1=1)
    # labels = dr.relabel(limit1=1, limit2=len(XX))
    # print('out')
    # n_clusters = len(set(labels)) - int(-1 in labels)
    # print('n_clusters2', n_clusters, set(labels))
    # # print('labels2', labels)
    # assert (n_clusters == 1+6+12)

    assert (0==1)

def test_loop_cube():
    return
    k = 100
    while k!=0:
        # print('+++++++++++++PREVED+++++++++++++', k - 1000)
        test_cube(showplot=False)
        # assert (False)
        k-=1

def test_loop_square():
    return
    k = 100
    while k!=0:
        # print('+++++++++++++PREVED+++++++++++++', k - 1000)
        test_square(False)
        # assert (False)
        k-=1

def test_druhg_sparse():
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    sparse_X = sparse.csr_matrix((data, (row, col)), shape=(3, 3))

    dr = DRUHG()
    dr.fit(sparse_X)
    print('sparse labels', dr.labels_)

def test_druhg_distance_matrix(filename=None):
    if filename is None:
        filename = test_druhg_distance_matrix.__name__
    D = distance.squareform(distance.pdist(X))
    D /= np.max(D)

    print(D.shape)
    dr = druhg(D, metric='precomputed')
    print(dr)
    n_clusters = len(set(dr[0])) - int(-1 in dr[0])
    print(n_clusters)
    if _plot_graph:
        plt.close('all')
        dr = DRUHG(metric="precomputed").fit(D)
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'1'+'.png')
    assert(n_clusters==4)

    dr = DRUHG(metric="precomputed", limit1=5).fit(D)
    labels = dr.labels_
    print(labels)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print(n_clusters)
    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'2'+'.png')

    assert(n_clusters==4)

def test_moons_and_blobs():
    XX = X
    dr = DRUHG(max_ranking=50, verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    # expecting 4 clusters
    print(labels)

    assert (n_clusters == 4)
#
def test_hdbscan_clusterable_data(filename=None):
    if filename is None:
        filename = test_hdbscan_clusterable_data.__name__
    XX = np.load('druhg\\tests\\clusterable_data.npy')
    dr = DRUHG(max_ranking=50, verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    uniques, counts = np.unique(labels, True)
    print(uniques, counts)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print(n_clusters)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'.png')

    assert (n_clusters==6)

def test_three_blobs(filename=None):
    if filename is None:
        filename = test_three_blobs.__name__
    XX = np.load('druhg\\tests\\three_blobs.npy')
    dr = DRUHG(max_ranking=3550, verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    uniques, counts = np.unique(labels, True)
    print(uniques, counts)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print(n_clusters)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'.png')

    assert (n_clusters==3)

def test_synthetic_outliers(filename=None):
    if filename is None:
        filename = test_synthetic_outliers.__name__
    XX = pd.read_csv('druhg\\tests\\synthetic.csv', sep=',')
    XX.drop(u'outlier', axis=1, inplace=True)
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limit2=len(XX), exclude=[1978,1973], verbose=False)
    dr.fit(XX)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'.png')

    # values, counts = np.unique(dr.labels_, return_counts=True)
    # for i, v in enumerate(values):
    #     print(v, counts[i])

    labels = dr.labels_
    # labels = dr.relabel(limit1=1)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print(labels)
    print('n_clusters', n_clusters)
    assert (n_clusters==6)


def test_compound1(filename=None):
    if filename is None:
        filename = test_compound1.__name__
    XX = pd.read_csv('druhg/tests/Compound.csv', sep=',', header=None).drop(2, axis=1)
    XX = np.array(XX)
    dr = DRUHG(max_ranking=1550, limit1=3, limit2=len(XX), verbose=False)
    dr.fit(XX)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'1'+'.png')

    labels = dr.labels_
    # labels = dr.relabel(limit1=1)
    n_clusters = len(set(labels)) - int(-1 in labels)
    # np.save('labels_compound', labels)
    print('n_clusters', n_clusters, set(labels))
    exc = labels[398]
    # dr.relabel(limit1=3, limit2=len(XX), exclude=[exc])
    # labels = dr.labels_
    n_clusters2 = len(set(labels)) - int(-1 in labels)
    print(exc, n_clusters2, set(labels))

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'2'+'.png')

    exc2 = labels[398]
    dr.relabel(limit1=3, limit2=len(XX), exclude=[exc, exc2])
    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'3'+'.png')


    assert (n_clusters==4)
    assert (False)


def test_compound_egg(filename=None):
    if filename is None:
        filename = test_compound_egg.__name__
    XX = pd.read_csv('druhg/tests/Compound.csv', sep=',', header=None)
    XX = XX[(XX[2]==6) | (XX[2]==5)]
    XX = XX.drop(2, axis=1)
    XX = np.array(XX)
    dr = DRUHG(max_ranking=2000, limit1=3, limit2=len(XX), verbose=False)
    dr.fit(XX)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'.png')

    labels = dr.labels_
    # labels = dr.relabel(limit1=1)
    n_clusters = len(set(labels)) - int(-1 in labels)
    # np.save('labels_compound', labels)
    print('n_clusters', n_clusters, set(labels))
    # dr.relabel(limit1=3, limit2=len(XX), exclude=[exc])
    # labels = dr.labels_
    n_clusters2 = len(set(labels)) - int(-1 in labels)
    print( n_clusters2, set(labels))

    assert (n_clusters==2)
    assert (False)

def test_copycat():
    XX = [[0]]*100
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, limit1=1, verbose=False)
    dr.fit(XX)
    print(dr.mst_[0], dr.mst_[1])
    print('pairs', dr.mst_)
    print('labels', dr.labels_)
    labels = dr.labels_
    assert (all(x == labels[0] for x in labels))

def test_copycats(): # should fail until weights are made
    XX = np.concatenate( ([[0]]*100, [[1]]*100))
    dr = DRUHG(max_ranking=10, limit1=1, verbose=False)
    dr.fit(XX)
    print(dr.mst_[0], dr.mst_[1])
    print('pairs', dr.mst_)
    print('labels', dr.labels_)
    labels = dr.labels_
    assert (not all(x == labels[0] for x in labels))
    assert (labels[0] != labels[-1])

    uniques, counts = np.unique(labels, True)
    print(uniques, counts)
    n_clusters = len(set(labels)) - int(-1 in labels)
    assert (n_clusters==2)

def test_copycats2():
    XX = np.concatenate( ([[0]]*10, [[1]]*10))
    dr = DRUHG(max_ranking=200, limit1=1, verbose=False)
    dr.fit(XX)
    print(dr.mst_[0], dr.mst_[1])
    print('pairs', dr.mst_)
    print('labels', dr.labels_)
    labels = dr.labels_
    assert (not all(x == labels[0] for x in labels))
    assert (labels[0] != labels[-1])

    uniques, counts = np.unique(labels, True)
    print(uniques, counts)
    n_clusters = len(set(labels)) - int(-1 in labels)
    assert (n_clusters==2)

def test_copycats3(): # should fail until weights are made
    XX = np.concatenate( ([[0]]*100, [[1]]*100, [[2]]*5) )
    dr = DRUHG(max_ranking=10, limit1=1, limit2=250, verbose=False)
    dr.fit(XX)
    print(dr.mst_[0], dr.mst_[1])
    print('pairs', dr.mst_)
    print('labels', dr.labels_)
    labels = dr.labels_
    assert (not all(x == labels[0] for x in labels))
    assert (labels[0] != labels[-1])

    uniques, counts = np.unique(labels, True)
    print(uniques, counts)
    n_clusters = len(set(labels)) - int(-1 in labels)
    assert (n_clusters==3)
#
# def test_speed0():
#     XX = []
#     size = 5
#     for i in range(0, size):
#         for j in range(0, size):
#             for k in range(0, size):
#                 XX.append([i, j, k])
#
#     XX = pd.read_csv('druhg\\tests\\synthetic.csv', sep=',')
#     XX.drop(u'outlier', axis=1, inplace=True)
#
#     return np.array(XX)
#
#
# def test_speed1(k=100):
#     XX = test_speed0()
#     while k!=0:
#         k -= 1
#         np.random.shuffle(XX)
#         dr = DRUHG(algorithm='fast', max_ranking=200, limit1=1, limit2=int(len(XX) / 2), verbose=False)
#         dr.fit(XX)
#
# def test_speed2(k=100):
#     XX = test_speed0()
#     while k!=0:
#         k -= 1
#         np.random.shuffle(XX)
#         dr = DRUHG(algorithm='fastminus', max_ranking=200, limit1=1, limit2=int(len(XX) / 2), verbose=False)
#         dr.fit(XX)
#
# def test_speed3(k=100):
#     XX = test_speed0()
#     while k!=0:
#         k -= 1
#         np.random.shuffle(XX)
#         dr = DRUHG(algorithm='slow+', max_ranking=200, limit1=1, limit2=int(len(XX) / 2), verbose=False)
#         dr.fit(XX)
#
# def test_speed4(k=100):
#     XX = test_speed0()
#     while k!=0:
#         k -= 1
#         np.random.shuffle(XX)
#         dr = DRUHG(algorithm='slow', max_ranking=200, limit1=1, limit2=int(len(XX) / 2), verbose=False)
#         dr.fit(XX)
#
# def test_speed5(k=100):
#     XX = test_speed0()
#     while k!=0:
#         k -= 1
#         np.random.shuffle(XX)
#         dr = DRUHG(algorithm='slow', max_ranking=200, limit1=1, limit2=int(len(XX) / 2), verbose=False)
#         dr.fit(XX)



def test_triangle(scale = 1., filename = None, all_clusters = True):
    if filename is None:
        filename = test_rightangle.__name__
    XX = [[0., 0.], [scale, 0], [0, scale*10.]]
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limit1=1, limit2=1000, verbose=False)
    dr.fit(XX)

    # two clusters
    # assert (len(dr.parents_) == 2)

    labels = dr.labels_
    print('pairs', dr.mst_)
    print('labels', dr.labels_)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)

    # assert (all(x == labels[0] for x in labels))

    dr.relabel(coords_arr=XX)
    assert (False)


def test_chameleon(filename=None):
    if filename is None:
        filename = test_chameleon.__name__
    XX = pd.read_csv('druhg\\tests\\chameleon.csv', sep='\t', header=None)
    XX = np.array(XX)
    dr = DRUHG(max_ranking=4200, limit1 = 1, limit2=len(XX), verbose=False, algorithm='slow')
    dr.fit(XX)
    labels = dr.labels_
    # labels = dr.relabel(limit1=1)
    values, counts = np.unique(labels, return_counts=True)
    n_clusters = 0
    for i, c in enumerate(counts):
        print(i, c, values[i])
        if c > 500 and values[i] >= 0:
            n_clusters += 1
    print('n_clusters', n_clusters)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'1'+'.png')

    dr = DRUHG(max_ranking=1200, limit1=1, limit2=int(len(XX)/4), exclude=[], verbose=False, algorithm='slow')
    dr.fit(XX)

    values, counts = np.unique(dr.labels_, return_counts=True)
    n_clusters2 = 0
    for i, c in enumerate(counts):
        print(i, c, values[i])
        if c > 500 and values[i] >= 0:
            n_clusters2 += 1
    print('n_clusters2', n_clusters2)

    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'2'+'.png')

    if _plot_graph:
        plt.close('all')
        dr.single_linkage_.plot()
        plt.savefig(filename +'plot'+ '.png')

    exc = dr.labels_[3024]
    dr.relabel(limit1=3, limit2=int(len(XX)/4), exclude=[exc])
    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'3'+'.png')

    exc = dr.labels_[3024]
    dr.relabel(limit1=3, limit2=int(len(XX)/8), exclude=[exc])
    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename+'4'+'.png')

    assert (n_clusters2==6)
    assert (n_clusters==6)
    assert False


# def test_boba(filename=None):
#     if filename is None:
#         filename = test_boba.__name__
#     XX = pd.read_csv('druhg\\tests\\chameleon.csv', sep='\t', header=None)
#     XX = np.array(XX)
#     lim = 1.
#     while lim >= 0:
#
#         dr = DRUHG(max_ranking=4200, limit1 = 1, limit2=lim, verbose=False, algorithm='slow')
#         dr.fit(XX)
#         if _plot_graph:
#             plt.close('all')
#             dr.minimum_spanning_tree_.plot()
#             plt.savefig(filename+str(lim)+'.png')
#         lim -= 0.05

def test_amt(filename=None):
    if filename is None:
        filename = test_amt.__name__
    XX = pd.read_csv('druhg\\tests\\chameleon.csv', sep='\t', header=None)
    XX = np.array(XX)
    dr = DRUHG(max_ranking=4200, limit1=1, limit2=0.25, verbose=False, algorithm='slow')
    dr.fit(XX)
    if _plot_graph:
        plt.close('all')
        dr.minimum_spanning_tree_.plot()
        plt.savefig(filename + '.png')
    assert (False)