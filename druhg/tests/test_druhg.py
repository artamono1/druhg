"""
Tests for DRUHG clustering algorithm
"""
# pytest -k "test_name"
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import sparse
from scipy import stats
import pytest  #delete __init__.py or it wont work
import time


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
_test_extra_visualisation = 0
_not_fail_all = True

def test_minitest(filename=None):
    if filename is None:
        filename = test_minitest.__name__
    XX = np.array([[ 9.82287700e-01, -4.91767541e-01],
       [ 7.79582720e-01,  6.08296786e-01],
       [-8.88193604e-01,  4.43296052e-01],
       [ 7.78691236e-02,  1.05579835e-01],
       [ 7.05833295e-01, -4.49687952e-01],
       [ 9.08054027e-01,  3.82957634e-01],
       [-8.44357575e-01,  4.76983140e-01],
       [-6.96863884e-01,  7.50031808e-01],
       [ 4.39031497e-01,  8.96900018e-01],
       [ 2.06198639e+00,  4.95933426e-01],
       [ 2.00697002e+00,  2.51940542e-01],
       [ 8.86671509e-01,  5.33899191e-01],
       [ 1.01472887e+00,  1.77887023e-01],
       [-9.99361908e-01,  8.14805761e-02],
       [ 1.66364569e+00, -2.72722628e-01],
       [ 1.45444706e-01, -3.60310809e-04],
       [ 2.47847446e-02,  2.47518860e-01],
       [ 5.33608226e-01, -4.16318654e-01],
       [-1.02367914e+00,  2.03681301e-01],
       [ 1.72252995e+00, -1.73532360e-01],
       [ 1.19371766e-01,  9.66962174e-01],
       [-9.96385260e-01,  1.51193473e-03],
       [ 1.97170679e+00,  1.10284595e-01],
       [-4.04580401e-03,  3.53573375e-01],
       [ 6.27338990e-01, -4.64420966e-01],
       [ 1.71681386e-01, -1.17028174e-01],
       [ 1.26919122e+00, -4.46221578e-01],
       [ 6.14585010e-02,  1.00584911e+00],
       [ 1.09577989e+00, -5.01458976e-01],
       [ 3.96411149e-01, -2.98249260e-01],
       [ 1.88598729e+00,  7.33595903e-03],
       [ 1.96974012e+00,  3.38327527e-01],
       [-5.08398106e-01,  8.60840352e-01],
       [-4.00736795e-01,  8.89316463e-01],
       [ 1.77070595e+00, -1.36976988e-01],
       [ 1.49886531e+00, -3.96515486e-01],
       [ 7.22042880e-01,  7.04074310e-01],
       [ 8.49944965e-01, -5.34124448e-01],
       [-9.09817816e-02,  1.03513561e+00],
       [ 5.14565924e-01,  7.87485818e-01],
       [ 4.02652800e-01,  9.37234351e-01],
       [-2.91295702e-01,  9.02204623e-01],
       [ 2.87744167e-01, -2.31116381e-01],
       [-3.31091977e-02,  4.88892172e-01],
       [ 2.41871950e-01,  9.92840358e-01],
       [ 1.06415165e+00, -1.65735096e-02],
       [ 1.37323752e+00, -4.65626874e-01],
       [-6.29552626e-01,  7.87828922e-01],
       [-8.27252851e-01,  5.81983413e-01],
       [ 1.00280073e+00,  2.68063630e-01],
       [-7.83496058e-01,  2.18322479e+00],
       [-9.92609950e-01,  2.02124136e+00],
       [-9.81354504e-01,  2.29182851e+00],
       [-4.98266423e-01,  1.78721081e+00],
       [-1.08969024e+00,  2.32889766e+00],
       [-1.08081235e+00,  1.98253144e+00],
       [-1.09687394e+00,  1.95677593e+00],
       [-1.00962708e+00,  1.98160320e+00],
       [-5.23444712e-01,  2.05973030e+00],
       [-5.27378118e-01,  2.01577480e+00],
       [-1.01076274e+00,  2.26189991e+00],
       [-1.07307453e+00,  1.59463042e+00],
       [-3.71408468e-02,  2.36246794e+00],
       [-4.78906511e-01,  2.09108831e+00],
       [-1.36496960e+00,  1.75639407e+00],
       [-9.63722898e-01,  1.77177389e+00],
       [-8.57835037e-01,  2.85261536e+00],
       [-9.01706144e-01,  1.96899908e+00],
       [-7.83591885e-01,  1.97077401e+00],
       [-7.53906201e-01,  2.01930595e+00],
       [-1.06416195e+00,  2.09696002e+00],
       [-1.06425933e+00,  1.92864353e+00],
       [-5.59921254e-01,  1.92873234e+00],
       [-2.93844091e-01,  2.13445936e+00],
       [-1.08927720e+00,  2.22384623e+00],
       [-5.47674986e-01,  2.56744321e+00],
       [-8.32079069e-01,  1.91501237e+00],
       [-8.64079731e-01,  2.02405299e+00],
       [-9.43370303e-02,  1.53264890e+00],
       [-8.46705805e-01,  2.08845880e+00],
       [-7.64153884e-01,  1.95806975e+00],
       [-7.97244563e-01,  1.93409856e+00],
       [-6.11964372e-01,  1.83346591e+00],
       [-7.97508190e-01,  2.18510057e+00],
       [-5.06448969e-01,  1.99409096e+00],
       [-6.53243323e-01,  2.52892190e+00],
       [-8.73821576e-01,  1.80817140e+00],
       [-7.03121501e-01,  1.95331014e+00],
       [-9.83658309e-01,  2.33860746e+00],
       [-1.01683507e+00,  1.94204773e+00],
       [-4.87343036e-01,  1.56494127e+00],
       [-7.79832602e-01,  2.07966975e+00],
       [-4.46877995e-01,  1.45948349e+00],
       [-3.51769879e-01,  1.59343158e+00],
       [-1.34234725e+00,  2.31765803e+00],
       [-9.56682748e-01,  1.53054479e+00],
       [-6.10630098e-01,  1.87378092e+00],
       [-7.42290479e-01,  1.84433031e+00],
       [-8.39334432e-01,  1.81906771e+00],
       [-9.27731599e-01,  2.29639091e+00],
       [ 1.42071742e+00,  3.30486324e+00],
       [ 1.55862485e+00,  1.30863477e+00],
       [ 1.24972581e+00,  2.42336599e+00],
       [ 5.33685073e-01,  2.59954396e+00],
       [ 7.68000568e-01,  2.08111755e+00],
       [ 2.35295779e+00,  3.68309710e+00],
       [ 1.54597101e+00,  1.84281151e+00],
       [ 8.12131013e-01,  3.29007476e+00],
       [ 7.94516457e-01,  2.41783502e+00],
       [ 7.98220091e-01,  9.82325962e-01],
       [ 4.41618515e-01,  2.86576268e+00],
       [ 2.28132907e+00,  2.62941621e+00],
       [ 1.30744465e+00,  9.21494187e-01],
       [ 2.89402898e+00,  3.07162590e+00],
       [ 2.59828872e+00,  3.48249198e+00],
       [ 2.39156834e-01,  1.13332844e+00],
       [ 1.28134650e+00,  1.69771109e+00],
       [ 2.47539905e+00,  3.84649616e+00],
       [ 8.47050787e-01,  2.07370998e+00],
       [ 1.70839331e+00,  2.03061588e+00],
       [ 1.86776491e+00,  2.64855805e+00],
       [ 1.82506092e+00,  2.65118340e+00],
       [ 7.38734334e-01,  2.46762804e+00],
       [ 2.05216302e+00,  3.09939872e-01],
       [ 2.08265157e+00,  2.68029197e+00],
       [ 5.48953797e-01,  1.63082304e+00],
       [ 6.66897628e-01,  2.19248368e+00],
       [ 7.21620016e-02,  1.92199689e+00],
       [ 9.35013031e-01,  1.14139910e+00],
       [ 2.10701578e+00,  1.56207334e+00],
       [ 1.32685731e+00,  3.02292533e+00],
       [-3.12399518e-01,  4.03870510e+00],
       [ 2.45342595e+00,  3.36391798e+00],
       [ 7.40975862e-01,  1.28206197e+00],
       [ 7.23212885e-01,  1.25423311e+00],
       [ 3.20586402e+00,  2.00176070e+00],
       [ 5.44831804e-01,  8.42644367e-01],
       [ 4.91531506e-01,  8.54574838e-01],
       [ 1.09246967e+00,  2.28977328e+00],
       [ 1.34148473e+00,  3.59316431e+00],
       [ 1.47390241e+00,  2.68453871e+00],
       [ 2.04938161e+00,  3.00980325e-01],
       [ 4.74061748e-01,  1.82517865e+00],
       [ 1.38255750e+00,  1.49441056e+00],
       [-1.81396229e-01,  1.67608102e+00],
       [ 1.61734753e+00,  1.62344100e+00],
       [ 2.27547713e+00,  3.02798895e+00],
       [ 1.26536052e+00,  2.74417246e+00],
       [ 2.22312311e+00,  1.59609013e+00],
       [ 7.16857960e-01,  1.30979190e+00],
       [ 1.69435113e+00,  2.58428292e+00],
       [ 2.56004668e+00,  1.90043678e+00],
       [ 4.00593052e-01,  2.07202927e+00],
       [ 1.09514002e+00,  8.32401990e-01],
       [ 3.44335366e-01,  2.12210586e+00],
       [ 1.66737772e+00,  2.58321428e+00],
       [ 6.30088239e-01,  3.34926791e+00],
       [ 1.71452800e+00,  4.05206484e+00],
       [ 1.56123347e+00,  2.73666162e+00],
       [ 2.25416915e+00,  3.33381046e+00],
       [ 1.92722848e+00,  2.84539606e+00],
       [ 1.94938390e+00,  2.02922499e+00],
       [ 1.33211707e+00,  7.77595470e-01],
       [ 1.44282596e+00,  3.03895516e+00],
       [ 2.53818450e+00,  2.09297521e+00],
       [ 7.06781846e-01,  2.88379784e+00],
       [ 8.75255708e-01,  2.19440080e+00],
       [ 1.03602323e+00,  1.95794841e+00],
       [ 2.05569358e+00,  2.81113483e+00],
       [ 1.29284658e+00,  3.66966168e+00]])
    dr = DRUHG(max_ranking=50, verbose=False,
               do_edges=True, size_range=[1, 1])
               # do_edges=True, limitL=25, limitH=0.3, fix_outliers=1)

    dr.fit(XX)
    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'1'+'.png')
        # dr.plot()

    # if _plot_graph and _test_extra_visualisation:
    #     plt.close('all')
    #     dr.single_linkage_.plot(dr.labels_)
    #
    #     plt.savefig(filename +'plot'+ '.png')
    assert _not_fail_all

def test_plotcluster(filename=None):
    if filename is None:
        filename = test_plotcluster.__name__
    # XX = datasets.load_iris()['data']
    XX = np.array(pd.read_csv('druhg/tests/chameleon.csv', sep='\t', header=None))
    dr = DRUHG(max_ranking=50, verbose=False, do_edges=True)
    dr.fit(XX)
    if _test_extra_visualisation:
        plt.close('all')
        # plt.style.use('dark_background')
        # dr.plot(dr.labels_)
        dr.plot()
        plt.savefig(filename+'1'+'.png')
    assert _not_fail_all

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
        dr.plot(labels)
        plt.savefig(filename+'1'+'.png')

    ari = adjusted_rand_score(iris['target'], labels)
    print('iris ari', ari)
    # assert (ari >= 0.50)
    # breaking biggest cluster
    labels = dr.relabel(limitL=0, limitH=int(len(XX)/2), fix_outliers=True)

    if _plot_graph:
        plt.close('all')
        dr.plot(labels)
        plt.savefig(filename + '2' + '.png')

    ari = adjusted_rand_score(iris['target'], labels)
    print('iris ari', ari)
    assert (ari >= 0.85)
    assert _not_fail_all

def test_plot_mst():
    iris = datasets.load_iris()
    XX = iris['data']
    dr = DRUHG(max_ranking=50, do_edges=True)
    dr.fit(XX)
    dr.plot(dr.labels_)
    assert _not_fail_all

def test_plot_dendrogram(filename=None):
    if not _test_extra_visualisation:
        return
    if filename is None:
        filename = test_plot_dendrogram.__name__
    iris = datasets.load_iris()
    XX = iris['data']
    dr = DRUHG(max_ranking=50, limitH=int(len(XX)/2), fix_outliers=1)  #, limitL=0, limitH=int(len(XX)/2), fix_outliers=1)
    dr.fit(XX)
    if _plot_graph:
        plt.close('all')
        dr.single_linkage_.plot(dr.labels_)
        plt.savefig(filename+'.png')
    assert _not_fail_all

def test_plot_one_dimension(): # пока не работает, нельзя дерево посторить!?
    iris = datasets.load_iris()
    XX = iris['data']
    XX = XX.flatten()
    XX = np.array(sorted(XX))
    dr = DRUHG(max_ranking=50)
    dr.fit(XX)
    dr.plot(dr.labels_)
    assert _not_fail_all

def test_2and3():
    cons = 10.
    XX = [[0.,0.],[1.,1.],[cons+3.,2.],[cons+4.,1.],[cons+5.,2.]]
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limitL = 1, limitH = 1000, verbose=False, do_edges=True)
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
    assert _not_fail_all

def test_flatrangle_scaled(filename=None):
    if filename is None:
        filename = test_flatrangle_scaled.__name__
    for i in range(-5, 4):
        test_line(size=3, scale=10 ** i, filename=filename, all_clusters=True)
    assert _not_fail_all

def test_rightangle(scale = 1., filename = None, all_clusters = True):
    if filename is None:
        filename = test_rightangle.__name__
    XX = [[0., 0.], [scale, 0], [0, scale]]
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limitL=1, limitH=1000, verbose=False, do_edges=True)
    dr.fit(XX)
    # two clusters
    # assert (len(dr.parents_) == 2)

    labels = dr.labels_
    print('pairs', dr.mst_)
    print('labels', dr.labels_)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)

    assert (all(x == labels[0] for x in labels))
    assert _not_fail_all

def test_rightangle_scaled(scale = 1., filename = None, all_clusters = True):
    if filename is None:
        filename = test_rightangle_scaled.__name__
    for i in range(-5, 4):
        test_rightangle(scale=10 ** i, filename=filename, all_clusters=True)
    assert _not_fail_all

def test_equilateral(scale = 1., filename = None, all_clusters = True):
    if filename is None:
        filename = test_equilateral.__name__
    XX = [[scale, 0., 0.], [0., scale, 0], [0., 0., scale]]
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limitL=1, limitH=1000, verbose=False, do_edges=True)
    dr.fit(XX)
    # two clusters
    # assert (len(dr.parents_) == 2)

    labels = dr.labels_
    print('pairs', dr.mst_)
    print('labels', dr.labels_)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)

    assert (all(x == labels[0] for x in labels))
    assert _not_fail_all

def test_equilateral_scaled(scale = 1., filename = None, all_clusters = True):
    if filename is None:
        filename = test_equilateral_scaled.__name__
    for i in range(-5, 4):
        test_equilateral(scale=10 ** i, filename=filename, all_clusters=True)
    assert _not_fail_all

def test_line(size=4, scale = 1., filename = None, all_clusters = False):
    if filename is None:
        filename = test_line.__name__
    XX = []
    for i in range(0,size):
        XX.append([0.,i*scale])
    XX = np.array(XX)

    dr = DRUHG(max_ranking=50, limitL=1, limitH=len(XX), verbose=False, do_edges=True)
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
    labels = dr.labels_
    if _plot_graph:
        plt.close('all')
        dr.plot(labels)
        plt.savefig(filename+'.png')

    if all_clusters:
        assert (all(x == labels[0] for x in labels))
        return

    assert (not all(x == labels[0] for x in labels))
    assert (labels[0] == labels[len(labels)-1])
    assert (labels[1] == labels[len(labels)-2])
    assert (labels[0] != labels[1])
    assert _not_fail_all

def test_linesix(filename=None):
    if filename is None:
        filename = test_linesix.__name__
    test_line(size=6, filename=filename)
    assert _not_fail_all

def test_linefive(filename=None):
    if filename is None:
        filename = test_linefive.__name__
    test_line(size=5, filename=filename)
    assert _not_fail_all

def test_linefour(filename=None):
    if filename is None:
        filename = test_linefour.__name__
    test_line(size=4, filename=filename)
    assert _not_fail_all

def test_linelong(filename=None):
    if filename is None:
        filename = test_linelong.__name__
    test_line(size=100, filename=filename)
    assert _not_fail_all

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
#     dr = DRUHG(max_ranking=10, limitL=1, limitH=len(XX), verbose=False, do_edges=True)
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
#     labels = dr.relabel(limitL=1)
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

    dr = DRUHG(max_ranking=2000, algorithm='slow',
               limitL=1, limitH=len(XX), verbose=False, do_edges=True)
    dr.fit(XX)

    if showplot and _plot_graph:
        plt.close('all')
        print('dr.labels_', dr.labels_)
        dr.plot(dr.labels_)
        # dr.plot(dr.labels_)
        plt.savefig(filename + 'plot' + '.png')

    s = 2 * len(XX) - 2
    # print(dr.mst_)
    # print(dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)
    print(dr.mst_)
    # print(XX)
    print(dr.labels_)
    # assert (n_clusters==1)
    # labels = dr.relabel(limitL=1, limitH=size*2)
    n_clusters = len(set(labels)) - int(-1 in labels)
    # print('n_clusters', n_clusters)
    # print('pairs', dr.mst_)
    print('labels', dr.labels_)

    un, cn = np.unique(labels, return_counts=True)
    for i in range(0, len(un)):
        print('square', un[i], cn[i] )

    labels = dr.labels_
    if showplot and _plot_graph:
        plt.close('all')
        dr.plot(labels)
        plt.savefig(filename+'.png')
    un, cn = np.unique(labels, return_counts=True)
    print('uniques', un, cn)

    sorteds = np.argsort(cn)
    print('hello')
    print('hello2', un[sorteds[0]], un[sorteds[1]])
    # assert (False)
    dr.relabel(limitL=1, limitH=size*2)
    if showplot and _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'2'+'.png')
    assert (n_clusters >= 5)
    assert (cn[sorteds[-1]] == (size-2)**2)
    assert (un[sorteds[0]] == -1 and cn[sorteds[0]] == 4)
    for i in range(1,5):
        assert (cn[sorteds[i]] == size - 2)
    assert _not_fail_all

def test_square_scaled(filename=None):
    if filename is None:
        filename = test_square_scaled.__name__
    for i in range(-4, 4):
        # test_square(scale=10 ** i, filename=filename + "i" + str(i) + "-")
        test_square(scale=10 ** i, filename=filename)
    assert _not_fail_all

def test_squares_two(filename=None):
    if filename is None:
        filename = test_squares_two.__name__
    XX = []
    size, scale = 6, 1
    for i in range(0, size):
        for j in range(0, size):
            XX.append([scale*i, scale*j])
            XX.append([2*size + scale*i, scale*j])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, verbose=False, do_edges=True)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print(dr.mst_)
    print(dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)

    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'.png')
    assert (n_clusters==2)
    assert _not_fail_all


def test_square_particles(filename=None):
    if filename is None:
        filename = test_square_particles.__name__
    XX = [[-0.51,1.5], [1.51,1.5]]
    for i in range(-3, 5):
        for j in range(-6, 1):
            XX.append([i,j])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, algorithm='slow', limitL=1, limitH=len(XX), verbose=False, do_edges=True)
    dr.fit(XX)

    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'.png')

    s = 2*len(XX) - 2
    print(dr.mst_)
    print(dr.labels_)
    print(dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    # two points are further metrically but close reciprocally
    assert (dr.mst_[s-4]*dr.mst_[s-3] == 0)
    assert (dr.mst_[s-4] + dr.mst_[s-3] == 1)
    assert _not_fail_all

def test_square_particles_big(filename=None):
    if filename is None:
        filename = test_square_particles_big.__name__
    XX = [[-0.51,1.5], [1.51,1.5]]
    for i in range(-5, 7):
        for j in range(-8, 1):
            XX.append([i,j])
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, algorithm='slow', limitL=1, limitH=len(XX), verbose=False, do_edges=True)
    dr.fit(XX)

    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'.png')

    s = 2*len(XX) - 2
    print(dr.mst_)
    print(dr.labels_)
    print(dr.mst_[s-1], dr.mst_[s-2], XX[dr.mst_[s-1]], XX[dr.mst_[s-2]])
    # two points are further metrically but close reciprocally
    assert (dr.mst_[s-4]*dr.mst_[s-3] == 0)
    assert (dr.mst_[s-4] + dr.mst_[s-3] == 1)
    assert _not_fail_all


def test_square_bomb():
    XX = [[0.,1.],[0.,2.],[0.,3.],[0.,4.],[0.,5.]]
    for i in range(-3, 4):
        for j in range(-6, 1):
            XX.append([i,j])
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limitL=1, limitH=len(XX), verbose=False, do_edges=True)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print(dr.mst_)
    x = 12
    labs = dr.labels_
    # fuse is separate
    print(labs)
    assert (labs[0]==labs[1]==labs[2]==labs[3])
    assert (np.count_nonzero(labs == labs[0]) == 4)
    assert _not_fail_all

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
        dr.plot(dr.labels_)
        plt.savefig(filename+'.png')
    # t-center is an outlier too
    assert (n_clusters == 3)
    assert (np.count_nonzero(labels == -1) == 4)
    assert _not_fail_all

#
def test_t_cross(filename=None):
    if filename is None:
        filename = test_t_cross.__name__
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
        dr.plot(dr.labels_)
        plt.savefig(filename+'.png')

    assert (n_clusters == 4)
    assert (np.count_nonzero(labels == -1) == 5)
    assert _not_fail_all


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
    dr = DRUHG(algorithm='slow', max_ranking=200, limitL=1, limitH=int(len(XX)/2), verbose=False, do_edges=True)
    dr.fit(XX)
    s = 2*len(XX) - 2
    print(dr.mst_)
    # assert (0==1)
    labels = dr.labels_
    unique, counts = np.unique(labels, return_counts=True)
    print(unique, counts)
    # labels = dr.relabel(limitL=1, limitH=len(XX)/2)

    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters, set(labels))
    # print('labels', labels)

    # sorteds = np.argsort(counts)
    # dr.relabel(limitL=1, limitH=len(XX), exclude=[unique[sorteds[0]], unique[sorteds[1]]])
    # labels = dr.labels_



    if _test_extra_visualisation and showplot and _plot_graph:
        import seaborn as sns

        # np.test() # helps if the plot not working
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
            print(i, XX[dr.mst_[2*i]], colors[dr.mst_[2*i]])
            x1, y1, z1 = list(XX[dr.mst_[2 * i]])
            x2, y2, z2 = list(XX[dr.mst_[2 * i + 1]])

            ax.plot([x1,x2], [y1,y2], [z1,z2], c=colors[dr.mst_[2*i]])
        # plt.pause(10)
        plt.show()
        plt.savefig(filename+'1'+'.png')

    if showplot and _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
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
    # # labels = dr.relabel(limitL=1)
    # labels = dr.relabel(limitL=1, limitH=len(XX))
    # print('out')
    # n_clusters = len(set(labels)) - int(-1 in labels)
    # print('n_clusters2', n_clusters, set(labels))
    # # print('labels2', labels)
    # assert (n_clusters == 1+6+12)

    assert _not_fail_all

def test_loop_cube():
    return
    k = 100
    while k!=0:
        # print('+++++++++++++PREVED+++++++++++++', k - 1000)
        test_cube(showplot=False)
        # assert (False)
        k-=1
    assert _not_fail_all

def test_loop_square():
    return
    k = 100
    while k!=0:
        # print('+++++++++++++PREVED+++++++++++++', k - 1000)
        test_square(False)
        # assert (False)
        k-=1
    assert _not_fail_all

def test_druhg_sparse():
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    sparse_X = sparse.csr_matrix((data, (row, col)), shape=(3, 3))

    dr = DRUHG()
    dr.fit(sparse_X)
    print('sparse labels', dr.labels_)
    assert _not_fail_all

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
        dr.plot(dr.labels_)
        plt.savefig(filename+'1'+'.png')
    assert(n_clusters==4)

    dr = DRUHG(metric="precomputed", limitL=5).fit(D)
    labels = dr.labels_
    print(labels)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print(n_clusters)
    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'2'+'.png')

    assert(n_clusters==4)
    assert _not_fail_all

def test_moons_and_blobs():
    XX = X
    dr = DRUHG(max_ranking=50, verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    # expecting 4 clusters
    print(labels)

    assert (n_clusters == 4)
    assert _not_fail_all
#
def test_two_moons():
    np.random.seed(0)

    n_samples = 1500
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    XX = noisy_moons[0]
    dr = DRUHG(max_ranking=3, size_range=[1, 1], verbose=False, do_edges=True)
    dr.fit(XX)

    plt.close('all')
    plt.style.use('dark_background')
    dr.plot()

    assert _not_fail_all

def test_infinity(filename=None):
    if filename is None:
        filename = test_infinity.__name__
    np.random.seed(0)

    n_samples = 1500
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    XX = noisy_moons[0]
    dr = DRUHG(max_ranking=1000, size_range=[1, 1], verbose=False, do_edges=True)
    dr.fit(XX)

    plt.close('all')
    plt.figure(figsize=(5, 5))
    dr.plot()
    assert _not_fail_all


def test_hdbscan_clusterable_data(filename=None):
    if filename is None:
        filename = test_hdbscan_clusterable_data.__name__
    XX = np.load('druhg/tests/clusterable_data.npy')
    dr = DRUHG(max_ranking=1000, verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    uniques, counts = np.unique(labels, True)
    print(uniques, counts)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print(n_clusters)

    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        dr.plot()
        plt.savefig(filename+'.png')

    assert (n_clusters==6)
    assert _not_fail_all

def test_blobs_three(filename=None):
    if filename is None:
        filename = test_blobs_three.__name__
    XX = np.load('druhg/tests/three_blobs.npy')
    dr = DRUHG(max_ranking=3550, verbose=False)
    dr.fit(XX)
    labels = dr.labels_
    uniques, counts = np.unique(labels, True)
    print(uniques, counts)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print(n_clusters)

    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'.png')

    assert (n_clusters==3)
    assert _not_fail_all

def test_synthetic_outliers(filename=None):
    if filename is None:
        filename = test_synthetic_outliers.__name__
    XX = pd.read_csv('druhg/tests/synthetic.csv', sep=',')
    XX.drop(u'outlier', axis=1, inplace=True)
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limitH=len(XX), exclude=[1978,1973], verbose=False)
    dr.fit(XX)

    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'.png')

    # values, counts = np.unique(dr.labels_, return_counts=True)
    # for i, v in enumerate(values):
    #     print(v, counts[i])

    labels = dr.labels_
    # labels = dr.relabel(limitL=1)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print(labels)
    print('n_clusters', n_clusters)
    assert (n_clusters==6)
    assert _not_fail_all


def test_compound1(filename=None):
    if filename is None:
        filename = test_compound1.__name__
    XX = pd.read_csv('druhg/tests/Compound.csv', sep=',', header=None).drop(2, axis=1)
    XX = np.array(XX)
    dr = DRUHG(max_ranking=1550, limitL=3, limitH=len(XX), verbose=False)
    dr.fit(XX)

    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'1'+'.png')

    labels = dr.labels_
    # labels = dr.relabel(limitL=1)
    n_clusters = len(set(labels)) - int(-1 in labels)
    # np.save('labels_compound', labels)
    print('n_clusters', n_clusters, set(labels))
    exc = labels[398]
    # dr.relabel(limitL=3, limitH=len(XX), exclude=[exc])
    # labels = dr.labels_
    n_clusters2 = len(set(labels)) - int(-1 in labels)
    print(exc, n_clusters2, set(labels))

    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'2'+'.png')

    exc2 = labels[398]
    dr.relabel(limitL=3, limitH=len(XX), exclude=[exc, exc2])
    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'3'+'.png')


    assert (n_clusters==4)
    assert _not_fail_all


def test_compound_egg(filename=None):
    if filename is None:
        filename = test_compound_egg.__name__
    XX = pd.read_csv('druhg/tests/Compound.csv', sep=',', header=None)
    XX = XX[(XX[2]==6) | (XX[2]==5)]
    XX = XX.drop(2, axis=1)
    XX = np.array(XX)
    dr = DRUHG(max_ranking=2000, limitL=3, limitH=len(XX), verbose=False)
    dr.fit(XX)

    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'.png')

    labels = dr.labels_
    # labels = dr.relabel(limitL=1)
    n_clusters = len(set(labels)) - int(-1 in labels)
    # np.save('labels_compound', labels)
    print('n_clusters', n_clusters, set(labels))
    # dr.relabel(limitL=3, limitH=len(XX), exclude=[exc])
    # labels = dr.labels_
    n_clusters2 = len(set(labels)) - int(-1 in labels)
    print( n_clusters2, set(labels))

    assert (n_clusters==2)
    assert _not_fail_all

def test_copycat():
    XX = [[0]]*100
    XX = np.array(XX)
    dr = DRUHG(max_ranking=200, limitL=1, verbose=False, do_edges=True)
    dr.fit(XX)
    print(dr.mst_[0], dr.mst_[1])
    print('pairs', dr.mst_)
    print('labels', dr.labels_)
    labels = dr.labels_
    assert (all(x == labels[0] for x in labels))
    assert _not_fail_all


def test_copycats(): # should fail until weights are made
    XX = np.concatenate( ([[0]]*100, [[1]]*100))
    dr = DRUHG(max_ranking=10, limitL=1, verbose=False, do_edges=True)
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
    assert _not_fail_all


def test_copycats2():
    XX = np.concatenate( ([[0]]*10, [[1]]*10))
    dr = DRUHG(max_ranking=200, limitL=1, verbose=False, do_edges=True)
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
    assert _not_fail_all


def test_copycats3(): # should fail until weights are made
    XX = np.concatenate( ([[0]]*100, [[1]]*100, [[2]]*5) )
    dr = DRUHG(max_ranking=10, limitL=1, limitH=250, verbose=False, do_edges=True)
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
    assert _not_fail_all

#
# def test_speed0():
#     XX = []
#     size = 5
#     for i in range(0, size):
#         for j in range(0, size):
#             for k in range(0, size):
#                 XX.append([i, j, k])
#
#     XX = pd.read_csv('druhg/tests/synthetic.csv', sep=',')
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
#         dr = DRUHG(algorithm='fast', max_ranking=200, limitL=1, limitH=int(len(XX) / 2), verbose=False)
#         dr.fit(XX)
#
# def test_speed2(k=100):
#     XX = test_speed0()
#     while k!=0:
#         k -= 1
#         np.random.shuffle(XX)
#         dr = DRUHG(algorithm='fastminus', max_ranking=200, limitL=1, limitH=int(len(XX) / 2), verbose=False)
#         dr.fit(XX)
#
# def test_speed3(k=100):
#     XX = test_speed0()
#     while k!=0:
#         k -= 1
#         np.random.shuffle(XX)
#         dr = DRUHG(algorithm='slow+', max_ranking=200, limitL=1, limitH=int(len(XX) / 2), verbose=False)
#         dr.fit(XX)
#
# def test_speed4(k=100):
#     XX = test_speed0()
#     while k!=0:
#         k -= 1
#         np.random.shuffle(XX)
#         dr = DRUHG(algorithm='slow', max_ranking=200, limitL=1, limitH=int(len(XX) / 2), verbose=False)
#         dr.fit(XX)
#
# def test_speed5(k=100):
#     XX = test_speed0()
#     while k!=0:
#         k -= 1
#         np.random.shuffle(XX)
#         dr = DRUHG(algorithm='slow', max_ranking=200, limitL=1, limitH=int(len(XX) / 2), verbose=False)
#         dr.fit(XX)



def test_triangle(scale = 1., height1 = 0.45, height2 = 0.5, dis=0,  filename = None, three_clusters = True):
    if filename is None:
        filename = test_triangle.__name__
    XX = [[scale*height1, 0.], [scale*height2, 0], [0, scale*dis]]
    XX = np.array(XX)
    dr = DRUHG(algorithm='slow', max_ranking=200, limitL=1, limitH=1000, verbose=False, do_edges=True)
    dr.fit(XX)

    print(XX)

    labels = dr.labels_
    print('pairs', dr.mst_)
    print('labels', dr.labels_)
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)
    return not all(x == labels[0] for x in labels)
    assert _not_fail_all


def test_triangle_full(filename = None):
    full_res = []
    res1 = []
    steps = 31
    for i in range(0, steps):
        dis = i*0.1

        if test_triangle(scale=1., dis=dis,
                         height1=0.57735026918962573, height2=-0.57735026918962573,
                         filename="test_triangle_equilateral_"+str(i)):
            res1.append(True)
        else:
            res1.append(dis)
        print("{:.4f}".format(dis), '======equilateral======')
    print('result equilateral', res1)

    res2 = []
    for i in range(0, steps):
        dis = i*0.1
        if test_triangle(scale=1., dis=dis,
                                 height1=1., height2=0,
                                 filename="test_triangle_right"+str(i)):
            res2.append(True)
        else:
            res2.append(dis)
        print("{:.4f}".format(dis), '======right======')
    print('result right', res2)
    print('result equilateral', res1)

    assert (True in res1)
    assert (True in res2)
    assert (not all(x for x in res1))
    assert (not all(x for x in res2))

    assert _not_fail_all

def test_triangle_line(filename = None):
    res = []
    for i in range(0, 17):
        h = 0.45+i*0.01
        res.append(test_triangle(scale=1., height1=h, filename="test_triangle_"+str(i)))
        print(h, '==========================')
    print(res)
    assert (False in res)
    assert (True in res)
    assert _not_fail_all

def test_run(filename=None):
    if filename is None:
        filename = test_run.__name__
    XX = pd.read_csv('druhg/tests/chameleon.csv', sep='\t', header=None)
    XX = np.array(XX)
    plt.style.use('dark_background')
    dr = DRUHG(max_ranking=4200, limitL=1, limitH=len(XX)/4, do_edges=True)
    dr.fit(XX)
    if _plot_graph:
        plt.close('all')
        dr.plot(core_color='brown')
        plt.savefig(filename+'1'+'.png')
    assert _not_fail_all

def test_chameleon(filename=None):
    if filename is None:
        filename = test_chameleon.__name__
    XX = pd.read_csv('druhg/tests/chameleon.csv', sep='\t', header=None)
    XX = np.array(XX)
    # using time module

    print('start', time.time())
    dr = DRUHG(max_ranking=4200, limitL=1, limitH=len(XX), verbose=False, algorithm='slow')
    print('init', time.time())
    dr.fit(XX)
    print('afterfit',time.time())
    labels = dr.labels_
    # labels = dr.relabel(limitL=1)
    values, counts = np.unique(labels, return_counts=True)
    n_clusters = 0
    for i, c in enumerate(counts):
        print(i, c, values[i])
        if c > 500 and values[i] >= 0:
            n_clusters += 1
    print('n_clusters', n_clusters)

    print('preplot',time.time())
    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'1'+'.png')
    # assert (False)
    print('afterplot',time.time())

    dr = DRUHG(max_ranking=1200, limitL=1, limitH=int(len(XX)/4), exclude=[], verbose=False, algorithm='slow')
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
        dr.plot(dr.labels_)
        plt.savefig(filename+'2'+'.png')

    if _plot_graph and _test_extra_visualisation:
        plt.close('all')
        dr.single_linkage_.plot(dr.labels_)
        plt.savefig(filename +'plot'+ '.png')

    exc = dr.labels_[3024]
    dr.relabel(limitL=3, limitH=int(len(XX)/4), exclude=[exc])
    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'3'+'.png')

    exc = dr.labels_[3024]
    dr.relabel(limitL=3, limitH=int(len(XX)/8), exclude=[exc])
    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'4'+'.png')

    assert (n_clusters2==6)
    assert (n_clusters==6)
    assert _not_fail_all


# def test_boba(filename=None):
#     if filename is None:
#         filename = test_boba.__name__
#     XX = pd.read_csv('druhg/tests/chameleon.csv', sep='\t', header=None)
#     XX = np.array(XX)
#     lim = 1.
#     while lim >= 0:
#
#         dr = DRUHG(max_ranking=4200, limitL = 1, limitH=lim, verbose=False, algorithm='slow')
#         dr.fit(XX)
#         if _plot_graph:
#             plt.close('all')
#             dr.plot(dr.labels_)
#             plt.savefig(filename+str(lim)+'.png')
#         lim -= 0.05

def test_cham_amt(filename=None):
    if filename is None:
        filename = test_cham_amt.__name__
    XX = pd.read_csv('druhg/tests/chameleon.csv', sep='\t', header=None)
    XX = np.array(XX)
    dr = DRUHG(max_ranking=4200, limitL=1, limitH=0.25, verbose=False, algorithm='slow')
    dr.fit(XX)
    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename + '.png')
    assert _not_fail_all

def test_zero_distances(filename=None):
    if filename is None:
        filename = test_zero_distances.__name__
    XX = []
    for j in range(0, 90):
        XX.append([0.,5.])
    for i in range(0, 2000):
        XX.append([0., i])
    XX = np.array(XX)
    np.random.shuffle(XX)
    dr = DRUHG(max_ranking=200, verbose=False)
    dr.fit(XX)
    print(XX)
    # center is an outlier
    labels = dr.labels_
    n_clusters = len(set(labels)) - int(-1 in labels)
    print('n_clusters', n_clusters)
    print('labels', len(labels), labels)

    if _plot_graph:
        plt.close('all')
        dr.plot(dr.labels_)
        plt.savefig(filename+'.png')

    assert (n_clusters == 4)
    assert (np.count_nonzero(labels == -1) == 5)
    assert _not_fail_all

