# -*- coding: utf-8 -*-
# Author: Pavel Artamonov
# License: 3-clause BSD


import numpy as np
import collections
from math import exp, log


# from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from warnings import warn


class Joint(object):
    def __init__(self, size=1, dist=0.):
        self.size = size
        self.val = dist
        self.L = None  # L.size >= R.size
        self.R = None
        self.T = None
        self.c = 'gray'

    def walkup(self):
        t = self
        while t.T:
            t = t.T
        return t

    def walkdown(self):
        l = self
        while l.L:
            l = l.L
        return l


class SingleLinkage(object):
    # todo: make it pretty
    def __init__(self, mst_pairs, values, labels):
        self._mst_pairs = mst_pairs
        self._values = values
        # self._data = data
        self._labels = labels

        self.default_scale_coef = 10.
        self.max_value = 1.

    def draw_dendrogram(self, ax, pairs, values, labels, lw=20., alpha=0.4, cmap='viridis'):
        # Step 1: Fill and connect Joints. Evaluate scales.
        # Step 2: Arrange outstanding trees.
        # Step 3: Depth-First search. Apply radial or flat visualisation func.

        min_index, max_index = min(pairs), max(pairs)
        if min_index < 0:
            raise ValueError('Indices should be non-negative')
        min_value, self.max_value = min(values), max(values[values != np.inf])
        if min_value < 0:
            raise ValueError('Values should be non-negative')

        cm = self.get_dendro_colors(labels)

        scale_sum = 0.

        size = int(len(pairs) / 2 + 1)
        tops = {}
        # trees building
        for j in range(0, size - 1):
            a, b = pairs[2 * j], pairs[2 * j + 1]

            v = values[j]
            if v == np.inf:
                scale_sum += self.max_value * self.default_scale_coef
            else:
                scale_sum += v

            topjoi = Joint()

            ta, sa = None, 1
            if a in tops:
                ta = tops[a].walkup()
                sa = ta.size
                ta.T = topjoi

            tb, sb = None, 1
            if b in tops:
                tb = tops[b].walkup()
                sb = tb.size
                tb.T = topjoi

            topjoi.size = sa + sb
            topjoi.val = v
            if sa >= sb:
                topjoi.L, topjoi.R = ta, tb
            else:
                topjoi.L, topjoi.R = tb, ta

            if labels[a] == labels[b]:
                topjoi.c = cm[labels[a]]
            tops[a] = topjoi
            tops[b] = topjoi

        # arranging trees in case they are not connected
        trees = collections.OrderedDict()
        for t in tops:
            joi = tops[t].walkup()
            if joi not in trees:
                trees[joi] = joi.size

        scale_sum += (len(trees) - 1) * self.max_value * self.default_scale_coef

        self.plot_radial(ax, trees, labels, scale_sum)



    def plot_radial(self, ax, trees, labels, scale_sum):
        try:
            from matplotlib import collections as mc
            from matplotlib.pyplot import Arrow
            from matplotlib.pyplot import Normalize
            from matplotlib.pyplot import cm
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot the minimum spanning tree.')

        # norm = len(labels)
        # sm = cm.ScalarMappable(cmap=cmap,
        #                            norm=Normalize(0, norm))
        # sm.set_array(norm)
        # colors = self.get_dendro_colors(labels)
        # c = 'gray'

        # depth first search
        x = 0.
        # print('so many treeess', len(trees))

        for tree in trees:
            l = tree.walkdown()
            while l:
                l, x = self.dps_flat_plot(ax, l, x)
            x += self.max_value * self.default_scale_coef

        ax.set_xticks([])
        for side in ('right', 'top', 'bottom'):
            ax.spines[side].set_visible(False)
        ax.set_ylabel('distance')

        return ax

    def dps_flat_plot(self, ax, joi, x):
        c = joi.c
        v = joi.val
        vexp = (1.+v)**2
        x1 = x
        nx = x + vexp
        x2 = nx
        ht = vexp
        h1 = h2 = (1. + v / 2.)**2
        s1 = s2 = 1
        if joi.L:
            h1 = joi.L.val
            s1 = joi.L.size
        if joi.R:
            h2 = joi.R.val
            s2 = joi.R.size

        ax.plot([x1, x2], [ht, ht], color=c, linewidth=log(1+s2))
        ax.plot([x1, x1], [ht, h1], color=c)
        ax.plot([x2, x2], [ht, h2], color=c)
        # print(x1, x2, ht, h1, h2)
        # print('hello', s1, s2, v, joi.L, joi.R, joi.T)

        if joi.R is not None:
            r = joi.R.walkdown()
            while r!=joi:
                r, nx = self.dps_flat_plot(ax, r, nx)

        return joi.T, nx

    def get_dendro_colors(self, labels):
        try:
            import seaborn as sns
        except ImportError:
            raise ImportError('You must install the seaborn library to draw colored labels.')

        unique, counts = np.unique(labels, return_counts=True)
        sorteds = np.argsort(counts)
        s = len(sorteds)

        i = sorteds[s - 1]
        max_size = counts[i]
        if unique[i] < 0:
            max_size = counts[sorteds[s - 2]]

        color_map = {}
        palette = sns.color_palette('bright', s + 1)
        col = 0
        a = (1. - 0.3) / (max_size - 1)
        b = 0.3 - a
        while s:
            s -= 1
            i = sorteds[s]
            if unique[i] < 0:  # outliers
                color_map[unique[i]] = (0., 0., 0., 0.15)
                continue
            alpha = a * counts[i] + b
            color_map[unique[i]] = palette[col] + (alpha,)
            col += 1

        return color_map

    def plot(self, axis=None):
        """Plot the dendrogram.

        Parameters
        ----------

        axis : matplotlib axis, optional
               The axis to render the plot to

        Returns
        -------

        axis : matplotlib axis
                The axis used the render the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot the minimum spanning tree.')

        if axis is None:
            # axis = plt.gca()
            # axis.set_axis_off()
            fig = plt.figure(figsize=[130, 50])
            axis = plt.subplot(111)


        axis = self.draw_dendrogram(axis, self._mst_pairs, self._values, self._labels)

        return axis


class MinimumSpanningTree(object):
    def __init__(self, mst_pairs, data, labels):
        self._mst_pairs = mst_pairs
        self._raw_data = data
        self._labels = labels

    def decrease_dimensions(self):
        if self._raw_data.shape[1] > 2:
            # Get a 2D projection; if we have a lot of dimensions use PCA first
            if self._raw_data.shape[1] > 32:
                # Use PCA to get down to 32 dimension
                data_for_projection = PCA(n_components=32).fit_transform(self._raw_data)
            else:
                data_for_projection = self._raw_data

            projection = TSNE().fit_transform(data_for_projection)
        elif self._raw_data.shape[1] == 2:
            projection = self._raw_data.copy()
        else:
            # one dimensional. We need to add dimension
            projection = self._raw_data.copy()
            projection = np.array([e for e in enumerate(projection)], np.int)

        return projection

    def get_node_colors(self, labels):
        try:
            import seaborn as sns
        except ImportError:
            raise ImportError('You must install the seaborn library to draw colored labels.')

        unique, counts = np.unique(labels, return_counts=True)
        sorteds = np.argsort(counts)
        s = len(sorteds)

        i = sorteds[s - 1]
        max_size = counts[i]
        if unique[i] < 0:
            max_size = counts[sorteds[s - 2]]

        color_map = {}
        palette = sns.color_palette('bright', s + 1)
        col = 0
        a = (1. - 0.3) / (max_size - 1)
        b = 0.3 - a
        while s:
            s -= 1
            i = sorteds[s]
            if unique[i] < 0:  # outliers
                color_map[unique[i]] = (0., 0., 0., 0.15)
                continue
            alpha = a * counts[i] + b
            color_map[unique[i]] = palette[col] + (alpha,)
            col += 1

        return [color_map[x] for x in labels]

    def fast_find(self, unionfind, n):
        n = int(n)
        p = unionfind[n]
        if p == 0:
            return n

        while unionfind[p] != 0:
            p = unionfind[p]

        # label up to the root
        while p != n:
            temp = unionfind[n]
            unionfind[n] = p
            n = temp

        return p

    def draw_edges(self, ax, pairs, pos, cols, lw=20., alpha=0.4, vary_line_width = None):
        try:
            from matplotlib import collections as mc
            from matplotlib.pyplot import Arrow
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot the minimum spanning tree.')

        min_index, max_index = min(pairs), max(pairs)
        if min_index < 0:
            raise ValueError('Indices should be non-negative')

        size = int(len(pairs) / 2 + 1)

        union_size = size
        if max_index > union_size - 1:
            union_size = max_index + 1
        union_size += 2

        uf, sz = np.zeros(2 * union_size, dtype=int), np.ones(union_size)
        next_label = union_size + 1

        line_width = 0.5

        outliner_color = (0, 0, 0, alpha)
        clusters_color = (0, 0, 0, min(alpha*2.,1.))

        min_arrow_width = 0.002
        max_arrow_width = lw
        max_collision = np.sqrt(union_size)
        thick_a = (max_arrow_width - min_arrow_width) / (1. * max_collision - 1)
        thick_b = max_arrow_width - 1. * max_collision * thick_a
        for j in range(0, size - 1):
            a, b = pairs[2 * j], pairs[2 * j + 1]
            start, end = pos[a], pos[b]

            color = outliner_color
            if self._labels[a] < 0 or self._labels[b] < 0:
                color = outliner_color
            elif self._labels[a] == self._labels[b]:
                color = cols[a]
                color = (color[0], color[1], color[2], alpha)
            else:
                color = clusters_color

            i = next_label - union_size
            aa, bb = self.fast_find(uf, a), self.fast_find(uf, b)

            a = (uf[a] != 0) * (aa - union_size)
            b = (uf[b] != 0) * (bb - union_size)
            if aa!=bb:
                uf[aa] = uf[bb] = next_label
                next_label += 1

                na, nb = sz[a], sz[b]
                sz[i] = na + nb

            size_reflection = np.sqrt(min(na, nb))
            if vary_line_width:
                line_width = size_reflection * thick_a + thick_b

            arr = Arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], color=color, width=line_width)
            ax.add_patch(arr)

        # line_collection.set_array(self._mst[:, 2].T)

    def plot(self, axis=None, node_size=40, node_color=None,
             node_alpha=0.8, edge_alpha=0.15, edge_linewidth=8, vary_line_width=True,
             core_color='purple'):
        """Plot the minimum spanning tree (as projected into 2D by t-SNE if required).

        Parameters
        ----------

        axis : matplotlib axis, optional
               The axis to render the plot to

        node_size : int, optional (default 40)
                The size of nodes in the plot.

        node_color : matplotlib color spec, optional
                By default draws colors according to labels
                where alpha regulated by cluster size.

        node_alpha : float, optional (default 0.8)
                The alpha value (between 0 and 1) to render nodes with.

        edge_alpha : float, optional (default 0.4)
                The alpha value (between 0 and 1) to render nodes with.

        edge_linewidth : float, optional (default 2)
                The linewidth to use for rendering edges.

        vary_line_width : bool, optional (default True)
                By default, edge thickness and color depends on the size of
                the clusters connected by it.
                Thicker edge connects a bigger clusters.
                Red color indicates emergence of the cluster.

        core_color : matplotlib color spec, optional (default 'purple')
                Plots colors at the node centers.
                Can be omitted by passing None.

        Returns
        -------

        axis : matplotlib axis
                The axis used the render the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot the minimum spanning tree.')

        if self._raw_data.shape[0] > 32767:
            warn('Too many data points for safe rendering of a minimal spanning tree!')
            return None

        if axis is None:
            axis = plt.gca()
            axis.set_axis_off()

        pos = self.decrease_dimensions()

        cols = node_color
        if node_color is None:
            cols = self.get_node_colors(self._labels)
            axis.scatter(pos.T[0], pos.T[1], c=cols, s=node_size, alpha=node_alpha)
        else:
            axis.scatter(pos.T[0], pos.T[1], c=cols, s=node_size)

        self.draw_edges(axis, self._mst_pairs, pos, cols, edge_linewidth, edge_alpha, vary_line_width)


        # axis.set_xticks([])
        # axis.set_yticks([])

        if core_color is not None:
            # adding dots at the node centers
            axis.scatter(pos.T[0], pos.T[1], c=core_color, marker='.', s=node_size / 10)

        return axis

    def to_numpy(self):
        """Return a numpy array of pairs of from and to in the minimum spanning tree
        """
        return self._mst_pairs.copy()

    def to_pandas(self):
        """Return a Pandas dataframe of the minimum spanning tree.

        Each row is an edge in the tree; the columns are `from` and `to`
        which are indices into the dataset
        """
        try:
            from pandas import DataFrame
        except ImportError:
            raise ImportError('You must have pandas installed to export pandas DataFrames')

        result = DataFrame({'from': self._mst_pairs[::2].astype(int),
                            'to': self._mst_pairs[1::2].astype(int),
                            'distance': None})
        return result

    def to_networkx(self):
        """Return a NetworkX Graph object representing the minimum spanning tree.

        Nodes have a `data` attribute attached giving the data vector of the
        associated point.
        """
        try:
            from networkx import Graph, set_node_attributes
        except ImportError:
            raise ImportError('You must have networkx installed to export networkx graphs')

        result = Graph()
        size = int(len(self._mst_pairs) / 2)
        for i in range(0, size):
            result.add_edge(self._mst_pairs[2 * i], self._mst_pairs[2 * i + 1])

        data_dict = {index: tuple(row) for index, row in enumerate(self._raw_data)}
        set_node_attributes(result, data_dict, 'data')

        return result


class Frames(object):
    def __init__(self, druhg):
        self._druhg = druhg

    # https://matplotlib.org/stable/tutorials/introductory/animation_tutorial.html
    def animate(self, XX, ax=None, fig=None, frames=40, interval_ms=500, xlim=[-20, 20], ylim=[-20, 20], xlabel='x', ylabel='y'):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
        except ImportError:
            raise ImportError('You must install the matplotlib library to animate the data.')
        print('frames=', frames, 'interval=', interval_ms)
        if XX.shape[0] > 32767:
            warn('Too many data points for safe rendering!')
            return None

        self._druhg.allocate_buffers(XX)
        self._run_times = 0

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        time_text = ax.text(0.95, 0.01, 'frame: ',
               verticalalignment='bottom', horizontalalignment='right',
               transform=ax.transAxes,
               color='green', fontsize=15)

        scat = ax.scatter(0, 0, c="b", s=5)
        scat.set_offsets(self._druhg._buffer_data1)
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

        def update_frame(frame):
            if self._run_times <= frame:
                self._run_times += 1

                self._druhg.buffer_develop()
                scat.set_offsets(self._druhg.new_data_)
                time_text.set_text('frame: '+str(frame))

            return (scat, time_text)

        def empty_init():
            pass


        ani = animation.FuncAnimation(fig=fig, func=update_frame, init_func=empty_init(),
                                      frames=frames, interval=interval_ms,)
        plt.show()

        return self
