# -*- coding: utf-8 -*-
# Author: Pavel Artamonov
# License: 3-clause BSD

import datetime

import numpy as np
import collections
from math import exp, log

# from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from warnings import warn

class UF(object): # shadows _druhg_unionfind
    def __init__(self, parents_arr, size):
        self.parent = parents_arr
        self.p_size = size

    def get_offset(self):
        return self.p_size + 1


class ClusterTree(object):
    def __init__(self, uf_arr, data_arr, values_arr=None, sizes_arr=None, clusters_arr=None, mst_pairs=None, num_edges_=0,
                 interactive=False):
        self._U = UF(uf_arr, len(data_arr))
        self._raw_data = data_arr
        self._values_arr = [-1., 0] if values_arr is None else values_arr
        self._has_values_arr = not (values_arr is None)
        self._num_edges = num_edges_ if num_edges_ > 0 else len(uf_arr)
        self._sizes_arr = sizes_arr
        self._clusters_arr = clusters_arr

        self._static_labels = None

        self._mst_pairs = mst_pairs # TODO: rebuild it if null?
        self._sum_coords = None

        self.clusters_pallete_ = np.zeros(len(self._raw_data), (np.double, 4))
        self.node_colors_ = np.zeros(len(self._raw_data), (np.double, 4))
        self.scat_ = None
        self.quiver_ = None
        self.quiver_colors_ = None
        self.annotation_ = None
        self.outlier_color_ = None
        self._timer_text = None

        self.dis_slider = None
        self.qty_slider = None

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

    def get_cluster(self, e, top_dis, range_size):
        ret_index = -1
        while self._U.parent[e] != 0:
            p = self._U.parent[e]
            pc = p - self._U.get_offset()
            if self._sizes_arr[pc] > range_size[1]:
                break
            if self._has_values_arr and top_dis < self._values_arr[pc]:
                break
            if self._clusters_arr[pc] > 0:  # it is a cluster
                if range_size[0] <= self._sizes_arr[pc]:
                    ret_index = pc
            e = p
        return ret_index

    def _avg_color(self, colora, colorb):
        color = (colora + colorb) / 2.
        return color

    def _plot_edges(self, ax, pos, node_colors, top_dis):
        try:
            from matplotlib import collections as mc
            from matplotlib.pyplot import Arrow
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot the minimum spanning tree.')

        num_edges = self._num_edges
        # start, end = pos[self._mst_pairs[num_edges//2]], pos[self._mst_pairs[num_edges//2 + 1]]
        if self.quiver_ is None:
            # line_width = min(1., ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5) / 2.  # медианный размер
            x, y, u, v = [], [], [], []
            for i in range(0, num_edges):
                a, b = self._mst_pairs[2*i], self._mst_pairs[2*i+1]
                if a==b:
                    break
                start, end = pos[a], pos[b]
                x.append(start[0])
                y.append(start[1])
                u.append(end[0] - start[0])
                v.append(end[1] - start[1])
            self.quiver_ = ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale = 1)

        if self.quiver_colors_ is None:
            self.quiver_colors_ = np.zeros(len(self._raw_data), (np.double, 4))
        for i in range(0, num_edges):
            a, b = self._mst_pairs[2*i], self._mst_pairs[2*i+1]
            if a == b:
                break
            color = self._avg_color(node_colors[a], node_colors[b])
            self.quiver_colors_[i][0] = color[0]
            self.quiver_colors_[i][1] = color[1]
            self.quiver_colors_[i][2] = color[2]
            self.quiver_colors_[i][3] = color[3] * 0.75
            if self._has_values_arr and self._values_arr[i] >= top_dis:
                self.quiver_colors_[i][3] = 0.
            self.quiver_.set_color(self.quiver_colors_)

    def convert_labels_to_colors(self, palette, base_node_alpha):
        different_colors = len(palette)
        num_points = len(self._static_labels)
        for i in range(0, num_points):
            lbl = self._static_labels[i]
            if lbl < 0:
                self.clusters_pallete_[i] = self.outlier_color_
            else:
                new_col = palette[lbl % different_colors]
                self.clusters_pallete_[i][0] = new_col[0]
                self.clusters_pallete_[i][1] = new_col[1]
                self.clusters_pallete_[i][2] = new_col[2]
                self.clusters_pallete_[i][3] = base_node_alpha

    def bg_colors_and_pallete(self, palette, base_node_alpha):
        different_colors = len(palette)
        rev_color = 0
        num_points = len(self._raw_data)
        size_uf = len(self._U.parent)
        slider_sizes_bg = np.zeros(num_points + 1)
        for i in range(num_points+1, size_uf):
            if self._U.parent[i] == 0:
                continue
            # cl = i - self._U.get_offset()
            pc = self._U.parent[i] - self._U.get_offset()
            if self._clusters_arr[pc] > 0:
                slider_sizes_bg[self._sizes_arr[pc]] += 1

            pc = self._U.parent[i] - self._U.get_offset()
            col = self.clusters_pallete_[pc]
            new_col = palette[rev_color]
            # making sure that parent has different color
            if col[0] == new_col[0] and col[1] == new_col[1] and col[2] == new_col[2]:
                new_col = palette[rev_color]

            self.clusters_pallete_[pc][0] = new_col[0]
            self.clusters_pallete_[pc][1] = new_col[1]
            self.clusters_pallete_[pc][2] = new_col[2]
            self.clusters_pallete_[pc][3] = base_node_alpha

            rev_color += 1
            if rev_color >= different_colors:
                rev_color = 0

        for i in range(0, num_points+1):
            if slider_sizes_bg[i] == 0: # making more visible on the axis
                slider_sizes_bg[i] = np.nan
        return slider_sizes_bg

    def restricted_labeling(self, top_dis, range_size):
        num_points = len(self._raw_data)
        for i in range(0, num_points):
            pc = self.get_cluster(i, top_dis, range_size)
            if pc > 0:
                self.node_colors_[i] = self.clusters_pallete_[pc]
                # num_clusters += 1
            else:
                self.node_colors_[i] = self.outlier_color_
        return self.node_colors_

    def on_pick(self, event):
        annotation_visible = self.annotation_.get_visible()
        if event.inaxes == self.axs[0,0]:
            is_contained, annotation_index = self.scat_.contains(event)
            if is_contained:
                point_loc = self.scat_.get_offsets()[annotation_index['ind'][0]]
                self.annotation_.xy = point_loc
                ind = annotation_index['ind'][0]
                pc = self.get_cluster(ind, self._values_arr[int(self.dis_slider.val)]*1.0001, self.qty_slider.val)
                ss = 1
                cl = 0
                dis = 0
                if pc >= 0:
                    ss = self._sizes_arr[pc]
                    cl = self._clusters_arr[pc] + 1
                    if self._has_values_arr:
                        dis = self._values_arr[pc]
                text_label = 'label:' + str(pc) \
                             + '\n dis: {:.4f}'.format(dis) \
                             + '\n size: ' + str(ss) \
                             + '\nparts: ' + str(cl) \
                             + '\n(p: '+ str(ind)+')'
                self.annotation_.set_text(text_label)
                self.annotation_.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if annotation_visible:
                    self.annotation_.set_visible(False)
                    self.fig.canvas.draw_idle()


    def on_key_press(self, event):
        if event.key == 'left' and self.dis_slider.val > 0:
            self.dis_slider.set_val(self.dis_slider.val - 1)
        elif event.key == 'right' and self.dis_slider.val < self.dis_slider.valmax:
            self.dis_slider.set_val(self.dis_slider.val + 1)

        v1, v2 = self.qty_slider.val
        if event.key == 'up' and v1+1 < v2:
            self.qty_slider.set_val([v1+1, v2])
        elif event.key == 'down' and v2 > 0:
            self.qty_slider.set_val([v1 - 1, v2])
        elif event.key == 'shift+up' and v1 < self.dis_slider.valmax:
            self.qty_slider.set_val([v1, v2 + 1])
        elif event.key == 'shift+down' and v1+1 < v2:
            self.qty_slider.set_val([v1, v2-1])

    def _apply(self, event):
        axbtn = self.axs[1, 1]
        axbtn.set_visible(False)
        self.update_plot(None)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


    def update_qty_slider(self, val):
        axbtn = self.axs[1, 1]
        num_points = len(self._raw_data)

        self.qty_slider.poly.set_xy([[0, 0], [1, 0],
                               [1, self.qty_slider.val[0]], [0, self.qty_slider.val[0]],
                               [0, self.qty_slider.val[1]], [1, self.qty_slider.val[1]],
                               [1, num_points], [0, num_points]])

        if val is not None and self.btn_apply is not None and not axbtn.get_visible():
            self.update_plot(val)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update_dis_slider(self, val):
        axbtn = self.axs[1, 1]
        num_points = len(self._raw_data)

        dis = self._values_arr[int(self.dis_slider.val)]
        self.dis_slider.valtext.set_text("{:.4f}".format(dis))
        # dis_slider.poly.set_xy([[dis_slider.val, 0.], [dis_slider.val, 2.], [num_points, 2.], [num_points, 0.]])
        self.dis_slider.poly.set(xy=[self.dis_slider.val, 0.], height=2., width=(num_points - self.dis_slider.val + 1))

        if val is not None and self.btn_apply is not None and not axbtn.get_visible():
            self.update_plot(val)
        if self.btn_apply is None or not axbtn.get_visible():
            self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update_plot(self, val):
        # axis.cla()
        axmain = self.axs[0, 0]
        axbtn = self.axs[1, 1]
        now = datetime.datetime.now()

        if self.dis_slider is None:
            dis = np.inf
        else:
            dis = self._values_arr[int(self.dis_slider.val)] * 1.0001

        if self.qty_slider is None:
            range_ = [0, np.inf]
        else:
            range_ = self.qty_slider.val

        if self._static_labels is None:
            cc = self.restricted_labeling(dis, range_)
        else:
            cc = self.clusters_pallete_

        if self._mst_pairs is not None:
            self._plot_edges(axmain, self.pos, cc, dis)  # edge_linewidth, edge_alpha, vary_line_width)

        if self.scat_ is None:
            self.scat_ = axmain.scatter(self.pos.T[0], self.pos.T[1], c=cc, s=self.node_size, alpha=self.node_alpha)
            axmain.set_axis_off()

            if self.fig is not None and self._static_labels is None:
                self.annotation_ = axmain.annotate(
                    text='',
                    xy=(0, 0),
                    xytext=(10, 15),
                    textcoords='offset points',
                    bbox={'boxstyle': 'round'},
                    arrowprops={'arrowstyle': '->'}
                )
                self.annotation_.set_visible(False)
                # fig.canvas.mpl_connect('motion_notify_event', motion_hover)
                self.fig.canvas.mpl_connect('button_press_event', self.on_pick)

            if self.core_color is not None:
                # adding (red)dots at the node centers
                axmain.scatter(self.pos.T[0], self.pos.T[1], c=self.core_color, marker='.', s=self.node_size / 10)
        else:
            self.scat_.set_color(cc)

        td = datetime.datetime.now() - now
        if self._static_labels is None:
            if self._timer_text is None:
                self._timer_text = axmain.text(0.05, 0.95, f'{td.total_seconds():.3f}' + " sec",
                                               transform=self.fig.transFigure,
                                               # transform=plt.gcf().transFigure,
                                               verticalalignment='top', horizontalalignment='left')
            else:
                self._timer_text.set_text(f'{td.total_seconds():.3f}' + " sec")
        if self.scat_ is not None and self.btn_apply is not None and not axbtn.get_visible():
            if td.total_seconds() < 3:
                axbtn.set_visible(False)
            else:
                axbtn.set_visible(True)

    def plot(self, static_labels=None, axis=None, interactive=True,
             node_size=40, node_color=None,
             node_alpha=0.8, edge_alpha=0.15, edge_linewidth=8,
             core_color='purple'):
        """Plot the cluster tree with slider controls.

        Parameters
        ----------
        static_labels : array, optional
                If passed - no slider widgets

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
            from matplotlib.widgets import Slider, Button, RangeSlider
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot cluster tree.')

        try:
            import seaborn as sns
        except ImportError:
            raise ImportError('You must install the seaborn library to draw colored labels.')

        if self._raw_data.shape[0] > 32767:
            warn('Too many data points for safe rendering of a cluster tree!')
            return None

        self._static_labels = static_labels

        self.pos = self.decrease_dimensions()
        self.core_color = core_color
        self.node_size = node_size
        self.node_alpha = node_alpha

        self.fig = None
        self.axs = np.array([[None, None],[None,None]])
        self.btn_apply = None

        different_colors = 10
        base_node_alpha = 0.8

        if axis is not None:
            axmain = axis
            self.axs[0, 0] = axis
        elif self ._static_labels is not None:
            # fig = plt.figure()
            axmain = plt.gca()
            axmain.set_axis_off()
            self.axs[0, 0] = axmain
        else:
            self.fig, self.axs = plt.subplots(2, 2, width_ratios=[0.9, 0.1], height_ratios=[0.95, 0.05])
            axmain = self.axs[0, 0]

        self.outlier_color_ = axmain.get_facecolor()
        self.outlier_color_ = (1. - self.outlier_color_[0], 1. - self.outlier_color_[1], 1. - self.outlier_color_[2], 0.5)

        if self._static_labels is not None:
            self.convert_labels_to_colors(sns.color_palette('bright', different_colors+2), base_node_alpha)
        else:
            slider_sizes_bg = self.bg_colors_and_pallete(sns.color_palette('bright', different_colors+2), base_node_alpha)

        if axis is not None or self ._static_labels is not None:
            self.update_plot(None)
            # plt.show()
            return axmain

        # dynamic with sliders
        # рисуем полотно и выводим два слайдера
        # выводить статы по времени отрисовки
        #   если время превышает, то выводить кнопку Run и выводить только при её нажатии
        axvals = self.axs[1, 0]
        axqty = self.axs[0, 1]
        axbtn = self.axs[1, 1]
        num_points = len(self._raw_data)

        _num_edges = 2
        if self._has_values_arr:
            _num_edges = self._num_edges
        axvals.plot(self._values_arr[:_num_edges], scaley='log', color=self.outlier_color_)

        self.dis_slider = Slider(axvals, 'Values', valmin=0, valmax=_num_edges-1,
                            valstep=1.,
                            valinit=_num_edges-1,
                            color=(self.outlier_color_[0],self.outlier_color_[1],self.outlier_color_[2], 0.2),
                            track_color=(0.5, 0.5, 0.5, 0.05),
                            handle_style={"": "|", "size": 30},
                        )
        if self._num_edges + 1 != num_points:
            _plural = "s" if num_points - self._num_edges - 1 > 1 else ""
            axvals.text(0.5, 0.5, "missing "+ str(num_points - self._num_edges - 1) + " edge"+_plural,
                        transform=axvals.transAxes,
                        horizontalalignment='center', verticalalignment='center_baseline',
                        weight="ultralight",
                        alpha=0.7)


        axqty.plot(slider_sizes_bg, range(0, len(slider_sizes_bg)), 'k_', scalex='log', color=self.outlier_color_)
        self.qty_slider = RangeSlider(axqty, "Qty", valmin=0, valmax=num_points,
                                 valstep=1., orientation="vertical",
                                 color=(self.outlier_color_[0],self.outlier_color_[1],self.outlier_color_[2],0.2),
                                 track_color=(0.5, 0.5, 0.5, 0.05),
                                 handle_style={"": "_", "size": 30},
        )

        self.btn_apply = Button(axbtn, 'Run', color='gray', hovercolor='green')
        self.btn_apply.on_clicked(self._apply)
        axbtn.set_visible(False)

        self.qty_slider.on_changed(self.update_qty_slider)
        self.dis_slider.on_changed(self.update_dis_slider)

        cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # init
        self.update_dis_slider(None)
        self.update_qty_slider(None)
        self.update_plot(None)

        plt.show()

        return axmain
