# -*- coding: utf-8 -*-
# Author: Pavel Artamonov
# License: 3-clause BSD

import datetime
import logging

import numpy as np
from math import pow

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from warnings import warn

# 'bright' - gray
_palette = [(0.00784313725490196, 0.24313725490196078, 1.0),
(1.0, 0.48627450980392156, 0.0),
(0.10196078431372549, 0.788235294117647, 0.2196078431372549),
(0.9098039215686274, 0.0, 0.043137254901960784),
(0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
(0.6235294117647059, 0.2823529411764706, 0.0),
(0.9450980392156862, 0.2980392156862745, 0.7568627450980392),
(1.0, 0.7686274509803922, 0.0),
(0.0, 0.8431372549019608, 1.0),
]


class UF(object): # shadows _druhg_unionfind
    def __init__(self, parents_arr, size):
        self.parent = parents_arr
        self.p_size = size

    def get_offset(self):
        return self.p_size + 1


class ClusterTree(object):
    def __init__(self, uf_arr, data_arr, values_arr=None, sizes_arr=None, clusters_arr=None, mst_pairs=None,
                 num_edges_=0,
                 interactive=False):
        self._U = UF(uf_arr, len(data_arr))
        self._raw_data = data_arr
        self._values_arr = [-1., 0] if values_arr is None else values_arr
        self._has_values_arr = not (values_arr is None)
        self._num_edges = num_edges_ if num_edges_ > 0 else len(uf_arr)//2 - 1
        self._sizes_arr = sizes_arr
        self._clusters_arr = clusters_arr

        self._static_labels = None

        self._mst_pairs = mst_pairs # TODO: rebuild it if null?
        self._sum_coords = None

        self.clusters_elder_ = None # np.zeros(len(self._raw_data), int)
        self.clusters_depth_ = None # np.zeros(len(self._raw_data), int)

        self.clusters_sum_edges_ = None # np.zeros(len(self._raw_data), np.double)
        self.clusters_cumsum_edges_ = None # np.zeros(len(self._raw_data), np.double)
        self.clusters_cumsum_heights_ = None # np.zeros(len(self._raw_data), int)

        self.clusters_pallete_ = np.zeros(len(self._raw_data), (np.double, 4))
        self.node_colors_ = np.zeros(len(self._raw_data), (np.double, 4))
        self.core_colors_ = None
        self.scat_ = None
        self.quiver_ = None
        self.quiver_colors_ = None
        self.annotation_ = None
        self.outlier_color_ = None
        self._timer_text = None

        self.dis_slider = None
        self.qty_slider = None
        self._last_drawn = None

    def decrease_dimensions(self):
        raw = np.asarray(self._raw_data)
        n_features = 1 if raw.ndim == 1 else raw.shape[1]
        if n_features > 2:
            # Get a 2D projection; if we have a lot of dimensions use PCA first
            if n_features > 32:
                data_for_projection = PCA(n_components=32).fit_transform(raw)
            else:
                data_for_projection = raw

            projection = TSNE().fit_transform(data_for_projection)
        elif n_features == 2:
            projection = np.asarray(raw, dtype=np.float64).copy()
        else:
            values = np.asarray(raw, dtype=np.float64).reshape(-1)
            projection = np.column_stack((
                np.arange(values.shape[0], dtype=np.float64),
                values,
            ))

        return projection

    def get_cluster(self, e, top_dis, range_size):
        ret_index = -1
        ret_depth = -1
        ret_outlier = -1

        offset = self._U.get_offset()
        while self._U.parent[e] != 0:
            ret_depth += 1
            p = self._U.parent[e]
            pc = p - offset

            if self._sizes_arr[pc] > range_size[1]:
                break
            if self._has_values_arr and top_dis < self._values_arr[pc]:
                break
            if self._clusters_arr[pc] < 0:  # it is a cluster
                if range_size[0] <= self._sizes_arr[pc]:
                    ret_index = pc
            elif ret_index < 0:
                ret_outlier = pc
            e = p

        return ret_index, ret_depth, ret_outlier

    def _avg_color(self, colora, colorb):
        color = (colora + colorb) / 2.
        return color

    def _plot_edges(self, ax, pos, node_colors, top_dis):
        try:
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
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
        # посчитаем 'вес' каждого узла,
        # для определиния яркости точек и сбалансированности кластеров
        offset_const = self._U.get_offset()
        different_colors = len(palette)
        num_points = len(self._raw_data)
        size_uf = len(self._U.parent)
        slider_sizes_bg = np.zeros(num_points + 1)
        for i in range(num_points+1, size_uf):
            if self._U.parent[i] == 0:
                # ! протестировать
                continue

            cc = i - offset_const
            pc = self._U.parent[i] - offset_const

            if self._clusters_arr[pc] < 0: # it is a cluster
                slider_sizes_bg[self._sizes_arr[pc]] += 1

            depth = self.clusters_depth_[cc]
            if depth == 0:
                depth = 1
                self.clusters_depth_[cc] = 1
                self.clusters_elder_[cc] = i

            pc_depth = self.clusters_depth_[pc]
            if pc_depth < depth + 1:
                elder = self.clusters_elder_[cc]
                self.clusters_elder_[pc] = elder

                new_col = palette[elder % different_colors]

                self.clusters_pallete_[pc][0] = new_col[0]
                self.clusters_pallete_[pc][1] = new_col[1]
                self.clusters_pallete_[pc][2] = new_col[2]
                self.clusters_pallete_[pc][3] = base_node_alpha

                pc_depth = depth
            else:
                self.clusters_pallete_[pc] = self.clusters_pallete_[cc]
            self.clusters_depth_[pc] = pc_depth + 1

            edge_weight = self._values_arr[cc] if self._values_arr is not None else 1.

            self.clusters_sum_edges_[cc] += edge_weight
            self.clusters_sum_edges_[pc] += self.clusters_sum_edges_[cc]

            self.clusters_cumsum_edges_[cc] += edge_weight*self._sizes_arr[cc]
            self.clusters_cumsum_edges_[pc] += self.clusters_cumsum_edges_[cc]

            self.clusters_cumsum_heights_[cc] += self._sizes_arr[cc]
            self.clusters_cumsum_heights_[pc] += self.clusters_cumsum_heights_[cc]


        for i in range(0, num_points+1):
            if slider_sizes_bg[i] == 0: # making more visible on the axis
                slider_sizes_bg[i] = np.nan
        return slider_sizes_bg

    def dynamic_labeling_and_coloring(self, top_dis, range_size):
        # Цвет кластера определяется лейблом самой глубокой ноды, тогда при соединении будет эффект поглощения
        # альфа точки равна 1 для самой глубокой ноды, последние выбросы будут самыми прозрачными
        min_node_alpha = 0.5
        max_node_alpha = 1.
        plus_core_alpha = 0.25

        num_points = len(self._raw_data)
        for i in range(0, num_points):
            pc, point_depth, outlier_pc = self.get_cluster(i, top_dis, range_size)
            if pc > 0:
                color = self.clusters_pallete_[pc]
                cluster_depth = self.clusters_depth_[pc]
                ## scaling
                coef = min_node_alpha + pow(point_depth,2) * (max_node_alpha - min_node_alpha)/(1 + pow(cluster_depth,2))
                self.node_colors_[i][0] = max(0, min(1.0, 1. - coef*(1. - color[0])))
                self.node_colors_[i][1] = max(0, min(1.0, 1. - coef*(1. - color[1])))
                self.node_colors_[i][2] = max(0, min(1.0, 1. - coef*(1. - color[2])))
                self.node_colors_[i][3] = color[3]

                if self.core_color is not None:
                    color = self.core_color
                    coef += plus_core_alpha
                    self.core_colors_[i][0] = max(0, min(1.0, 1. - coef*(1. - color[0])))
                    self.core_colors_[i][1] = max(0, min(1.0, 1. - coef*(1. - color[1])))
                    self.core_colors_[i][2] = max(0, min(1.0, 1. - coef*(1. - color[2])))
                    self.core_colors_[i][3] = color[3]
                # print(self.node_colors_[i], coef, cluster_depth, point_depth)
                # num_clusters += 1
            else:
                color = self.outlier_color_
                ## mixing with white
                coef = 1. if outlier_pc < 0 else \
                        min_node_alpha + pow(point_depth,2) * (max_node_alpha - min_node_alpha)/(1 + pow(self.clusters_depth_[outlier_pc],2))

                self.node_colors_[i][0] = max(0, min(1.0, coef*color[0] + 1 - coef))
                self.node_colors_[i][1] = max(0, min(1.0, coef*color[1] + 1 - coef))
                self.node_colors_[i][2] = max(0, min(1.0, coef*color[2] + 1 - coef))
                self.node_colors_[i][3] = color[3]

                if self.core_color is not None:
                    color = self.core_color
                    coef += plus_core_alpha
                    self.core_colors_[i][0] = max(0, min(1.0, coef*color[0] + 1 - coef))
                    self.core_colors_[i][1] = max(0, min(1.0, coef*color[1] + 1 - coef))
                    self.core_colors_[i][2] = max(0, min(1.0, coef*color[2] + 1 - coef))
                    self.core_colors_[i][3] = color[3]

        return self.node_colors_, self.core_colors_

    def on_pick(self, event):
        annotation_visible = self.annotation_.get_visible()

        if event.inaxes == self.axs[0,0]:
            if annotation_visible and self.annotation_.get_tightbbox().contains(event.x, event.y):
                if event.button!=1:
                    # right click inside annotation = dismiss
                    self.annotation_.set_visible(False)
                    self.fig.canvas.draw_idle()
                return

            is_contained, annotation_index = self.scat_.contains(event)
            if is_contained:
                point_loc = self.scat_.get_offsets()[annotation_index['ind'][0]]
                self.annotation_.xy = point_loc
                ind = annotation_index['ind'][0]
                pc, point_depth, outlier_pc = self.get_cluster(ind, self._values_arr[int(self.dis_slider.val)]*1.0001, self.qty_slider.val)
                ss = -1
                cl = -1
                dis = -1.
                text_label = ''
                if pc >= 0:
                    text_label += 'label:'
                else:
                    pc = outlier_pc
                    text_label += 'outlier:'
                ss = self._sizes_arr[pc]
                cl = -self._clusters_arr[pc]
                cluster_depth = self.clusters_depth_[pc]
                if self._has_values_arr:
                    dis = self._values_arr[pc]
                    val = self._values_arr[self._U.parent[ind] - self._U.get_offset()]

                text_label += str(pc) \
                            + '\n #' + str(ss)
                # text_label += '\n parts: ' + str(cl)
                text_label += '\n depth: ' + str(cluster_depth) \
                            + '\n  dis: {:.4f}'.format(dis) \
                            + '\n   ' + str(point_depth) + '↕' + ' xx%' \
                            + '\n  dis: {:.4f}'.format(val) \
                            + '\nid: '+ str(ind)

                text_label += '\n' + '{:.0f}'.format(self.clusters_cumsum_heights_[pc])+ '  {:.2f}'.format(self.clusters_cumsum_edges_[pc])
                text_label += '\n' + '{:.2f}'.format(self.clusters_sum_edges_[pc])


                self.annotation_.set_text(text_label)
                self.annotation_.set_visible(True)
                self.fig.canvas.draw_idle()
                return
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

        num_points = len(self._raw_data)
        self.qty_slider.poly.set_xy([[0, 0], [1, 0],
                               [1, self.qty_slider.val[0]], [0, self.qty_slider.val[0]],
                               [0, self.qty_slider.val[1]], [1, self.qty_slider.val[1]],
                               [1, num_points], [0, num_points]])

        axbtn = self.axs[1, 1]
        if val is not None and self.btn_apply is not None and not axbtn.get_visible():
            self.update_plot(val)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update_dis_slider(self, val):

        # if val is not None and val==self.dis_slider.poly.get:
        #     # against dragging right or left
        #     return

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
            cc, core_cc = self.dynamic_labeling_and_coloring(dis, range_)
        else:
            cc = self.clusters_pallete_
            core_cc = self.core_color

        if self._mst_pairs is not None:
            self._plot_edges(axmain, self.pos, cc, dis)  # edge_linewidth, edge_alpha, vary_line_width)

        if self.scat_ is None:
            self.scat_ = axmain.scatter(self.pos.T[0], self.pos.T[1], c=cc, s=self.node_size) # , alpha=self.node_alpha
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
                axmain.scatter(self.pos.T[0], self.pos.T[1], c=core_cc, marker='.', s=self.node_size / 10)
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
            axbtn.set_visible(td.total_seconds() >= 3)

    def plot(self, static_labels=None, axis=None, interactive=True,
             node_size=40, node_color=None,
             node_alpha=0.8, edge_alpha=0.15, edge_linewidth=8,
             core_color='purple',
             depth_formula=None,
             color_palette=None):
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
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Slider, Button, RangeSlider
            import matplotlib.colors as Colors
        except ImportError:
            raise ImportError('You must install the matplotlib library to plot cluster tree.')

        if color_palette is not None:
            try:
                import seaborn as sns
                color_palette = sns.color_palette(color_palette, 10+2)
            except ImportError:
                raise ImportError('You must install the seaborn library to use color palette codes.', color_palette)

        else:
            color_palette = _palette

        if self._raw_data.shape[0] > 32767:
            warn('Too many data points for safe rendering of a cluster tree!')
            return None

        self._static_labels = static_labels

        self.pos = self.decrease_dimensions()
        self.core_color = core_color
        if self.core_color is not None:
            self.core_color = Colors.to_rgba(self.core_color)
            self.core_colors_ = np.zeros(len(self._raw_data), (np.double, 4))
        self.node_size = node_size
        self.node_alpha = node_alpha

        self.fig = None
        self.axs = np.array([[None, None],[None,None]])
        self.btn_apply = None

        self._last_drawn = None

        base_node_alpha = 0.8

        if axis is not None:
            axmain = axis
            self.axs[0, 0] = axis
        elif self._static_labels is not None:
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
            self.convert_labels_to_colors(color_palette, base_node_alpha)
        else:
            if self.clusters_elder_ is None:
                self.clusters_elder_ = np.zeros(len(self._raw_data), int)
            if self.clusters_depth_ is None:
                self.clusters_depth_ = np.zeros(len(self._raw_data), int)
            if self.clusters_sum_edges_ is None:
                self.clusters_sum_edges_ = np.zeros(len(self._raw_data), np.double)
            if self.clusters_cumsum_edges_ is None:
                self.clusters_cumsum_edges_ = np.zeros(len(self._raw_data), np.double)
            if self.clusters_cumsum_heights_ is None:
                self.clusters_cumsum_heights_ = np.zeros(len(self._raw_data), int)

            slider_sizes_bg = self.bg_colors_and_pallete(color_palette, base_node_alpha)

        if axis is not None or self._static_labels is not None:
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
        self.qty_slider.drawon = False
        self.dis_slider.drawon = False

        cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self._key_cid = cid

        # init
        self.update_dis_slider(None)
        self.update_qty_slider(None)
        self.update_plot(None)

        plt.show()

        return axmain
