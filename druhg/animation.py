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
        # try:
        #     import datetime
        # except ImportError:
        #     raise ImportError('You must install the datetime library to animate the data.')

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
            print('update_frame', self._run_times, frame)
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
