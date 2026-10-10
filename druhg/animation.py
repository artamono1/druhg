# -*- coding: utf-8 -*-
# Author: Pavel Artamonov
# License: 3-clause BSD

import logging
from warnings import warn

class Frames(object):
    def __init__(self, druhg):
        self._druhg = druhg

    # https://matplotlib.org/stable/tutorials/introductory/animation_tutorial.html
    def animate(self, XX, ax=None, fig=None, frames=40, interval_ms=500,
                xlim=[-20, 20], ylim=[-20, 20], xlabel='x', ylabel='y'):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('You must install the matplotlib library to animate the data.')

        logger = logging.getLogger(__package__)
        logger.info('frames=%s interval=%s', frames, interval_ms)
        if XX.shape[0] > 32767:
            warn('Too many data points for safe rendering!')
            return None

        self._druhg.allocate_buffers(XX)
        self._run_times = 0

        was_interactive = plt.isinteractive()
        plt.ion()
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        time_text = ax.text(
            0.95, 0.01, 'frame: ',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='green', fontsize=15)

        scat = ax.scatter(0, 0, c="b", s=5)
        scat.set_offsets(self._druhg.new_data_)
        ax.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

        def update_frame(frame):
            logger.debug('update_frame %s %s', self._run_times, frame)
            if self._run_times <= frame:
                self._run_times += 1
                self._druhg.buffer_develop()
                scat.set_offsets(self._druhg.new_data_)
                time_text.set_text('frame: ' + str(frame))
            return (scat, time_text)

        frame_seq = range(frames) if isinstance(frames, int) else frames
        # plt.pause() calls show(), which raises the window. On Wayland that
        # prints "QWindow::requestActivate" once per frame.
        fig.canvas.draw_idle()
        try:
            for frame in frame_seq:
                if not plt.fignum_exists(fig.number):
                    break
                update_frame(frame)
                fig.canvas.draw_idle()
                if interval_ms > 0:
                    fig.canvas.start_event_loop(interval_ms / 1000.0)
                else:
                    fig.canvas.flush_events()
        finally:
            if plt.fignum_exists(fig.number):
                plt.close(fig)
            if not was_interactive:
                plt.ioff()
        return self
