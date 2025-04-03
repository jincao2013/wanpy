# Copyright (C) 2023 Jin Cao
#
# This file is distributed as part of the wanpy code and
# under the terms of the GNU General Public License. See the
# file LICENSE in the root directory of the wanpy
# distribution, or http://www.gnu.org/licenses/gpl-3.0.txt
#
# The wanpy code is hosted on GitHub:
#
# https://github.com/jincao2013/wanpy

__date__ = "Nov. 25, 2023"

import numpy as np
import wanpy as wp

__all__ = [
    'plot_band',
]

def plot_band(kpath_car, eig, xlabel, eemin=-3.0, eemax=3.0, yticks=None, savefig=False, size=[3.8,3]):
    import matplotlib
    import matplotlib.pyplot as plt

    nline = len(xlabel) - 1
    kdist = wp.kmold(kpath_car)
    nk, nw = eig.shape

    if savefig:
        matplotlib.use('Agg')

    '''
      * plot band
    '''
    fig = plt.figure('band', figsize=size, dpi=150)
    fig.clf()
    ax = fig.add_subplot(111)

    ax.axis([kdist.min(), kdist.max(), eemin, eemax])
    ax.axhline(0, color='k', linewidth=0.6, zorder=101, linestyle="--")

    ax.plot(kdist, eig, linewidth=1, linestyle="-", color='b', alpha=1, zorder=12)

    for i in range(1, nline):
        ax.axvline(x=kdist[i * nk // nline], linestyle='--', color='k', linewidth=0.6, alpha=1, zorder=101)

    if xlabel is not None:
        plt.xticks(kdist[np.arange(len(xlabel)) * (nk // nline)], xlabel)
    ax.set_ylabel('$E$ (eV)')
    if yticks is not None:
        ax.set_yticks(yticks)
    else:
        ax.set_yticks(np.linspace(eemin, eemax, 3))

    plt.tick_params(direction="in", pad=8)
    fig.tight_layout()

    if savefig:
        plt.savefig('band.png')
    else:
        plt.show()
