# Copyright (C) 2020 Jin Cao
#
# This file is distributed as part of the wanpy code and
# under the terms of the GNU General Public License. See the
# file LICENSE in the root directory of the wanpy
# distribution, or http://www.gnu.org/licenses/gpl-3.0.txt
#
# The wanpy code is hosted on GitHub:
#
# https://github.com/jincao2013/wanpy

__date__ = "Mar. 31, 2025"

import os
import numpy as np
from numpy import linalg as LA
import wanpy as wp
import wanpy.response as res
from wanpyProjects.demo.lib import libwtb
from wanpyProjects.demo.lib.libplot import *
import matplotlib.pyplot as plt

wdir = os.path.join(wp.ROOT_WDIR, r'demo')
input_dir = os.path.join(wp.ROOT_WDIR, r'demo/htblib')

if __name__ == '__main__':
    '''
      * Job = 
      ** band
      ** dos
    '''
    job = 'band'

    htb_fname = r'htb.h5'

    os.chdir(input_dir)
    htb = wp.Htb()
    htb.load_h5(htb_fname)
    os.chdir(wdir)

    ''' 
      * K path tags
    '''
    nk_path = 101

    G = np.array([0.0, 0.0, 0.0])
    X = np.array([0.5, 0.0, 0.0])
    Y = np.array([0.0, 0.5, 0.0])
    M = np.array([0.5, 0.5, 0.0])

    kpath_HSP = np.array([G, X, M, Y, G])
    xlabel = ['G', 'X', 'M', 'Y', 'G']

'''
  * Band Calculators
'''
def cal_band(htb, kpath):
    nk = kpath.shape[0]
    bandE = np.zeros([nk, htb.nw], dtype='float64')
    for i, k in zip(range(nk), kpath):
        res.printk(i, nk, k)
        bandE[i] = res.cal_band(htb, k, tbgauge=False, use_ws_distance=False)
    return bandE

'''
  * Plot
'''
if __name__ == '__main__' and job == 'band':
    os.chdir(wdir)
    kpath = wp.make_kpath(kpath_HSP, nk_path - 1)
    kpath_car = LA.multi_dot([htb.lattG, kpath.T]).T
    bandE = cal_band(htb, kpath)

    plot_band(kpath_car, bandE, xlabel, eemin=-0.3, eemax=0.3)

if __name__ == '__main__' and job == 'dos':
    npdata = np.load(r'dos.npz')
    nkmesh = npdata['nkmesh']
    kmesh = npdata['kmesh']
    lattG = npdata['lattG']
    dos = npdata['dos']
    del npdata

    kmesh_car = (lattG @ kmesh.T).T
    # plot_contour(dos, kmesh_car, XX=None, YY=None, vmin=0, vmax=dos.max(), vmin_show=0, vmax_show=dos.max(), plotref=False, cmap='Blues', figsize=[4,3.6])
    plt.xlabel(r'$k_x~(\AA^{-1})$')
    plt.ylabel(r'$k_y~(\AA^{-1})$')
    plt.axis([-1.4, 1.4, -1.4, 1.4])
    plt.xticks(np.linspace(-1.2, 1.2, 3))
    plt.yticks(np.linspace(-1.2, 1.2, 3))
    plt.tight_layout()
