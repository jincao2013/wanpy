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

import time
import os
import argparse
import numpy as np
from numpy import linalg as LA
from mpi4py import MPI
import wanpy as wp
import wanpy.response as res
from wanpy.MPI import Config
from wanpy.MPI import MPI_Reduce, MPI_Gather, init_kmesh, init_htb_response_data
from wanpyProjects.template.lib import libwtb
from wanpyProjects.template.lib.plot import *

if wp.PYGUI:
    import matplotlib.pyplot as plt

if wp.PYGUI:
    wdir = os.path.join(wp.ROOT_WDIR, r'templateProject')
    config_dir = os.path.join(wp.ROOT_WDIR, r'templateProject/configs')
    config_path = os.path.join(config_dir, "config.toml")
else:
    wdir = os.getcwd()
    # load config_path from command-line argument
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("-t", "--toml", required=True, help="Path to config.toml")
    args = parser.parse_args()
    config_path = args.toml

if __name__ == '__main__':
    # Load config.toml
    cf = Config(MPI)
    cf.load_config(config_path)

    # Initialize htb object and loads htb data from .h5 for main MPI rank
    htb = wp.Htb()
    if cf.MPI_main: htb.load_h5(os.path.join(cf.htb_dir, cf.htb_fname))
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
  * MPI calculator
'''
@MPI_Gather(MPI, iterprint=500, dtype='float64')
def cal_Fermi_surface(k, dim):
    return libwtb.cal_Fermi_surface(htb, k, cf.omega, eta=cf.ewidth_imag)

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
  * main
'''
if __name__ == '__main__' and cf.job is not None:
    T0 = time.time()

    if cf.MPI_main:
        print('Running on   {} total cores'.format(cf.MPI_ncore))
        print('WANPY version: {}'.format(wp.__version__))
        print('Author: {}'.format(wp.__author__))
        print()

    cf.print_config()
    htb = init_htb_response_data(MPI, htb, tmin_h=cf.tmin_h, tmin_r=cf.tmin_r, open_boundary=cf.open_boundary, istb=cf.istb, use_wcc=cf.use_wcc, atomic_wcc=cf.atomic_wcc)
    kmesh = init_kmesh(MPI, cf.nkmesh, random_k=cf.random_k, kcube=cf.kcube, kmesh_shift=cf.kmesh_shift)

    NK = kmesh.shape[0] if cf.MPI_main else None
    kmesh_car = (htb.lattG @ kmesh.T).T if cf.MPI_main else None

    T1 = time.time()
    if cf.MPI_main:
        print('{} job start at {}. unix time: {}'.format(cf.job, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), time.time()))

if __name__ == '__main__' and cf.job == 'band':
    os.chdir(wdir)
    kpath = wp.make_kpath(kpath_HSP, nk_path - 1)
    kpath_car = LA.multi_dot([htb.lattG, kpath.T]).T
    bandE = cal_band(htb, kpath)
    if wp.PYGUI:
        plot_band(kpath_car, bandE, xlabel, eemin=-0.3, eemax=0.3)

if __name__ == '__main__' and cf.job == 'dos':
    dim = [1]
    savefname = r'dos.npz'

    RS = cal_Fermi_surface(kmesh, dim)

    if cf.MPI_main:
        dos = RS
        kmesh_car = (htb.lattG @ kmesh.T).T
        np.savez_compressed(savefname, omega=cf.omega, ewidth_imag=cf.ewidth_imag, lattG=htb.lattG,
                            nkmesh=cf.nkmesh, kmesh=kmesh, dos=dos
                            )

    if cf.MPI_main and wp.PYGUI:
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

if __name__ == '__main__' and cf.job == 'debug':
    pass

'''
  * End 
'''
if __name__ == '__main__' and cf.job is not None:
    if cf.MPI_main:
        T2 = time.time()
        print('Time consuming in total {:.3f}s ({:.3f}h).'.format(T2 - T0, (T2 - T0) / 3600))
        print('Time consuming {:.2f}ms on each of {} kpoints per core.'.format(1000 * cf.MPI_ncore * (T2 - T1) / NK, NK))
    else:
        pass
