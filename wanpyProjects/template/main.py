# Copyright (C) 2025 Jin Cao
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

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
import time
import os
import numpy as np
from numpy import linalg as LA
from mpi4py import MPI
import wanpy as wp
import wanpy.response as res
from wanpy.MPI import MPI_Reduce, MPI_Gather, init_kmesh, init_htb_response_data
from wanpyProjects import lib as libwpy
from wanpyProjects.lib.plot import *

COMM = MPI.COMM_WORLD
MPI_rank = COMM.Get_rank()
MPI_ncore = COMM.Get_size()
MPI_main = not MPI_rank

if wp.PYGUI:
    import matplotlib.pyplot as plt

if wp.PYGUI:
    wdir = os.path.join(wp.ROOT_WDIR, r'...')
    input_dir = os.path.join(wp.ROOT_WDIR, r'...')
else:
    wdir = os.getcwd()
    input_dir = r'./'

'''
  * Input
'''
if __name__ == '__main__':
    '''
      * Job = 
      ** band
    '''
    Job = 'band'
    setup = True

    '''
      * System
    '''
    htb_fname = r'htb.soc.symmr.h5'
    pt_symmetric = False

    os.chdir(input_dir)
    htb = wp.Htb()
    if MPI_main: htb.load_h5(htb_fname)
    os.chdir(wdir)

    '''
      * DOS tags
    '''
    # ne = 101
    # emin = -0.3
    # emax = 0.3  # 0.6, 0.5, 0.3
    # ee = np.linspace(emin, emax, ne)

    # ewidth = 0.005  # 5meV or 10meV

    omega = -0.1

    '''
      * Fermi surface tags
    '''
    ngate = 101
    gateMIN = -0.3
    gateMAX = 0.3
    gate = np.linspace(gateMIN, gateMAX, ngate)

    temperature = np.array([20, 50, 100])
    ntemperature = temperature.shape[0]

    ewidth_imag = 1e-6
    tau = 0.01  # in unit of ps, i.e., 1e-12s

    '''
      * htb tags
    '''
    tmin_h = -1e-6
    tmin_r = -1e-6
    open_boundary = -1
    istb = True
    use_wcc = True
    atomic_wcc = True

    '''
      * BZ tags
    '''
    nkmesh = np.array([24, 24, 1])
    # nkmesh = np.array([1920, 1920, 1])
    kcube = np.identity(3)
    kmesh_shift = np.array([0, 0, 0])
    random_k = False
    centersym = False

    ''' 
      * K path tags
    '''
    nk_path = 101

    G = np.array([0.0, 0.0, 0.0])
    M = np.array([0.5, 0.0, 0.0])
    K = np.array([1/3, 1/3, 0.0])

    kpath_HSP = np.array([G, M, K, G])
    xlabel = ['G', 'M', 'K', 'G']


'''
  * MPI calculator
'''
@MPI_Gather(MPI, iterprint=500, dtype='float64')
def cal_Fermi_surface(k, dim):
    return libwpy.cal_Fermi_surface(htb, k, omega, eta=ewidth_imag)

'''
  * Band Calculators
'''
def cal_band(htb, kpath):
    nk = kpath.shape[0]
    bandE = np.zeros([nk, htb.nw], dtype='float64')
    # bandE = np.zeros([nk, 24], dtype='float64')
    for i, k in zip(range(nk), kpath):
        res.printk(i, nk, k)
        bandE[i] = res.cal_band(htb, k, tbgauge=False, use_ws_distance=False)
        # bandE[i] = libwtb.debug_cal_band_symm(htb, k)
    return bandE

'''
  * main
'''
if __name__ == '__main__' and setup == True:
    T0 = time.time()

    if MPI_main:
        print('Running on   {} total cores'.format(MPI_ncore))
        print('WANPY version: {}'.format(wp.__version__))
        print('Author: {}'.format(wp.__author__))
        print()

    htb = init_htb_response_data(MPI, htb, tmin_h=tmin_h, tmin_r=tmin_r, open_boundary=open_boundary, istb=istb, use_wcc=use_wcc, atomic_wcc=atomic_wcc)
    kmesh = init_kmesh(MPI, nkmesh, random_k=random_k, kcube=kcube, kmesh_shift=kmesh_shift)

    NK = kmesh.shape[0] if MPI_main else None
    kmesh_car = (htb.lattG @ kmesh.T).T if MPI_main else None

    T1 = time.time()
    if MPI_main:
        print('{} job start at {}. unix time: {}'.format(Job, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), time.time()))

if __name__ == '__main__' and Job == 'band':
    os.chdir(wdir)

    kpath = wp.make_kpath(kpath_HSP, nk_path - 1)
    kpath_car = LA.multi_dot([htb.lattG, kpath.T]).T

    bandE = cal_band(htb, kpath)

    if wp.PYGUI:
        plot_band(kpath_car, bandE, xlabel, eemin=-1, eemax=1, figsize=[3.3, 3])

'''
  * End 
'''
if __name__ == '__main__' and setup == True:
    if MPI_rank == 0:
        T2 = time.time()
        print('Time consuming in total {:.3f}s ({:.3f}h).'.format(T2 - T0, (T2 - T0) / 3600))
        print('Time consuming {:.2f}ms on each of {} kpoints per core.'.format(1000 * MPI_ncore * (T2 - T1) / NK, NK))
    else:
        pass
