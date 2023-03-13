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

__date__ = "Apr. 14, 2020"


import time
import os
import sys
import getpass
sys.path.append(os.environ.get('PYTHONPATH'))
from wanpy.env import *

if os.environ.get('PYGUI') == 'True':
    from pylab import *
    from wanpy.core.plot import *
    from wanpy.response.response_plot import *

import numpy as np
from numpy import linalg as LA

from wanpy.core.units import *
from wanpy.core.mesh import make_kpath, make_mesh
from wanpy.core.structure import BandstructureHSP
# from wanpy.response.response import Htb_response
# from wanpy.core.read_write import Htb, Cell
from wanpy.core.structure import Htb, Cell
from wanpy.core.greenfunc import self_energy
from wanpy.core.trans_hr import Supercell_Htb

import wanpy.response.response as res

from wanpy.core.bz import gauss_Delta_func, adapted_gauss_Delta_func
from wanpy.core.bz import get_adaptive_ewide_II, get_adaptive_ewide_III
from wanpy.core.bz import get_adaptive_ewide_II_slab, get_adaptive_ewide_III_slab
from wanpy.core.bz import FD_zero, FD, MPgauss, gauss, lorentz, get_is_surf

from wanpy.MPI.MPI import MPI_Reduce, MPI_Gather
from wanpy.MPI.init_data import init_kmesh, init_htb_response_data

from mpi4py import MPI

COMM = MPI.COMM_WORLD
MPI_rank = COMM.Get_rank()
MPI_ncore = COMM.Get_size()

'''
  * Models
'''
class KP(object):

    def __init__(self):
        self.nb = None
        self.fermi = 0
        self.latt = np.eye(3)
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)

    def get_hk(self, kc):
        # periodical kp are needed for surface states calculations
        pass

    # def get_hR(self, kc, N, openboundary=1):
    #     if N%2 != 0:
    #         print('N must be even number')
    #         sys.exit(1)
    #     kx, ky, kz = kc
    #     kk = np.linspace(0, 1, N+1)[:-1]
    #     kkc = 2 * np.pi * kk
    #     R = np.arange(N+1) - N // 2
    #     degen = np.ones_like(R)
    #     degen[np.array([0, -1])] = 2
    #     nR = N+1
    #     hk = np.zeros([N, self.nb, self.nb], dtype='complex128')
    #     for i in range(N):
    #         kci = np.array([kx, ky, kz])
    #         kci[openboundary] = kkc[i]
    #         hk[i] = self.get_hk(kci)
    #     eikR = np.exp(-2j * np.pi * np.einsum('k,R->kR', kk, R))
    #     hR = np.einsum('kR,kmn->Rmn', eikR, hk) / N
    #     return nR, hR

    def get_hR(self, k, N, openboundary=1):
        if N%2 != 0:
            print('N must be even number')
            sys.exit(1)

        kk = np.kron(np.ones([N, 1]), k)
        kk.T[openboundary] = np.linspace(0, 1, N+1)[:-1]
        kkc = LA.multi_dot([self.lattG, kk.T]).T
        R = np.zeros([N+1, 3])
        R.T[openboundary] = np.arange(N+1) - N // 2
        degen = np.ones_like(R)
        degen[np.array([0, -1])] = 2
        nR = N+1

        hk = np.zeros([N, self.nb, self.nb], dtype='complex128')
        for i in range(N):
            hk[i] = self.get_hk(kkc[i])
        eikR = np.exp(-2j * np.pi * np.einsum('ka,Ra->kR', kk, R))
        hR = np.einsum('kR,kmn->Rmn', eikR, hk) / N
        return nR, hR

    def get_hk_slab(self, kc, N=100, M=2, openboundary=1):
        k = LA.multi_dot([LA.inv(self.lattG), kc])
        nR, hR = self.get_hR(k, N, openboundary)
        hk_slab = np.kron(np.eye(N), hR[nR // 2])
        for i in range(M):
            j = i+1
            hk_slab += np.kron(np.eye(N, N, j), hR[nR//2+j]) + \
                       np.kron(np.eye(N, N, -j), hR[nR//2-j])
        nb = self.nb * N
        return hk_slab

    def get_surf_green(self, ee, kc, nprin=2, N=100, M=1, openboundary=1, eps=0.01):
        k = LA.multi_dot([LA.inv(self.lattG), kc])
        kx, ky, kz = kc
        ne = ee.shape[0]
        if M >= nprin:
            M = nprin - 1

        nR, hR = self.get_hR(k, N, openboundary)

        # build h0, h1R and h1L
        h0 = np.kron(np.eye(nprin), hR[nR//2])
        h1R = np.zeros_like(h0)
        h1L = np.zeros_like(h0)
        nbs = h0.shape[0]
        for i in range(M):
            j = i+1
            h0 += np.kron(np.eye(nprin, nprin, j), hR[nR//2+j]) + \
                  np.kron(np.eye(nprin, nprin, -j), hR[nR//2-j])
            h1R += np.kron(np.eye(nprin, nprin, nprin - j), hR[nR // 2 + j])
            h1L += np.kron(np.eye(nprin, nprin, nprin - j), hR[nR // 2 - j])

        # get surface green's functions
        dos_L = np.zeros([ne], dtype='float64')
        dos_R = np.zeros([ne], dtype='float64')
        for i, _e in zip(range(ne), ee):
            eI = (_e + 1j * eps) * np.identity(nbs)
            selfenLr = self_energy(eI - h0, eI - h0, h1L.T.conj(), h1L)
            selfenRr = self_energy(eI - h0, eI - h0, h1R.T.conj(), h1R)
            GLr = LA.inv(eI - h0 - selfenLr)
            GRr = LA.inv(eI - h0 - selfenRr)
            dos_L[i] = -2 * np.imag(np.trace(GLr))
            dos_R[i] = -2 * np.imag(np.trace(GRr))

        return dos_L, dos_R

    # '''
    #   * htb & htbsc
    # '''
    # def _init_ws_gridR(self, ngridR, latt):
    #     '''
    #     :param Rgrid:
    #     :return:
    #             nrpts: int
    #             ndegen: list
    #             irvec: [array([i1,i2,i3])]
    #
    #     '''
    #     # ***********
    #     # init
    #     # ***********
    #     a1 = latt.T[0]
    #     a2 = latt.T[1]
    #     a3 = latt.T[2]
    #
    #     # ***********
    #     # main
    #     # ***********
    #     nR = 0  # -1
    #     ndegen = []
    #     gridR = []
    #
    #     g_matrix = np.dot(np.array([a1, a2, a3]),
    #                       np.array([a1, a2, a3]).T)
    #
    #     for n1 in range(-ngridR[0], ngridR[0] + 1):
    #         for n2 in range(-ngridR[1], ngridR[1] + 1):
    #             for n3 in range(-ngridR[2], ngridR[2] + 1):
    #                 # Loop 125 R
    #                 icnt = -1
    #                 dist = np.zeros((125))
    #                 for i1 in [-2, -1, 0, 1, 2]:
    #                     for i2 in [-2, -1, 0, 1, 2]:
    #                         for i3 in [-2, -1, 0, 1, 2]:
    #                             icnt += 1
    #                             ndiff = np.array([
    #                                 n1 - i1 * ngridR[0],
    #                                 n2 - i2 * ngridR[1],
    #                                 n3 - i3 * ngridR[2]
    #                             ])
    #                             dist[icnt] = ndiff.dot(g_matrix).dot(ndiff)
    #                 # print(dist)
    #
    #                 # dist_min = min(dist.tolist())
    #                 dist_min = np.min(dist)
    #                 if np.abs((dist[62] - dist_min)) < 10 ** -7:
    #                     # nrpts += 1
    #                     ndegen.append(0)
    #                     for i in range(0, 125):
    #                         if np.abs(dist[i] - dist_min) < 10 ** -7:
    #                             ndegen[nR] += 1
    #                     nR += 1
    #
    #                     # irvec.append(n1 * a1 + n2 * a2 + n3 * a3)
    #                     gridR.append(np.array([n1, n2, n3]))
    #
    #     ndegen = np.array(ndegen, dtype='int64')
    #     gridR = np.array(gridR, dtype='int64')
    #     # print('nrpts={}'.format(nrpts_s))
    #     # print('ndegen=\n', ndegen_s)
    #     # print('irvec=\n')
    #     # pp.pprint(irvec_s)
    #     # print('*=============================================================================*')
    #     # print('|                                   R Grid                                     |')
    #     # print('|    number of R Grid = {:4>}                                                  |'.format(nR))
    #     # print('*=============================================================================*')
    #     # for i in range(nR):
    #     #     print('|{: 4}). {: 3} {: 3} {: 3}   *{:2>}  '.format(i + 1, gridR[i, 0], gridR[i, 1], gridR[i, 2], ndegen[i]),
    #     #           end='')
    #     #     if (i + 1) % 3 == 0:
    #     #         print('|')
    #     # print('')
    #     # print('*--------------------------------------------------------------------------------*')
    #     return nR, ndegen, gridR
    #
    # def _init_kmesh(self, nmesh):
    #     N1, N2, N3 = nmesh
    #     N = N1 * N2 * N3
    #     n2, n1, n3 = np.meshgrid(np.arange(N2), np.arange(N1), np.arange(N3))
    #     mesh = np.array([n1.reshape(N), n2.reshape(N), n3.reshape(N)], dtype='float64').T
    #     mesh /= nmesh
    #     nk = mesh.shape[0]
    #     return nk, mesh

    # def build_htb(self, ngridk):
    #     nR, ndegen, gridR = self._init_ws_gridR(ngridk, self.latt)
    #     nk, gridk = self._init_kmesh(ngridk)
    #     gridkc = LA.multi_dot([self.lattG, gridk.T]).T
    #
    #     hk = np.array([
    #         self.get_hk(gridkc[ik])
    #         for ik in range(nk)
    #     ], dtype='complex128')
    #     eikR = np.exp(-2j * np.pi * np.einsum('ka,Ra->kR', gridk, gridR))
    #     h_Rmn = np.einsum('kR,kmn->Rmn', eikR, hk) / nk
    #
    #     htb = Htb()
    #     htb.name = 'W90_interpolation'
    #     htb.fermi = self.fermi
    #     htb.nw = self.nb
    #     htb.nR = nR
    #     htb.R = gridR
    #     htb.Rc = LA.multi_dot([self.latt, gridR.T]).T
    #     htb.ndegen = ndegen
    #
    #     htb.latt = self.latt
    #     htb.lattG = self.lattG
    #
    #     htb.wcc = np.zeros([self.nb, 3])
    #     htb.wccf = np.zeros([self.nb, 3])
    #     htb.hr_Rmn = h_Rmn
    #     htb.r_Ramn = np.array([0])
    #
    #     cell = Cell()
    #     cell.name = ''
    #     cell.spec = ['H'] * self.nb
    #     cell.lattice = self.latt
    #     cell.latticeG = self.lattG
    #     cell.ions = np.zeros([self.nb, 3])
    #     cell.ions_car = np.zeros([self.nb, 3])
    #     htb.cell = cell
    #
    #     self.htb = htb
    #     return htb
    #
    # def build_htb_slab(self, RGrid, n1=1, n2=11, n3=1, open_boundary=1):
    #     transM = np.array([
    #         [n1, 0., 0.],
    #         [0., n2, 0.],
    #         [0., 0., n3],
    #     ])
    #     supercell_htb = Supercell_Htb(self.htb, transM, RGrid)
    #     htbsc = supercell_htb.get_htb_slab(open_boundary, returntb=True)
    #     htbsc.nb = htbsc.nw
    #     self.htbsc = htbsc
    #     return htbsc


    '''
      * calculators
    '''
    def cal_band(self, kpath):
        kpath_car = LA.multi_dot([self.lattG, kpath.T]).T
        nk = kpath.shape[0]
        bandE = np.zeros([nk, self.nb], dtype='float64')
        for i, kc in zip(range(nk), kpath_car):
            print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i + 1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                kc[0], kc[1], kc[2]
                )
            )
            hk = self.get_hk(kc)
            E, U = LA.eigh(hk)
            bandE[i] = E - self.fermi
        return bandE

    def cal_slab_band(self, kpath, N, M, openboundary):
        kpath_car = LA.multi_dot([self.lattG, kpath.T]).T
        nk = kpath.shape[0]
        nb = self.nb * N
        bandE = np.zeros([nk, nb], dtype='float64')
        for i, kc in zip(range(nk), kpath_car):
            print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i + 1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                kc[0], kc[1], kc[2]
                )
            )
            hk = self.get_hk_slab(kc, N, M, openboundary)
            # hk = self.htb.get_hk(k)
            E, U = LA.eigh(hk)
            bandE[i] = E - self.fermi
        return bandE

    def cal_surfdos(self, kpath, ee, nprin=2, N=100, M=1, openboundary=1, eps=0.01):
        kpath_car = LA.multi_dot([self.lattG, kpath.T]).T
        nk = kpath.shape[0]
        ne = ee.shape[0]

        dos_L = np.zeros([nk, ne], dtype='float64')
        dos_R = np.zeros([nk, ne], dtype='float64')

        for i, kc in zip(range(nk), kpath_car):
            print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i + 1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                kc[0], kc[1], kc[2]
                )
            )
            dos_L[i], dos_R[i] = self.get_surf_green(ee, kc, nprin, N, M, openboundary, eps)
        return dos_L, dos_R

    def cal_surfcontour(self, meshk, ee, nprin=2, N=100, M=1, openboundary=1, eps=0.01):
        meshk_car = LA.multi_dot([self.lattG, meshk.T]).T
        nk = meshk.shape[0]
        ne = ee.shape[0]
        dos_L = np.zeros([nk, ne], dtype='float64')
        dos_R = np.zeros([nk, ne], dtype='float64')

        for i, kc in zip(range(nk), meshk_car):
            print(kc)
            print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                    i+1, nk,
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    kc[0], kc[1], kc[2]
                    )
                  )
            dos_L[i], dos_R[i] = self.get_surf_green(ee, kc, nprin, N, M, openboundary, eps)

        return dos_L, dos_R

class Double_Weyl(KP):
    def __init__(self):
        KP.__init__(self)
        self.nb = 2
        self.fermi = 0
        self.latt = np.eye(3)
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)

        self.M0 = 1
        self.M1 = 1
        self.M2 = 1
        self.A = 1

    # def get_hk(self, kc):
    #     kx, ky, kz = kc
    #     # Mk = self.M0 - self.M1 * (kx**2 + ky**2) - self.M2 * kz ** 2
    #     Mk = (self.M0 - 2*self.M1 - 4*self.M2) + 2*self.M1 * (np.cos(kx) + np.cos(ky)) + 2*self.M2 * np.cos(kz)
    #     hk = Mk * sigmaz + np.array([
    #         [0, (np.sin(kx) - 1j*np.sin(ky))**1],
    #         [(np.sin(kx) + 1j*np.sin(ky))**1, 0]
    #     ], dtype='complex128')
    #     return hk

    def get_hk(self, kc):
        Chern = 2
        kx, ky, kz = kc
        Mk = (self.M0 - 2*self.M1 - 4*self.M2) + 2*self.M1 * (np.cos(kx) + np.cos(ky)) + 2*self.M2 * np.cos(kz)
        hk = Mk * sigmaz + np.array([
            [0, (np.sin(kx) - 1j*np.sin(ky))**Chern],
            [(np.sin(kx) + 1j*np.sin(ky))**Chern, 0]
        ], dtype='complex128')
        return hk

class Two_C2_Weyl(KP):
    def __init__(self):
        KP.__init__(self)
        self.nb = 4
        self.fermi = 0
        self.latt = np.eye(3)
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)

        self.M0 = 1
        self.M1 = 1
        self.M2 = 1

        self.Ax = 1
        self.Ay = 1

        c = 0.5
        self.c5 = c
        self.c7 = c
        self.c9 = c
        self.c11 = c

    def get_hk(self, kc):
        kx, ky, kz = kc
        Mk = (self.M0 - 2*self.M1 - 4*self.M2) + 2*self.M1 * (np.cos(kx) + np.cos(ky)) + 2*self.M2 * np.cos(kz)
        hc1k = Mk * sigmaz + self.Ax * np.sin(kx) * sigmax + self.Ay * np.sin(ky) * sigmay
        hk = np.kron(sigma0, hc1k) + \
             self.c5 * np.sin(kx) * np.kron(sigmay, sigmay) + \
             self.c7 * np.sin(ky) * np.kron(sigmax, sigmax) + \
             -self.c9 * np.sin(kx) * np.kron(sigmax, sigmay) + \
             self.c11 * np.sin(ky) * np.kron(sigmay, sigmax)

        return hk

class Two_C2_Weyl_110(KP):

    def __init__(self):
        KP.__init__(self)
        self.nb = 4
        self.fermi = 0
        self.latt = np.array([
            [ 0.5,-0.5, 0],
            [ 0.5, 0.5, 0],
            [ 0.0, 0.0, 1],
        ]).T
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)

        self.C0 = 0
        self.C1 = 0.#1
        self.C2 = 0.#1
        self.C3 = -0.#15

        self.M0 = 0.8
        self.M1 = 0.25
        self.M3 = 0.25

        self.Ax = 0.2
        self.Ay = 0.2

        c = 0.0
        self.c5 = c
        self.c7 = c
        self.c9 = c
        self.c11 = c

        self.kcw = np.arccos(1 - 0.5 * (self.M0/self.M3))
        self.kw = np.arccos(1 - 0.5 * (self.M0/self.M3)) / np.pi / 2

    def get_hk(self, kc):
        k = LA.multi_dot([LA.inv(self.lattG), kc])
        k1, k2, k3 = k * 2 * np.pi
        Ck = (self.C0 + 2*self.C3 + 4*self.C1 + 4*self.C2) - 2*self.C3*np.cos(k3) - \
             2*self.C1*(np.cos(k1)+np.cos(k2)+np.sin(k1)*np.sin(k2)) - 2*self.C2*(np.cos(k1)+np.cos(k2)-np.sin(k1)*np.sin(k2))
        Mk = (self.M0 - 2*self.M3 - 8*self.M1) + 2*self.M3*np.cos(k3) + \
             4*self.M1*(np.cos(k1)+np.cos(k2))
        hc1k = Ck*sigma0 + Mk * sigmaz + \
               self.Ax * (np.sin(k1)+np.sin(k2)) * sigmax + \
               self.Ay * (-np.sin(k1)+np.sin(k2)) * sigmay
        hk = np.kron(sigma0, hc1k) + \
             self.c5 * (np.sin(k1)+np.sin(k2)) * np.kron(sigmay, sigmay) + \
             self.c7 * (-np.sin(k1)+np.sin(k2)) * np.kron(sigmax, sigmax) + \
             -self.c9 * (np.sin(k1)+np.sin(k2)) * np.kron(sigmax, sigmay) + \
             self.c11 * (-np.sin(k1)+np.sin(k2)) * np.kron(sigmay, sigmax)
        return hk

class Weyl(KP):

    def __init__(self):
        KP.__init__(self)
        self.nb = 2
        self.fermi = 0
        self.latt = np.eye(3)
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)

        self.C0 = 0
        self.C1 = 0#.1
        self.C3 = -0.#15

        self.M0 = 0.5
        self.M1 = 0.5
        self.M3 = 0.5

        self.Ax = 0.25
        self.Ay = 0.25

        self.kcw = np.arccos(1 - 0.5 * (self.M0/self.M3))
        self.kw = np.arccos(1 - 0.5 * (self.M0/self.M3)) / np.pi / 2

    def get_hk(self, kc):
        k = LA.multi_dot([LA.inv(self.lattG), kc])
        k1, k2, k3 = k * 2 * np.pi
        Ck = (self.C0 + 2*self.C3 + 4*self.C1) - 2*self.C3*np.cos(k3) - 2*self.C1*(np.cos(k1) + np.cos(k2))
        Mk = (self.M0 - 2*self.M3 - 4*self.M1) + 2*self.M3*np.cos(k3) + 2*self.M1*(np.cos(k1) + np.cos(k2))
        hk = Ck*sigma0 + Mk * sigmaz + \
             self.Ax * np.sin(k1) * sigmax + \
             self.Ay * np.sin(k2) * sigmay
        return hk

@MPI_Gather(MPI, iterprint=100)
def cal_surfcontour(k, dim, nprin=2, N=100, M=1, openboundary=1, eps=0.01):
    kc = LA.multi_dot([ham.lattG, k])
    dos_L, dos_R = ham.get_surf_green(ee, kc, nprin, N, M, openboundary, eps)
    return dos_L, dos_R


'''
  * Plot
'''
def plot_contour(dos, meshk_plot, cmap='inferno'):
    import matplotlib.pyplot as plt

    nk1 = len(set(meshk_plot.T[0]))
    nk2 = len(set(meshk_plot.T[1]))
    # cmap = 'seismic'
    # cmap = 'hot'
    # cmap = 'inferno'

    # nk, nw = bandE.shape
    # dos = np.log(dos)

    fig = plt.figure('dist')
    fig.clf()
    ax = fig.add_subplot(111)

    XX_MIN = meshk_plot.T[0].min()
    XX_MAX = meshk_plot.T[0].max()
    YY_MIN = meshk_plot.T[1].min()
    YY_MAX = meshk_plot.T[1].max()

    ax.axis([XX_MIN, XX_MAX, YY_MIN, YY_MAX])
    ax.axhline(0, color='white', linewidth=0.5, zorder=101)
    ax.axvline(0, color='white', linewidth=0.5, zorder=101)

    vmax = np.max(dos)
    vmin = np.min(dos)
    levels = np.linspace(vmin, vmax, 500)

    cs = ax.contourf(meshk_plot.T[0].reshape(nk1,nk2), meshk_plot.T[1].reshape(nk1,nk2), dos.reshape(nk1,nk2), levels, vmax=vmax, vmin=vmin, cmap=cmap)
    # plt.xlabel('$k_x$')
    # plt.ylabel('$k_y$')
    # plt.title('Fermi={:.4f} eV')

    cbar = plt.colorbar(cs)
    # cbar.set_label('Density of States')
    cbar.set_ticks(np.linspace(vmin, vmax, 5))

    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()

    plt.show()

if os.environ.get('PYGUI') == 'True':
    wdir = os.path.join(ROOT_WDIR, r'surfResponse/')
    input_dir = os.path.join(ROOT_WDIR, r'surfResponse/hamlib')
else:
    wdir = os.getcwd()
    # input_dir = r'/home/jincao/1-Works/8_NLO/8_weyl/htblib'


if __name__ == '__main__':
    '''
      * Job = 
      ** band
      ** surfdos
      ** surfcontor
    '''
    Job = r'band'

if __name__ == '__main__' and Job == 'band':

    # ham = Double_Weyl()
    ham = Two_C2_Weyl()
    ham = Two_C2_Weyl_110()
    ham = Weyl()

    nk1 = 51
    kpath_HSP = np.array([
        [ 0.5, 0.0, 0.0], #  a
        [ 0.0, 0.0, 0.0], #  G
        [ 0.0, 0.0, 0.5], #  c
    ])
    xlabel = ['a', 'G', 'c']

    kpath_HSP = np.array([
        [-0.5, 0.0, ham.kw-0.1], #  a
        [ 0.0, 0.0, ham.kw-0.1], #  G
        [ 0.5, 0.0, ham.kw-0.1], #  c
    ])
    xlabel = ['-a', 'G', 'a']

    # kpath_HSP = np.array([
    #     [ 0.1, 0.1,-0.5], #  a
    #     [ 0.1, 0.1, 0.0], #  G
    #     [ 0.1, 0.1, 0.5], #  c
    # ])
    # kpath_HSP = np.array([
    #     [ 0.1, 0.1,-0.5], #  a
    #     [ 0.1, 0.1, 0.0], #  G
    #     [ 0.1, 0.1, 0.5], #  c
    # ])
    # xlabel = ['-c', 'G', 'c']
    # kpath_HSP = np.array([
    #     [-0.5, 0.0, 0.05], #  a
    #     [ 0.0, 0.0, 0.05], #  G
    #     [ 0.5, 0.0, 0.05], #  c
    # ])
    # xlabel = ['-a', 'G', 'a']

    kpath = make_kpath(kpath_HSP, nk1 - 1)
    kpath_car = LA.multi_dot([ham.lattG, kpath.T]).T

    bandE = ham.cal_band(kpath)
    bandE = ham.cal_slab_band(kpath, N=50, M=2, openboundary=1)

    bandstructure_hsp = BandstructureHSP()
    bandstructure_hsp.HSP_list = kpath_HSP
    bandstructure_hsp.HSP_path_frac = kpath
    bandstructure_hsp.HSP_path_car = kpath_car
    bandstructure_hsp.HSP_name = xlabel

    bandstructure_hsp.eig = bandE
    bandstructure_hsp.nk, bandstructure_hsp.nb = bandE.shape

    if os.environ.get('PYGUI') == 'True':
        # bandstructure_hsp.plot_band(eemin=-1.5, eemax=1.5, unit='C')
        bandstructure_hsp.plot_band(eemin=-1, eemax=1, unit='D')
        plt.axvline(0.5+0.05, linestyle='--', color='k', linewidth=0.5)
        plt.axvline(0.5+0.1, linestyle='--', color='k', linewidth=0.5)

        gate = np.array([-0.5, -0.4, -0.3, -0.2, -0.1])
        for i in gate:
            plt.axhline(i, linestyle='-', color='k', linewidth=0.5)

if __name__ == '__main__' and Job == 'surfdos':

    ham = Double_Weyl()
    ham = Two_C2_Weyl()
    ham = Two_C2_Weyl_110()

    nk1 = 101
    kpath_HSP = np.array([
        [ 0.5, 0.0, 0.0], #  a
        [ 0.0, 0.0, 0.0], #  G
        [ 0.0, 0.0, 0.5], #  c
    ])
    xlabel = ['a', 'G', 'c']
    # kpath_HSP = np.array([
    #     [-0.5, 0.0, 0.0], #  a
    #     [ 0.0, 0.0, 0.0], #  G
    #     [ 0.5, 0.0, 0.0], #  c
    # ])
    # xlabel = ['-a', 'G', 'a']
    # aa = 0.05
    # kpath_HSP = np.array([
    #     [aa, 0.0,-0.5], #  a
    #     [aa, 0.0, 0.0], #  G
    #     [aa, 0.0, 0.5], #  c
    # ])
    # xlabel = ['-c', 'G', 'c']
    eemin = -1.0
    eemax = 1.0
    ne = 300

    kpath = make_kpath(kpath_HSP, nk1 - 1)
    kpath_car = LA.multi_dot([ham.lattG, kpath.T]).T
    ee = np.linspace(eemin, eemax, ne)

    dos_L, dos_R = ham.cal_surfdos(kpath, ee, nprin=3, N=100, M=2, openboundary=1, eps=0.01)

    bandstructure_hsp = BandstructureHSP()
    bandstructure_hsp.HSP_list = kpath_HSP
    bandstructure_hsp.HSP_path_frac = kpath
    bandstructure_hsp.HSP_path_car = kpath_car
    bandstructure_hsp.HSP_name = xlabel
    bandstructure_hsp.ee = ee

    bandstructure_hsp.surfDOS1 = dos_L
    bandstructure_hsp.surfDOS2 = dos_R

    if os.environ.get('PYGUI') == 'True':
        bandstructure_hsp.plot_surfDOS(ee, np.log(dos_L), eemin, eemax, unit='C', cmap='seismic')

if __name__ == '__main__' and Job == 'surfcontor':
    savefname = r'greencontor.npz'
    ham = Double_Weyl()
    ham = Two_C2_Weyl()
    ham = Two_C2_Weyl_110()

    ee = np.array([0, -0.1, -0.2, -0.3, -0.4])
    ee = np.array([-0.1, -0.2])
    nkmesh = [48, 1, 48]
    kmesh_shift = np.array([-0.5, 0, -0.5])

    kmesh = make_mesh(nkmesh) + kmesh_shift
    # dos_L, dos_R = ham.cal_surfcontour(meshk, ee, nprin=3, N=100, M=2, openboundary=1, eps=0.01)

    dim = [2, ee.shape[0]]
    dos_L, dos_R = cal_surfcontour(kmesh, dim, nprin=3, N=100, M=2, openboundary=1, eps=0.01)

    if MPI_rank == 0:

        np.savez_compressed(savefname,
                            ee=ee,
                            nkmesh=nkmesh,
                            kmesh_shift=kmesh_shift,
                            kmesh=kmesh,
                            dos_L=dos_L,
                            dos_R=dos_R,
                            )

        if os.environ.get('PYGUI') == 'True':
            npdata = np.load(savefname)
            ee = npdata['ee']
            dos_L = npdata['dos_L']
            dos_R = npdata['dos_R']
            kmesh = npdata['kmesh']
            npdata.close()

            dos = dos_L.T[0]
            plot_contour(np.log(dos), np.delete(kmesh, 1, 1))

