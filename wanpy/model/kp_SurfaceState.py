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

__date__ = "Nov. 9, 2019"


import time
import os
import sys
import getpass
sys.path.append(os.environ.get('PYTHONPATH'))

if os.environ.get('PYGUI') == 'True':
    from pylab import *
    from wanpy.core.plot import *
    from wanpy.response.response_plot import *

import numpy as np
from numpy import linalg as LA

from wanpy.core.units import *
from wanpy.core.mesh import make_kpath
from wanpy.core.structure import BandstructureHSP
from wanpy.core.greenfunc import self_energy

from wanpy.MPI.MPI import MPI_Reduce, MPI_Gather
from wanpy.MPI.init_data import init_kmesh, init_htb_response_data

from mpi4py import MPI

COMM = MPI.COMM_WORLD
MPI_rank = COMM.Get_rank()
MPI_ncore = COMM.Get_size()


class Graphene_kpmodel(object):

    def __init__(self):
        self.nb = 2

        self.grapheneA = 2.46
        self.tAB = 1.8

        self.latt = self.grapheneA * np.array([
            [np.sqrt(3)/2, -1/2, 0],
            [np.sqrt(3)/2, 1/2, 0],
            [0, 0, 1],
        ]).T
        self.latt[2, 2] = 10
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)
        self.K1 = LA.multi_dot([self.lattG, np.array([2/3, 1/3, 0])])
        self.K2 = LA.multi_dot([self.lattG, np.array([1/3, 2/3, 0])])
        self.tauA = (self.latt.T[0] + self.latt.T[1]) / 3
        self.tauB = 2 * (self.latt.T[0] + self.latt.T[1]) / 3
        self.d1 = self.tauB - self.tauA
        self.d2 = self.tauB - self.tauA - self.latt.T[0]
        self.d3 = self.tauB - self.tauA - self.latt.T[1]

    def get_hk(self, kc):
        fk = np.exp(1j * kc.dot(self.d1)) + np.exp(1j * kc.dot(self.d2)) + np.exp(1j * kc.dot(self.d3))
        hk = self.tAB * np.array([
            [0, fk],
            [fk.conj(), 0]
        ], dtype='complex128')
        return hk

    def get_hk_slab(self, k, N2=20, ribbon='zigzag'):
        '''
          ribbon 
        ''' #

        if ribbon == 'armchair':
            surlattG = np.array([
                self.lattG.T[0] + self.lattG.T[1],
                self.lattG.T[1] - self.lattG.T[0],
                self.lattG.T[2]
            ]).T
            surlatt = LA.inv(surlattG.T / 2 / np.pi)
        elif ribbon == 'zigzag':
            surlattG = np.array([
                self.lattG.T[1] - self.lattG.T[0],
                self.lattG.T[0] + self.lattG.T[1],
                self.lattG.T[2]
            ]).T
            surlatt = LA.inv(surlattG.T / 2 / np.pi)
        else:
            surlattG = None
            surlatt = None

        kk = np.zeros([N2, 3], dtype='float64')
        kk.T[0] = k[0]
        kk.T[1] = np.linspace(0, 1, N2+1)[:-1]
        kkc = LA.multi_dot([surlattG, kk.T]).T

        nR = N2+1
        R = np.zeros([nR, 3], dtype='float64')
        degen = np.zeros([nR], dtype='int64')
        R.T[1] = np.arange(nR) - N2 // 2
        degen[np.array([0, -1])] = 2

        hk = np.array([
            self.get_hk(kkc[ik])
            for ik in range(N2)
        ])
        eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', kk, R))
        hR = np.einsum('kR,kmn->Rmn', eikR, hk) / N2

        # h0 = hR[nR//2]
        # hl1 = hR[nR//2-1]
        # hr1 = hR[nR//2+1]
        h0 = np.block([
            [hR[nR//2], hR[nR//2+1]],
            [hR[nR//2-1], hR[nR//2]],
        ])
        hr1 = np.block([
            [hR[nR//2+2], hR[nR//2+3]],
            [hR[nR//2+1], hR[nR//2+2]],
        ])
        hl1 = hr1.T.conj()
        hk_slab = np.kron(np.eye(N2), h0) + \
                  np.kron(np.eye(N2, N2, 1), hr1) + \
                  np.kron(np.eye(N2, N2, -1), hl1)
        return hk_slab

    def get_slab_green(self, ee, k, N2=20, eps=0.01, ribbon='zigzag'):
        ne = ee.shape[0]

        if ribbon == 'armchair':
            surlattG = np.array([
                self.lattG.T[0] + self.lattG.T[1],
                self.lattG.T[1] - self.lattG.T[0],
                self.lattG.T[2]
            ]).T
            surlatt = LA.inv(surlattG.T / 2 / np.pi)
        elif ribbon == 'zigzag':
            surlattG = np.array([
                self.lattG.T[1] - self.lattG.T[0],
                self.lattG.T[0] + self.lattG.T[1],
                self.lattG.T[2]
            ]).T
            surlatt = LA.inv(surlattG.T / 2 / np.pi)
        else:
            surlattG = None
            surlatt = None

        '''
          * F.T. to real space
        '''
        kk = np.zeros([N2, 3], dtype='float64')
        kk.T[0] = k[0]
        kk.T[1] = np.linspace(0, 1, N2+1)[:-1]
        kkc = LA.multi_dot([surlattG, kk.T]).T

        nR = N2+1
        R = np.zeros([nR, 3], dtype='float64')
        degen = np.zeros([nR], dtype='int64')
        R.T[1] = np.arange(nR) - N2 // 2
        degen[np.array([0, -1])] = 2

        hk = np.array([
            self.get_hk(kkc[ik])
            for ik in range(N2)
        ])
        eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', kk, R))
        hR = np.einsum('kR,kmn->Rmn', eikR, hk) / N2

        h0 = hR[nR//2]
        h1R = hR[nR//2+1]
        h1L = hR[nR//2-1]
        # h0 = np.block([
        #     [hR[nR//2], hR[nR//2+1]],
        #     [hR[nR//2-1], hR[nR//2]],
        # ])
        # h1R = np.block([
        #     [hR[nR//2+2], hR[nR//2+3]],
        #     [hR[nR//2+1], hR[nR//2+2]],
        # ])
        # h1L = h1R.T.conj()

        '''
          * green func
        '''
        GLr = np.zeros([ne, self.nb, self.nb], dtype='complex128')
        GRr = np.zeros([ne, self.nb, self.nb], dtype='complex128')
        DOS_L = np.zeros([ne], dtype='float64')
        DOS_R = np.zeros([ne], dtype='float64')

        for i, _e in zip(range(ne), ee):
            e = (_e + 1j * eps) * np.identity(self.nb)
            selfenLr = self_energy(e - h0, e - h0, h1L, h1L.T.conj())
            selfenRr = self_energy(e - h0, e - h0, h1R, h1R.T.conj())
            GLr[i] = LA.inv(e - h0 - selfenLr)
            GRr[i] = LA.inv(e - h0 - selfenRr)
            DOS_L[i] = -2 * np.imag(np.trace(GLr[i]))
            DOS_R[i] = -2 * np.imag(np.trace(GRr[i]))

        return DOS_L, DOS_R


class QAH_kpmodel(object):

    def __init__(self):
        self.nb = 2

        self.A = 0.5
        self.B = 0.5
        self.m = 0.5
        self.chern_number = (np.sign(self.m) + np.sign(self.B)) / 2

        self.a = 1
        self.latt = self.a * np.identity(3)
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)

    def get_hk(self, kc):
        kx, ky, kz = kc
        dx = self.A * np.sin(kx * self.a)
        dy = self.A * np.sin(ky * self.a)
        dz = self.m - 4 * self.B * (np.sin(kx*self.a/2) ** 2 + np.sin(ky*self.a/2) ** 2)
        hk = dx * sigma_x + dy * sigma_y + dz * sigma_z
        return hk

    def get_hk_slab(self, kc, Ny):
        kx, ky, kz = kc

        kky = 2 * np.pi * np.linspace(0, 1, Ny+1)[:-1]
        Ry = np.arange(Ny+1) - Ny // 2
        degen = np.ones_like(Ry)
        degen[np.array([0, -1])] = 2
        nR = Ny+1

        hk = np.array([
            self.get_hk(np.array([kx, kky[ik], 0]))
            for ik in range(Ny)
        ])
        eikR = np.exp(1j * np.einsum('k,R->kR', kky, Ry))
        hR = np.einsum('kR,kmn->Rmn', eikR, hk) / Ny

        h0 = np.block([
            [hR[nR//2], hR[nR//2+1]],
            [hR[nR//2-1], hR[nR//2]],
        ])
        hr1 = np.block([
            [hR[nR//2+2], hR[nR//2+3]],
            [hR[nR//2+1], hR[nR//2+2]],
        ])
        hl1 = hr1.T.conj()
        hk_slab = np.kron(np.eye(Ny), h0) + \
                  np.kron(np.eye(Ny, Ny, 1), hr1) + \
                  np.kron(np.eye(Ny, Ny, -1), hl1)

        hk_slab = np.kron(np.eye(Ny), hR[nR//2]) + \
                  np.kron(np.eye(Ny, Ny, 1), hR[nR//2+1]) + \
                  np.kron(np.eye(Ny, Ny, -1), hR[nR//2-1])
        return hk_slab

    def get_slab_green(self, ee, kc, Ny=20, eps=0.01):
        kx, ky, kz = kc
        ne = ee.shape[0]

        GLr = np.zeros([ne, self.nb, self.nb], dtype='complex128')
        GRr = np.zeros([ne, self.nb, self.nb], dtype='complex128')
        DOS_L = np.zeros([ne], dtype='complex128')
        DOS_R = np.zeros([ne], dtype='complex128')

        kky = 2 * np.pi * np.linspace(0, 1, Ny + 1)[:-1]
        Ry = np.arange(Ny + 1) - Ny // 2
        degen = np.ones_like(Ry)
        degen[np.array([0, -1])] = 2
        nR = Ny + 1

        hk = np.array([
            self.get_hk(np.array([kx, kky[ik], 0]))
            for ik in range(Ny)
        ])
        eikR = np.exp(1j * np.einsum('k,R->kR', kky, Ry))
        hR = np.einsum('kR,kmn->Rmn', eikR, hk) / Ny

        h0 = hR[nR//2]
        h1L = hR[nR//2-1]
        h1R = hR[nR//2+1]

        for i, _e in zip(range(ne), ee):
            e = (_e + 1j * eps) * np.identity(self.nb)
            selfenLr = self_energy(e - h0, e - h0, h1L, h1L.T.conj())
            selfenRr = self_energy(e - h0, e - h0, h1R, h1R.T.conj())
            GLr[i] = LA.inv(e - h0 - selfenLr)
            GRr[i] = LA.inv(e - h0 - selfenRr)
            DOS_L[i] = -2 * np.imag(np.trace(GLr[i]))
            DOS_R[i] = -2 * np.imag(np.trace(GRr[i]))

        return DOS_L, DOS_R


class Weyl_kpmodel(object):
    # Inversion breaking Weyl model with four nodes(WPs)

    def __init__(self):
        self.nb = 4

        self.A = 2
        self.B1 = 1
        self.B2 = 0.25
        self.m = 1
        self.v = 1

        self.kw = np.sqrt(self.m*self.v**2/self.B2)
        self.k0 = 0 # self.kw

        self.a = 1
        self.latt = self.a * np.identity(3)
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)

    def get_h0(self, kx, ky, kz):
        t1 = 1
        t2 = 1
        M = 1
        t = 1
        kw = 2 * np.arcsin(M/4/t1)
        Mk = M - 4 * t1 * np.sin(kz/2)**2 - 4 * t2 * (np.sin(kx/2)**2+np.sin(ky/2)**2)
        hk = 2 * t * (np.sin(kx) * sigma_x + np.sin(ky) * sigma_y) + Mk * sigma_z
        return hk

    # def get_h0(self, kx, ky, kz):
    #     Mk = self.m * self.v ** 2 - self.B1 * kx ** 2 - self.B1 * ky ** 2 - self.B2 * kz ** 2
    #     h = self.A * (kx * sigma_x + ky * sigma_y) + Mk * sigma_z
    #     return h

    def get_hk(self, kc):
        kx, ky, kz = kc

        k0 = 0.5
        h11 = self.get_h0(kx-k0, ky, kz)
        h22 = np.conj(self.get_h0(-kx-k0, -ky, kz))
        zero = np.zeros([2, 2], dtype='complex128')
        hk = np.block([
            [h11, zero],
            [zero, h22]
        ])
        return hk



    def get_hk_slab(self, kc, Ny):
        kx, ky, kz = kc

        kky = 2 * np.pi * np.linspace(0, 1, Ny+1)[:-1]
        Ry = np.arange(Ny+1) - Ny // 2
        degen = np.ones_like(Ry)
        degen[np.array([0, -1])] = 2
        nR = Ny+1

        hk = np.array([
            self.get_hk(np.array([kx, kky[ik], kz]))
            for ik in range(Ny)
        ])
        eikR = np.exp(1j * np.einsum('k,R->kR', kky, Ry))
        hR = np.einsum('kR,kmn->Rmn', eikR, hk) / Ny

        hk_slab = np.kron(np.eye(Ny), hR[nR//2])
        for i in range(1, nR//2-1):
            hk_slab += np.kron(np.eye(Ny, Ny, i), hR[nR//2+i]) + \
                       np.kron(np.eye(Ny, Ny, -i), hR[nR//2-i])
        # hk_slab[0:4, :] = 1e10
        # hk_slab[-5:-1, :] = 1e10
        # hk_slab = 0.5 * (hk_slab+hk_slab.T.conj())
        return hk_slab


'''
  * Calculators
'''
def cal_band(ham, kpath, nb):
    nk = kpath.shape[0]
    bandE = np.zeros([nk, nb], dtype='float64')

    for i, k in zip(range(nk), kpath):
        print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i+1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                k[0], k[1], k[2]
                )
              )
        hk = ham.get_hk(k)
        E, U = LA.eigh(hk)
        bandE[i] = E

    return bandE

def cal_slab_band(ham, kpath, nb):
    nk = kpath.shape[0]
    bandE = np.zeros([nk, nb], dtype='float64')

    for i, k in zip(range(nk), kpath):
        print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i+1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                k[0], k[1], k[2]
                )
              )
        hk = ham.get_hk_slab(k, Ny)
        E, U = LA.eigh(hk)
        bandE[i] = E

    return bandE

def cal_surfDOS(ham, kpath, ne):
    nk = kpath.shape[0]
    surfDOS_L = np.zeros([nk, ne], dtype='float64')
    surfDOS_R = np.zeros([nk, ne], dtype='float64')

    for i, kc in zip(range(nk), kpath):
        print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i+1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                kc[0], kc[1], kc[2]
                )
              )
        surfDOS_L[i], surfDOS_R[i] = ham.get_slab_green(ee, kc, Ny, eps)

    return surfDOS_L, surfDOS_R

'''
  * sys = 
    graphene
    QAH
    weyl
'''
sys = r'weyl'

if __name__ == '__main__':
    pass
    # ham = QAH_kpmodel()
    # ham = Graphene_kpmodel()
    # ham = Weyl_kpmodel()


if __name__ == '__main__' and sys == 'QAH':
    ham = QAH_kpmodel()

    eps = 0.01
    ne = 100
    ee = np.linspace(-6, 6, ne)

    Ny = 20
    nk1 = 101
    kpath_HSP = np.array([
        [-1/2, 0.0, 0.0], #  X
        [ 0.0, 0.0, 0.0], #  G
        [ 1/2, 0.0, 0.0], #  X
    ])
    xlabel = ['-X', 'G', 'X']

    kpath = make_kpath(kpath_HSP, nk1 - 1)
    kpath_car = multi_dot([ham.lattG, kpath.T]).T

    # bandE = cal_band(ham, kpath_car, ham.nb)
    bandE = cal_slab_band(ham, kpath_car, Ny*ham.nb*1)
    # surfDOS_L, surfDOS_R = cal_surfDOS(ham, kpath_car, ne)

    bandstructure_hsp = BandstructureHSP()
    bandstructure_hsp.HSP_list = kpath_HSP
    bandstructure_hsp.HSP_path_frac = kpath
    bandstructure_hsp.HSP_path_car = kpath_car
    bandstructure_hsp.HSP_name = xlabel
    bandstructure_hsp.ee = ee

    # bandstructure_hsp.surfDOS1 = surfDOS_L
    # bandstructure_hsp.surfDOS2 = surfDOS_R

    bandstructure_hsp.eig = bandE
    bandstructure_hsp.nk, bandstructure_hsp.nb = bandE.shape

    if os.environ.get('PYGUI') == 'True':
        bandstructure_hsp.plot_band(eemin=-3, eemax=3, unit='C')
        # bandstructure_hsp.plot_surfDOS(ee, surfDOS_L, eemin=-6, eemax=6, unit='C')
        # bandstructure_hsp.plot_surfDOS(ee, surfDOS_R, eemin=-6, eemax=6, unit='C')


if __name__ == '__main__' and sys == 'graphene':
    ham = Graphene_kpmodel()

    eps = 0.01
    ne = 100
    ee = np.linspace(-6, 6, ne)

    Ny = 40
    nk1 = 101
    kpath_HSP = np.array([
        [-1/3, 1/3, 0.0], #  K
        [ 0.0, 0.0, 0.0], #  G
        [ 0.0, 1/2, 0.0], #  M
        [ 1/3, 2/3, 0.0], # -K
    ])
    xlabel = ['K', 'G', 'M', '-K']

    kpath_HSP = np.array([
        [-1/2, 0.0, 0.0], #  X
        [ 0.0, 0.0, 0.0], #  G
        [ 1/2, 0.0, 0.0], #  X
    ])
    xlabel = ['-X', 'G', 'X']

    kpath = make_kpath(kpath_HSP, nk1 - 1)
    kpath_car = multi_dot([ham.lattG, kpath.T]).T

    # bandE = cal_band(ham, kpath, ham.nb)
    bandE = cal_slab_band(ham, kpath, Ny*ham.nb*2)
    # surfDOS_L, surfDOS_R = cal_surfDOS(ham, kpath, ne)

    bandstructure_hsp = BandstructureHSP()
    bandstructure_hsp.HSP_list = kpath_HSP
    bandstructure_hsp.HSP_path_frac = kpath
    bandstructure_hsp.HSP_path_car = kpath_car
    bandstructure_hsp.HSP_name = xlabel
    bandstructure_hsp.ee = ee

    # bandstructure_hsp.surfDOS1 = surfDOS_L
    # bandstructure_hsp.surfDOS2 = surfDOS_R

    bandstructure_hsp.eig = bandE
    bandstructure_hsp.nk, bandstructure_hsp.nb = bandE.shape

    if os.environ.get('PYGUI') == 'True':
        bandstructure_hsp.plot_band(eemin=-6, eemax=6, unit='C')
        # bandstructure_hsp.plot_surfDOS(ee, surfDOS_L, eemin=-6, eemax=6, unit='C')
        # bandstructure_hsp.plot_surfDOS(ee, surfDOS_R, eemin=-6, eemax=6, unit='C')

        # np.savez_compressed(r'RobbonBand.npz',
        #                     bandstructure_hsp=bandstructure_hsp,
        #                     )
        #
        # npdata = np.load(r'RobbonBand.npz')
        # bandHSP = npdata['bandstructure_hsp'].item()


if __name__ == '__main__' and sys == 'weyl':
    ham = Weyl_kpmodel()

    eps = 0.01
    ne = 100
    ee = np.linspace(-6, 6, ne)

    Ny = 20 * 2 # Ny should be even number
    nk1 = 51
    kpath_HSP = np.array([
        [ 0, 0.0, -1/2], #  X
        [ 0.0, 0.0, 0.0], #  G
        [ 0, 0.0, 1/2], #  X
    ])
    xlabel = ['-X', 'G', 'X']

    # kpath_HSP = np.array([
    #     [0, -1/2, 0.0], #  X
    #     [ 0.0, 0.0, 0.0], #  G
    #     [ 0, 1/2, 0.0], #  X
    # ])
    # xlabel = ['-X', 'G', 'X']

    kpath = make_kpath(kpath_HSP, nk1 - 1)
    kpath_car = multi_dot([ham.lattG, kpath.T]).T
    kpath_car[:,0] = -0.5

    # bandE = cal_band(ham, kpath_car, ham.nb)
    bandE = cal_slab_band(ham, kpath_car, Ny*ham.nb)
    # surfDOS_L, surfDOS_R = cal_surfDOS(ham, kpath, ne)

    bandstructure_hsp = BandstructureHSP()
    bandstructure_hsp.HSP_list = kpath_HSP
    bandstructure_hsp.HSP_path_frac = kpath
    bandstructure_hsp.HSP_path_car = kpath_car
    bandstructure_hsp.HSP_name = xlabel
    bandstructure_hsp.ee = ee

    # bandstructure_hsp.surfDOS1 = surfDOS_L
    # bandstructure_hsp.surfDOS2 = surfDOS_R

    bandstructure_hsp.eig = bandE
    bandstructure_hsp.nk, bandstructure_hsp.nb = bandE.shape

    if os.environ.get('PYGUI') == 'True':
        bandstructure_hsp.plot_band(eemin=-10, eemax=10, unit='C')
        # bandstructure_hsp.plot_surfDOS(ee, surfDOS_L, eemin=-6, eemax=6, unit='C')
        # bandstructure_hsp.plot_surfDOS(ee, surfDOS_R, eemin=-6, eemax=6, unit='C')

        # np.savez_compressed(r'RobbonBand.npz',
        #                     bandstructure_hsp=bandstructure_hsp,
        #                     )
        #
        # npdata = np.load(r'RobbonBand.npz')
        # bandHSP = npdata['bandstructure_hsp'].item()



