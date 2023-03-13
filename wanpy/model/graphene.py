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

__date__ = "Sep. 5, 2019"


import os
import sys

sys.path.append(os.environ.get('PYTHONPATH'))

import numpy as np
import numpy.linalg as LA

from wanpy.core.DEL.read_write import Cell
from wanpy.response._DEL.response import Htb_response

from wanpy.core.DEL.read_write import Htb


class GrapheneTB(Htb):
    '''
      * SK for graphene
    ''' #

    def __init__(self, *args, **kwargs):
        Htb.__init__(self, *args, **kwargs)

        self.grapheneA = 2.46
        self.grapheneC = 3.015

        self.name = 'Graphene TB Model'
        self.fermi = 0
        self.nw = 2
        self.nR = 5
        self.R = np.array([
            [ 0, -1,  0],
            [-1,  0,  0],
            [ 0,  0,  0],
            [ 1,  0,  0],
            [ 0,  1,  0],
        ], dtype='int64')
        self.ndegen = np.ones([self.nR], dtype='int64')

        self.cell = self._init_cell()
        self.wcc = self.cell.ions_car
        self.wccf = self.cell.ions

        self.hr_Rmn = self._init_hr_Rmn()
        self.r_Ramn = self._init_r_Ramn()

    def _init_cell(self):
        cell = Cell()
        cell.name = 'Graphene'
        cell.lattice = self.grapheneA * np.array([
            [1, 0,  0],
            [1/2, 3**0.5/2,  0],
            [0,  0, 100/self.grapheneA],
        ]).T
        cell.latticeG = cell.get_latticeG()
        cell.N = 2
        cell.spec = ['C', 'C']
        cell.ions = np.array([
            [1/3, 1/3, 0],
            [2/3, 2/3, 0],
        ])
        cell.ions.T[2] = 0 # 3.015 / 100
        cell.ions_car = cell.get_ions_car()
        return cell

    def _init_hr_Rmn(self):
        hr_Rmn = np.zeros([self.nR, self.nw, self.nw], dtype='complex128')
        t = -2.7
        hr_Rmn[0] = t * np.array([
            [0, 1],
            [0, 0],
        ])
        hr_Rmn[1] = t * np.array([
            [0, 1],
            [0, 0],
        ])
        hr_Rmn[2] = t * np.array([
            [0, 1],
            [1, 0],
        ])
        hr_Rmn[3] = t * np.array([
            [0, 0],
            [1, 0],
        ])
        hr_Rmn[4] = t * np.array([
            [0, 0],
            [1, 0],
        ])
        return hr_Rmn

    def _init_r_Ramn(self):
        r_Ramn = np.zeros([self.nR, 3, self.nw, self.nw], dtype='complex128')
        noise_OD = 0.1 * np.random.random()
        noise_OD = np.array([
            [0, noise_OD],
            [noise_OD, 0],
        ])
        # r_Ramn += 0.03 * np.random.random([self.nR, 3, self.nw, self.nw])
        r_Ramn[2, 0] = np.diag(self.cell.ions_car.T[0]) #+ noise_OD
        r_Ramn[2, 1] = np.diag(self.cell.ions_car.T[1]) #+ noise_OD
        r_Ramn[2, 2] = np.diag(self.cell.ions_car.T[2]) #+ noise_OD

        return r_Ramn

    def get_hk(self, kc):
        return self.get_hk_wgauge(kc)

    def get_hk0(self, k):
        # Rc = LA.multi_dot([self.cell.lattice, self.R.T]).T
        eikr = np.exp(2j * np.pi * np.einsum('a,Ra', k, self.R))
        hk = np.einsum('R,Rmn->mn', eikr, self.hr_Rmn, optimize=True)
        hk = 0.5 * (hk + hk.conj().T)
        return hk

    def get_hk_wgauge(self, kc):
        Rc = LA.multi_dot([self.cell.lattice, self.R.T]).T
        eikr = np.exp(1j * np.einsum('a,Ra', kc, Rc))
        hk = np.einsum('R,Rmn->mn', eikr, self.hr_Rmn, optimize=True)
        hk = 0.5 * (hk + hk.conj().T)
        return hk

    def get_hk_tbgauge(self, kc):
        sk = np.exp(-1j * np.einsum('a,ia', kc, self.wcc))

        Rc = LA.multi_dot([self.cell.lattice, self.R.T]).T
        eikr = np.exp(1j * np.einsum('a,Ra', kc, Rc))
        hk = np.einsum('R,m,Rmn,n->mn', eikr, sk, self.hr_Rmn, sk.conj(), optimize=True)
        hk = 0.5 * (hk + hk.conj().T)
        return hk

    def w90gethk(self, kc):
        return self.get_hk(kc)


class GrapheneTB_double_layer(Htb):
    '''
      * Slonczewski–Weiss–McClure model for double layer graphene
    ''' #

    def __init__(self, *args, **kwargs):
        Htb.__init__(self, *args, **kwargs)

        self.grapheneA = 2.46
        self.grapheneC = 3.015

        self.delta_prime = 1*0.050 # represents the on-site potential of dimer sites with respect to nondimer sites
        self.gamma1 = 0.400 #  coupling between dimer sites
        self.gamma3 = 0.320 #  trigonal warping
        self.gamma4 = 0.044 #  electron-hole asymmetry

        self.v3 = 0.5 * 3 ** 0.5 * self.grapheneA * self.gamma3
        self.v4 = 0.5 * 3 ** 0.5 * self.grapheneA * self.gamma4
        self.vF = 2.1354 * self.grapheneA # graphene dirac fermi verlocity

        self.name = 'Double layer graphene TB Model'
        self.fermi = 0
        self.nw = 4
        self.nR = 7
        self.R = np.array([
            [-1, -1,  0],
            [ 0, -1,  0],
            [-1,  0,  0],
            [ 0,  0,  0],
            [ 1,  0,  0],
            [ 0,  1,  0],
            [ 1,  1,  0],
        ], dtype='int64')
        self.ndegen = np.ones([self.nR], dtype='int64')

        self.cell = self._init_cell()
        self.wcc = self.cell.ions_car
        self.wccf = self.cell.ions

        self.hr_Rmn = self._init_hr_Rmn()
        self.r_Ramn = self._init_r_Ramn()

    def _init_cell(self):
        cell = Cell()
        cell.name = 'Graphene'
        vac = 100
        cell.lattice = self.grapheneA * np.array([
            [1, 0,  0],
            [1/2, 3**0.5/2,  0],
            [0,  0, vac/self.grapheneA],
        ]).T
        cell.latticeG = cell.get_latticeG()
        cell.N = 2
        cell.spec = ['C', 'C', 'C', 'C']
        cell.ions = np.array([
            [3/6, 3/6, 0], # (L1, A)
            [5/6, 5/6, 0], # (L1, B)
            [1/6, 1/6, self.grapheneC], # (L2, A)
            [3/6, 3/6, self.grapheneC], # (L2, B)
        ])
        cell.ions.T[2] /= vac
        cell.ions_car = cell.get_ions_car()
        return cell

    def _init_hr_Rmn(self):
        hr_Rmn = np.zeros([self.nR, self.nw, self.nw], dtype='complex128')
        gamma0 = -3.16
        gamma1 = 0.381
        gamma3 = 0.38
        gamma4 = 0.14
        delta = 0.022
        delta_prime = 0.022

        # L1 Dirac
        hr_Rmn[3, 0, 1] = gamma0
        hr_Rmn[3, 1, 0] = gamma0
        hr_Rmn[4, 1, 0] = gamma0
        hr_Rmn[5, 1, 0] = gamma0
        # L2 Dirac
        hr_Rmn[3, 3, 2] = gamma0
        hr_Rmn[3, 2, 3] = gamma0
        hr_Rmn[4, 3, 2] = gamma0
        hr_Rmn[5, 3, 2] = gamma0
        # on-site potential of dimer sites
        hr_Rmn[3, 1, 1] = delta_prime
        hr_Rmn[3, 2, 2] = delta_prime
        # gamma1: coupling between dimer sites
        hr_Rmn[3, 1, 2] = gamma1
        hr_Rmn[3, 2, 1] = gamma1
        # v3: trigonal warping, layer A-B
        hr_Rmn[4, 3, 0] = gamma3
        hr_Rmn[5, 3, 0] = gamma3
        hr_Rmn[6, 3, 0] = gamma3
        # v4: electron-hole asymmetry, layer A-A and B-B
        hr_Rmn[3, 0, 2] = gamma4
        hr_Rmn[3, 2, 0] = gamma4
        hr_Rmn[4, 2, 0] = gamma4
        hr_Rmn[5, 2, 0] = gamma4

        hr_Rmn[3, 1, 3] = gamma4
        hr_Rmn[3, 3, 1] = gamma4
        hr_Rmn[4, 3, 1] = gamma4
        hr_Rmn[5, 3, 1] = gamma4

        hr_Rmn[2] = hr_Rmn[4].conj().T
        hr_Rmn[1] = hr_Rmn[5].conj().T
        hr_Rmn[0] = hr_Rmn[6].conj().T

        return hr_Rmn

    def _init_r_Ramn(self):
        r_Ramn = np.zeros([self.nR, 3, self.nw, self.nw], dtype='complex128')
        r_Ramn[self.nR//2, 0] = np.diag(self.cell.ions_car.T[0])
        r_Ramn[self.nR//2, 1] = np.diag(self.cell.ions_car.T[1])
        r_Ramn[self.nR//2, 2] = np.diag(self.cell.ions_car.T[2])
        return r_Ramn

    def get_hk(self, kc):
        return self.get_hk_wgauge(kc)

    def get_hk0(self, k):
        # Rc = LA.multi_dot([self.cell.lattice, self.R.T]).T
        eikr = np.exp(2j * np.pi * np.einsum('a,Ra', k, self.R))
        hk = np.einsum('R,Rmn->mn', eikr, self.hr_Rmn, optimize=True)
        hk = 0.5 * (hk + hk.conj().T)
        return hk

    def get_hk_wgauge(self, kc):
        Rc = LA.multi_dot([self.cell.lattice, self.R.T]).T
        eikr = np.exp(1j * np.einsum('a,Ra', kc, Rc))
        hk = np.einsum('R,Rmn->mn', eikr, self.hr_Rmn, optimize=True)
        hk = 0.5 * (hk + hk.conj().T)
        return hk

    def get_hk_tbgauge(self, kc):
        sk = np.exp(-1j * np.einsum('a,ia', kc, self.wcc))

        Rc = LA.multi_dot([self.cell.lattice, self.R.T]).T
        eikr = np.exp(1j * np.einsum('a,Ra', kc, Rc))
        hk = np.einsum('R,m,Rmn,n->mn', eikr, sk, self.hr_Rmn, sk.conj(), optimize=True)
        hk = 0.5 * (hk + hk.conj().T)
        return hk

    def w90gethk(self, kc):
        return self.get_hk(kc)



if __name__ == '__main__':
    wdir = r'C:\Users\Jin\Research\test_wan90_setup\graphene_scell'
    wdir = r'C:\Users\Jin\Research\test_wan90_setup\tdbg'
    os.chdir(wdir)
    #
    # ham = GrapheneTB()
    # ham.save_htb()

    ham = GrapheneTB_double_layer()
    ham.save_htb()

    htb = Htb_response()
    htb.copy_dict_from_htb(ham)

    nk1 = 301
    kpath_HSP = np.array([
        [-1/3, 1/3, 0.0], #  K
        [ 0.0, 0.0, 0.0], #  G
        [ 0.0, 1/2, 0.0], #  M
        [ 1/3, 2/3, 0.0], # -K
    ])
    HSP_name = ['K', 'G', 'M', '-K']

    # nk1 = 10001
    # kpath_HSP = np.array([
    #     [ 1/3, 2/3, 0.0], #  K
    #     [ 0.0, 0.0, 0.0], #  G
    # ])
    # HSP_name = ['K', 'G']

    # kpath_HSP = np.array([
    #     [-2/3,-1/3, 0.0], # -K
    #     [ 0.0, 0.0, 0.0], #  G
    #     [ 2/3, 1/3, 0.0], #  K
    #     [ 0.5, 0.5, 0.0], #  M
    #     [ 0.0, 0.0, 0.0], #  G
    #     [-0.5,-0.5, 0.0], # -M
    #     [-2/3,-1/3, 0.0], # -K
    # ])
    # HSP_name = None

    bandstructure_hsp = htb.cal_band_HSP(kpath_HSP, nk1, HSP_name=HSP_name)

    if os.environ.get('PYGUI') == 'True':
        bandstructure_hsp.plot_band(eemin=-10.1, eemax=10.1, unit='C')

