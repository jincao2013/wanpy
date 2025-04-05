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

__date__ = "Jan. 3, 2019"

import os
import time
import numpy as np
from numpy import linalg as LA
from wanpy.core.structure import Htb, Cell
from wanpy.core.mesh import make_kmesh_dev001, make_mesh, make_ws_gridR

__all__ = [
    'Supercell_Htb',
    'FT_htb',
]

'''
  * High level module
    which means one can be treated as black box. 
'''
class Supercell_Htb(object):

    """
      * Shift Bloch Phase method for building supercell
      ** Level 0
         build_sposcar
         _creat_ws_of_supercell
         _get_hk
         _get_rk
         _get_U
      ** Level 1
         get_hK
         get_rK
         get_hR
         get_rR

      * N.B.
        1. def() in same level do not ref with each other
        2. _** named function is forbiden with outer env.
        3. even number of RGrid is recommanded.
        4. latt_sc = latt @ transM

      * procudure
          grid_k  grid_K  grid_R
        hr  ->  hk  ->  hK  ->  hR
        rr  ->  rk  ->  rK  ->  rR

        If he basis of htb in order of
        |w1> |w2>
        then the basis of transformed htb in order of
        |w1 s1> |w2 s1>; |w1 s2> |w2 s2>; ...; |w1 sn> |w2 sn>
        where n is supercell index
        wcc.reshape(Ncell, nw_uc) # Ncell = nw_sc//nw_uc

        Notice SPOSCAR is in order of
        |atom1 s1> |atom1 s2> ... |atom1 sn>; |atom2 s1> |atom2 s2> ... |atom2 sn>; ...
        due to the convention of POSCAR.vasp
    """

    def __init__(self, htb, transM, RGrid, wcc_shift=0.):

        self.transM = transM
        self.Nsuper = int(np.round(LA.det(transM)))
        self.RGrid = RGrid
        self.sR = self._decompose_transM_R(transM, self.Nsuper)
        self.sk = self._decompose_transM_k(transM, self.Nsuper)
        # print(self.sR)

        # ***********
        # read htb
        # ***********
        self.htb = htb
        self.htb_sc = Htb()

        # ***********
        # read POSCAR
        # ***********
        self.ucell = self.htb.cell
        self.scell = self.build_sposcar()

        self.sR_car = LA.multi_dot([self.sR, self.ucell.lattice.T])

        # ***********
        # read wannier90
        # ***********
        self.hr_Rmn = htb.hr_Rmn
        self.rr_Ramn = htb.r_Ramn
        self.nR = htb.nR
        self.nw = htb.nw
        self.Nw = self.nw * self.Nsuper

        nR_sc, ndegen_sc, grid_R = self._creat_ws_of_supercell(self.scell.lattice)
        self.nR_sc = nR_sc
        self.ndegen_sc = ndegen_sc
        self.grid_r = htb.R
        self.grid_R = grid_R
        self.grid_K = make_kmesh_dev001(RGrid[0], RGrid[1], RGrid[2], 'G') # shape = (NK, 3)
        self.grid_k = self._creat_kmesh()       # shape=(nk, 3)
        self.NK = self.grid_K.shape[0]
        self.nk = self.grid_k.shape[0]
        self.estimate_memoery()

        self.wcc_uc = htb.wcc + wcc_shift             # in car. coor. shape=(nw, 3)
        self.wcc_sc, self.wccf_sc = self.build_wcc_sc()

        self.Uk = self._get_Uk()                # shape=(NK, N_extend, Nw, nw)

        # self.Trans2tb = self.get_Trans2tb()

    '''
       Level 0
    '''
    def _creat_ws_of_supercell(self, lattice):
        '''
        :param Rgrid:
        :return:
                nrpts: int
                ndegen: list
                irvec: [array([i1,i2,i3])]

        '''
        # ***********
        # init
        # ***********
        a1 = lattice.T[0]
        a2 = lattice.T[1]
        a3 = lattice.T[2]
        RGrid = self.RGrid

        # ***********
        # main
        # ***********
        nrpts_s = 0  # -1
        ndegen_s = []
        irvec_s = []

        g_matrix = np.dot(np.array([a1, a2, a3]),
                          np.array([a1, a2, a3]).T)

        for n1 in range(-RGrid[0], RGrid[0] + 1):
            for n2 in range(-RGrid[1], RGrid[1] + 1):
                for n3 in range(-RGrid[2], RGrid[2] + 1):
                    # Loop 125 R
                    icnt = -1
                    dist = np.zeros((125))
                    for i1 in [-2, -1, 0, 1, 2]:
                        for i2 in [-2, -1, 0, 1, 2]:
                            for i3 in [-2, -1, 0, 1, 2]:
                                icnt += 1
                                ndiff = np.array([
                                    n1 - i1 * RGrid[0],
                                    n2 - i2 * RGrid[1],
                                    n3 - i3 * RGrid[2]
                                ])
                                dist[icnt] = ndiff.dot(g_matrix).dot(ndiff)
                    # print(dist)

                    # dist_min = min(dist.tolist())
                    dist_min = np.min(dist)
                    if np.abs((dist[62] - dist_min)) < 10 ** -7:
                        # nrpts += 1
                        ndegen_s.append(0)
                        for i in range(0, 125):
                            if np.abs(dist[i] - dist_min) < 10 ** -7:
                                ndegen_s[nrpts_s] += 1
                        nrpts_s += 1

                        # irvec.append(n1 * a1 + n2 * a2 + n3 * a3)
                        irvec_s.append(np.array([n1, n2, n3]))

        ndegen_s = np.array(ndegen_s)
        irvec_s = np.array(irvec_s)
        # print('nrpts={}'.format(nrpts_s))
        # print('ndegen=\n', ndegen_s)
        # print('irvec=\n')
        # pp.pprint(irvec_s)
        print('*=============================================================================*')
        print('|                            R Grid of Super Cell                             |')
        print('|    number of R Grid = {:4>}                                                  |'.format(nrpts_s))
        print('*=============================================================================*')
        for i in range(nrpts_s):
            print('|{: 4}). {: 3} {: 3} {: 3}   *{:2>}  '.format(i + 1, irvec_s[i, 0], irvec_s[i, 1], irvec_s[i, 2], ndegen_s[i]),
                  end='')
            if (i + 1) % 3 == 0:
                print('|')
        print('')
        print('*--------------------------------------------------------------------------------*')
        return nrpts_s, ndegen_s, irvec_s

    def _decompose_transM_R(self, transM, nsuper):
        '''
           * Find a n1 * n2 * n3 times large UCell which is
             equivalent with SCell.
             rs = inv(T) * ru
        '''
        T = transM # self.transM
        N = nsuper # self.Nsuper

        sR = np.unique(
            np.round(
                np.remainder(
                    np.array([
                        LA.multi_dot([LA.inv(T), np.array([i, 0, 0])])  # ru -> rs
                        for i in range(N)
                    ]), 1.0
                ), 10
            ), axis=0
        )
        n1 = sR.shape[0]
        if n1 == N:
            return LA.multi_dot([T, sR.T]).T # rs -> ru

        sR = np.unique(
            np.round(
                np.remainder(
                    np.array([
                        LA.multi_dot([LA.inv(T), np.array([i, j, 0])])
                        for i in range(n1)
                        for j in range(N // n1)
                    ]), 1.0
                ), 10
            ), axis=0
        )
        n2 = sR.shape[0] // n1
        if n1 * n2 == N:
            return LA.multi_dot([T, sR.T]).T

        sR = np.unique(
            np.round(
                np.remainder(
                    np.array([
                        LA.multi_dot([LA.inv(T), np.array([i, j, k])])
                        for i in range(n1)
                        for j in range(n2)
                        for k in range(N // (n1 * n2))
                    ]), 1.0
                ), 10
            ), axis=0
        )
        n3 = sR.shape[0] // (n1*n2)
        return LA.multi_dot([T, sR.T]).T

    def _decompose_transM_k(self, transM, nsuper):
        '''
           * Find a n1 * n2 * n3 times large SBZ which is
             equivalent with UBZ.
             ku = inv(T).T * ks
        '''
        T = transM # self.transM
        N = nsuper # self.Nsuper

        sk = np.unique(
            np.round(
                np.remainder(
                    np.array([
                        LA.multi_dot([LA.inv(T).T, np.array([i, 0, 0])])  # ks -> ku
                        for i in range(N)
                    ]), 1.0
                ), 10
            ), axis=0
        )
        n1 = sk.shape[0]
        if n1 == N:
            return LA.multi_dot([T.T, sk.T]).T # ku -> ks

        sk = np.unique(
            np.round(
                np.remainder(
                    np.array([
                        LA.multi_dot([LA.inv(T).T, np.array([i, j, 0])])
                        for i in range(n1)
                        for j in range(N // n1)
                    ]), 1.0
                ), 10
            ), axis=0
        )
        n2 = sk.shape[0] // n1
        if n1 * n2 == N:
            return LA.multi_dot([T.T, sk.T]).T

        sk = np.unique(
            np.round(
                np.remainder(
                    np.array([
                        LA.multi_dot([LA.inv(T).T, np.array([i, j, k])])
                        for i in range(n1)
                        for j in range(n2)
                        for k in range(N // (n1 * n2))
                    ]), 1.0
                ), 10
            ), axis=0
        )
        n3 = sk.shape[0] // (n1*n2)
        return LA.multi_dot([T.T, sk.T]).T

    def estimate_memoery(self):
        '''
          * hr_Rmn rr_Ramn Uk(NK, Nw, Nw)
          * hk(k, nw, nw) rk(k, a, nw, nw)
          * hK(K, Nw, Nw) rK(K, a, Nw, Nw)
          * hR(R, Nw, Nw) rR(R, a, Nw, Nw)
        '''
        unit = self.hr_Rmn.nbytes / self.nR / 1024 ** 3
        unit_sc = unit * self.Nsuper ** 2
        # print('hr_Rmn = {:12.6} Gb'.format(unit * self.nR))
        # print('rr_Ramn = {:12.6} Gb'.format(unit * self.nR * 3))
        # print('Uk = {:12.6} Gb'.format(unit_sc * self.NK))
        # print('hk = {:12.6} Gb'.format(unit * self.nk))
        # print('rk = {:12.6} Gb'.format(unit * self.nk * 3))
        # print('hK = {:12.6} Gb'.format(unit_sc * self.NK))
        # print('rK = {:12.6} Gb'.format(unit_sc * self.NK * 3))
        # print('hR = {:12.6} Gb'.format(unit_sc * self.nR_sc))
        # print('rR = {:12.6} Gb'.format(unit_sc * self.nR_sc * 3))
        # print('Memoery consuming in total = {:12.6} Gb'.format(
        #     unit * (self.nR + self.nk) * 4 +
        #     unit_sc * (self.nR_sc + self.NK) * 4
        # ))
        print('*============================================================================*')
        print('|                    BUILDING SUPER CELL MEMORY ESTIMATE                     |')
        print('|         Minimum RAM allocated during each phase of the calculation         |')
        print('*============================================================================*')
        print('|                   htb, Uk: {:10.2f} Gb                                   |'.format(unit * self.nR * 4 + unit_sc * self.NK))
        print('|                        hk: {:10.2f} Gb                                   |'.format(unit * self.nk))
        print('|                        rk: {:10.2f} Gb                                   |'.format(unit * self.nk * 3))
        print('|                        hK: {:10.2f} Gb                                   |'.format(unit_sc * self.NK))
        print('|                        rK: {:10.2f} Gb                                   |'.format(unit_sc * self.NK * 3))
        print('|                        hR: {:10.2f} Gb                                   |'.format(unit_sc * self.nR_sc))
        print('|                        rR: {:10.2f} Gb                                   |'.format(unit_sc * self.nR_sc * 3))
        print('|                     Total: {:10.2f} Gb                                   |'.format(
            unit * (self.nR + self.nk) * 4 + unit_sc * (self.nR_sc + self.NK) * 4
        ))
        print('*----------------------------------------------------------------------------*')

    '''
       Level 1
    '''
    def _creat_kmesh(self):
        '''
         grid_k
           in basis of G_uc
        '''
        grid_K = self.grid_K
        NK = grid_K.shape[0]

        # in basis of G_sc
        kmesh = np.kron(np.ones([self.Nsuper, 1]), grid_K) + \
                np.kron(self.sk, np.ones([NK, 1]))

        # translate to basis of G_uc
        kmesh = LA.multi_dot([LA.inv(self.transM).T, kmesh.T]).T
        kmesh = np.remainder(kmesh + 1e-10, np.array([1., 1., 1.]))

        return kmesh

    def build_sposcar(self, save_sposcar=True):
        print('[From Supercell_Htb at {}] Building sposcar procudure ... '.format(time.asctime()))
        scell = Cell()
        lattice = self.ucell.lattice
        spec = self.ucell.spec
        ions = self.ucell.ions

        lattice_sc = LA.multi_dot([lattice, self.transM])
        latticeG_sc = 2 * np.pi * LA.inv(lattice_sc.T)
        spec_sc = [ i for i in spec for j in range(self.Nsuper)]
        ions_sc = np.remainder(
            np.einsum(
                'mn,an->am', LA.inv(self.transM),
                np.array(
                    np.kron(ions, np.ones([self.Nsuper, 1])) + \
                    np.kron(np.ones([ions.shape[0], 1]), self.sR)
                )
            ), 1.0
        )
        ions_car_sc = LA.multi_dot([lattice_sc, ions_sc.T]).T

        scell.name = 'SPOSCAR of ' + self.ucell.name
        scell.lattice = lattice_sc
        scell.latticeG = latticeG_sc
        scell.spec = spec_sc
        scell.ions = ions_sc
        scell.ions_car = ions_car_sc
        scell.N = ions_sc.shape[0]

        print('[From Supercell_Htb {}] Complite building sposcar.'.format(time.asctime()))

        if save_sposcar:
            if os.path.exists('SPOSCAR.vasp'):
                os.remove('SPOSCAR.vasp')
            scell.save_poscar(fname=r'SPOSCAR.vasp')
            print('[From Supercell_Htb {}] sposcar have been saved.'.format(time.asctime()))

        return scell

    def build_wcc_sc(self):
        lattice = self.ucell.lattice
        lattice_sc = self.scell.lattice
        wcc_uc = self.wcc_uc

        wccf_uc = LA.multi_dot([LA.inv(lattice), wcc_uc.T]).T
        wccf_sc = np.kron(np.ones([self.Nsuper, 1]), wccf_uc) + \
                     np.kron(self.sR, np.ones([self.nw, 1]))   # (Nw, 3)
        wccf_sc = LA.multi_dot([LA.inv(self.transM), wccf_sc.T]).T
        wccf_sc = np.remainder(wccf_sc, np.array([1, 1, 1]))
        wcc_sc = LA.multi_dot([lattice_sc, wccf_sc.T]).T

        return wcc_sc, wccf_sc

    def _get_Uk(self):
        '''
          * shape = (NK, Nw, Nw)
             1    2    . N
           1 k0r0 k0r1 . k0rN
           2 k1r0 k1r1 . k1rN
           . .    .    . .
           N kNr0 kNr1 . kNrN
        '''
        nw = self.nw
        Nw = self.Nw
        NK = self.NK
        Nsuper = self.Nsuper
        Uk = np.zeros([NK, Nw, Nw], dtype='complex128')
        grid_k = self.grid_k.reshape(Nsuper, NK, 3)

        wcc_uc = np.kron(np.ones([Nsuper, 1]), self.wcc_uc)
        wcc_sc = self.wcc_sc

        d = wcc_sc - wcc_uc
        # print(d)
        d = LA.multi_dot([LA.inv(self.ucell.lattice), d.T]).T.reshape(Nsuper, nw, 3)
        self.d = d

        for i in range(Nsuper):
            for j in range(Nsuper):
                I = i * nw
                J = j * nw
                eikr = np.exp(-2j * np.pi * np.einsum('Ka,na->Kn', grid_k[i], d[j]))
                eikr = np.array([
                    np.diag(dd)
                    for dd in eikr
                ])
                Uk[:, I:I + nw, J:J + nw] = eikr

        Uk = Uk / np.sqrt(Nsuper) # shape = (NK, Nw, Nw)

        return Uk

    def _get_onsite_R(self, R):
        i = 0
        for _R in R:
            if (_R == 0).all():
                break
            i += 1
        return i

    def _get_hk(self):
        print('[From Supercell_Htb {}] Building hk ...'.format(time.asctime()))
        eikr = np.exp(2j * np.pi * np.einsum('ka,ra->kr', self.grid_k, self.grid_r)) / self.htb.ndegen
        hk = np.einsum('kr,rmn->kmn', eikr, self.hr_Rmn, optimize=True)
        hk = 0.5 * (hk + np.einsum('kmn->knm', hk).conj())
        print('[From Supercell_Htb {}] Complite building hk.'.format(time.asctime()))
        self.hk = hk
        return hk

    def _get_rk(self):
        print('[From Supercell_Htb {}] Building rk ...'.format(time.asctime()))
        eikr = np.exp(2j * np.pi * np.einsum('ka,ra->kr', self.grid_k, self.grid_r)) / self.htb.ndegen
        rk = np.einsum('kr,ramn->kamn', eikr, self.rr_Ramn, optimize=True)
        rk = 0.5 * (rk + np.einsum('kamn->kanm', rk).conj())
        print('[From Supercell_Htb {}] Complite building rk.'.format(time.asctime()))
        self.rk = rk
        return rk

    '''
       Level 2
    '''
    def _get_hK(self, _hk):
        '''
          * sum kc={K} in car. coor.
        '''
        print('[From Supercell_Htb {}] Building hK ...'.format(time.asctime()))
        nw = self.nw
        Nw = self.Nw

        _hk = _hk.reshape(self.Nsuper, self.NK, nw, nw)
        hk = np.zeros([self.NK, Nw, Nw], dtype='complex128')
        for i in range(self.Nsuper):
            I = i * nw
            hk[:, I:I + nw, I:I + nw] = _hk[i]
        Uk = self.Uk
        UkH = np.einsum('KMm->KmM', Uk.conj())
        hK = np.einsum('KNn,Knm,KmM->KNM', UkH, hk, self.Uk, optimize=True)

        hK = 0.5 * (hK + np.einsum('Kmn->Knm', hK).conj())
        print('[From Supercell_Htb {}] Complite building hK.'.format(time.asctime()))
        self.hK = hK
        return hK

    def _get_rK(self, _rk):
        print('[From Supercell_Htb {}] Building rK ...'.format(time.asctime()))
        nw = self.nw
        Nw = self.Nw

        _rk = _rk.reshape(self.Nsuper, self.NK, 3, nw, nw) # aKAnm
        rk = np.zeros([self.NK, 3, Nw, Nw], dtype='complex128')
        for i in range(self.Nsuper):
            for a in range(3):
                I = i * nw
                rk[:, a, I:I + nw, I:I + nw] = _rk[i, :, a, :, :]
        Uk = self.Uk
        UkH = np.einsum('KMm->KmM', Uk.conjugate())
        rK = np.einsum('KNn,Kanm,KmM->KaNM', UkH, rk, Uk, optimize=True)

        # rK_diag = np.array([
        #     [
        #         np.kron(np.diag(_sr), np.identity(nw)) # (nw*N_extend, nw*N_extend)
        #         for _sr in self.sR_car.T # 3
        #     ]
        #     for i in range(self.NK) # NK
        # ]) # (NK, 3, NW, NW)
        # rK += rK_diag

        rK = 0.5 * (rK + np.einsum('Kamn->Kanm', rK).conj())
        print('[From Supercell_Htb {}] Complite building rK.'.format(time.asctime()))
        self.rK = rK
        return rK

    # def _get_hK_par(self, _hk):
    #     '''
    #       * sum kc={K} in car. coor.
    #     '''
    #     print('[From Supercell_BPhase {}] Par Building hK ...'.format(time.asctime()))
    #     nw = self.nw
    #     Nw = self.Nw
    #
    #     _hk = _hk.reshape(self.Nsuper, self.NK, nw, nw)
    #     hk = np.zeros([self.NK, Nw, Nw], dtype='complex128')
    #     for i in range(self.Nsuper):
    #         I = i * nw
    #         hk[:, I:I + nw, I:I + nw] = _hk[i]
    #     Uk = self.Uk
    #
    #     UkH = np.einsum('KMm->KmM', Uk.conj())
    #     hK = np.einsum('KNn,Knm,KmM->KNM', UkH, hk, Uk, optimize=True)
    #
    #     print('[From Supercell_BPhase {}] Complite Par building hK.'.format(time.asctime()))
    #     self.hK = hK
    #     return hK

    '''
       Level 3
    '''
    def build_hR(self):
        hk = self._get_hk()
        hK = self._get_hK(hk)

        print('[From Supercell_Htb {}] Building hR ...'.format(time.asctime()))
        eiKR = np.exp(-2j * np.pi * np.einsum('ka,ra->kr', self.grid_K, self.grid_R))
        hR_Rmn = np.einsum('KR,KMN->RMN', eiKR, hK, optimize=True) / self.NK

        print('[From Supercell_Htb {}] Complite building hR.'.format(time.asctime()))
        self.hr_Rmn = hR_Rmn
        return hR_Rmn

    def build_rR(self, returnzero=False, returntb=False):
        if returnzero or self.rr_Ramn.dtype in [np.dtype('O'), None]:
            print('[From Supercell_Htb {}] rR seted as zero'.format(time.asctime()))
            r_Ramn = np.zeros([self.nR_sc, 3, self.Nw, self.Nw], dtype='complex128')
            self.r_Ramn = r_Ramn
            return r_Ramn
        elif returntb:
            print('[From Supercell_Htb {}] rR seted in tb form'.format(time.asctime()))
            r_Ramn = np.zeros([self.nR_sc, 3, self.Nw, self.Nw], dtype='complex128')
            r_Ramn[self.nR_sc//2, 0] += np.diag(self.wcc_sc.T[0])
            r_Ramn[self.nR_sc//2, 1] += np.diag(self.wcc_sc.T[1])
            r_Ramn[self.nR_sc//2, 2] += np.diag(self.wcc_sc.T[2])
            self.r_Ramn = r_Ramn
            return r_Ramn

        rk = self._get_rk()
        rK = self._get_rK(rk)

        print('[From Supercell_Htb {}] Building rR ...'.format(time.asctime()))
        eiKR = np.exp(-2j * np.pi * np.einsum('ka,ra->kr', self.grid_K, self.grid_R))
        r_Ramn = np.einsum('KR,KANM->RAMN', eiKR, rK, optimize=True) / self.NK

        onsite = self._get_onsite_R(self.grid_R)
        for a in range(3):
            for i in range(self.Nw):
                r_Ramn[onsite, a, i, i] = self.wcc_sc[i, a]

        print('[From Supercell_Htb {}] Complite building rR.'.format(time.asctime()))
        self.r_Ramn = r_Ramn
        return r_Ramn

    def get_htb_sc(self, returnzeror=False, returntb=False):
        hr_Rmn = self.build_hR()
        r_Ramn = self.build_rR(returnzeror, returntb)
        # self._head_container = [
        #     'name', 'fermi', 'nw', 'nR', 'R', 'ndegen', 'cell',
        #     'wout', 'wcc', 'wccf',
        # ]
        self.htb_sc.name = self.scell.name
        self.htb_sc.fermi = self.htb.fermi
        self.htb_sc.nw = self.Nw
        self.htb_sc.cell = self.scell
        self.htb_sc.wcc, self.htb_sc.wccf = self.wcc_sc, self.wccf_sc
        self.htb_sc.latt = self.scell.lattice
        self.htb_sc.lattG = self.scell.latticeG

        self.htb_sc.nR = self.nR_sc
        self.htb_sc.ndegen = self.ndegen_sc
        self.htb_sc.R = self.grid_R
        self.htb_sc.hr_Rmn = hr_Rmn
        self.htb_sc.r_Ramn = r_Ramn
        return self.htb_sc

    def get_htb_slab(self, open_boundary=1, returnzero=False, returntb=False):
        hr_Rmn = self.build_hR()
        r_Ramn = self.build_rR(returnzero, returntb)

        nR = self.nR_sc
        nR_remain = 0
        slab_remain = np.zeros(nR, dtype='int')
        for i in range(nR):
            if self.grid_R[i, open_boundary] == 0:
                slab_remain[i] = 1 # True
                nR_remain += 1
            else:
                slab_remain[i] = 0 # False

        # self._head_container = [
        #     'name', 'fermi', 'nw', 'nR', 'R', 'ndegen', 'N_ucell', 'cell',
        #     'wout', 'wcc', 'wccf',
        #     'latt', 'lattG',
        #     'nD', 'D_namelist',
        # ]
        self.htb_sc.name = self.scell.name
        self.htb_sc.fermi = self.htb.fermi
        self.htb_sc.nw = self.Nw
        self.htb_sc.cell = self.scell
        self.htb_sc.wcc, self.htb_sc.wccf = self.wcc_sc, self.wccf_sc
        self.htb_sc.latt = self.scell.lattice
        self.htb_sc.lattG = self.scell.latticeG

        self.htb_sc.nR = nR_remain
        self.htb_sc.ndegen = np.array([self.ndegen_sc[i] for i in range(nR) if slab_remain[i]], dtype='int64')
        self.htb_sc.R = np.array([self.grid_R[i] for i in range(nR) if slab_remain[i]], dtype='int64')
        self.htb_sc.hr_Rmn = np.array([hr_Rmn[i] for i in range(nR) if slab_remain[i]], dtype='complex128')
        self.htb_sc.r_Ramn = np.array([r_Ramn[i] for i in range(nR) if slab_remain[i]], dtype='complex128')
        return self.htb_sc


'''
  * Low level module
'''
class FT_htb(object):

    def __init__(self, ngridR, latt):
        self.latt = latt
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)

        self.meshk = make_mesh(nmesh=ngridR)
        self.meshkc = LA.multi_dot([self.lattG, self.meshk.T]).T
        self.nk = self.meshk.shape[0]

        self.ngridR = ngridR
        self.nR, self.ndegen, self.gridR = make_ws_gridR(ngridR, latt, info=False)

        self.eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', self.meshk, self.gridR, optimize=True))

    '''
      * F.T. of H 
        hr_Rmn = <0m|H|Rn>
        hk_kmn = <mk|H|nk>
    '''
    def get_hr_Rmn(self, hk_kmn):
        hr_Rmn = np.einsum('kR,kmn->Rmn', self.eikR.conj(), hk_kmn, optimize=True) / self.nk
        return hr_Rmn

    def get_hk_kmn(self, hr_Rmn):
        hk_kmn = np.einsum('R,kR,Rmn->kmn', 1/self.ndegen, self.eikR, hr_Rmn, optimize=True)
        return hk_kmn

    '''
      * F.T. of r 
        r_Ramn = <0m|r|Rn>
        rk = <mk|r|nk>
      * <0m|r|Rn>
    '''
    def get_r_Ramn(self, rk):
        hr_Rmn = np.einsum('kR,Ramn->kamn', self.eikR.conj(), rk, optimize=True) / self.nk
        return hr_Rmn

    def get_rk(self, r_Ramn):
        rk = np.einsum('R,kR,Ramn->kamn', 1/self.ndegen, self.eikR, r_Ramn, optimize=True)
        return rk

    '''
      * from htb
    '''
    def get_hk_kmn_from_htb(self, htb):
        eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', self.meshk, htb.R, optimize=True))
        hk_kmn = np.einsum('R,kR,Rmn->kmn', 1/htb.ndegen, eikR, htb.hr_Rmn, optimize=True)
        return hk_kmn

    def get_rk_from_htb(self, htb):
        eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', self.meshk, htb.R, optimize=True))
        np.einsum('R,Ramn->amn', eikR, htb.r_Ramn, optimize=True)
        rk = np.einsum('R,kR,Rmn->kmn', 1/htb.ndegen, eikR, htb.r_Ramn, optimize=True)
        return rk

class Trans_htb(object):
    def __init__(self):
        pass

    '''
      * resort wcc
    '''
    def resort_wcc(self, wcc, axis=2, returetransM=False):
        # set rearraged_band_index
        nw = wcc.shape[0]
        orbindex = np.argsort(wcc.T[axis])

        # get transM
        transM = np.zeros([nw, nw], dtype='float64')
        for i in range(nw):
            transM[i, orbindex[i]] = 1

        # trans wcc
        wcc_new = transM @ wcc
        if returetransM:
            return wcc_new, transM
        else:
            return wcc_new

    def resort_wcc_hr_Rmn(self, wcc, hr_Rmn, axis=2):
        # this func can be directly applied for hk_kmn
        wcc_new, transM = self.resort_wcc(wcc, axis, returetransM=True)
        transM = np.array(transM, dtype='complex128')
        hr_Rmn = np.einsum('mi,Rij,jn->Rmn', transM, hr_Rmn, transM.T, optimize=True)
        return hr_Rmn

    def resort_wcc_r_Ramn(self, wcc, r_Ramn, axis=2):
        wcc_new, transM = self.resort_wcc(wcc, axis, returetransM=True)
        transM = np.array(transM, dtype='complex128')
        r_Ramn = np.einsum('mi,Raij,jn->Ramn', transM, r_Ramn, transM.T, optimize=True)
        return r_Ramn

    '''
      * shift wcc
        # using translation operator
        # H'(k) = e^(-i*k*tau) H(k) e^(i*k*tau)
        # tau = np.diag([tau_1, tau_2, ...])
    '''
    def shift_wcc(self, htb, wcc_shift):
        wccf_new = (LA.inv(htb.latt) @ (htb.wcc + wcc_shift).T).T
        wccf_new = np.remainder(wccf_new, np.array([1, 1, 1]))
        wcc_new = (htb.latt @ wccf_new.T).T
        return wcc_new

    def shift_wcc_hr_Rmn(self, htb, wcc_shift, ngridR):
        wcc_new = self.shift_wcc(htb, wcc_shift)
        delta_wcc = wcc_new - htb.wcc

        fthtb = FT_htb(ngridR=ngridR, latt=htb.latt)
        meshkc = fthtb.meshkc
        hk_kmn = fthtb.get_hk_kmn_from_htb(htb)

        eiktau = np.exp(1j * np.einsum('ka,ra->kr', meshkc, delta_wcc))
        hk_kmn = np.einsum('km,kmn,kn->kmn', eiktau, hk_kmn, eiktau.conj(), optimize=True)

        hr_Rmn = fthtb.get_hr_Rmn(hk_kmn)
        return hr_Rmn

    def shift_wcc_r_Ramn(self, htb, wcc_shift, ngridR):
        wcc_new = self.shift_wcc(htb, wcc_shift)
        delta_wcc = wcc_new - htb.wcc

        fthtb = FT_htb(ngridR=ngridR, latt=htb.latt)
        meshkc = fthtb.meshkc
        rk = fthtb.get_rk_from_htb(htb)

        eiktau = np.exp(1j * np.einsum('ka,ra->kr', meshkc, delta_wcc))
        rk = np.einsum('km,kamn,kn->kamn', eiktau, rk, eiktau.conj(), optimize=True)

        r_Ramn = fthtb.get_r_Ramn(rk)
        return r_Ramn


'''
  * Test model
'''
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

class Toy_model_1d(Htb):

    def __init__(self, *args, **kwargs):
        Htb.__init__(self, *args, **kwargs)

        self.name = 'one dimensional toy model'
        self.fermi = 0
        self.nw = 3
        self.nR = 3
        self.R = np.array([
            [-1,  0,  0],
            [ 0,  0,  0],
            [ 1,  0,  0],
        ], dtype='int64')
        self.ndegen = np.ones([self.nR], dtype='int64')

        self.cell = self._init_cell()
        self.latt = self.cell.lattice
        self.lattG = self.cell.latticeG
        self.wcc = self.cell.ions_car
        self.wccf = self.cell.ions

        self.hr_Rmn = self._init_hr_Rmn()
        self.r_Ramn = self._init_r_Ramn()

    def _init_cell(self):
        cell = Cell()
        cell.name = 'one dimensional toy model'
        cell.lattice = np.diag([self.nw, 1.0, 1.0])
        cell.latticeG = cell.get_latticeG()
        cell.N = self.nw
        cell.spec = ['C'] * self.nw
        cell.ions = np.zeros([self.nw, 3], dtype=np.float64)
        cell.ions.T[0] = np.arange(self.nw) / self.nw
        cell.ions_car = cell.get_ions_car()
        return cell

    def _init_hr_Rmn(self):
        hr_Rmn = np.zeros([self.nR, self.nw, self.nw], dtype='complex128')
        hr_Rmn[1] = np.array([
            [0, 1, 0],
            [1, 0, 2],
            [0, 2, 0],
        ])
        hr_Rmn[0] = np.array([
            [0, 0, 3],
            [0, 0, 0],
            [0, 0, 0],
        ])
        hr_Rmn[2] = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [3, 0, 0],
        ])
        return hr_Rmn

    def _init_r_Ramn(self):
        r_Ramn = np.zeros([self.nR, 3, self.nw, self.nw], dtype='complex128')
        r_Ramn[2, 0] = np.diag(self.cell.ions_car.T[0]) #+ noise_OD
        r_Ramn[2, 1] = np.diag(self.cell.ions_car.T[1]) #+ noise_OD
        r_Ramn[2, 2] = np.diag(self.cell.ions_car.T[2]) #+ noise_OD

        return r_Ramn

def hr_free_ele(save_npz=True):
    nw = 1
    nR = 7
    lattice = np.identity(3)
    latticeG = 2 * np.pi * LA.inv(lattice.T)
    ions = np.array([
        [0.0, 0.0, 0.0]
    ])
    R = np.array([
        [0, 0, 0],
        [1, 0, 0], [-1,  0,  0],
        [0, 1, 0], [0,  -1,  0],  # 3, 4
        [0, 0, 1], [0,   0, -1],  # 5, 6
    ], dtype='int32')
    ndegen = np.ones(nR, dtype='int32')

    hr_Rmn = np.zeros([nR, nw, nw], dtype='complex128')
    r_Ramn = np.zeros([nR, 3, nw, nw], dtype='complex128')

    hr_Rmn[0] = 3
    hr_Rmn[1] = -0.5
    hr_Rmn[2] = -0.5
    hr_Rmn[3] = -0.5
    hr_Rmn[4] = -0.5
    hr_Rmn[5] = -0.5
    hr_Rmn[6] = -0.5

    for i in range(3):
        r_Ramn[0:i] = ions[0, i]

    cell = {
        'lattice': lattice,
        'latticeG': latticeG,
        'spec': ['H', 'H'],
        'ions': ions,
        'ion_unit': 'D',
    }
    wc = {
        'wcenter': ions,
        'wcenterf': None,
        'wborden': None,
    }
    head = {
        'name': 'IB_WeylSM_Bulk',
        'cell': cell,
        'wc': wc,
        'fermi': 0,
        'nw': nw,
        'nR': nR,
        'R': R,
        'ndegen': ndegen,
    }
    htb = {
        'head': head,
        'hr_Rmn': hr_Rmn,
        'r_Ramn': r_Ramn,
    }
    if save_npz:
        np.savez_compressed(r'htb.npz',
                            head=head,
                            hr_Rmn=hr_Rmn,
                            r_Ramn=r_Ramn,
                            )
        np.savez_compressed(r'wannier90_hr.dat',
                            degeneracy=np.ones(nR),
                            r=R,
                            hr_mn=hr_Rmn)
        np.savez_compressed(r'wannier90_r.dat',
                            R=R,
                            r_mn=r_Ramn)
        np.savez_compressed(r'wc.npz',
                            lattice=lattice,
                            wcenter=ions,
                            wcenterf=ions,
                            )

    return htb

'''
  * Plot
'''
def plot_htb_grid(scell_SBP):
    import matplotlib.gridspec as gridspec
    from matplotlib.pyplot import subplot
    import matplotlib.pyplot as plt

    G = gridspec.GridSpec(2, 2)

    grid_K = scell_SBP.grid_K
    grid_k = scell_SBP.grid_k
    grid_r = scell_SBP.grid_r
    grid_R = scell_SBP.grid_R

    ax = subplot(G[0, 0])
    plt.title('grid_K')
    ax.scatter(grid_K[:,0], grid_K[:,1], alpha=0.5)
    ax.axis('equal')

    ax = subplot(G[1, 0])
    plt.title('grid_k')
    ax.scatter(grid_k[:324, 0], grid_k[:324, 1], alpha=0.5)
    ax.scatter(grid_k[324:, 0], grid_k[324:, 1], alpha=0.5, color='red')
    ax.axis('equal')

    # ax = subplot(G[2, 0])
    # grid_K = scell_BP.grid_K
    # NK = grid_K.shape[0]
    # kmesh = np.kron(np.matlib.ones(scell_BP.N_extend).T, grid_K) + \
    #         np.kron(scell_BP.SR, np.matlib.ones(NK).T)
    # # kmesh = np.einsum('mn,an->am', LA.inv(scell_BP.transM), np.array(kmesh))
    # kmesh = multi_dot([np.array(kmesh), LA.inv(scell_BP.transM).T])
    #
    # kmesh = np.remainder(kmesh + 1e-10, np.array([1., 1., 1.]))
    # ax.scatter(kmesh[:, 0], kmesh[:, 1], alpha=0.5)
    # ax.axis('equal')

    ax = subplot(G[0, 1])
    plt.title('grid_r')
    ax.scatter(grid_r[:,0], grid_r[:,1], alpha=0.5)
    ax.axis('equal')

    ax = subplot(G[1, 1])
    plt.title('grid_R')
    ax.scatter(grid_R[:, 0], grid_R[:, 1], alpha=0.5)
    ax.axis('equal')


def plot_wcenter(wc, arc_weigh=None):
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns

    lattice = wc['lattice']
    wcenter = wc['wcenter']
    nw = wcenter.shape[0]
    a = lattice[:2,0]
    b = lattice[:2,1]
    c = lattice[:2,2]
    if arc_weigh == None:
        arc_weigh = 30 * np.ones(nw)

    fig, ax = plt.subplots()

    Path = mpath.Path
    path_data = [
        (Path.MOVETO, [0, 0]),
        (Path.LINETO, a),
        (Path.LINETO, a+b),
        (Path.LINETO, b),
        (Path.LINETO, [0, 0]),
    ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='#607d8b', alpha=0.3)
    ax.add_patch(patch)

    # plot wanier center
    cmap = sns.diverging_palette(255, 133, l=60, n=101, center="dark", as_cmap=True)
    # for i in range(nw):
    #     x, y = wcenter[i,:2]
    #     # s = wborden[i] * 50
    #     s = np.abs(arc_weigh[i]) * 10
    #     ax.scatter(x, y, s, color='#303f9f', marker='o', alpha=0.5, zorder=10)
    #     # ax.text(x, y, '{}'.format(i), va='top', fontsize=8)
    x, y = wcenter[:,0], wcenter[:,1]
    s = np.abs(arc_weigh) * 10
    c = arc_weigh * 10
    # ax.scatter(x, y, s, color='#303f9f', marker='o', alpha=0.5, zorder=10)
    ax.scatter(x, y, s, c, cmap=cmap, marker='o', alpha=0.5, zorder=10)

    ax.grid()
    ax.axis('equal')
    plt.show()


if __name__ == '__main__':
    from wanpy import ROOT_WDIR
    wdir = os.path.join(ROOT_WDIR, r'test')
    '''
      * Job list
      ** scell
      ** shift_wcc
    '''
    Job = r'shift_wcc'

    if Job == r'scell':
        htb_fname = r'htb.h5'
        transM = np.array([
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 1],
            ])
        RGrid = [12, 12, 1]
        wcc_shift = 0 # np.array([0.01, 0.00, 0.01])

        '''
          * Run
        '''
        os.chdir(wdir)
        htb = Htb()
        htb.load_htb(htb_fname)
        htb.save_wcc()
        print('ucell fermi = {} eV'.format(htb.fermi))

        scell_SBP = Supercell_Htb(htb, transM, RGrid, wcc_shift=wcc_shift)
        htb_sc = scell_SBP.get_htb_sc()
        print('scell fermi = {} eV'.format(htb_sc.fermi))
        htb_sc.save_wcc(r'wcc_sc.vasp')
        htb_sc.save_htb(r'htb_sc.h5')


    if Job == r'shift_wcc':
        htb = Toy_model_1d()

        fthtb = FT_htb(ngridR=[12, 1, 1], latt=htb.latt)
        nR = fthtb.nR
        meshkc = fthtb.meshkc

        hk_kmn = fthtb.get_hk_kmn_from_htb(htb)

        wcc_shift = np.array([1.1, 0, 0])
        wccf_new = (LA.inv(htb.latt) @ (htb.wcc + wcc_shift).T).T
        wccf_new = np.remainder(wccf_new, np.array([1, 1, 1]))
        wcc_new = (htb.latt @ wccf_new.T).T
        delta_wcc = wcc_new - htb.wcc


        eiktau = np.exp(1j * np.einsum('ka,ra->kr', meshkc, delta_wcc))
        hk_kmn = np.einsum('km,kmn,kn->kmn', eiktau, hk_kmn, eiktau.conj(), optimize=True)


        hr_Rmn = fthtb.get_hr_Rmn(hk_kmn)

        with np.printoptions(precision=5, suppress=True):
            print(hr_Rmn[nR//2])
            print(hr_Rmn[nR//2-1])
            print(hr_Rmn[nR//2+1])
            # print(np.abs(hr_Rmn[nR//2]))
            # print(np.abs(hr_Rmn[nR//2-1]))
            # print(np.abs(hr_Rmn[nR//2+1]))