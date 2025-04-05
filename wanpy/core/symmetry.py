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

__date__ = "Mar. 30, 2023"

from collections import namedtuple
import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance_matrix
from scipy.linalg import block_diag
import sympy as sp
from wanpy.core.errorhandler import WanpyError, WanpyInputError
from wanpy.core.mesh import make_mesh, make_ws_gridR
from wanpy.core.utils import get_op_cartesian, check_valid_symmops
from wanpy.core.units import *

__all__ = [
    'Symmetrize_Htb_rspace',
    'Symmetrize_Htb_kspace',
    'get_proj_info',
    'parse_symmetry_inputfile',
]

"""
* Note 
  Dec 2, 2024
    The wannier interface (for both v1.2 and v3.1) wannier_setup retures proj_site according to POSCAR,
    and the atomic positions are typically in range of [0, 1), for example by generated from VESTA or phonopy.
    Starting from vasp version 6, when proj_site are obtained from wannier_setup, it will be refined in range 
    of [-0.5, 0.5), see line 1154 of mlwf.F in vasp 6.4.3. For example, for a POSCAR: 
    
    0.0000000000000000  0.5999686590902087  0.3517463977113349 <--- pos 1
    0.5000000000000000  0.3998626154247874  0.1156965177456744 <--- pos 2
    
    We will have the following in OUTCAR: 
   LOCPROJ orbitals
     ----------------------------------------------------------------------------------------------
      n  l  m   za        pos                       proj_x                    proj_z
     ----------------------------------------------------------------------------------------------
      1  2  1   1.890     0.000  -0.400   0.352     1.000   0.000   0.000     0.000   0.000   1.000 <--- pos 1
      ... 
      1  2  1   1.890    -0.500   0.400   0.116     1.000   0.000   0.000     0.000   0.000   1.000 <--- pos 2
      ... 
    
    Therefore, for amn generted by vasp 6, we need set input of atoms_pos in range of [-0.5, 0.5). 
    
"""

class Symmetrize_Htb_rspace(object):

    def __init__(self, htb, symmops, atoms_pos, atoms_orbi, soc, iprint=1):
        self.htb = htb
        self.latt = htb.latt
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)
        self.nR = htb.nR
        self.R = htb.R
        self.Rc = htb.Rc
        self.nw = htb.nw

        self.symmops = symmops
        self.atoms_pos = atoms_pos
        self.atoms_orbi = atoms_orbi
        self.soc = soc

        # self.atoms_pos = [np.remainder(i + np.array([100.5, 100.5, 100.5]), 1.0) - 0.5 * np.ones(3) for i in atoms_pos]

        self.pos_car = (self.latt @ np.vstack(self.atoms_pos).T).T

        self.nsymmops = symmops.shape[0]
        self.ntype_atoms = len(atoms_pos)
        self.npos = self.pos_car.shape[0]

        self.list_nw_at_pos = np.zeros(self.npos, dtype='int64')
        i = 0
        for itype in range(self.ntype_atoms):
            for ipos in range(self.atoms_pos[itype].shape[0]):
                for l in self.atoms_orbi[itype]:
                    self.list_nw_at_pos[i] += get_rep_atomic(self.symmops[0], int(l)).shape[0]
                i += 1

        self.nw_spinless = self.list_nw_at_pos.sum()

        self.Rcij = None

        self.wcc, self.wccf = self.get_atomic_wcc()
        self.iprint = iprint

    def use_ngridR(self, ngridR):
        """
          resample the htb using larger ngridR
        """

        print('resampling htb with user specified ngridR = {} * {} * {}'.format(ngridR[0], ngridR[1], ngridR[2]))
        ''' resample htb '''
        htb = self.htb
        nR, ndegen, gridR = make_ws_gridR(ngridR, self.latt, info=False)

        meshk_resampled = make_mesh(nmesh=ngridR)
        # meshkc = LA.multi_dot([self.lattG, meshk.T]).T
        nk = meshk_resampled.shape[0]

        hk_kmn_resampled = self.ft_gridR_to_meshk(htb.hr_Rmn, meshk_resampled, htb.R, htb.ndegen, tbgauge=False)
        r_kamn_resampled = self.ft_gridR_to_meshk(htb.r_Ramn, meshk_resampled, htb.R, htb.ndegen, tbgauge=False)

        hr_Rmn_resampled = self.ft_meshk_to_gridR(hk_kmn_resampled, meshk_resampled, gridR, tbgauge=False)
        r_Ramn_resampled = self.ft_meshk_to_gridR(r_kamn_resampled, meshk_resampled, gridR, tbgauge=False)

        ''' update self '''
        self.nR = nR
        self.R = gridR
        self.Rc = (self.latt @ gridR.T).T

        ''' update htb '''
        htb.nR = nR
        htb.R = self.R
        htb.Rc = self.Rc
        htb.ndegen = ndegen
        htb.hr_Rmn = hr_Rmn_resampled
        htb.r_Ramn = r_Ramn_resampled

    def run(self):
        check_valid_symmops(self.symmops)
        symmops, atoms_pos, atoms_orbi, soc = self.symmops, self.atoms_pos, self.atoms_orbi, self.soc
        nw, nR, npos = self.nw, self.nR, self.npos

        # set ndegen = ones
        htb = self.htb
        hr_Rmn = np.einsum('R,Rmn->Rmn', 1 / htb.ndegen, htb.hr_Rmn, optimize=True)
        r_Ramn = np.einsum('R,Ramn->Ramn', 1 / htb.ndegen, htb.r_Ramn, optimize=True)
        ndegen = np.ones(nR, dtype='int64')

        couting_init = np.ones([nR, npos, npos], dtype='float64')
        couting_Rij = np.zeros([nR, npos, npos], dtype='float64')
        hr_Rmn_symm = np.zeros([nR, nw, nw], dtype='complex128')
        # r_Ramn_symm = np.zeros([htb.nR, 3, htb.nw, htb.nw], dtype='complex128')

        self.Rcij = np.einsum('ijRa->Rija', np.full((self.npos, self.npos, nR, 3), self.Rc)) + \
                    np.einsum('Rija->Rija', np.full((nR, self.npos, self.npos, 3), self.pos_car)) - \
                    np.einsum('Rjia->Rija', np.full((nR, self.npos, self.npos, 3), self.pos_car))

        for isym in range(self.nsymmops):
            print('Symmetrizing hr_kmn in real space {}/{}'.format(isym+1, self.nsymmops))
            symmop = symmops[isym]
            TR, det, alpha, nx, ny, nz, taux, tauy, tauz = symmop
            tau = np.array([taux, tauy, tauz])
            tau_car = self.latt @ tau

            if self.iprint >= 2: print('\t', 'cal corep ')
            corep, rep_pos = get_corep(symmop, self.latt, atoms_pos, atoms_orbi, soc)
            rep_TRij, rep_TRmn = self.get_rep_TR(symmop, rep_pos)

            if self.iprint >= 2: print('\t', 'track relevant matrix elements with symmetry ')
            couting_Rij += np.einsum('TRij,mi,Rij,jn->Tmn', rep_TRij, rep_pos.T, couting_init, rep_pos, optimize=True)

            if self.iprint >= 2: print('\t', 'rot hr_Rmn ')
            if int(TR) == 0:
                hr_Rmn_symm += np.einsum('TRij,mi,Rij,jn->Tmn', rep_TRmn, corep.T.conj(), hr_Rmn, corep, optimize=True)
            elif int(TR) == 1:
                hr_Rmn_symm += np.einsum('TRij,mi,Rij,jn->Tmn', rep_TRmn, corep.T.conj(), hr_Rmn, corep, optimize=True).conj()

            if self.iprint >= 2: print()

        hr_Rmn_symm /= self.nsymmops
        # r_Ramn_symm /= self.nsymmops

        print('process hopping at boundary of the super Wigner–Seitz cell')
        couting_Rmn = self.extend_couting_Rij(couting_Rij)
        symm_relevant = np.array(np.array(couting_Rmn, dtype='int64') == self.nsymmops, dtype='complex128')
        hr_Rmn_symm = hr_Rmn_symm * symm_relevant

        num_nonzero = np.count_nonzero(np.array(symm_relevant.real, dtype='int'))
        print('{}% of irrelevant hopping with symmetry are dropped.'.format(
            np.round(100 * (symm_relevant.size - num_nonzero) / symm_relevant.size, 1)
        ))

        # update ndegen, hr_Rmn, r_Ramn
        print('update htb object ...')
        htb.ndegen = ndegen
        htb.hr_Rmn = hr_Rmn_symm
        # self.htb.r_Ramn = r_Ramn_symm

        # update wcc, wccf, and diagonal part of r_Ramn
        htb.wcc, htb.wccf = self.wcc, self.wccf
        if np.nonzero(self.R[nR//2])[0].size != 0: raise WanpyError('R[nR//2] is not [0 0 0].')
        for i in range(3):
            htb.r_Ramn[nR//2, i] *= 1 - np.eye(nw)
            htb.r_Ramn[nR//2, i] += np.diag(self.wcc.T[i])

    '''
      * F.T. between R-space and k-space
    '''
    def ft_gridR_to_meshk(self, Rmn, meshk, gridR, ndegen, tbgauge):
        eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', meshk, gridR, optimize=True))
        if tbgauge:
            eiktau = np.exp(2j * np.pi * np.einsum('ka,ja->kj', meshk, self.wccf))
            kmn = np.einsum('R,kR,km,R...mn,kn->k...mn', 1/ndegen, eikR, eiktau.conj(), Rmn, eiktau, optimize=True)
        else:
            kmn = np.einsum('R,kR,R...mn->k...mn', 1/ndegen, eikR, Rmn, optimize=True)
        return kmn

    def ft_meshk_to_gridR(self, kmn, meshk, gridR, tbgauge):
        nk = meshk.shape[0]
        eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', meshk, gridR, optimize=True))
        if tbgauge:
            eiktau = np.exp(2j * np.pi * np.einsum('ka,ja->kj', meshk, self.wccf))
            Rmn = np.einsum('kR,km,k...mn,kn->R...mn', eikR.conj(), eiktau, kmn, eiktau.conj(), optimize=True) / nk
        else:
            Rmn = np.einsum('kR,k...mn->R...mn', eikR.conj(), kmn, optimize=True) / nk
        return Rmn

    def get_rep_TR(self, symmop, rep_pos):
        """
          Note (need proofread):

          In matrix form, the real space transformation can be expressed as

              H_{m'n'}(R') = P^{latt}_{mn}(R',R) P^{orbi}_{m'm} P^{orbi}_{n'n} H_{mn}(R)

          where we have assumed the real atomic basis. Here, P^{orbi}_{n'n} is the same
          corepresentation matrix as in reciprocal space case,
          P^{latt}_{mn}(R',R) = 1 if under transformation Q={g|v}T

              Q: <tau_{m}|R+tau_{n}> -> <Q(tau_{m})|Q(R+tau_{n})> =  <tau_{Q_m}|R'+tau_{Q_n}>

          with tau_{Q_n} = Q tau_n mode Rj. Therefore, for each pair of (m,n) P^{latt}_{mn}(R',R)
          is determined by

              g(R + tau_n - tau_m) = R' + tau_{Q_n} - tau_{Q_m},

          or alternatively,

              g(R + tau_{Q_n} - tau_{Q_m}) = R' + tau_n - tau_m.

        """
        npos, nR = self.npos, self.nR
        nw_spinless = self.list_nw_at_pos.sum()

        pos_car_rot = np.einsum('ij,ja->ia', rep_pos, self.pos_car, optimize=True)
        Rcij_rot = np.einsum('ijRa->Rija', np.full((npos, npos, nR, 3), self.Rc)) + \
                   np.einsum('Rija->Rija', np.full((nR, npos, npos, 3), pos_car_rot)) - \
                   np.einsum('Rjia->Rija', np.full((nR, npos, npos, 3), pos_car_rot))
        rot_car = get_op_cartesian(symmop)
        Rcij_rot = np.einsum('ab,Rijb->Rija', rot_car, Rcij_rot, optimize=True)

        rep_TRij = np.zeros([nR, nR, npos, npos])
        rep_TRmn = np.zeros([nR, nR, nw_spinless, nw_spinless])

        if self.iprint >= 2: print('\t', 'cal rep_TRij ')
        for ii in range(npos):
            for jj in range(npos):
                dismat = distance_matrix(Rcij_rot[:, ii, jj, :], self.Rcij[:, ii, jj, :])  # 1179 ms
                rep_TRij[:, :, ii, jj] = np.array(dismat < 0.1, dtype='float64')

        if self.iprint >= 2: print('\t', 'cal rep_TRmn ')
        index_i = 0
        for ii in range(npos):
            if self.iprint >= 2: print('\t', 'cal rep_TRmn at position {}/{} '.format(ii+1, npos))
            index_j = 0
            num_i = self.list_nw_at_pos[ii]
            for jj in range(npos):
                num_j = self.list_nw_at_pos[jj]
                rep_TRmn[:, :, index_i:index_i + num_i, index_j:index_j + num_j] = \
                    np.repeat(rep_TRij[:, :, ii, jj], num_i * num_j, axis=1).reshape([nR, nR, num_i, num_j])
                # np.einsum('mnRT->RTmn', np.kron(rep_TRij[:, :, ii, jj], np.ones([num_i, num_j, 1, 1])))
                index_j += self.list_nw_at_pos[jj]
            index_i += self.list_nw_at_pos[ii]

        if self.soc:
            rep_TRmn = np.kron(np.ones([1, 1, 2, 2]), rep_TRmn)
        return rep_TRij, rep_TRmn

    def extend_couting_Rij(self, couting_Rij):
        # extend couting_Rij to couting_Rmn with shape (nR, nw, nw)

        couting_Rmn = np.zeros([self.nR, self.nw_spinless, self.nw_spinless], dtype='float64')
        index_i = 0
        for ii in range(self.npos):
            index_j = 0
            num_i = self.list_nw_at_pos[ii]
            for jj in range(self.npos):
                num_n = self.list_nw_at_pos[jj]
                couting_Rmn[:, index_i:index_i + num_i, index_j:index_j + num_n] = \
                    np.einsum('mnR->Rmn', np.kron(couting_Rij[:, ii, jj], np.ones([num_i, num_n, 1])))
                index_j += self.list_nw_at_pos[jj]
            index_i += self.list_nw_at_pos[ii]
        if self.soc:
            couting_Rmn = np.kron(np.ones([1, 2, 2]), couting_Rmn)
        return couting_Rmn

    '''
      * update wannier center
    '''
    def get_atomic_wcc(self):
        """ get wcc and wccf from input: atoms_orbi """
        _wccf = []
        for i in range(self.ntype_atoms):
            ion = self.atoms_pos[i]
            nw_ion_i = np.sum(2 * np.array(self.atoms_orbi[i]) + 1)
            _wccf.append(np.kron(ion, np.ones([nw_ion_i, 1])))
        wccf = np.vstack(_wccf)
        if self.soc:
            wccf = np.kron(np.ones([2, 1]), wccf)
        wcc = (self.latt @ wccf.T).T
        return wcc, wccf

class Symmetrize_Htb_kspace(object):
    """
      * Note
      1. Only support htb generated by v1.2 version VASP2WANNIER interface
         if higher version in interface was used, one should rearrange the order of
         the wannier basis as v1.2:
         for spin:
             for atoms:
                 for atomic orbitals:
                     ...
      2. Atomic orbitals should follow the default sequence of wannier90.
         For example, for p orbitals: pz, px, py.
         Refer to Chapter 3 of the user guide of wannier90 for more details.
      3. Always ensure the band structure is reproduced correctly before proceeding with further calculations.
      4. The Symmetrize_Htb does not depend on htb.worbi
    """

    def __init__(self, ngridR, htb, symmops, atoms_pos, atoms_orbi, soc, iprint=2):
        self.htb = htb
        self.symmops = symmops
        self.atoms_pos = atoms_pos
        self.atoms_orbi = atoms_orbi
        self.soc = soc

        self.nsymmops = symmops.shape[0]
        self.ntype_atoms = len(atoms_pos)

        self.latt = htb.latt
        self.lattG = 2 * np.pi * LA.inv(self.latt.T)

        self.meshk = make_mesh(nmesh=ngridR)
        self.meshkc = LA.multi_dot([self.lattG, self.meshk.T]).T
        self.nk = self.meshk.shape[0]

        self.ngridR = ngridR
        self.nR, self.ndegen, self.gridR = make_ws_gridR(ngridR, self.latt, info=False)

        self.wcc, self.wccf = self.get_atomic_wcc()
        # self.eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', self.meshk, self.gridR, optimize=True))
        # self.eiktau = np.exp(2j * np.pi * np.einsum('ka,ja->kj', self.meshk, wccf))

        self.iprint = iprint

    def run(self, tmin=1e-6):
        check_valid_symmops(self.symmops)
        htb = self.htb
        symmops, atoms_pos, atoms_orbi, soc = self.symmops, self.atoms_pos, self.atoms_orbi, self.soc

        # write symmops info into htb
        htb.symmops = symmops

        hk_kmn_symm = np.zeros([self.nk, htb.nw, htb.nw], dtype='complex128')
        r_kamn_symm = np.zeros([self.nk, 3, htb.nw, htb.nw], dtype='complex128')
        for i in range(self.nsymmops):
            print('Symmetrizing hr_kmn and r_kamn in reciprocal space {}/{}'.format(i+1, self.nsymmops))

            if self.iprint >= 2: print('\t', 'rot meshk')
            TR = symmops[i][0]
            invg = LA.inv(get_op_cartesian(symmops[i]))
            meshkc_invg = (invg @ self.meshkc.T).T * (-1)**TR
            meshk_invg = (LA.inv(self.lattG) @ meshkc_invg.T).T

            if self.iprint >= 2: print('\t', 'cal corep')
            corep = self.get_corep(symmops[i])

            if self.iprint >= 2: print('\t', 'cal hk_kmn and r_kamn on roted meshk')
            hk_kmn = self.ft_gridR_to_meshk(htb.hr_Rmn, meshk_invg, htb.R, htb.ndegen, tbgauge=True)
            r_kamn = self.ft_gridR_to_meshk(htb.r_Ramn, meshk_invg, htb.R, htb.ndegen, tbgauge=True)

            if self.iprint >= 2: print('\t', 'rot hk_kmn and r_kamn')
            if int(TR) == 0:
                hk_kmn_symm += np.einsum('mi,kij,jn->kmn', corep, hk_kmn, corep.T.conj(), optimize=True)
                r_kamn_symm += np.einsum('mi,kaij,jn->kamn', corep, r_kamn, corep.T.conj(), optimize=True)
            elif int(TR) == 1:
                hk_kmn_symm += np.einsum('mi,kij,jn->kmn', corep, np.conj(hk_kmn), corep.T.conj(), optimize=True)
                r_kamn_symm += np.einsum('mi,kaij,jn->kamn', corep, np.conj(r_kamn), corep.T.conj(), optimize=True)

            if self.iprint >= 2: print()

        hk_kmn_symm /= self.nsymmops
        r_kamn_symm /= self.nsymmops

        # get h and r in gridR from symmetrize hr_kmn and r_kamn
        print('cal hr_Rmn and r_Ramn in new gridR (larger one) from symmetrize hr_kmn and r_kamn ...')
        hr_Rmn_symmtric = self.ft_meshk_to_gridR(hk_kmn_symm, self.meshk, self.gridR, self.ndegen, tbgauge=True)
        r_Ramn_symmtric = self.ft_meshk_to_gridR(r_kamn_symm, self.meshk, self.gridR, self.ndegen, tbgauge=True)

        # update nR, ndegen, R, Rc, hr_Rmn, r_Ramn
        print('updating htb object ...')
        nonzero_indicator_hr = np.array([1 if np.abs(hr_Rmn_symmtric[i]).max() > tmin else 0 for i in range(self.nR)])
        nonzero_indicator_r = np.array([1 if np.abs(r_Ramn_symmtric[i]).max() > tmin else 0 for i in range(self.nR)])
        nonzero = np.where((nonzero_indicator_hr + nonzero_indicator_r) > 0)[0]

        self.htb.nR = nonzero.size
        self.htb.hr_Rmn = hr_Rmn_symmtric[nonzero]
        self.htb.r_Ramn = r_Ramn_symmtric[nonzero]
        self.htb.ndegen = self.ndegen[nonzero]
        self.htb.R = self.gridR[nonzero]
        self.htb.Rc = (self.latt @ self.htb.R.T).T

        # update wcc, wccf, and diagonal part of r_Ramn
        self.htb.wcc, self.htb.wccf = self.wcc, self.wccf
        if np.nonzero(htb.R[htb.nR//2])[0].size != 0: raise WanpyError('R[nR//2] is not [0 0 0].')
        for i in range(3):
            self.htb.r_Ramn[self.htb.nR//2, i] *= 1 - np.eye(self.htb.nw)
            self.htb.r_Ramn[self.htb.nR//2, i] += np.diag(self.wcc.T[i])

    '''
      * F.T. between R-space and k-space
    '''
    def ft_gridR_to_meshk(self, Rmn, meshk, gridR, ndegen, tbgauge):
        eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', meshk, gridR, optimize=True))
        if tbgauge:
            eiktau = np.exp(2j * np.pi * np.einsum('ka,ja->kj', meshk, self.wccf))
            kmn = np.einsum('R,kR,km,R...mn,kn->k...mn', 1/ndegen, eikR, eiktau.conj(), Rmn, eiktau, optimize=True)
        else:
            kmn = np.einsum('R,kR,R...mn->k...mn', 1/ndegen, eikR, Rmn, optimize=True)
        return kmn

    def ft_meshk_to_gridR(self, kmn, meshk, gridR, ndegen, tbgauge):
        eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', meshk, gridR, optimize=True))
        if tbgauge:
            eiktau = np.exp(2j * np.pi * np.einsum('ka,ja->kj', meshk, self.wccf))
            Rmn = np.einsum('kR,km,k...mn,kn->R...mn', eikR.conj(), eiktau, kmn, eiktau.conj(), optimize=True) / self.nk
        else:
            Rmn = np.einsum('kR,k...mn->R...mn', eikR.conj(), kmn, optimize=True) / self.nk
        return Rmn

    '''
      * update wannier center
    '''
    def get_atomic_wcc(self):
        """ get wcc and wccf from input: atoms_orbi """
        _wccf = []
        for i in range(self.ntype_atoms):
            ion = self.atoms_pos[i]
            nw_ion_i = np.sum(2 * np.array(self.atoms_orbi[i]) + 1)
            _wccf.append(np.kron(ion, np.ones([nw_ion_i, 1])))
        wccf = np.vstack(_wccf)
        if self.soc:
            wccf = np.kron(np.ones([2, 1]), wccf)
        wcc = (self.latt @ wccf.T).T
        return wcc, wccf

    '''
      * corep of magnetic space group
    '''
    # def get_corep(self, symmop):
    #     corep, rep_pos = get_corep(symmop, self.latt, self.atoms_pos, self.atoms_orbi, self.soc)
    #     return corep

    def get_corep(self, symmop):
        """
          The same with wanpy.core.symmetry.get_corep, I will remove this function later.

          Get corep of magnetic space group operation.
          Note the k-independent corep is only applicable for tb-gauge case. For the lattice-gauge case, the corep
          depends on k.

          For the unitary operator:
            D_k^{latt}(g) = S(k).D^{tb}(g).S(k)*,   (1)
          where S(k) = e^(ik\tau), and \tau is the coordinates of orbital center.
          On the other hand, for the anti-unitary operator, we have:
            D_k^{latt}(g) = S(k).D^{tb}(g).S(k).    (2)

        """
        # atoms_pos = np.array([
        #     [
        #         [0, 0, 0],
        #         [0.5, 0.5, 0],
        #     ],
        #     [
        #         [0.5, 0, 0.5],
        #         [0, 0.5, 0.5],
        #     ]
        # ])
        # atoms_orbi_l = [
        #     [0, 2],
        #     [0, 1, 2]
        # ]

        atoms_pos, atoms_orbi, soc = self.atoms_pos, self.atoms_orbi, self.soc
        TR, det, alpha, nx, ny, nz, taux, tauy, tauz = symmop
        tau = np.array([taux, tauy, tauz])
        tau_car = self.latt @ tau

        _corep = []
        for i in range(self.ntype_atoms):
            ion = atoms_pos[i]
            ion = np.remainder(ion, 1) # define the un-roted ion in [0, 1) # added at v0.15.2
            orbi_l = atoms_orbi[i]

            # get op_orbi
            _rep_orbi = [get_rep_atomic(symmop, l) for l in orbi_l]
            rep_orbi = block_diag(*_rep_orbi)

            # get op_pos
            rot = get_op_cartesian(symmop)

            ion_car = (self.latt @ ion.T).T
            ion_car_rot = (rot @ ion_car.T).T + tau_car

            ion_rot = (LA.inv(self.latt) @ ion_car_rot.T).T
            # ion_rot = np.remainder(ion_rot + 1e-5, 1.0)    # move ion at 1- to 0+ # removed at v0.15.2
            ion_rot = np.remainder(ion_rot, 1.0)  # move ion_rot at [0, 1) # added at v0.15.2
            ion_car_rot = (self.latt @ ion_rot.T).T

            dismat = distance_matrix(ion_car, ion_car_rot)
            rep_pos = np.array(dismat < 0.01, dtype='float')

            assert LA.matrix_rank(rep_pos) == ion.shape[0] # check if a valid symmop

            # get op
            _corep.append(np.kron(rep_pos, rep_orbi))
        corep = block_diag(*_corep)
        if soc:
            rep_spin = get_rep_spin(symmop)
            corep = np.kron(rep_spin, corep)
        return corep

def get_rep_spin(symmop):
    TR, det, theta, nx, ny, nz, taux, tauy, tauz = symmop
    axis = np.array([nx, ny, nz]) / LA.norm(np.array([nx, ny, nz]))
    D = np.cos(theta/2) * sigma0 - 1j * np.sin(theta/2) * np.einsum('amn,a->mn', sigma, axis)
    # if TR == 1: D = -1j * sigmay @ D # this line is incorrect, see operation rule for corepresentation.
    if TR == 1: D = D @ (-1j * sigmay)
    D[np.abs(D)<1e-10] = 0
    return D

def get_corep(symmop, latt, atoms_pos, atoms_orbi, soc):
    """
      Get corep of magnetic space group operation.
      Note the k-independent corep is only applicable for tb-gauge case. For the lattice-gauge case, the corep
      depends on k.

      For the unitary operator:
        D_k^{latt}(g) = S(k).D^{tb}(g).S(k)*,   (1)
      where S(k) = e^(ik\tau), and \tau is the coordinates of orbital center.
      On the other hand, for the anti-unitary operator, we have:
        D_k^{latt}(g) = S(k).D^{tb}(g).S(k).    (2)

      Regulation of the basis order:
       - spin index
         - atomic positions
           - atomic orbitals
             ...
           ...
         ...
      for example in monolayer graphene with s and p orbitals at each C atoms and with SOC, the basis is
          |u,C1,s> |u,C1,pz> |u,C1,px> |u,C1,py>
          |u,C2,s> |u,C2,pz> |u,C2,px> |u,C2,py>
          |d,C1,s> |d,C1,pz> |d,C1,px> |d,C1,py>
          |d,C2,s> |d,C2,pz> |d,C2,px> |d,C2,py>
    """

    # atoms_pos, atoms_orbi, soc = self.atoms_pos, self.atoms_orbi, self.soc
    ntype_atoms = len(atoms_pos)
    TR, det, alpha, nx, ny, nz, taux, tauy, tauz = symmop
    tau = np.array([taux, tauy, tauz])
    tau_car = latt @ tau

    _corep = []
    _rep_pos = []
    for i in range(ntype_atoms):
        ion = atoms_pos[i]
        ion = np.remainder(ion, 1) # define the un-roted ion in [0, 1) # added at v0.15.2
        orbi_l = atoms_orbi[i]

        # get op_orbi
        _rep_orbi_i = [get_rep_atomic(symmop, l) for l in orbi_l]
        rep_orbi_i = block_diag(*_rep_orbi_i)

        # get op_pos
        rot = get_op_cartesian(symmop)

        ion_car = (latt @ ion.T).T
        ion_car_rot = (rot @ ion_car.T).T + tau_car

        ion_rot = (LA.inv(latt) @ ion_car_rot.T).T
        ion_rot = np.remainder(ion_rot + 1e-5, 1.0)  # move ion at 1- to 0+ # removed at v0.15.2
        # ion_rot = np.remainder(ion_rot, 1.0)  # move ion_rot at [0, 1) # added at v0.15.2
        ion_car_rot = (latt @ ion_rot.T).T

        dismat = distance_matrix(ion_car, ion_car_rot)
        rep_pos_i = np.array(dismat < 0.01, dtype='float')

        # print('ion:', ion)
        # print('ion_rot:', ion_rot)
        assert LA.matrix_rank(rep_pos_i) == ion.shape[0]  # check if a valid symmop

        # get op
        _corep.append(np.kron(rep_pos_i, rep_orbi_i))
        _rep_pos.append(rep_pos_i)
    corep = block_diag(*_corep)
    rep_pos = block_diag(*_rep_pos)
    if soc:
        rep_spin = get_rep_spin(symmop)
        corep = np.kron(rep_spin, corep)
    return corep, rep_pos

def get_rep_atomic(symmop, l):
    """
    This function generates representation of a given O(3) group element in the atomic orbital basis.

    The rep $D_{nm}$ obeys the following equation:
    $\hat{P}_{g}Y_{lm}\left(\boldsymbol{r}\right)=Y_{lm}\left(g^{-1}\boldsymbol{r}\right)=\sum_{n}Y_{ln}\left(\boldsymbol{r}\right)D_{nm}\left(g\right),$
    where $g\in O(3)$.
    """
    x, y, z = sp.symbols('x, y, z')
    r = sp.Matrix([x, y, z])

    rot = get_op_cartesian(symmop)
    invrot = LA.inv(rot)

    if l == 0: return np.eye(1)

    D = np.zeros([2*l+1, 2*l+1], dtype='float64')
    for m in range(2*l+1):
        expr = Ylm(l, m, invrot@r)
        func_rot = expr.expand()
        if l == 1:
            D[0, m] = func_rot.coeff(z)     # pz
            D[1, m] = func_rot.coeff(x)     # px
            D[2, m] = func_rot.coeff(y)     # py
        elif l == 2:
            D[0, m] = func_rot.coeff(z**2) * np.sqrt(3)                   # dz2
            D[1, m] = func_rot.coeff(x).coeff(z)                          # dxz
            D[2, m] = func_rot.coeff(y).coeff(z)                          # dyz
            D[3, m] = func_rot.coeff(x**2) * 2 + D[0, m] / np.sqrt(3)     # dx2-y2
            D[4, m] = func_rot.coeff(x).coeff(y)                          # dxy
        elif l == 3:
            D[0, m] = func_rot.coeff(z**3) * np.sqrt(15)                                # fz3
            D[1, m] = func_rot.coeff(x).coeff(z**2) * np.sqrt(10) / 2                   # fxz2
            D[2, m] = func_rot.coeff(y).coeff(z**2) * np.sqrt(10) / 2                   # fyz2
            D[3, m] = func_rot.coeff(z).coeff(x**2) * 2 + D[0, m] * 3 / np.sqrt(15)     # fz(x2-y2)
            D[4, m] = func_rot.coeff(x).coeff(y).coeff(z)                               # fxyz
            D[5, m] = (func_rot.coeff(x**3) * 2 + D[1, m] / np.sqrt(10)) * np.sqrt(6)   # fx(x2-3y2)
            D[6, m] = (func_rot.coeff(y**3) * -2 - D[2, m] / np.sqrt(10)) * np.sqrt(6)  # fy(3x2-y2)
    D[np.abs(D)<1e-10] = 0
    return D

def Ylm(l, m, r):
    """
    Get real valued spherical harmonics Ylm, mapping the sphere S^2 to R.
    The parameters l and m align with wannier90 projectors as described in Chap. 3 of the user guide.
    Refer to https://en.wikipedia.org/wiki/Table_of_spherical_harmonics for the explicit expression of Ylm.
    """
    x, y, z = r
    if l == 0: return 1     # s
    elif l == 1:
        if m == 0: return z      # pz
        elif m == 1: return x    # px
        elif m == 2: return y    # py
    elif l == 2:
        if m == 0: return (2*z**2 - x**2 - y**2)/2/np.sqrt(3)   # dz2
        if m == 1: return x * z                                 # dxz
        if m == 2: return y * z                                 # dyz
        if m == 3: return (x**2 - y**2)/2                       # dx2-y2
        if m == 4: return x * y                                 # dxy
    elif l == 3:
        if m == 0: return z*(2*z**2 - 3*x**2 - 3*y**2)/2/np.sqrt(15)    # fz3
        if m == 1: return x*(4*z**2 -x**2 - y**2)/2/np.sqrt(10)         # fxz2
        if m == 2: return y*(4*z**2 -x**2 - y**2)/2/np.sqrt(10)         # fyz2
        if m == 3: return z*(x**2 - y**2)/2                             # fz(x2-y2)
        if m == 4: return x*y*z                                         # fxyz
        if m == 5: return x*(x**2 - 3*y**2)/2/np.sqrt(6)                # fx(x2-3y2)
        if m == 6: return y*(3*x**2 - y**2)/2/np.sqrt(6)                # fy(3x2-y2)

def get_proj_info(htb, wannier_center_def):
    """
      * Get atoms_pos, atoms_orbi from htb
        used to simply the input of Symmetrize_Htb
    """
    orbital_dict = {0:'s', 1:'p', 2:'d', 3:'f'}
    atoms_pos = []
    atoms_spec = []
    atoms_orbi = []

    if htb.worbi.soc:
        nw = htb.nw//2
    else:
        nw = htb.nw

    proj_l = htb.worbi.proj_lmr.T[0][:nw]
    proj_wccf = htb.worbi.proj_wccf[:nw]
    ion = htb.cell.ions
    spec = htb.cell.spec
    if wannier_center_def.lower() == 'ws':
        # refined in range of [-0.5, 0.5) to keep in line with the wannier center
        # used in calculating amn in VASP 6.4.3.
        proj_wccf = np.remainder(proj_wccf + 100.5, 1) - 0.5
        ion = np.remainder(ion + 100.5, 1) - 0.5
    elif wannier_center_def.lower() == 'poscar':
        # proj_wccf origins from wannier_setup, and are in line with POSCAR,
        # if wannier_center_def = poscar, do nothing here.
        pass
    else:
        WanpyInputError('wannier_center_def should be poscar or ws')

    i = 0
    while True:
        # print(i)
        if i >= nw: break
        wccf = proj_wccf[i]
        # find one index of ion closing to proj_wccf[i]
        index_ion_first = np.argmin(LA.norm(wccf - ion, axis=1))
        # get name of the jth spec of this proj_wccf[i]
        atoms_spec_j = spec[index_ion_first]
        # find indexes of all atoms of the jth spec
        index_ion_all = np.where(np.array(spec) == atoms_spec_j)[0]
        n_ion = index_ion_all.size
        # find all atomic positions of the jth spec
        atoms_pos_j = ion[index_ion_all]

        # how many proj_wccf share the same position of proj_wccf[i],
        # i.e., the number of orbitals on this atomic position
        n = np.where(np.isclose(proj_wccf, wccf).all(axis=1))[0].size
        # get l of jth spec
        _, index_orbi = np.unique(proj_l[i:i+n], return_index=True)
        atoms_orbi_j = proj_l[i:i+n][np.sort(index_orbi)]

        i += n_ion * n

        atoms_pos.append(atoms_pos_j)
        atoms_spec.append(atoms_spec_j)
        atoms_orbi.append(atoms_orbi_j)

    # if standardize_atoms_pos:
    #     # refine atoms_pos in range of [-0.5, 0.5)
    #     atoms_pos = [np.remainder(i + 100.5, 1) - 0.5 for i in atoms_pos]

    print('wanpy found the following projections:')
    for j in range(len(atoms_pos)):
        print(atoms_spec[j], ':', ' '.join([orbital_dict[i] for i in atoms_orbi[j]]) )

    return atoms_pos, atoms_orbi

def parse_symmetry_inputfile(fname='symmetry.in'):

    def str_to_bool(s):
        return {'t': True, 'f': False}.get(s.lower()[0], None)

    # set default values
    symmetric_method = 'kspace'
    rspace_use_ngridR = False
    parse_symmetry = 'man'
    wannier_center_def = None
    ngridR = None
    symprec = 1e-5
    magmoms = []
    symmops = []

    # update values from file
    with open(fname, 'r') as f:
        while True:
            _pos = f.tell()
            inline = f.readline().split('#')[0]
            inline_keyword = inline.split('=')[0].strip()

            if _pos == f.tell(): break          # If no new line was read, exit the loop

            if inline_keyword == 'symmetric_method':
                symmetric_method = str(inline.split('=')[1].split()[0]).lower()

            if inline_keyword == 'rspace_use_ngridR':
                rspace_use_ngridR = str_to_bool(str(inline.split('=')[1].split()[0]).lower())

            if inline_keyword == 'parse_symmetry':
                parse_symmetry = str(inline.split('=')[1].split()[0]).lower()

            if inline_keyword == 'wannier_center_def':
                wannier_center_def = str(inline.split('=')[1].split()[0]).lower()

            if inline_keyword == 'ngridR':
                ngridR = [int(i) for i in inline.split('=')[1].split()]

            if inline_keyword == 'symprec':
                symprec = float(inline.split('=')[1].split()[0])

            if inline_keyword == '&magmoms':
                while '/' not in inline:
                    inline = f.readline().split('#')[0]
                    magmoms_i = inline.split()
                    if len(magmoms_i) == 3:
                        magmoms_i = [float(i) for i in magmoms_i]
                        magmoms.append(magmoms_i)

            if inline_keyword == '&symmops':
                while '/' not in inline:
                    inline = f.readline().split('#')[0]
                    symmops_i = inline.split()
                    if len(symmops_i) == 9:
                        symmops_i = [float(i) for i in symmops_i]
                        symmops_i[2] = symmops_i[2]/180 * np.pi
                        symmops.append(symmops_i)
    symmops = np.array(symmops)
    magmoms = np.array(magmoms)
    f.close()

    check_valid_symmops(symmops)

    adict = {
        'symmetric_method': symmetric_method,
        'rspace_use_ngridR': rspace_use_ngridR,
        'parse_symmetry': parse_symmetry,
        'wannier_center_def': wannier_center_def,
        'ngridR': ngridR,
        'symprec': symprec,
        'magmoms': magmoms,
        'symmops': symmops,
    }
    wanpynamedtuple = namedtuple('WanpyNamedTuple', adict.keys())
    win = wanpynamedtuple(**adict)
    return win

if __name__ == "__main__":
    pass
