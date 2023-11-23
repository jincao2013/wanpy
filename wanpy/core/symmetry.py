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

__date__ = "Mar. 30, 2023"

from collections import namedtuple
import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance_matrix
from scipy.linalg import block_diag
import sympy as sp
from wanpy.core.errorhandler import WanpyError
from wanpy.core.mesh import make_mesh, make_ws_gridR
from wanpy.core.utils import get_op_cartesian
from wanpy.core.units import *

# class MPointGroup(object):
#
#     def __init__(self):
#         self.name = None
#         self.latt = None
#         self.lattG = None
#         self.n_op = None
#         self.elements = None
#         self.op_axis = None
#         self.op_fraction = None
#         self.op_cartesian = None
#         self.TR = None
#
#     def get_op_cartesian(self):
#         self.n_op = len(self.op_axis)
#         self.op_cartesian = np.zeros([self.n_op, 3, 3], dtype='float64')
#         self.TR = np.zeros([self.n_op], dtype='float64')
#         for i in range(self.n_op):
#             _TR, det, theta, nx, ny, nz = self.op_axis[i]
#             n = np.array([nx, ny, nz]) / LA.norm(np.array([nx, ny, nz]))
#             rot = scipy_rot.from_rotvec(theta * n)
#             self.op_cartesian[i] = det * rot.as_matrix()
#             self.TR[i] = _TR


class Symmetrize_Htb(object):
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

    def __init__(self, ngridR, htb, symmops, atoms_pos, atoms_orbi, soc):
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

    def run(self, tmin=1e-6):
        self.check_valid_symmops()
        htb = self.htb
        symmops, atoms_pos, atoms_orbi, soc = self.symmops, self.atoms_pos, self.atoms_orbi, self.soc

        # write symmops info into htb
        htb.symmops = symmops

        # get h and r in meshk from original htb
        # print('[Symmetrize_Htb] calculating h and r in meshk from original htb ...')
        # hk_kmn = self.get_hk_kmn_from_htb(self.htb, tbgauge=True)
        # r_kamn = self.get_rk_from_htb(self.htb, tbgauge=True)


        # symmetry hr_kmn and r_kamn
        # print('[Symmetrize_Htb] symmetrizing hr_kmn and r_kamn ...')
        # wcc, wccf = self.get_atomic_wcc()

        # eiktau = np.exp(2j * np.pi * np.einsum('ka,ja->kj', self.meshk, wccf))

        # hk_kmn_symm = np.zeros_like(hk_kmn)
        # r_kamn_symm = np.zeros_like(r_kamn)
        # for i in range(self.nsymmops):
        #     invgk = self.get_index_invgk(symmops[i])
        #     corep_tb = self.get_corep(symmops[i], atoms_pos, atoms_orbi, soc)
        #     if int(symmops[i, 0]) == 1:
        #         # convert corep to lattice gauge
        #         # corep_latt = np.einsum('ki,ij,kj->kij', eiktau, corep_tb, eiktau.conj(), optimize=True)
        #         hk_kmn_symm += np.einsum('mi,kij,jn->kmn', corep_tb, hk_kmn[invgk], corep_tb.T.conj(), optimize=True)
        #         r_kamn_symm += np.einsum('mi,kaij,jn->kamn', corep_tb, r_kamn[invgk], corep_tb.T.conj(), optimize=True)
        #         # hk_kmn_symm += np.einsum('kmi,kij,knj->kmn', corep_latt, hk_kmn[invgk], corep_latt.conj(), optimize=True)
        #         # r_kamn_symm += np.einsum('kmi,kaij,knj->kamn', corep_latt, r_kamn[invgk], corep_latt.conj(), optimize=True)
        #     elif int(symmops[i, 0]) == -1:
        #         # convert corep to lattice gauge
        #         # corep_latt = np.einsum('ki,ij,kj->kij', eiktau, corep_tb, eiktau, optimize=True)
        #         hk_kmn_symm += np.einsum('mi,kij,jn->kmn', corep_tb, np.conj(hk_kmn[invgk]), corep_tb.T.conj(), optimize=True)
        #         r_kamn_symm += np.einsum('mi,kaij,jn->kamn', corep_tb, np.conj(r_kamn[invgk]), corep_tb.T.conj(), optimize=True)
        #         # hk_kmn_symm += np.einsum('kmi,kij,knj->kmn', corep_latt, np.conj(hk_kmn[invgk]), corep_latt.conj(), optimize=True)
        #         # r_kamn_symm += np.einsum('kmi,kaij,knj->kamn', corep_latt, np.conj(r_kamn[invgk]), corep_latt.conj(), optimize=True)

        hk_kmn_symm = np.zeros([self.nk, htb.nw, htb.nw], dtype='complex128')
        r_kamn_symm = np.zeros([self.nk, 3, htb.nw, htb.nw], dtype='complex128')
        for i in range(self.nsymmops):
            print('[Symmetrize_Htb] symmetrizing hr_kmn and r_kamn {}/{}'.format(i+1, self.nsymmops))
            TR = symmops[i][0]
            invg = LA.inv(get_op_cartesian(symmops[i]))
            meshkc_invg = (invg @ self.meshkc.T).T * (-1)**TR
            meshk_invg = (LA.inv(self.lattG) @ meshkc_invg.T).T
            corep = self.get_corep(symmops[i])
            hk_kmn = self.ft_gridR_to_meshk(htb.hr_Rmn, meshk_invg, htb.R, htb.ndegen, tbgauge=True)
            r_kamn = self.ft_gridR_to_meshk(htb.r_Ramn, meshk_invg, htb.R, htb.ndegen, tbgauge=True)
            if int(TR) == 0:
                hk_kmn_symm += np.einsum('mi,kij,jn->kmn', corep, hk_kmn, corep.T.conj(), optimize=True)
                r_kamn_symm += np.einsum('mi,kaij,jn->kamn', corep, r_kamn, corep.T.conj(), optimize=True)
            elif int(TR) == 1:
                hk_kmn_symm += np.einsum('mi,kij,jn->kmn', corep, np.conj(hk_kmn), corep.T.conj(), optimize=True)
                r_kamn_symm += np.einsum('mi,kaij,jn->kamn', corep, np.conj(r_kamn), corep.T.conj(), optimize=True)

        hk_kmn_symm /= self.nsymmops
        r_kamn_symm /= self.nsymmops

        # get h and r in gridR from symmetrize hr_kmn and r_kamn
        # self.htb.hr_Rmn = self.ft_meshk_to_gridR(hk_kmn_symm, self.meshk, self.gridR, self.ndegen, tbgauge=True)
        # self.htb.r_Ramn = self.ft_meshk_to_gridR(r_kamn_symm, self.meshk, self.gridR, self.ndegen, tbgauge=True)
        # self.htb.nR = self.nR
        # self.htb.ndegen = self.ndegen
        # self.htb.R = self.gridR
        # self.htb.Rc = (self.latt @ self.gridR.T).T

        # get h and r in gridR from symmetrize hr_kmn and r_kamn
        print('[Symmetrize_Htb] get h and r in gridR from symmetrize hr_kmn and r_kamn ...')
        hr_Rmn_symmtric = self.ft_meshk_to_gridR(hk_kmn_symm, self.meshk, self.gridR, self.ndegen, tbgauge=True)
        r_Ramn_symmtric = self.ft_meshk_to_gridR(r_kamn_symm, self.meshk, self.gridR, self.ndegen, tbgauge=True)

        # update nR, ndegen, R, Rc, hr_Rmn, r_Ramn
        print('[Symmetrize_Htb] updating htb object ...')
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

    def check_valid_symmops(self):
        symmops = self.symmops
        assert set([int(i) for i in symmops.T[0]]) == {0, 1}
        assert set([int(i) for i in symmops.T[1]]) == {-1, 1}

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
      * F.T.: from R-space to k-space
        hr_Rmn = <0m|H|Rn>
        hk_kmn = <mk|H|nk>
        r_Ramn = <0m|r|Rn>
        r_kamn = <mk|r|nk>
    '''
    # def get_hk_kmn(self, hr_Rmn, meshk, gridR, ndegen, tbgauge):
    #     eikR = np.exp(2j * np.pi * np.einsum('R,ka,Ra->kR', 1/ndegen, meshk, gridR, optimize=True))
    #     if tbgauge:
    #         eiktau = np.exp(2j * np.pi * np.einsum('ka,ja->kj', meshk, self.wccf))
    #         hk_kmn = np.einsum('kR,km,Rmn,kn->kmn', eikR, eiktau.conj(), hr_Rmn, eiktau, optimize=True)
    #     else:
    #         hk_kmn = np.einsum('kR,Rmn->kmn', eikR, hr_Rmn, optimize=True)
    #     return hk_kmn
    #
    # def get_r_kamn(self, r_Ramn, meshk, gridR, ndegen, tbgauge):
    #     eikR = np.exp(2j * np.pi * np.einsum('R,ka,Ra->kR', 1/ndegen, meshk, gridR, optimize=True))
    #     if tbgauge:
    #         r_kamn = np.einsum('kR,km,Ramn,kn->kamn', self.eiktau.conj(), self.eikR, r_Ramn, self.eiktau, optimize=True)
    #     else:
    #         r_kamn = np.einsum('kR,Ramn->kamn', eikR, r_Ramn, optimize=True)
    #     return r_kamn
    #
    # def get_hk_kmn_from_htb(self, htb, meshk, tbgauge):
    #     eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', meshk, htb.R, optimize=True))
    #     if tbgauge:
    #         eiktau = np.exp(2j * np.pi * np.einsum('ka,ja->kj', meshk, self.wccf))
    #         hk_kmn = np.einsum('R,kR,km,Rmn,kn->kmn', 1/htb.ndegen, eikR, self.eiktau.conj(), htb.hr_Rmn, self.eiktau, optimize=True)
    #     else:
    #         hk_kmn = np.einsum('R,kR,Rmn->kmn', 1/htb.ndegen, eikR, htb.hr_Rmn, optimize=True)
    #     return hk_kmn
    '''
      * F.T.: from k-space to R-space
    '''
    # def get_hr_Rmn(self, hk_kmn, meshk, gridR, ndegen, tbgauge):
    #     eikR = np.exp(2j * np.pi * np.einsum('R,ka,Ra->kR', 1/ndegen, meshk, gridR, optimize=True))
    #     if tbgauge:
    #         hr_Rmn = np.einsum('kR,km,kmn,kn->Rmn', self.eikR.conj(), self.eiktau, hk_kmn, self.eiktau.conj(), optimize=True) / self.nk
    #     else:
    #         hr_Rmn = np.einsum('kR,kmn->Rmn', self.eikR.conj(), hk_kmn, optimize=True) / self.nk
    #     return hr_Rmn
    #
    # def get_r_Ramn(self, r_kamn, meshk, gridR, ndegen, tbgauge):
    #     eikR = np.exp(2j * np.pi * np.einsum('R,ka,Ra->kR', 1/ndegen, meshk, gridR, optimize=True))
    #     if tbgauge:
    #         r_Ramn = np.einsum('kR,km,kamn,kn->Ramn', self.eikR.conj(), self.eiktau, r_kamn, self.eiktau.conj(), optimize=True) / self.nk
    #     else:
    #         r_Ramn = np.einsum('kR,kamn->Ramn', self.eikR.conj(), r_kamn, optimize=True) / self.nk
    #     return r_Ramn
    #
    # def get_rk_from_htb(self, htb, tbgauge):
    #     eikR = np.exp(2j * np.pi * np.einsum('ka,Ra->kR', self.meshk, htb.R, optimize=True))
    #     if tbgauge:
    #         rk = np.einsum('R,kR,km,Ramn,kn->kamn', 1/htb.ndegen, eikR, self.eiktau.conj(), htb.r_Ramn, self.eiktau, optimize=True)
    #     else:
    #         rk = np.einsum('R,kR,Ramn->kamn', 1/htb.ndegen, eikR, htb.r_Ramn, optimize=True)
    #     return rk

    '''
      * update wannier center
    '''
    def get_atomic_wcc(self):
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
      * rotate the k-mesh
    '''
    # def get_index_invgk(self, symmop):
    #     TR = symmop[0]
    #     invg = LA.inv(get_op_cartesian(symmop))
    #     meshkc_rot = (invg @ self.meshkc.T).T * (-1)**TR
    #     meshk_rot = (LA.inv(self.lattG) @ meshkc_rot.T).T
    #     meshk_rot = np.remainder(meshk_rot + 1e-10, 1.0)
    #     meshkc_rot = (self.lattG @ meshk_rot.T).T
    #
    #     dismat = distance_matrix(meshkc_rot, self.meshkc)
    #     index_gk = np.argmin(dismat, 1)
    #     # self.meshkc[index] - meshkc_rot
    #     x, y = np.where(dismat < 1e-5)
    #     if x.size != self.nk:
    #         raise WanpyError('error in find index_gk')
    #     return index_gk

    '''
      * corep of magnetic space group
    '''
    def get_corep(self, symmop):
        """
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
            orbi_l = atoms_orbi[i]

            # get op_orbi
            _rep_orbi = [get_rep_atomic(symmop, l) for l in orbi_l]
            rep_orbi = block_diag(*_rep_orbi)

            # get op_pos
            rot = get_op_cartesian(symmop)

            ion_car = (self.latt @ ion.T).T
            ion_car_rot = (rot @ ion_car.T).T + tau_car

            ion_rot = (LA.inv(self.latt) @ ion_car_rot.T).T
            ion_rot = np.remainder(ion_rot + 1e-5, 1.0)    # move ion at 1- to 0+
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
            D[1, m] = func_rot.coeff(x).coeff(z)                                # dxz
            D[2, m] = func_rot.coeff(y).coeff(z)                                # dyz
            D[3, m] = func_rot.coeff(x**2) * 2 + D[0, m] / np.sqrt(3)     # dx2-y2
            D[4, m] = func_rot.coeff(x).coeff(y)                                # dxy
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

def get_proj_info(htb):
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

    i = 0
    while True:
        # print(i)
        if i >= nw: break
        wccf = proj_wccf[i]
        index_ion_first = np.argmin(LA.norm(wccf - htb.cell.ions, axis=1))
        index_ion_all = np.where(np.array(htb.cell.spec) == htb.cell.spec[index_ion_first])[0]
        n_ion = index_ion_all.size
        atoms_spec_j = htb.cell.spec[index_ion_first]
        atoms_pos_j = htb.cell.ions[index_ion_all]

        n = np.where(np.isclose(proj_wccf, wccf).all(axis=1))[0].size
        _, index_orbi = np.unique(proj_l[i:i+n], return_index=True)
        atoms_orbi_j = proj_l[i:i+n][np.sort(index_orbi)]

        i += n_ion * n

        atoms_pos.append(atoms_pos_j)
        atoms_spec.append(atoms_spec_j)
        atoms_orbi.append(atoms_orbi_j)

    print('wanpy found the following projections:')
    for j in range(len(atoms_pos)):
        print(atoms_spec[j], ':', ' '.join([orbital_dict[i] for i in atoms_orbi[j]]) )

    return atoms_pos, atoms_orbi

# def read_symmetry_inputfile(fname='symmetry.in'):
#     symmops = []
#     with open(fname, 'r') as f:
#         while True:
#             inline = f.readline().split('#')[0]
#
#             if not inline:
#                 break
#
#             if 'ngridR' in inline:
#                 ngridR = [int(i) for i in inline.split()[-3:]]
#
#             if 'symmops' in inline:
#                 while '/' not in inline:
#                     inline = f.readline().split('#')[0]
#                     symmops_i = inline.split()
#                     if len(symmops_i) == 9:
#                         symmops_i = [float(i) for i in symmops_i]
#                         symmops_i[2] = symmops_i[2]/180 * np.pi
#                         symmops.append(symmops_i)
#                         assert int(symmops_i[0]) in [0, 1]
#                         assert int(symmops_i[1]) in [-1, 1]
#     symmops = np.array(symmops)
#     f.close()
#     return ngridR, symmops

def parse_symmetry_inputfile(fname='symmetry.in'):
    symmops = []
    magmoms = []
    symprec = 1e-5
    with open(fname, 'r') as f:
        while True:
            _pos = f.tell()
            inline = f.readline().split('#')[0]

            if _pos == f.tell(): break

            if 'ngridR' in inline:
                ngridR = [int(i) for i in inline.split('=')[1].split()]

            if 'parse_symmetry' in inline:
                parse_symmetry = str(inline.split('=')[1].split()[0])

            if 'symprec' in inline:
                symprec = float(inline.split('=')[1].split()[0])

            if 'magmoms' in inline:
                while '/' not in inline:
                    inline = f.readline().split('#')[0]
                    magmoms_i = inline.split()
                    if len(magmoms_i) == 3:
                        magmoms_i = [float(i) for i in magmoms_i]
                        magmoms.append(magmoms_i)

            if 'symmops' in inline:
                while '/' not in inline:
                    inline = f.readline().split('#')[0]
                    symmops_i = inline.split()
                    if len(symmops_i) == 9:
                        symmops_i = [float(i) for i in symmops_i]
                        symmops_i[2] = symmops_i[2]/180 * np.pi
                        symmops.append(symmops_i)
                        assert int(symmops_i[0]) in [0, 1]
                        assert int(symmops_i[1]) in [-1, 1]
    symmops = np.array(symmops)
    magmoms = np.array(magmoms)
    f.close()

    adict = {
        'ngridR': ngridR,
        'parse_symmetry': parse_symmetry,
        'symprec': symprec,
        'magmoms': magmoms,
        'symmops': symmops,
    }
    wanpynamedtuple = namedtuple('WanpyNamedTuple', adict.keys())
    win = wanpynamedtuple(**adict)
    return win

if __name__ == "__main__":
    import os
    # from wanpy.core.plot import *
    # import matplotlib.pyplot as plt
    from wanpy.env import ROOT_WDIR
    from wanpy.core.structure import Htb

    wdir = os.path.join(ROOT_WDIR, r'symmtric_htb')
    input_dir = os.path.join(ROOT_WDIR, r'symmtric_htb/htblib')

    # htb_fname = r'htb.MnPd.afm100.h5'
    # htb_fname = r'htb.MnPt.afm100.h5'
    htb_fname = r'htb.vs2vs.afmy.h5'

    os.chdir(input_dir)
    htb = Htb()
    htb.load_h5(htb_fname)
    os.chdir(wdir)

    # soc = True
    # ngridR = np.array([14, 14, 14])
    # symmops = np.array([
    #     # TR, det, alpha, nx, ny, nz, taux, tauy, tauz
    #     [0, 1, 0, 0, 0, 1, 0, 0, 0], # e
    #     [0, 1, np.pi, 1, 0, 0, 0, 0, 0], # c2x
    #     [0, 1, np.pi, 0, 1, 0, 0.5, 0.5, 0], # c2y
    #     [0, 1, np.pi, 0, 0, 1, 0.5, 0.5, 0], # c2z
    #     [0, -1, 0, 0, 0, 1, 0, 0, 0], # P
    #     [0, -1, np.pi, 1, 0, 0, 0, 0, 0], # mx
    #     [0, -1, np.pi, 0, 1, 0, 0.5, 0.5, 0], # my
    #     [0, -1, np.pi, 0, 0, 1, 0.5, 0.5, 0], # mz
    #     # anti unitary
    #     [1, 1, 0, 0, 0, 1, 0.5, 0.5, 0], # T
    #     [1, 1, np.pi, 1, 0, 0, 0.5, 0.5, 0], # c2xT
    #     [1, 1, np.pi, 0, 1, 0, 0, 0, 0], # c2yT
    #     [1, 1, np.pi, 0, 0, 1, 0, 0, 0], # c2zT
    #     [1, -1, 0, 0, 0, 1, 0.5, 0.5, 0], # PT
    #     [1, -1, np.pi, 1, 0, 0, 0.5, 0.5, 0], # mxT
    #     [1, -1, np.pi, 0, 1, 0, 0, 0, 0], # myT
    #     [1, -1, np.pi, 0, 0, 1, 0, 0, 0], # mzT
    # ])
    # # symmops = np.array([
    # #     # TR, det, alpha, nx, ny, nz, taux, tauy, tauz
    # #     [0, 1, 0, 0, 0, 1, 0, 0, 0], # e
    # #     [1, -1, 0, 0, 0, 1, 0.5, 0.5, 0], # PT
    # # ])
    # # atoms_pos = [
    # #     np.array([
    # #         [0, 0, 0],
    # #         [0.5, 0.5, 0],
    # #     ]),
    # #     np.array([
    # #         [0.5, 0, 0.5],
    # #         [0, 0.5, 0.5],
    # #     ])
    # # ]
    # # atoms_orbi = [
    # #     [0, 2],
    # #     [0, 1, 2]
    # # ]
    # atoms_pos, atoms_orbi = get_proj_info(htb)
    # symmhtb = Symmetrize_Htb(ngridR, htb, symmops, atoms_pos, atoms_orbi, soc)
    # symmhtb.run()
    # # htb.r_Ramn = None
    # # htb.save_htb(r'htb.MnPd.afm100.symmpt.e6.h5', decimals=12)
    # htb.save_htb(r'htb.MnPt.afm100.symm.h5')

    ''' VS2VS '''
    soc = True
    ngridR = np.array([6, 16, 6])
    symmops = np.array([
        # TR, det, alpha, nx, ny, nz, taux, tauy, tauz
        [0, 1, 0, 0, 0, 1, 0, 0, 0], # e
        [0, -1, 0, 0, 0, 1, 0, 0, 0], # P
        [0, -1, np.pi, 0, 1, 0, 0, 0, 0], # My
        [0, 1, np.pi, 0, 1, 0, 0, 0, 0], # C2y
        # anti unitary
        [1, 1, 0, 0, 0, 1, 0.5, 0.5, 0], # ~T
        [1, -1, 0, 0, 0, 1, 0.5, 0.5, 0], # ~TP
        [1, -1, np.pi, 0, 1, 0, 0.5, 0.5, 0], # ~TMy
        [1, 1, np.pi, 0, 1, 0, 0.5, 0.5, 0], # ~TC2y
    ])
    atoms_pos = [
        np.array([
            [0.1552357247222010, 0.0000000000000000, 0.0059324618969288],
            [0.0861959795410315, 0.0000000000000000, 0.3088876646940996],
            [0.0705769787938255, 0.5000000000000000, 0.6792078946571625],
            [0.9294230212061745, 0.5000000000000000, 0.3207921053428374],
            [0.9138040204589686, 0.0000000000000000, 0.6911123353059004],
            [0.8447642752777991, 0.0000000000000000, 0.9940675381030712],
            [0.7390035718037002, 0.5000000000000000, 0.6598011335297749],
            [0.7609964274058485, 0.0000000000000000, 0.3401988631552862],
            [0.6552357206563747, 0.5000000000000000, 0.0059324567625196],
            [0.5861959779801355, 0.5000000000000000, 0.3088876630709361],
            [0.5705769793101882, 0.0000000000000000, 0.6792078957778224],
            [0.4294230206898117, 0.0000000000000000, 0.3207921042221775],
            [0.4138040220198645, 0.5000000000000000, 0.6911123369290638],
            [0.3447642793436252, 0.5000000000000000, 0.9940675432374804],
            [0.2390035725941515, 0.0000000000000000, 0.6598011368447139],
            [0.2609964281963000, 0.5000000000000000, 0.3401988664702250],
        ]),
        np.array([
            [0.7044177935899905, 0.0000000000000000, 0.8534962109885530],
            [0.1511460161078381, 0.5000000000000000, 0.4847941460770144],
            [0.0000000000000000, 0.0000000000000000, 0.5000000000000000],
            [0.8488539838921618, 0.5000000000000000, 0.5152058539229856],
            [0.7955822078552005, 0.5000000000000000, 0.1465037814517498],
            [0.2044177921447997, 0.5000000000000000, 0.8534962185482500],
            [0.6511460157838294, 0.0000000000000000, 0.4847941387978736],
            [0.5000000000000000, 0.5000000000000000, 0.5000000000000000],
            [0.3488539842161779, 0.0000000000000000, 0.5152058612021265],
            [0.2955822064100094, 0.0000000000000000, 0.1465037890114469],
        ])
    ]
    atoms_orbi = [
        [1],
        [2]
    ]
    # atoms_pos, atoms_orbi = get_proj_info(htb)
    symmhtb = Symmetrize_Htb(ngridR, htb, symmops, atoms_pos, atoms_orbi, soc)
    symmhtb.run()
    # htb.r_Ramn = None
    # htb.save_htb(r'htb.vs2vs.afmy.symm.h5')
    # htb.save_wannier90_hr_dat(fmt='18.12')
