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

__date__ = "May. 24, 2020"

import numpy as np
from numpy import linalg as LA
from enum import Enum
from wanpy.core.utils import commdot, anticommdot
from wanpy.core.bz import gauss_Delta_func, adapted_gauss_Delta_func
from wanpy.core.bz import get_adaptive_ewide_II, get_adaptive_ewide_III
from wanpy.core.bz import FD_zero, fermi_dirac_func
from wanpy.core.units import *

__all__ = [
    'Res_dim_coeff',
    'Res_unit',
    'get_fft001',
    'get_fft_d2',
    'get_fft_withzeeman',
    'get_hk',
    'get_Dk',
    'cal_band',
    'cal_berry_curvature',
    'cal_berry_curvature_apply_Efield',
    'cal_berry_curvature_apply_Bfield',
    'cal_dos',
    'cal_int_dos',
    'cal_jdos',
    'cal_static_polarization_function',
    'cal_dielectric_function',
    'cal_dielectric_function_manyEf',
    'get_epsilon_rpa',
    'get_vqchi0',
    'cal_shift_current_Rshift_abc',
    'cal_shift_current_Rshift_all',
    'cal_shift_current_Retarded',
    'cal_LPGE_Retarded_d1_211',
    'cal_LPGE_Retarded_d2_220',
    'cal_LPGE_Retarded_d2_210',
    'cal_LPGE_Retarded_d1',
    'cal_LPGE_Retarded_d2',
    'cal_CPGE_2D',
    'cal_CPGE_3D',
    'cal_CPGE',
    'cal_CPGE_Retarded',
    'cal_bc_dipole',
    'cal_SHG_prb1998',
    'cal_shg_nagaosa',
    'cal_linear_L_MOKE',
    'cal_2nd_L_MOKE',
]

'''
  * unit 
'''
class Res_dim_coeff(Enum):
    dos = 1.0
    jdos = 1.0
    dielectric_RPA = EV / Epsilon0 * 1e10
    bc_dipole = 1.0
    shift_current_2D = EV / Hbar * 1e6  # in unit uA Ans V^-2
    shift_current_3D = EV / Hbar * 1e6  # in unit uA V^-2
    cpge_2D = EV / Hbar * 1e6  # in unit uA Ans V^-2
    cpge_3D = EV / Hbar * 1e6  # in unit uA V^-2
    shg = 1.0
    linear_L_MOKE = 1.0
    nl_2nd_L_MOKE = 1.0


class Res_unit(Enum):
    dos = 'a.u.'
    jdos = 'a.u.'
    dielectric_RPA = ''
    bc_dipole = r'\mathring{A}'
    shift_current_2D = r'\times\mathrm{10}^{2}\left(\mu\mathrm{A}\cdot\mathrm{\mathring{A}}\cdot\mathrm{V}^{-2}\right)'
    shift_current_3D = r'\times\mathrm{10}^{2}\left(\mu\mathrm{A}\cdot\mathrm{V}^{-2}\right)'
    cpge_2D = r'\times\mathrm{10}^{2}\left(\mu\mathrm{A}\cdot\mathrm{\mathring{A}}\cdot\mathrm{V}^{-2}\right)'
    cpge_3D = r'\times\mathrm{10}^{2}\left(\mu\mathrm{A}\cdot\mathrm{V}^{-2}\right)'
    shg = 'a.u.'
    linear_L_MOKE = 'a.u.'
    nl_2nd_L_MOKE = 'a.u.'
    

'''
  * basic functions on htb
'''
def get_fft001(htb, k, sym_h=True, sym_r=True, imaxgap='null', maxgap=-1.0, randomgauge=False):
    # maxgap determine if reture zero response
    # maxgap < 0 do not enforce to returen zero response
    eikr_hr = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_hr))
    hk = np.einsum('R,Rmn->mn', eikr_hr, htb.hr_Rmn, optimize=True)

    E, U = LA.eigh(hk)
    E = E - htb.fermi
    if imaxgap == 'I':
        if np.abs(E).min() > maxgap:
            return None, None, None, None, None, None, None
    elif imaxgap == 'II':
        if 0 < maxgap < (E[E > 0].min() - E[E < 0].max()):
            return None, None, None, None, None, None, None
    else:
        pass

    eikr_r = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_r))
    hkk = 1j * np.einsum('Ra,R,Rmn->amn', htb.Rc_hr, eikr_hr, htb.hr_Rmn, optimize=True)
    Awk = np.einsum('R,Ramn->amn', eikr_r, htb.r_Ramn, optimize=True)

    if sym_h:
        hk = 0.5 * (hk + hk.conj().T)
        hkk = 0.5 * (hkk + np.einsum('amn->anm', hkk.conj()))
    if sym_r:
        Awk = 0.5 * (Awk + np.einsum('amn->anm', Awk.conj()))

    vw = hkk + 1j * (np.einsum('lm,amn->aln', hk, Awk, optimize=True) - np.einsum('alm,mn->aln', Awk, hk, optimize=True))
    if randomgauge:
        U = np.einsum('jn,n->jn', U, np.exp(2j * np.pi * np.random.random(htb.nw)))
    v = np.einsum('mi,aij,jn->amn', U.conj().T, vw, U, optimize=True)
    # v = U.conj().T @ vw @ U
    return v, vw, E, U, hk, hkk, Awk

def get_fft_d2(htb, k, index, imaxgap='null', maxgap=-1.0):
    # v, vw, w, ww, E, U = get_fft_d2(htb, k, index)
    nindex = index.shape[0]
    eikr_hr = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_hr))
    hk = np.einsum('R,Rmn->mn', eikr_hr, htb.hr_Rmn, optimize=True)

    E, U = LA.eigh(hk)
    E = E - htb.fermi
    if imaxgap == 'I':
        if np.abs(E).min() > maxgap:
            return None, None, None, None, None, None
    elif imaxgap == 'II':
        if 0 < maxgap < (E[E > 0].min() - E[E < 0].max()):
            return None, None, None, None, None, None
    else:
        pass

    eikr_r = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_r))
    hkk = 1j * np.einsum('Ra,R,Rmn->amn', htb.Rc_hr, eikr_hr, htb.hr_Rmn, optimize=True)
    Awk = np.einsum('R,Ramn->amn', eikr_r, htb.r_Ramn, optimize=True)
    hk = 0.5 * (hk + hk.conj().T)
    hkk = 0.5 * (hkk + np.einsum('amn->anm', hkk.conj()))
    Awk = 0.5 * (Awk + np.einsum('amn->anm', Awk.conj()))

    vw = hkk + 1j * (
            np.einsum('lm,amn->aln', hk, Awk, optimize=True) - np.einsum('alm,mn->aln', Awk, hk, optimize=True))
    v = np.einsum('mi,aij,jn->amn', U.conj().T, vw, U, optimize=True)

    hkkk = np.zeros([nindex, htb.nw, htb.nw], dtype='complex128')
    Awkk = np.zeros([nindex, htb.nw, htb.nw], dtype='complex128')
    ww = np.zeros([nindex, htb.nw, htb.nw], dtype='complex128')
    w = np.zeros([nindex, htb.nw, htb.nw], dtype='complex128')
    for i in range(nindex):
        a, b = index[i]
        hkkk[i] = -np.einsum('R,R,R,Rmn->mn', htb.Rc_hr.T[a], htb.Rc_hr.T[b], eikr_hr, htb.hr_Rmn, optimize=True)
        Awkk[i] = 1j * np.einsum('R,R,Rmn->mn', htb.Rc_r.T[a], eikr_r, htb.r_Ramn[:, b, :, :], optimize=True)
        hkkk[i] = 0.5 * (hkkk[i] + hkkk[i].T.conj())
        Awkk[i] = 0.5 * (Awkk[i] + Awkk[i].T.conj())
        ww[i] = hkkk[i] + 1j * (commdot(hkk[a], Awk[b]) + commdot(hk, Awkk[i]) + commdot(vw[b], Awk[a]))
        # ww[i] = 0.5 * (ww[i] + ww[i].conj().T)
        w[i] = U.conj().T @ ww[i] @ U
    return v, vw, w, ww, E, U

def _get_fft_d3(htb, k, a, b, c):
    # v, vw, vabc, vwabc, E, U = get_fft_d3(htb, k, a, b, c)
    eikr_hr = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_hr))
    eikr_r = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_r))

    hk = np.einsum('R,Rmn->mn', eikr_hr, htb.hr_Rmn, optimize=True)
    hkk = 1j * np.einsum('Ra,R,Rmn->amn', htb.Rc_hr, eikr_hr, htb.hr_Rmn, optimize=True)
    Awk = np.einsum('R,Ramn->amn', eikr_r, htb.r_Ramn, optimize=True)
    E, U = LA.eigh(hk)
    E = E - htb.fermi
    vw = hkk + 1j * (
            np.einsum('lm,amn->aln', hk, Awk, optimize=True) - np.einsum('alm,mn->aln', Awk, hk, optimize=True))
    v = np.einsum('mi,aij,jn->amn', U.conj().T, vw, U, optimize=True)

    hkkk_ac = -np.einsum('R,R,R,Rmn->mn', htb.Rc_hr.T[a], htb.Rc_hr.T[c], eikr_hr, htb.hr_Rmn, optimize=True)
    hkkk_bc = -np.einsum('R,R,R,Rmn->mn', htb.Rc_hr.T[b], htb.Rc_hr.T[c], eikr_hr, htb.hr_Rmn, optimize=True)
    Awkk_ab = 1j * np.einsum('R,R,Rmn->mn', htb.Rc_r.T[a], eikr_r, htb.r_Ramn[:, b, :, :], optimize=True)
    Awkk_ca = 1j * np.einsum('R,R,Rmn->mn', htb.Rc_r.T[c], eikr_r, htb.r_Ramn[:, a, :, :], optimize=True)
    Awkk_bc = 1j * np.einsum('R,R,Rmn->mn', htb.Rc_r.T[b], eikr_r, htb.r_Ramn[:, c, :, :], optimize=True)
    Awkk_cb = 1j * np.einsum('R,R,Rmn->mn', htb.Rc_r.T[c], eikr_r, htb.r_Ramn[:, b, :, :], optimize=True)
    hkkkk_abc = -1j * np.einsum('R,R,R,R,Rmn->mn', htb.Rc_hr.T[a], htb.Rc_hr.T[b], htb.Rc_hr.T[c], eikr_hr,
                                htb.hr_Rmn, optimize=True)
    Awkkk_acb = -np.einsum('R,R,R,Rmn->mn', htb.Rc_r.T[a], htb.Rc_r.T[c], eikr_r, htb.r_Ramn[:, b, :, :],
                           optimize=True)

    ww_bc = hkkk_bc + 1j * (commdot(hkk[b], Awk[c]) + commdot(hk, Awkk_bc) + commdot(vw[c], Awk[b]))

    vwabc = hkkkk_abc + 1j * (
            commdot(ww_bc, Awk[a]) + commdot(vw[b], Awkk_ca)
            + commdot(hkkk_ac, Awk[b]) + commdot(hk, Awkkk_acb)
            + commdot(hkk[c], Awkk_ab) + commdot(hkk[a], Awkk_cb)
    )
    vwabc = 0.5 * (vwabc + vwabc.conj().T)
    vabc = LA.multi_dot([U.conj().T, vwabc, U])

    return v, vw, vabc, vwabc, E, U

def get_fft_withzeeman(htb, k, B, sym_h=True, sym_r=True):

    # N.B. this function can only be used for soc htb
    #      only spin zeeman is included

    op_spin = np.array([
        np.kron(sigmax, np.eye(htb.nw//2)),
        np.kron(sigmay, np.eye(htb.nw//2)),
        np.kron(sigmaz, np.eye(htb.nw//2)),
    ])

    eikr_hr = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_hr))
    eikr_r = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_r))
    hk_0 = np.einsum('R,Rmn->mn', eikr_hr, htb.hr_Rmn, optimize=True)
    hkk = 1j * np.einsum('Ra,R,Rmn->amn', htb.Rc_hr, eikr_hr, htb.hr_Rmn, optimize=True)
    Awk = np.einsum('R,Ramn->amn', eikr_r, htb.r_Ramn, optimize=True)

    hz_S = NiuB * np.einsum('amn,a', op_spin, B)        # -niu_s * B, niu_s = -2 niuB S/hbar
    hk = hk_0 + hz_S

    E, U = LA.eigh(hk)
    E = E - htb.fermi

    if sym_h:
        hk = 0.5 * (hk + hk.conj().T)
        hkk = 0.5 * (hkk + np.einsum('amn->anm', hkk.conj()))
    if sym_r:
        Awk = 0.5 * (Awk + np.einsum('amn->anm', Awk.conj()))

    vw = hkk + 1j * (np.einsum('lm,amn->aln', hk, Awk, optimize=True) - np.einsum('alm,mn->aln', Awk, hk, optimize=True))
    v = np.einsum('mi,aij,jn->amn', U.conj().T, vw, U, optimize=True)
    return v, vw, E, U, hk, hkk, Awk

'''
  * band structure calculators 
'''
def get_hk(htb, k, tbgauge=False, use_ws_distance=False):
    eikr = np.exp(2j * np.pi * np.einsum('a,Ra', k, htb.R_hr)) / htb.ndegen
    hr_Rmn = htb.hr_Rmn

    # In test case, use_ws_distance is 10 times slower !
    if use_ws_distance:
        eikT = np.sum(htb.invndegenT * np.exp(2j * np.pi * np.einsum('a,NRmna->NRmn', k, htb.wsvecT)), axis=0)
        hr_Rmn = htb.hr_Rmn * eikT

    if not tbgauge:
        hk = np.einsum('R,Rmn->mn', eikr, hr_Rmn)
    else:
        eiktau = np.exp(2j * np.pi * np.einsum('a,na', k, htb.wccf))
        hk = np.einsum('R,m,Rmn,n->mn', eikr, eiktau.conj(), hr_Rmn, eiktau, optimize=True)
    return hk

def get_Dk(htb, k, i, tbgauge=False):
    eikr = np.exp(2j * np.pi * np.einsum('a,Ra', k, htb.R_hr)) / htb.ndegen
    if not tbgauge:
        Dk = np.einsum('R,Rmn->mn', eikr, htb.D_iRmn[i], optimize=True)
    else:
        eiktau = np.exp(2j * np.pi * np.einsum('a,na', k, htb.wccf))
        Dk = np.einsum('R,m,Rmn,n->mn', eikr, eiktau.conj(), htb.D_iRmn[i], eiktau, optimize=True)
    return Dk

def cal_band(htb, k, returnU=False, tbgauge=False, use_ws_distance=False):
    # eikr = np.exp(2j * np.pi * np.einsum('a,Ra', k, htb.R_hr)) / htb.ndegen
    # hk = np.einsum('R,Rmn->mn', eikr, htb.hr_Rmn)
    hk = get_hk(htb, k, tbgauge, use_ws_distance)
    E, U = LA.eigh(hk)
    E = E - htb.fermi
    if returnU:
        return E, U
    else:
        return E

# def cal_band_HSP(htb, HSP_list, nk1=101, HSP_name=None):
#     kpath = make_kpath(HSP_list, nk1)
#     kpath_car = LA.multi_dot([htb.cell.latticeG, kpath.T]).T
#
#     nk = kpath.shape[0]
#     bandE = np.zeros([nk, htb.nw], dtype='float64')
#
#     for i, k in zip(range(nk), kpath):
#         print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
#             i + 1, nk,
#             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
#             k[0], k[1], k[2]
#         )
#         )
#         bandE[i] = cal_band(htb, k)
#
#     bandstructure_hsp = BandstructureHSP()
#     bandstructure_hsp.eig = bandE
#     bandstructure_hsp.HSP_list = HSP_list
#     bandstructure_hsp.HSP_path_frac = kpath
#     bandstructure_hsp.HSP_path_car = kpath_car
#     bandstructure_hsp.HSP_name = HSP_name
#     bandstructure_hsp.nk, bandstructure_hsp.nb = nk, htb.nw
#
#     return bandstructure_hsp

# def cal_band_apply_Efield(htb, k):
#     pass

def cal_berry_curvature(htb, k, ewide=0.02):
    # ewide should be seted as 0.1 times of the minimal energy difference, or using adapted smearing
    v, vw, E, U, hk, hkk, Awk = get_fft001(htb, k, sym_h=True, sym_r=True)

    e1, e2 = np.meshgrid(E, E)
    invE = np.real(1 / (e2 - e1 - 1j * ewide))
    invE = invE - np.diag(np.diag(invE))
    invE2 = invE ** 2

    # bc = np.zeros([3, htb.nw], dtype='float64')
    # bc[0] = -2. * np.imag(np.einsum('nm,mn,mn->n', v[1], v[2], invE2, optimize=True))
    # bc[1] = -2. * np.imag(np.einsum('nm,mn,mn->n', v[2], v[0], invE2, optimize=True))
    # bc[2] = -2. * np.imag(np.einsum('nm,mn,mn->n', v[0], v[1], invE2, optimize=True))

    bc = np.zeros([3, htb.nw], dtype='float64')
    bc[0] = np.einsum('mn,mn->n', -2. * np.imag(v[1].T * v[2]), invE2, optimize=True)
    bc[1] = np.einsum('mn,mn->n', -2. * np.imag(v[2].T * v[0]), invE2, optimize=True)
    bc[2] = np.einsum('mn,mn->n', -2. * np.imag(v[0].T * v[1]), invE2, optimize=True)

    return E, bc

def cal_berry_curvature_apply_Efield(htb, k, ewide=0.02):
    pass

def cal_berry_curvature_apply_Bfield(htb, k, B, ewide=0.02):
    v, vw, E, U, hk, hkk, Awk = get_fft_withzeeman(htb, k, B, sym_h=True, sym_r=True)

    e1, e2 = np.meshgrid(E, E)
    invE = np.real(1 / (e2 - e1 - 1j * ewide))
    invE = invE - np.diag(np.diag(invE))
    invE2 = invE ** 2

    bc = np.zeros([3, htb.nw], dtype='float64')
    bc[0] = np.einsum('mn,mn->n', -2. * np.imag(v[1].T * v[2]), invE2, optimize=True)
    bc[1] = np.einsum('mn,mn->n', -2. * np.imag(v[2].T * v[0]), invE2, optimize=True)
    bc[2] = np.einsum('mn,mn->n', -2. * np.imag(v[0].T * v[1]), invE2, optimize=True)

    return E, bc

# def cal_berry_curvature_HSP(htb, HSP_list, nk1=101, HSP_name=None, ewide=0.01):
#     kpath = make_kpath(HSP_list, nk1)
#     kpath_car = LA.multi_dot([htb.cell.latticeG, kpath.T]).T
#
#     nk = kpath.shape[0]
#     bandE = np.zeros([nk, htb.nw], dtype='float64')
#     BC = np.zeros([3, nk, htb.nw], dtype='float64')
#
#     for i, k in zip(range(nk), kpath):
#         print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
#             i + 1, nk,
#             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
#             k[0], k[1], k[2]
#         )
#         )
#         bandE[i], BC[:, i, :] = cal_berry_curvature(htb, k, ewide)
#
#     bandstructure_hsp = BandstructureHSP()
#     bandstructure_hsp.eig = bandE
#     bandstructure_hsp.BC = BC
#     bandstructure_hsp.HSP_list = HSP_list
#     bandstructure_hsp.HSP_path_frac = kpath
#     bandstructure_hsp.HSP_path_car = kpath_car
#     bandstructure_hsp.HSP_name = HSP_name
#     bandstructure_hsp.nk, bandstructure_hsp.nb = nk, htb.nw
#
#     return bandstructure_hsp


'''
  * Calculators for resonse functions 
'''
def cal_dos(htb, k, ee, ewide=0.005, ewide_min=0.005):
    v, vw, E, U, hk, hkk, Awk = get_fft001(htb, k)

    W = get_adaptive_ewide_II(v, htb.dk, ewide_min=ewide_min)
    D = np.meshgrid(E, ee)
    D = D[1] - D[0]
    delta_an = np.exp(-0.5 * (D / ewide) ** 2) / np.sqrt(2 * np.pi) / ewide
    delta_an_AD = np.exp(-0.5 * (D / W) ** 2) / np.sqrt(2 * np.pi) / W

    dos = np.einsum('an->a', delta_an, optimize=True)
    dos_ad = np.einsum('an->a', delta_an_AD, optimize=True)
    return dos, dos_ad

def cal_int_dos(htb, k, ee, ewidth=0.005):
    v, vw, E, U, hk, hkk, Awk = get_fft001(htb, k)

    npmesh = np.meshgrid(ee, E)
    f = fermi_dirac_func(npmesh[1] - npmesh[0], smear=ewidth, ismear=-1)
    int_dos = np.einsum('ne->e', f)
    return int_dos

def cal_jdos(htb, k, ee, ewide=0.005, ewide_min=0.005, isADwide=True):
    v, vw, E, U, hk, hkk, Awk = get_fft001(htb, k)

    e1, e2 = np.meshgrid(E, E)

    f_ln = FD_zero(e2) - FD_zero(e1)
    if isADwide:
        delta_nla_AD = adapted_gauss_Delta_func(E, ee, v, htb.dk, ewide_min=ewide_min)
        jdos = np.einsum('ln,nla', f_ln, delta_nla_AD, optimize=True)
    else:
        delta_nla = gauss_Delta_func(E, ee, ewide)
        jdos = np.einsum('ln,nla', f_ln, delta_nla, optimize=True)

    return jdos

'''
  * dielectric functions 
    Updated: Feb. 10, 2022
'''
def cal_static_polarization_function(htb, k, qq, gate, ewide, selec_band=...):
    nq = qq.shape[0]
    nw = htb.nw
    nws = nw if selec_band is Ellipsis else selec_band.shape[0]

    hk = get_hk(htb, k, tbgauge=False)
    E0, U0 = LA.eigh(hk)
    E0 = E0[selec_band] - htb.fermi
    f0 = FD_zero(E0 - gate)
    U0 = U0[:, selec_band]

    U = np.zeros([nq, nw, nws], dtype='complex128')
    E = np.zeros([nq, nws], dtype='float64')
    for i in range(nq):
        hk = get_hk(htb, k + qq[i], tbgauge=False)
        _E, _U = LA.eigh(hk)
        E[i] = _E[selec_band]
        U[i] = _U[:, selec_band]
        # E[i], U[i] = LA.eigh(hk)
    E = E - htb.fermi
    f = FD_zero(E - gate)
    fq = np.einsum('qnm->qmn', np.full((nq, nws, nws), f0)) - \
         np.einsum('mqn->qmn', np.full((nws, nq, nws), f))
    invEq = 1 / (
            np.einsum('qnm->qmn', np.full((nq, nws, nws), E0)) -
            np.einsum('mqn->qmn', np.full((nws, nq, nws), E)) +
            1j * ewide
    )
    invEq = np.real(invEq)
    eiqr = np.exp(-2j * np.pi * np.einsum('qa,na->qn', qq, htb.wccf))
    Mmn = np.einsum('mj,qj,qjn->qmn', U0.T.conj(), eiqr, U, optimize=True)
    # Mmn = np.einsum('mj,qjn->qmn', U0.T.conj(), U, optimize=True)
    chi0 = np.einsum('qmn,qmn,qmn->q', fq, invEq, np.abs(Mmn) ** 2, optimize=True)
    return chi0

def cal_dielectric_function(htb, ee, k, qq, gate, ewide, man=False, selec_band=...):
    """
      dev note.
      * I have tested that using tbgauge without eiqr, and using wannier
        gauge with eiqr, both give the same results, but the later one is about
        1.5 times faster.
    """
    ne = ee.shape[0]
    nq = qq.shape[0]
    nw = htb.nw
    nws = nw if selec_band is Ellipsis else selec_band.shape[0]

    hk = get_hk(htb, k, tbgauge=False)
    E0, U0 = LA.eigh(hk)
    E0 = E0[selec_band] - htb.fermi
    f0 = FD_zero(E0 - gate)
    U0 = U0[:, selec_band]

    U = np.zeros([nq, nw, nws], dtype='complex128')
    E = np.zeros([nq, nws], dtype='float64')
    chi0 = np.zeros([nq, ne], dtype='complex128')
    for i in range(nq):
        hk = get_hk(htb, k + qq[i], tbgauge=False)
        _E, _U = LA.eigh(hk)
        E[i] = _E[selec_band]
        U[i] = _U[:, selec_band]
        # E[i], U[i] = LA.eigh(hk)
    E = E - htb.fermi
    f = FD_zero(E - gate)
    fq = np.einsum('qnm->qmn', np.full((nq, nws, nws), f0)) - \
         np.einsum('mqn->qmn', np.full((nws, nq, nws), f))
    invEqe = 1 / (
            np.einsum('qenm->qmne', np.full((nq, ne, nws, nws), E0)) -
            np.einsum('emqn->qmne', np.full((ne, nws, nq, nws), E)) +
            np.einsum('qmne->qmne', np.full((nq, nws, nws, ne), ee)) +
            1j * ewide
    )
    eiqr = np.exp(-2j * np.pi * np.einsum('qa,na->qn', qq, htb.wccf))
    Mmn = np.einsum('mj,qj,qjn->qmn', U0.T.conj(), eiqr, U, optimize=True)
    # Mmn = np.einsum('mj,qjn->qmn', U0.T.conj(), U, optimize=True)
    if man:
        chi0_man = np.einsum('qmn,qmne,qmn->qemn', fq, invEqe, np.abs(Mmn) ** 2, optimize=True)
        chi0_intra = np.einsum('qemn,mn->qe', chi0_man, np.eye(nws), optimize=True)
        chi0_inter = np.einsum('qemn,mn->qe', chi0_man, 1-np.eye(nws), optimize=True)
        chi0 = np.array([chi0_intra+chi0_inter, chi0_intra, chi0_inter])
    else:
        chi0 = np.einsum('qmn,qmne,qmn->qe', fq, invEqe, np.abs(Mmn) ** 2, optimize=True)
    return chi0

def cal_dielectric_function_manyEf(htb, ee, k, qq, gate, ewide, man=False, selec_band=...):
    """
      dev note.
      * see note of cal_dielectric_function
    """
    ne = ee.shape[0]
    nq = qq.shape[0]
    nw = htb.nw
    nws = nw if selec_band is Ellipsis else selec_band.shape[0]
    ngate = gate.shape[0]

    hk = get_hk(htb, k, tbgauge=False)
    E0, U0 = LA.eigh(hk)
    E0 = E0[selec_band] - htb.fermi
    U0 = U0[:, selec_band]
    f0 = np.zeros([ngate, nws], dtype='float64')  # fm(k) -> gm
    for i in range(ngate):
        f0[i] = FD_zero(E0 - gate[i])

    U = np.zeros([nq, nw, nws], dtype='complex128')
    E = np.zeros([nq, nws], dtype='float64')
    chi0 = np.zeros([nq, ne], dtype='complex128')
    for i in range(nq):
        hk = get_hk(htb, k + qq[i], tbgauge=False)
        _E, _U = LA.eigh(hk)
        E[i] = _E[selec_band]
        U[i] = _U[:, selec_band]
        # E[i], U[i] = LA.eigh(hk)
    E = E - htb.fermi

    f = np.zeros([ngate, nq, nws], dtype='float64')  # fn(k+q) -> gqn
    for i in range(ngate):
        f[i] = FD_zero(E - gate[i])

    fq = np.einsum('qngm->gqmn', np.full((nq, nws, ngate, nws), f0)) - \
         np.einsum('mgqn->gqmn', np.full((nws, ngate, nq, nws), f))
    fq = np.ascontiguousarray(fq)
    invEqe = 1 / (
            np.einsum('qenm->qmne', np.full((nq, ne, nws, nws), E0)) -
            np.einsum('emqn->qmne', np.full((ne, nws, nq, nws), E)) +
            np.einsum('qmne->qmne', np.full((nq, nws, nws, ne), ee)) +
            1j * ewide
    )
    eiqr = np.exp(-2j * np.pi * np.einsum('qa,na->qn', qq, htb.wccf))
    Mmn = np.einsum('mj,qj,qjn->qmn', U0.T.conj(), eiqr, U, optimize=True)
    # Mmn = np.einsum('mj,qjn->qmn', U0.T.conj(), U, optimize=True)
    if man:
        chi0_man = np.einsum('gqmn,qmne,qmn->gqemn', fq, invEqe, np.abs(Mmn) ** 2, optimize=True)
        chi0_intra = np.einsum('qemn,mn->qe', chi0_man, np.eye(nws), optimize=True)
        chi0_inter = np.einsum('qemn,mn->qe', chi0_man, 1-np.eye(nws), optimize=True)
        chi0 = np.array([chi0_intra+chi0_inter, chi0_intra, chi0_inter])
    else:
        chi0 = np.einsum('gqmn,qmne,qmn->gqe', fq, invEqe, np.abs(Mmn) ** 2, optimize=True)
    return chi0

def get_epsilon_rpa(qc, chi0, background_epsilon=1, dim=2):
    qcnorm = LA.norm(qc, axis=1)
    invqc = np.real(1 / (qcnorm + 1j * 1e-50))

    if dim == 2:
        # dim_coeff = EV / (4*np.pi*Epsilon0) * 1e10  # in IS
        dim_coeff = Bohr * Hartree  # in a.u., the two should be the same due to the dimonsionless
        vq = 2 * np.pi * invqc / background_epsilon
    elif dim == 3:
        dim_coeff = Bohr * Hartree
        vq = 4 * np.pi * invqc ** 2 / background_epsilon
    else:
        dim_coeff = None
        vq = None

    epsilon_rpa = 1 - dim_coeff * np.einsum('q,...qe->...qe', vq, chi0)
    loss = -1 * np.imag(1 / epsilon_rpa)
    return qcnorm, epsilon_rpa, loss

def get_vqchi0(qc, chi0, background_epsilon=1, dim=2):
    qcnorm = LA.norm(qc, axis=1)
    invqc = np.real(1 / (qcnorm + 1j * 1e-50))

    if dim == 2:
        # dim_coeff = EV / (4*np.pi*Epsilon0) * 1e10  # in IS
        dim_coeff = Bohr * Hartree  # in a.u., the two should be the same due to the dimonsionless
        vq = 2 * np.pi * invqc / background_epsilon
    elif dim == 3:
        dim_coeff = Bohr * Hartree
        vq = 4 * np.pi * invqc ** 2 / background_epsilon
    else:
        dim_coeff = None
        vq = None

    vqchi0 = dim_coeff * np.einsum('q,q...->q...', vq, chi0)
    return qcnorm, vqchi0


'''
  * Shift current
'''
# def cal_shift_current_2D(htb, k, ee, gate, rank3Index, ewide=0.005, ewide_min=0.005, isADwide=True):
#     return cal_shift_current_Rshift(htb, k, ee, gate, rank3Index, ewide, ewide_min, isADwide)
#
#
# def cal_shift_current_3D(htb, k, ee, gate, rank3Index, ewide=0.005, ewide_min=0.005, isADwide=True):
#     return cal_shift_current_Rshift(htb, k, ee, gate, rank3Index, ewide, ewide_min, isADwide)

def cal_shift_current_Rshift_abc(htb, k, ee, rank3Index, ewide, ewide_min, isADwide=True, include_w=True):
    def _get_w_ab(a, b, htb, hk, hkk, vw, U, Awk, eikr_hr, eikr_r):
        hkkk_ab = -np.einsum('R,R,R,Rmn->mn', htb.Rc_hr.T[a], htb.Rc_hr.T[b], eikr_hr, htb.hr_Rmn, optimize=True)
        Awkk_ab = 1j * np.einsum('R,R,Rmn->mn', htb.Rc_r.T[a], eikr_r, htb.r_Ramn[:, b, :, :], optimize=True)
        ww_ab = hkkk_ab + 1j * (commdot(hkk[a], Awk[b]) + commdot(hk, Awkk_ab) + commdot(vw[b], Awk[a]))
        w_ab = U.conj().T @ ww_ab @ U
        w_ab = w_ab - np.diag(w_ab)
        return w_ab

    def _get_rba(a, b, invE, v, vv, w_ab):
        rba = 1j * invE * (
                invE * v[b] * vv[a] +
                np.einsum('nl,lm,lm->nm', v[b], v[a], invE, optimize=True) -
                np.einsum('lm,nl,nl->nm', v[b], v[a], invE, optimize=True) -
                w_ab
        )
        return rba

    a, b, c = rank3Index  # (1, 1, 1)

    eikr_hr = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_hr))
    eikr_r = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_r))
    hk = np.einsum('R,Rmn->mn', eikr_hr, htb.hr_Rmn, optimize=True)
    hkk = 1j * np.einsum('Ra,R,Rmn->amn', htb.Rc_hr, eikr_hr, htb.hr_Rmn, optimize=True)
    Awk = np.einsum('R,Ramn->amn', eikr_r, htb.r_Ramn, optimize=True)

    E, U = LA.eigh(hk)
    E = E - htb.fermi

    e1, e2 = np.meshgrid(E, E)
    invE = np.real(1 / (e2 - e1 - 1j * ewide))
    invE = invE - np.diag(np.diag(invE))

    vw = hkk + 1j * (
                np.einsum('lm,amn->aln', hk, Awk, optimize=True) - np.einsum('alm,mn->aln', Awk, hk, optimize=True))
    v = np.einsum('mi,aij,jn->amn', U.conj().T, vw, U, optimize=True)

    if include_w:
        w_ab = _get_w_ab(a, b, htb, hk, hkk, vw, U, Awk, eikr_hr, eikr_r)
        if b != c:
            w_ac = _get_w_ab(a, c, htb, hk, hkk, vw, U, Awk, eikr_hr, eikr_r)
        else:
            w_ac = w_ab
    else:
        w_ab = 0
        w_ac = 0

    vv = np.zeros_like(v)
    for i in range(3):
        _vv = np.diag(v[i])
        _vv = np.meshgrid(_vv, _vv)
        vv[i] = _vv[1] - _vv[0]

    r = -1j * np.einsum('amn,mn->amn', v, invE)
    rba = _get_rba(a, b, invE, v, vv, w_ab)
    if b != c:
        rca = _get_rba(a, b, invE, v, vv, w_ac)
    else:
        rca = rba

    if isADwide:
        delta = adapted_gauss_Delta_func(E, ee, v, htb.dk, ewide_min=ewide_min) + \
                adapted_gauss_Delta_func(E, -ee, v, htb.dk, ewide_min=ewide_min)
    else:
        delta = gauss_Delta_func(E, ee, ewide) + \
                gauss_Delta_func(E, -ee, ewide)

    f = FD_zero(e2) - FD_zero(e1)

    # sc = -np.pi * np.imag(np.einsum('nm,mn,nm,mne->e', f, r[b], rba, delta, optimize=True))

    sc = np.einsum('nm,mn,nm,mne->e', f, r[b], rca, delta, optimize=True) + \
         np.einsum('nm,mn,nm,mne->e', f, r[c], rba, delta, optimize=True)
    sc = np.pi * np.real(1j * 0.5 * sc)

    return sc  # -> [ne]

def cal_shift_current_Rshift_all(htb, k, ee, ewide, ewide_min, isADwide=True, include_w=True):
    index2 = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [0, 1],
        [0, 2],
        [1, 2],
    ])

    '''
      * F.T.
    '''
    eikr_hr = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_hr))
    eikr_r = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_r))
    hk = np.einsum('R,Rmn->mn', eikr_hr, htb.hr_Rmn, optimize=True)
    hkk = 1j * np.einsum('Ra,R,Rmn->amn', htb.Rc_hr, eikr_hr, htb.hr_Rmn, optimize=True)
    Awk = np.einsum('R,Ramn->amn', eikr_r, htb.r_Ramn, optimize=True)

    E, U = LA.eigh(hk)
    E = E - htb.fermi

    e1, e2 = np.meshgrid(E, E)
    invE = np.real(1 / (e2 - e1 - 1j * ewide))
    invE = invE - np.diag(np.diag(invE))

    vw = hkk + 1j * (np.einsum('lm,amn->aln', hk, Awk, optimize=True) - np.einsum('alm,mn->aln', Awk, hk, optimize=True))
    v = np.einsum('mi,aij,jn->amn', U.conj().T, vw, U, optimize=True)

    if include_w:
        hkkk = np.zeros([6, htb.nw, htb.nw], dtype='complex128')
        Awkk = np.zeros([6, htb.nw, htb.nw], dtype='complex128')
        ww = np.zeros([6, htb.nw, htb.nw], dtype='complex128')

        i = 0
        for a, b in index2:
            hkkk[i] = -np.einsum('R,R,R,Rmn->mn', htb.Rc_hr.T[a], htb.Rc_hr.T[b], eikr_hr, htb.hr_Rmn, optimize=True)
            Awkk[i] = 1j * np.einsum('R,R,Rmn->mn', htb.Rc_r.T[a], eikr_r, htb.r_Ramn[:, b, :, :], optimize=True)
            ww[i] = hkkk[i] + 1j * (commdot(hkk[a], Awk[b]) + commdot(hk, Awkk[i]) + commdot(vw[b], Awk[a]))
            i += 1

        w = np.zeros([3, 3, htb.nw, htb.nw], dtype='complex128')

        i = 0
        for a, b in index2:
            wi = U.conj().T @ ww[i] @ U
            w[a, b] = wi
            if a != b:
                w[b, a] = wi
            i += 1
    else:
        w = np.zeros([3, 3, htb.nw, htb.nw], dtype='complex128')

    delta_v = np.zeros_like(v)
    for i in range(3):
        _vv = np.diag(v[i])
        _vv = np.meshgrid(_vv, _vv)
        delta_v[i] = _vv[1] - _vv[0]

    '''
      * r, rba
    '''
    r = -1j * np.einsum('amn,mn->amn', v, invE)

    rba = np.zeros([3, 3, htb.nw, htb.nw], dtype='complex128')
    for a in range(3):
        for b in range(3):
            rba[a, b] = 1j * invE * (
                    invE * v[b] * delta_v[a] +
                    np.einsum('nl,lm,lm->nm', v[b], v[a], invE, optimize=True) -
                    np.einsum('lm,nl,nl->nm', v[b], v[a], invE, optimize=True) -
                    w[a, b]
            )

    if isADwide:
        delta = adapted_gauss_Delta_func(E, ee, v, htb.dk, ewide_min=ewide_min) + \
                adapted_gauss_Delta_func(E, -ee, v, htb.dk, ewide_min=ewide_min)
    else:
        delta = gauss_Delta_func(E, ee, ewide) + \
                gauss_Delta_func(E, -ee, ewide)

    f = FD_zero(e2) - FD_zero(e1)

    sc = np.einsum('nm,cmn,abnm,mne->abce', f, r, rba, delta, optimize=True)
    sc = np.pi * np.real(1j * 0.5 * (sc + np.einsum('abce->acbe', sc)))

    # _sc = np.max(np.abs(sc))
    # # if _sc > 0.1:
    # print(_sc)

    return sc  # -> [ngate, ne]

def cal_shift_current_Retarded(htb, k, ee, gate, rank3Index, ewide=0.005, ewide_min=0.005, isADwide=True):
    a, b, c = rank3Index
    ne = ee.shape[0]
    ngate = gate.shape[0]
    v, vw, E, U, hk, hkk, Awk = get_fft001(htb, k)
    W = get_adaptive_ewide_III(v, htb.dk, np.sqrt(2), ewide_min=ewide_min)

    e1, e2 = np.meshgrid(E, E)
    if isADwide:
        invE_AD = 1 / (e2 - e1 - 1j * W)
        invE_AD = invE_AD - np.diag(np.diag(invE_AD))
    else:
        invE = 1 / (e2 - e1 - 1j * ewide)
        invE = invE - np.diag(np.diag(invE))

    npmesh = np.meshgrid(E, E, ee)
    if isADwide:
        invEe_AD = 1 / ((npmesh[1] - npmesh[0] - npmesh[2]).T - 1j * W.T).T + \
                   1 / ((npmesh[1] - npmesh[0] + npmesh[2]).T - 1j * W.T).T
    else:
        invEe = 1 / (npmesh[1] - npmesh[0] - npmesh[2] - 1j * ewide) + \
                1 / (npmesh[1] - npmesh[0] + npmesh[2] - 1j * ewide)

    f = np.zeros([ngate, htb.nw, htb.nw], dtype='float64')
    for i in range(ngate):
        f[i] = FD_zero(e2 - gate[i]) - FD_zero(e1 - gate[i])

    sc = np.zeros([ngate, ne], dtype='float64')
    if isADwide:
        sc = -1 * np.einsum('gln,ml,nle,lm,mn,nl->ge', f, invE_AD, invEe_AD, v[a], v[b], v[c], optimize=True).real
    else:
        sc = -1 * np.einsum('gln,ml,nle,lm,mn,nl->ge', f, invE, invEe, v[a], v[b], v[c], optimize=True).real

    # OR1 = np.real(OR1 / (ee ** 2 + 1j * 0.01))
    # OR1_AD = np.real(OR1_AD / (ee ** 2 + 1j * 0.01))

    # OR1 = -1 * np.einsum('ln,nl,ml,nla,lm,mn,nl', f, (invE.real) ** 2, invE, invEe, v[a], v[b], v[c], optimize=True).real
    # OR1_AD = -1 * np.einsum('ln,nl,ml,nla,lm,mn,nl', f, (invE.real) ** 2, invE_AD, invEe_AD, v[a], v[b], v[c], optimize=True).real

    return sc  # -> [ngate, ne]

'''
  * LPGE
'''
def cal_LPGE_Retarded_d1_211(htb, k, ee, gate, rank3Index, ewide=0.005, ewide_min=0.005, isADwide=True):
    return cal_LPGE_Retarded_d1(htb, k, ee, gate, rank3Index, ewide, ewide_min, isADwide)


def cal_LPGE_Retarded_d2_220(htb, k, ee, gate, rank3Index, ewide=0.005, ewide_min=0.005, isADwide=True):
    return cal_LPGE_Retarded_d2(htb, k, ee, gate, rank3Index, ewide, ewide_min, isADwide, whichterm='220')


def cal_LPGE_Retarded_d2_210(htb, k, ee, gate, rank3Index, ewide=0.005, ewide_min=0.005, isADwide=True):
    return cal_LPGE_Retarded_d2(htb, k, ee, gate, rank3Index, ewide, ewide_min, isADwide, whichterm='210')


def cal_LPGE_Retarded_d1(htb, k, ee, gate, rank3Index, ewide=0.005, ewide_min=0.005, isADwide=True):
    # 211 term
    a, b, c = rank3Index
    ne = ee.shape[0]
    ngate = gate.shape[0]
    v, vw, E, U, hk, hkk, Awk = get_fft001(htb, k)
    W = get_adaptive_ewide_III(v, htb.dk, np.sqrt(2), ewide_min=ewide_min)

    e1, e2 = np.meshgrid(E, E)
    if isADwide:
        invE_AD = 1 / (e2 - e1 + 1j * W)
        invE_AD = invE_AD - np.diag(np.diag(invE_AD))
        # invE_AD = np.real(invE_AD)
    else:
        invE = 1 / (e2 - e1 + 1j * ewide)
        invE = invE - np.diag(np.diag(invE))

    npmesh = np.meshgrid(E, E, ee)
    if isADwide:
        invEe_AD = 1 / ((npmesh[1] - npmesh[0] + npmesh[2]).T + 1j * W.T).T + \
                   1 / ((npmesh[1] - npmesh[0] - npmesh[2]).T + 1j * W.T).T
        # invEe_AD = 1j * np.imag(invEe_AD)
    else:
        invEe = 1 / (npmesh[1] - npmesh[0] + npmesh[2] + 1j * ewide) + \
                1 / (npmesh[1] - npmesh[0] - npmesh[2] + 1j * ewide)

    f = np.zeros([ngate, htb.nw, htb.nw], dtype='float64')
    for i in range(ngate):
        f[i] = FD_zero(e2 - gate[i]) - FD_zero(e1 - gate[i])

    # invE = np.real(invE)
    # invEe = np.imag(invEe)
    if isADwide:
        sc = -1 * np.einsum('gnl,lm,lne,lm,mn,nl->ge', f, invE_AD, invEe_AD, v[a], v[b], v[c], optimize=True).real + \
             -1 * np.einsum('gnl,lm,lne,lm,mn,nl->ge', f, invE_AD, invEe_AD, v[a], v[c], v[b], optimize=True).real
    else:
        sc = -1 * np.einsum('gnl,lm,lne,lm,mn,nl->ge', f, invE, invEe, v[a], v[b], v[c], optimize=True).real + \
             -1 * np.einsum('gnl,lm,lne,lm,mn,nl->ge', f, invE, invEe, v[a], v[c], v[b], optimize=True).real
    sc *= 0.5
    # sc = np.real(sc / (ee ** 2 + 1j * 0.005))
    # print(np.max(sc))
    return sc  # -> [ngate, ne]


def cal_LPGE_Retarded_d2(htb, k, ee, gate, rank3Index, ewide=0.005, ewide_min=0.005, isADwide=True, whichterm='220'):
    # 220 and 210 term
    a, b, c = rank3Index
    ngate = gate.shape[0]
    ne = ee.shape[0]
    index = np.array([[a, b], [b, c]], dtype='int')
    v, vw, w, ww, E, U = get_fft_d2(htb, k, index)
    W = get_adaptive_ewide_III(v, htb.dk, np.sqrt(2), ewide_min=ewide_min)

    e1, e2 = np.meshgrid(E, E)
    npmesh = np.meshgrid(E, E, ee)
    invEe = 1 / ((npmesh[1] - npmesh[0] + npmesh[2]).T + 1j * W.T).T

    f = np.zeros([ngate, htb.nw, htb.nw], dtype='float64')
    for i in range(ngate):
        f[i] = FD_zero(e2 - gate[i]) - FD_zero(e1 - gate[i])

    if whichterm == '220':
        lpge = np.einsum('gmn,mne,mn,nm->ge', f, invEe, w[0], v[c], optimize=True).real
    elif whichterm == '210':
        lpge = 0.5 * np.einsum('gmn,mne,mn,nm->ge', f, invEe, v[a], w[1], optimize=True).real
        # print(w[1].max())
    else:
        lpge = None
    return lpge  # -> [ngate, ne]

# def cal_LPGE_Retarded_d3(htb, k, ee, gate, rank3Index, ewide=0.005, ewide_min=0.005, isADwide=True):
#     a, b, c = rank3Index
#     ngate = gate.shape[0]
#     ne = ee.shape[0]
#     v, vw, vabc, vwabc, E, U = get_fft_d3(htb, k, a, b, c)
#     f = 0
#     lpge = np.einsum('gn,nn->g', f, vabc, optimize=True)
#     return lpge  # -> [ngate]


'''
  * CPGE or injection current
'''
def cal_CPGE_2D(htb, k, ee, gate, rank3Index, ewide=0.005, ewide_min=0.005, isADwide=True):
    return cal_CPGE(htb, k, ee, gate, rank3Index, ewide, ewide_min, isADwide)


def cal_CPGE_3D(htb, k, ee, gate, rank3Index, ewide=0.005, ewide_min=0.005, isADwide=True):
    return cal_CPGE(htb, k, ee, gate, rank3Index, ewide, ewide_min, isADwide)


def cal_CPGE(htb, k, ee, gate, rank3Index, ewide=0.005, ewide_min=0.005, isADwide=True):
    # Injection current, Sipe
    a, b, c = rank3Index
    ngate = gate.shape[0]
    ne = ee.shape[0]

    eikr_hr = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_hr))
    eikr_r = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_r))
    hk = np.einsum('R,Rmn->mn', eikr_hr, htb.hr_Rmn, optimize=True)
    hkk = 1j * np.einsum('Ra,R,Rmn->amn', htb.Rc_hr, eikr_hr, htb.hr_Rmn, optimize=True)
    Awk = np.einsum('R,Ramn->amn', eikr_r, htb.r_Ramn, optimize=True)

    E, U = LA.eigh(hk)
    E = E - htb.fermi

    e1, e2 = np.meshgrid(E, E)
    invE = np.real(1 / (e2 - e1 - 1j * ewide))
    invE = invE - np.diag(np.diag(invE))
    # f = FD_zero(e2) - FD_zero(e1)

    vw = hkk + 1j * (
            np.einsum('lm,amn->aln', hk, Awk, optimize=True) - np.einsum('alm,mn->aln', Awk, hk, optimize=True))
    v = np.einsum('mi,aij,jn->amn', U.conj().T, vw, U, optimize=True)

    vva = np.diag(v[a])
    vva = np.meshgrid(vva, vva)
    vva = vva[1] - vva[0]

    r = -1j * np.einsum('amn,mn->amn', v, invE)
    omegaz_mn = 1j * (np.einsum('mn,nm->mn', r[0], r[1]) - np.einsum('mn,nm->mn', r[1], r[0]))

    if isADwide:
        delta = adapted_gauss_Delta_func(E, ee, v, htb.dk, ewide_min=ewide_min)
    else:
        delta = gauss_Delta_func(E, ee, ewide) + gauss_Delta_func(E, -ee, ewide)

    f = np.zeros([ngate, htb.nw, htb.nw], dtype='float64')
    for i in range(ngate):
        f[i] = FD_zero(e2 - gate[i]) - FD_zero(e1 - gate[i])

    cpge = np.zeros([ngate, ne], dtype='float64')
    cpge = 0.5 * np.pi * np.real(np.einsum('gnm,mn,mn,mne->ge', f, vva, omegaz_mn, delta, optimize=True))

    return cpge  # -> [ngate, ne]


def cal_CPGE_Retarded(htb, k, ee, gate, rank3Index, ewide=0.005, ewide_min=0.005, isADwide=True):
    a, b, c = rank3Index
    ne = ee.shape[0]
    ngate = gate.shape[0]
    v, vw, E, U, hk, hkk, Awk = get_fft001(htb, k)
    W = get_adaptive_ewide_III(v, htb.dk, np.sqrt(2), ewide_min=ewide_min)

    e1, e2 = np.meshgrid(E, E)
    if isADwide:
        invE_AD = 1 / (e2 - e1 + 1j * W)
        invE_AD = invE_AD - np.diag(np.diag(invE_AD))
    else:
        invE = 1 / (e2 - e1 + 1j * ewide)
        invE = invE - np.diag(np.diag(invE))

    npmesh = np.meshgrid(E, E, ee)
    if isADwide:
        invEe_AD = 1 / ((npmesh[1] - npmesh[0] + npmesh[2]).T + 1j * W.T).T + \
                   -1 / ((npmesh[1] - npmesh[0] - npmesh[2]).T + 1j * W.T).T
    else:
        invEe = 1 / (npmesh[1] - npmesh[0] + npmesh[2] + 1j * ewide) + \
                -1 / (npmesh[1] - npmesh[0] - npmesh[2] + 1j * ewide)

    f = np.zeros([ngate, htb.nw, htb.nw], dtype='float64')
    for i in range(ngate):
        f[i] = FD_zero(e2 - gate[i]) - FD_zero(e1 - gate[i])

    if isADwide:
        cpge = -1 * np.einsum('gnl,lm,lne,lm,mn,nl->ge', f, invE_AD, invEe_AD, v[a], v[b], v[c], optimize=True).imag - \
               -1 * np.einsum('gnl,lm,lne,lm,mn,nl->ge', f, invE_AD, invEe_AD, v[a], v[c], v[b], optimize=True).imag
    else:
        cpge = -1 * np.einsum('gnl,lm,lne,lm,mn,nl->ge', f, invE, invEe, v[a], v[b], v[c], optimize=True).imag - \
               -1 * np.einsum('gnl,lm,lne,lm,mn,nl->ge', f, invE, invEe, v[a], v[c], v[b], optimize=True).imag
    cpge *= 0.5
    # cpge = np.real(cpge / (ee ** 2 + 1j * 0.005))

    return cpge  # -> [ngate, ne]


'''
  * Berry curvature dipole
'''
def cal_bc_dipole(htb, k, gate, temperature, ewide_berry=0.001):
    ngate = gate.shape[0]
    ntemperature = temperature.shape[0]
    borden = Kb * temperature
    # ewide should be seted as 0.1 times of the minimal energy difference, or using adapted smearing
    v, vw, E, U, hk, hkk, Awk = get_fft001(htb, k, sym_h=True, sym_r=True)

    e1, e2 = np.meshgrid(E, E)
    invE = np.real(1 / (e2 - e1 - 1j * ewide_berry))
    invE = invE - np.diag(np.diag(invE))
    invE2 = invE ** 2

    # vv = np.zeros_like(v)
    # for i in range(3):
    #     _vv = np.diag(v[i])
    #     _vv = np.meshgrid(_vv, _vv)
    #     vv[i] = _vv[1] - _vv[0]
    diagv = np.array([np.diag(v[0]), np.diag(v[1]), np.diag(v[2])]).real

    # bc = np.zeros([3, htb.nw], dtype='float64')
    # bc[0] = np.einsum('mn,mn->n', -2. * np.imag(v[1].T * v[2]), invE2, optimize=True)
    # bc[1] = np.einsum('mn,mn->n', -2. * np.imag(v[2].T * v[0]), invE2, optimize=True)
    # bc[2] = np.einsum('mn,mn->n', -2. * np.imag(v[0].T * v[1]), invE2, optimize=True)
    bcz = np.einsum('mn,mn->n', -2. * np.imag(v[0].T * v[1]), invE2, optimize=True)

    pf = np.zeros([ngate, ntemperature, htb.nw], dtype='float64')
    for i in range(ngate):
        for j in range(ntemperature):
            pf[i, j] = -np.exp(-0.5 * ((E - gate[i]) / borden[j]) ** 2) / np.sqrt(2 * np.pi) / borden[j]
    dos = -1 * np.sum(pf, axis=2)

    dipole = np.einsum('an,n,gTn->agT', diagv, bcz, pf, optimize=True)

    result = np.zeros([4, ngate, ntemperature], dtype='float64')
    result[:3] = dipole
    result[3] = dos

    return E, bcz, result

'''
  * second harmonic generation
'''
def _get_rba(a, b, invE, v, vv, w_ab):
    rba = 1j * invE * (
            invE * v[b] * vv[a] +
            np.einsum('nl,lm,lm->nm', v[b], v[a], invE, optimize=True) -
            np.einsum('nl,lm,nl->nm', v[a], v[b], invE, optimize=True) -
            w_ab
    )
    return rba

def _get_ww_ab(a, b, hk, vw, hkk, Awk, Awkk_ab, hkkk_ab):
    ww_ab = hkkk_ab + \
            1j * (LA.multi_dot([hkk[a], Awk[b]]) - LA.multi_dot([Awk[b], hkk[a]])) + \
            1j * (LA.multi_dot([hk, Awkk_ab]) - LA.multi_dot([Awkk_ab, hk])) + \
            1j * (LA.multi_dot([vw[b], Awk[a]]) - LA.multi_dot([Awk[a], vw[b]]))
    return ww_ab

def cal_SHG_prb1998(htb, k, ee, rank3Index, gate, ewide=0.01, ewide_berry=0.001):
    """
    The second-order susceptibility chi for the SHG.
    For the detail of the formula, see ref.[1-2].
    Note there is an Erratum of ref.[1],
    where the correct SHG formula is presented.

    Unit:
    For 2D system, chi(3D) = chi(2D) / d(Angs).
    For 3D system, using chi directly.
    Then,
    dim_coeff = EV / Epsilon0 * 1e12
    dim_coeff * chi is in unit of [pm/V].

    Ref.
    [1] PRB 96, 115147 (2017)
    [2] PRB 57, 3905 (1998)

    Note.
    This function has been tested with the GaAs results in Ref.[1].
    """
    a, b, c = rank3Index
    ngate = gate.shape[0]

    eikr_hr = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_hr))
    eikr_r = np.exp(2j * np.pi * np.einsum('a,ia', k, htb.R_r))
    hk = np.einsum('R,Rmn->mn', eikr_hr, htb.hr_Rmn, optimize=True)
    hkk = 1j * np.einsum('Ra,R,Rmn->amn', htb.Rc_hr, eikr_hr, htb.hr_Rmn, optimize=True)
    hkkk_ab = -np.einsum('R,R,R,Rmn->mn', htb.Rc_hr.T[a], htb.Rc_hr.T[b], eikr_hr, htb.hr_Rmn, optimize=True)
    hkkk_ac = -np.einsum('R,R,R,Rmn->mn', htb.Rc_hr.T[a], htb.Rc_hr.T[c], eikr_hr, htb.hr_Rmn, optimize=True)
    hkkk_bc = -np.einsum('R,R,R,Rmn->mn', htb.Rc_hr.T[b], htb.Rc_hr.T[c], eikr_hr, htb.hr_Rmn, optimize=True)
    Awk = np.einsum('R,Ramn->amn', eikr_r, htb.r_Ramn, optimize=True)
    Awkk_ab = 1j * np.einsum('R,R,Rmn->mn', htb.Rc_r.T[a], eikr_r, htb.r_Ramn[:, b, :, :], optimize=True)
    Awkk_ac = 1j * np.einsum('R,R,Rmn->mn', htb.Rc_r.T[a], eikr_r, htb.r_Ramn[:, c, :, :], optimize=True)
    Awkk_bc = 1j * np.einsum('R,R,Rmn->mn', htb.Rc_r.T[b], eikr_r, htb.r_Ramn[:, c, :, :], optimize=True)

    E, U = LA.eigh(hk)
    E = E - htb.fermi
    e1, e2 = np.meshgrid(E, E)

    f = np.zeros([ngate, htb.nw, htb.nw], dtype='float64')
    for i in range(ngate):
        f[i] = FD_zero(e2 - gate[i]) - FD_zero(e1 - gate[i])

    invE = np.real(1 / (e2 - e1 - 1j * ewide_berry))
    invE = invE - np.diag(np.diag(invE))

    npmesh = np.meshgrid(E, E, ee, indexing='ij')
    invEe = 1 / (npmesh[0] - npmesh[1] - npmesh[2] - 1j * ewide)
    invE2e = 1 / (npmesh[0] - npmesh[1] - 2 * npmesh[2] - 1j * ewide)

    npmesh = np.meshgrid(E, E, E, indexing='ij')
    invE3 = np.real(1 / (2 * npmesh[0] - npmesh[1] - npmesh[2] - 1j * ewide_berry))  # lmn

    vw = hkk + 1j * (np.einsum('lm,amn->aln', hk, Awk, optimize=True) - np.einsum('alm,mn->aln', Awk, hk, optimize=True))
    v = np.einsum('mi,aij,jn->amn', U.conj().T, vw, U, optimize=True)
    r = -1j * np.einsum('amn,mn->amn', v, invE)

    ww_ab = hkkk_ab + 1j * (commdot(hkk[a], Awk[b]) + commdot(hk, Awkk_ab) + commdot(vw[b], Awk[a]))
    ww_ac = hkkk_ac + 1j * (commdot(hkk[a], Awk[c]) + commdot(hk, Awkk_ac) + commdot(vw[c], Awk[a]))
    ww_bc = hkkk_bc + 1j * (commdot(hkk[b], Awk[c]) + commdot(hk, Awkk_bc) + commdot(vw[c], Awk[b]))
    w_ab = LA.multi_dot([U.conj().T, ww_ab, U])
    w_ac = LA.multi_dot([U.conj().T, ww_ac, U])
    w_bc = LA.multi_dot([U.conj().T, ww_bc, U])

    vv = np.zeros_like(v)
    for i in range(3):
        _vv = np.diag(v[i])
        _vv = np.meshgrid(_vv, _vv)
        vv[i] = _vv[1] - _vv[0]

    rba = _get_rba(a, b, invE, v, vv, w_ab)
    rab = _get_rba(b, a, invE, v, vv, w_ab)
    rac = _get_rba(c, a, invE, v, vv, w_ac)
    rca = _get_rba(a, c, invE, v, vv, w_ac)
    rbc = _get_rba(c, b, invE, v, vv, w_bc)
    rcb = _get_rba(b, c, invE, v, vv, w_bc)

    # \chi_e term
    chi_inter = 0.5 * (
            np.einsum('nm,ml,ln,lmn,gnm,mne->ge', r[a], r[b], r[c], invE3, f, invE2e, optimize=True) * 2 +
            np.einsum('nm,ml,ln,lmn,gnm,mne->ge', r[a], r[c], r[b], invE3, f, invE2e, optimize=True) * 2 +
            np.einsum('nm,ml,ln,lmn,gln,lne->ge', r[a], r[b], r[c], invE3, f, invEe, optimize=True) +
            np.einsum('nm,ml,ln,lmn,gln,lne->ge', r[a], r[c], r[b], invE3, f, invEe, optimize=True) +
            np.einsum('nm,ml,ln,lmn,gml,mle->ge', r[a], r[b], r[c], invE3, f, invEe, optimize=True) +
            np.einsum('nm,ml,ln,lmn,gml,mle->ge', r[a], r[c], r[b], invE3, f, invEe, optimize=True)
    )
    # \chi_i term
    chi_mix = 0.5j * (
            np.einsum('gnm,nm,mn,mn,mne->ge', f, r[a], rbc, invE, invE2e, optimize=True) * 2 +
            np.einsum('gnm,nm,mn,mn,mne->ge', f, r[a], rcb, invE, invE2e, optimize=True) * 2 +
            np.einsum('gnm,nm,mn,mn,mne->ge', f, rac, r[b], invE, invEe, optimize=True) +
            np.einsum('gnm,nm,mn,mn,mne->ge', f, rab, r[c], invE, invEe, optimize=True) +
            np.einsum('gnm,nm,mn,mn,mn,mne->ge', f, r[a], r[b], vv[c], invE ** 2, invEe, optimize=True) +
            np.einsum('gnm,nm,mn,mn,mn,mne->ge', f, r[a], r[c], vv[b], invE ** 2, invEe, optimize=True) +
            np.einsum('gnm,nm,mn,mn,mn,mne->ge', f, r[a], r[b], vv[c], invE ** 2, invE2e, optimize=True) * -4 +
            np.einsum('gnm,nm,mn,mn,mn,mne->ge', f, r[a], r[c], vv[b], invE ** 2, invE2e, optimize=True) * -4 +
            np.einsum('gnm,nm,mn,mn,mne->ge', f, rba, r[c], invE, invEe, optimize=True) * -0.5 +
            np.einsum('gnm,nm,mn,mn,mne->ge', f, rca, r[b], invE, invEe, optimize=True) * -0.5
    )

    chi = chi_inter + chi_mix

    return chi

def cal_shg_nagaosa(htb, ee, k, gate, ewide, tensorIndex, adewide=False):
    """
    For details of the formula, see
    [1]. PRB 94, 035117 (2016)
    [2]. Sci. Adv., 2 (5), e1501524.
    [3]. Nature Physics 13, 350–355 (2017)

    Unit:
    For 2D system, chi(3D) = chi(2D) / d(Angs).
    For 3D system, using chi directly.
    Then,
    dim_coeff = EV / Epsilon0 * 1e12
    dim_coeff * chi is in unit of [pm/V].
    """
    ngate = gate.shape[0]
    a, b, c = tensorIndex

    # kc = np.array([0.2, 0.1, 0.2])
    # v, vw, E, U, hk, hkk, Awk = res.get_fft001(htb, k)
    index = np.array([
        [a, b]
    ])
    v, vw, w, ww, E, U = get_fft_d2(htb, k, index)
    wwab = ww[0]

    e1, e2 = np.meshgrid(E, E)
    invE = np.real(1 / (e2 - e1 - 1j * 0.001))
    invE = invE - np.diag(np.diag(invE))

    f = np.zeros([ngate, htb.nw, htb.nw], dtype='float64')
    for i in range(ngate):
        f[i] = FD_zero(e2 - gate[i]) - FD_zero(e1 - gate[i])

    v = np.einsum('mi,aij,jn->amn', U.conj().T, vw, U, optimize=True)
    wab = np.einsum('mi,ij,jn->mn', U.conj().T, wwab, U, optimize=True)
    # wzz = 0

    vv = np.zeros_like(v)
    for i in range(3):
        _vv = np.diag(v[i])
        _vv = np.meshgrid(_vv, _vv)
        vv[i] = _vv[1] - _vv[0]

    r = -1j * np.einsum('amn,mn->amn', v, invE)
    rba = 1j * invE * (
            invE * v[b] * vv[a] +
            np.einsum('nl,lm,lm->nm', v[b], v[a], invE, optimize=True) -
            np.einsum('lm,nl,nl->nm', v[b], v[a], invE, optimize=True) -
            wab
    )
    Rab = np.imag(rba / (r[b] + 1e60 * np.eye(htb.nw)))

    if adewide:
        dk = htb.dk
        delta = -1 * adapted_gauss_Delta_func(E, ee, v, dk, ewide_min=ewide) + \
                0.5 * adapted_gauss_Delta_func(E, 2 * ee, v, dk, ewide_min=ewide)
    else:
        delta = -1 * gauss_Delta_func(E, ee, ewide) + 0.5 * gauss_Delta_func(E, 2*ee, ewide)

    # shg = np.einsum('v,c,vc,vc,cve->e', f, 1-f, np.abs(v[b])**2, Rab, delta, optimize=True)
    shg = np.einsum('gnm,nm,nm,mne->ge', f, Rab, np.abs(v[b])**2, delta, optimize=True)
    # shg = np.einsum('v,c,vc,vc', f, 1-f, np.abs(v[2])**2, Rzz)

    shg = 0.5 * np.pi * shg

    return shg


'''
  * linear and non-linear MO
'''
def cal_linear_L_MOKE(htb, dyy):
    """
    Linear Longitudinal MOKE
    * y-axis is chosen to be parallel to both the direction of the magnetization and the plane of incidence
    * theta_i=np.pi/4

    * temp code:
    RS1 = 1 + dim_coeff * (RS[0] / NK)
    RS2 = dim_coeff * (RS[1] / NK)
    dyy = RS1 + 1j * RS2
    """
    phi = -1 / np.sqrt(2) / (dyy - 1) ** 2
    phi_p = phi * (1 + 1 / np.sqrt(2 * dyy - 1)) * (360 / 2 / np.pi)
    phi_s = phi * (1 - 1 / np.sqrt(2 * dyy - 1)) * (360 / 2 / np.pi)
    return phi_p, phi_s


def cal_2nd_L_MOKE(htb):
    """
    2nd order Longitudinal MOKE
    * y-axis is chosen to be parallel to both the direction of the magnetization and the plane of incidence
    * theta_i=np.pi/4
    """

    pass



