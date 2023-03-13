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

__date__ = "Jan. 3, 2019"

import numpy as np
import numpy.linalg as LA
from wanpy.core.units import *
from wanpy.core.toolkits import eye_filter
from scipy import special


'''
  * Fermi-Dirac distribution
'''
def FD(e, T=None, fermi=0):
    if T is None:
        return fermi_dirac_dis_0T(e, fermi)
    else:
        return fermi_dirac_dis_finit_T(e, T, fermi=0.0)

def FD_zero(e, fermi=0.0):
    return fermi_dirac_dis_0T(e, fermi)

def fermi_dirac_dis_0T(e, fermi=0.0):
    return (1 - np.sign(e-fermi)) / 2

def fermi_dirac_dis_finit_T(e, T, fermi=0.0):
    x = (e - fermi) / (Kb * (T + 1e-5))
    x[x > 64] = 64
    # Fermi-Dirac Distribution
    return 1 / (np.exp(x) + 1)

'''
 Rect
'''
def Rect(e, a, b, T=300):
    return FD(e, T, fermi=b) * (1 - FD(e, T, fermi=a))

'''
  * Surface
'''
def get_is_surf(U, nw_sc, nw_uc, N_ucell, tolerance=0.3):
    u = np.abs(np.array(U).T) ** 2
    u = np.sum(u.reshape(nw_sc, N_ucell, nw_uc), axis=2)
    uq1 = np.array([
        np.fft.fft(u[i])[1].real
        for i in range(nw_sc)
    ])
    is_surf = uq1 > tolerance
    # print(is_surf)
    return is_surf


'''
  * Adaptive broaden
  ** get_adaptive_ewide_II 
  ** get_adaptive_ewide_III 
  ** get_adaptive_ewide_II_slab 
  ** get_adaptive_ewide_III_slab 
'''
def get_adaptive_ewide_II(v, dk, a=np.sqrt(2), ewide_min=0.001):
    '''
      EQ.34 in [PRB 75,195121(2007)]
      for DOS type calculation
      * dk = (2Pi / L) / Nk
    '''
    dk[dk < 0.001] = 0.001

    vx, vy, vz = np.real(np.diagonal(v.T))
    v = np.array([vx, vy, vz])
    W = (a * dk * v.T).T

    W = np.abs(W)
    W = np.maximum(W[0], W[1], W[2])
    W[W < ewide_min] = ewide_min
    return W


def get_adaptive_ewide_III(v, dk, a=np.sqrt(2), ewide_min=0.001, ewide_max=None, dk_min=0.001):
    '''
      EQ.35 in [PRB 75,195121(2007)]
      for JDOS type calculation
      * dk = (2Pi / L) / Nk
    '''
    dk[dk < dk_min] = dk_min
    nw = v.shape[-1]

    vx, vy, vz = np.real(np.diagonal(v.T))

    v = np.array([vx, vy, vz])
    W = (a * dk * v.T).T

    W = np.array([
        np.meshgrid(W[0], W[0]),
        np.meshgrid(W[1], W[1]),
        np.meshgrid(W[2], W[2]),
    ])
    W = np.array([
        W[0, 1] - W[0, 0],
        W[1, 1] - W[1, 0],
        W[2, 1] - W[2, 0],
    ])
    W = np.abs(W)
    W = np.maximum(W[0], W[1], W[2])
    W[W < ewide_min] = ewide_min
    if ewide_max is not None:
        W[W > ewide_max] = ewide_max
    for i in range(nw):
        W[i, i] = 1e8
    # print('W = ', np.sort(W.flatten())[::-1])
    return W


def get_adaptive_ewide_II_slab(v, E, U, dk, N_ucell, open_boundary=-1, a=np.sqrt(2), ewide_min=0.001, dk_min=0.001, win=None):
    '''
      Slab version of get_adaptive_ewide_II
      recover to get_adaptive_ewide_II when open_boundary = -1

      Algorithm:
      * For open boundary direction
    '''
    dk[dk < dk_min] = dk_min
    nw = v.shape[-1]
    nw_uc = nw // N_ucell
    if type(win) is int:
        win *= nw
    v_ob = np.imag(v[open_boundary])

    vx, vy, vz = np.real(np.diagonal(v.T))

    vv = np.array([vx, vy, vz])
    W = (a * dk * vv.T).T
    is_surf = None

    if open_boundary >= 0:
        filter = eye_filter(nw, win)

        e1, e2 = np.meshgrid(E, E)
        invE = np.real(1 / (e2 - e1 - 1j * 1e-8))
        r_ob = v_ob * invE * filter
        r_ob = np.argmax(np.abs(r_ob), axis=0)
        W[open_boundary] = a * np.array([E[j] - E[i] for i, j in zip(range(nw), r_ob)])

        # surface state
        is_surf = get_is_surf(U, nw, nw_uc, N_ucell, tolerance=0.3)
        for i in range(nw):
            if is_surf[i]:
                W[open_boundary, i] = ewide_min

    W = np.abs(W)
    W = np.maximum(W[0], W[1], W[2])
    W[W < ewide_min] = ewide_min
    return W


def get_adaptive_ewide_III_slab(v, E, U, dk, N_ucell, open_boundary=-1, a=np.sqrt(2), ewide_min=0.001, inf_CenterW=False, win=None):
    '''
      Slab version of get_adaptive_ewide_III
      * recover to get_adaptive_ewide_III when open_boundary = -1.
      * The bulk quantum well states excitation are adapted as
        adaptive broaden.
      * The broaden of surface states are seted as minimal ewide.
      * htb.dk at open_boundary direction affect nothing.

      Algorithm descriptions:
      * For open boundary direction
        idea 1. W_nm = (max_{l} |v_nl| - max_{l} |v_ml|) * dk   (2)
        idea 2.
          w_n = E_n - E_l                           (3)
          l = { l | max( <u_n|partial_k u_l> )}     (4)
          W_nm = |w_n - w_m|                        (5)
    '''
    dk[dk < 0.001] = 0.001
    nw = v.shape[-1]
    nw_uc = nw // N_ucell
    if type(win) is int:
        win *= nw
    v_ob = np.imag(v[open_boundary])

    vx, vy, vz = np.real(np.diagonal(v.T))

    vv = np.array([vx, vy, vz])
    W = (a * dk * vv.T).T
    is_surf = None

    if open_boundary >= 0:
        filter = eye_filter(nw, win)

        e1, e2 = np.meshgrid(E, E)
        invE = np.real(1 / (e2 - e1 - 1j * 1e-8))
        r_ob = v_ob * invE * filter
        r_ob = np.argmax(np.abs(r_ob), axis=0)
        W[open_boundary] = a * np.array([E[j] - E[i] for i, j in zip(range(nw), r_ob)])

        # set surface state with minimal ewide
        is_surf = get_is_surf(U, nw, nw_uc, N_ucell, tolerance=0.3)
        for i in range(nw):
            if is_surf[i]:
                W[open_boundary, i] = ewide_min

    W = np.array([
        np.meshgrid(W[0], W[0]),
        np.meshgrid(W[1], W[1]),
        np.meshgrid(W[2], W[2]),
    ])
    W = np.array([
        W[0, 1] - W[0, 0],
        W[1, 1] - W[1, 0],
        W[2, 1] - W[2, 0],
    ])
    W = np.abs(W)
    W = np.maximum(W[0], W[1], W[2])
    W[W < ewide_min] = ewide_min

    # for i in range(nw):
    #     W[i, i] = 1e8
    W = W + 1e10 * np.eye(nw)

    # print('W = ', np.sort(W.flatten())[::-1])

    # set surface states with infinite ewitde
    # this is only used for excluding surface contributions
    if (open_boundary >= 0) and inf_CenterW:
        for i in range(nw):
            if is_surf[i]:
                W[i, :] = 1e10
                W[:, i] = 1e10

    return W


'''
  * Integration
'''
def gauss_Delta_func(E, ee, ewide=0.02):
    D = np.meshgrid(E, E, ee)
    D = D[1] - D[0] - D[2]
    D = np.exp(-0.5 * (D / ewide) ** 2) / np.sqrt(2*np.pi) / ewide  # shape = (num_wann, num_wann, ne)
    return D


def lorentz_Delta_func(E, ee, ewide=0.02):
    D = np.meshgrid(E, E, ee)
    D = D[1] - D[0] - D[2]
    D = ewide / np.pi / (ewide ** 2 + D ** 2)  # shape = (num_wann, num_wann, ne)
    return D


def adapted_gauss_Delta_func(E, ee, v, dk, a=np.sqrt(2), ewide_min=0.001, ewide_max=None):
    D = np.meshgrid(E, E, ee)
    D = D[1] - D[0] - D[2]
    W = get_adaptive_ewide_III(v, dk, a=a, ewide_min=ewide_min, ewide_max=ewide_max)
    D = np.exp(-0.5 * (D.T / W.T) ** 2) / W.T / np.sqrt(2 * np.pi)
    D = D.T
    return D


def adapted_gauss_Delta_func_slab(E, ee, v, U, dk, N_ucell, open_boundary=-1, a=np.sqrt(2), ewide_min=0.001, inf_CenterW=False, win=2):
    D = np.meshgrid(E, E, ee)
    D = D[1] - D[0] - D[2]
    W = get_adaptive_ewide_III_slab(v, E, U, dk, N_ucell,
                                     open_boundary=open_boundary, a=a, ewide_min=ewide_min, inf_CenterW=inf_CenterW, win=win)
    D = np.exp(-0.5 * (D.T / W.T) ** 2) / W.T / np.sqrt(2 * np.pi)
    D = D.T
    return D



'''
  * Lorentz 
  * gauss
  * Methfessel and Paxton(1989) Gaussian-broadened sampling integration
'''
def lorentz(e, ewide=0.1):
    e = e / ewide
    D = 1 / np.pi / (1 + e ** 2)
    D = D / ewide
    return D


def gauss(e, ewide=0.1):
    e = e / ewide
    D = np.exp(-0.5 * e ** 2) / np.sqrt(2*np.pi)
    D = D / ewide
    return D


def MPgauss(e, ewide=0.1, N=1):
    e = e / ewide
    D = np.zeros_like(e)
    for i in range(N+1):
        Ai = (-1) ** i / (np.math.factorial(i) * 4 ** i * np.sqrt(np.pi))
        hermite = special.hermite(2 * i)
        D += Ai * hermite(e)
    D *= np.exp(-e ** 2)
    D = D / ewide
    return D
