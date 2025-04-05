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

import math
import numpy as np
import numpy.linalg as LA
from wanpy.core.units import *
from scipy import special

__all__ = [
    'fermi_dirac_func',
    'delta_func',
]

'''
  * K-point integration: special points way
'''
def fermi_dirac_func(E, smear=0, ismear=0):
    """
    smear = Kb*T = beta^-1
    ismear = -1: Fermi-Dirac function
    ismear = 0: Gauss error function
    ismear >= 1: Methfessel-Paxton smearing
    """

    # zero temperature:
    if np.isclose(smear, 0):
        f = (1. - np.sign(E)) / 2.
        return f

    # finite temperature:
    if ismear == -1:
        x = E / smear
        x = np.clip(x, -32, 32)
        f = 1. / (np.exp(x) + 1.)
        return f
    elif ismear == 0:
        x = E / smear
        f = 0.5 * (1. - special.erf(x))
    elif ismear >= 1:
        x = E / smear
        f = 0.5 * (1. - special.erf(x))
        for n in range(1, ismear + 1):
            An = (-1) ** n / math.factorial(n) / 4 ** n / np.sqrt(np.pi)
            hermite = special.hermite(2 * n - 1)
            f += An * hermite(x) * np.exp(-x ** 2)

    return f

def delta_func(E, smear=0.01, ismear=0):
    """
    smear = Kb*T = beta^-1
    ismear = -2: Lorentzian function
    ismear = -1: derivation of Fermi-Dirac function: -(df/dE)
    ismear = 0: Gauss function
    ismear >= 1: Methfessel-Paxton smearing
    """
    if ismear == 0:
        x = E / smear
        x = np.clip(x, -32, 32)
        delta = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi) / smear
    elif ismear == -1:
        x = E / smear / 2
        x = np.clip(x, -32, 32)
        delta = 0.25 * np.cosh(x)**-2 / smear
    elif ismear == -2:
        x = E / smear
        delta = smear / np.pi / (smear ** 2 + x ** 2)
    elif ismear >= 1:
        delta = None

    return delta

def delta_func_adaptive(E, v, dk, scaling=1, ewidth_max=0.1, ewidth_min=0.0001):
    smear = get_adaptive_ewidth_typeII(v, dk=dk, scaling=scaling, ewidth_max=ewidth_max, ewidth_min=ewidth_min)
    x = E / smear
    delta = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi) / smear
    return delta

'''
  * Fermi-Dirac distribution
'''
def FD(e, T=None, fermi=0):
    if T is None or T<1e-3:
        return _fermi_dirac_dis_0T(e, fermi)
    else:
        return _fermi_dirac_dis_finit_T(e, T, fermi=0.0)

def FD_zero(e, fermi=0.0):
    return _fermi_dirac_dis_0T(e, fermi)

def _fermi_dirac_dis_0T(e, fermi=0.0):
    return (1 - np.sign(e-fermi)) / 2

def _fermi_dirac_dis_finit_T(e, T, fermi=0.0):
    x = (e - fermi) / (Kb * (T + 1e-5))
    x[x > 64] = 64
    # Fermi-Dirac Distribution
    return 1 / (np.exp(x) + 1)


'''
  * Adaptive broaden
  ** get_adaptive_ewide_II 
  ** get_adaptive_ewide_III 
  ** get_adaptive_ewide_II_slab 
  ** get_adaptive_ewide_III_slab 
'''
def get_adaptive_ewidth_typeII(v, dk, scaling=1, ewidth_max=0.1, ewidth_min=0.0001):
    """
      This function retures a state-depending broadening width for type-II BZ intrgral.
      The adaptive broadening width is given based on Eq. (34) in [PRB 75, 195121 (2007)].
      Here, dk = (|b1|, |b2|, |b3|) / Nk.
    """
    dk_min = 0.001
    dk[dk < dk_min] = dk_min

    vx, vy, vz = np.real(np.diagonal(v.T))
    W = (dk * np.array([vx, vy, vz]).T).T
    W = scaling * LA.norm(np.abs(W), axis=0)

    W[W > ewidth_max] = ewidth_max
    W[W < ewidth_min] = ewidth_min
    return W

def get_adaptive_ewidth_typeIII(v, dk, scaling=1, ewidth_max=0.1, ewidth_min=0.0001):
    """
      This function retures a state-depending broadening width for type-II BZ intrgral.
      The adaptive broadening width is given based on Eq. (35) in [PRB 75, 195121 (2007)].
      Here, dk = (|b1|, |b2|, |b3|) / Nk.
    """
    dk_min = 0.001
    dk[dk < dk_min] = dk_min

    diagv = np.diagonal(v, axis1=1, axis2=2).real
    diff_diagv = np.array([np.meshgrid(diagv[i], diagv[i]) for i in range(3)])
    diff_diagv = np.array([diff_diagv[i, 1] - diff_diagv[i, 0] for i in range(3)])

    W = scaling * LA.norm(np.abs((dk * diff_diagv.T).T), axis=0)

    W[W > ewidth_max] = ewidth_max
    W[W < ewidth_min] = ewidth_min
    W = W + 1e12 * np.diag(np.diag(np.ones_like(W)))

    # print('W = ', np.sort(W.flatten())[::-1])
    return W

def get_adaptive_ewide_II(v, dk, scaling=1, ewide_min=0.001):
    """
      This function retures a state-depending broadening width for type-II BZ intrgral.
      The adaptive broadening width is given based on Eq. (34) in [PRB 75, 195121 (2007)].
      for DOS type calculation
      * dk = (2Pi / L) / Nk

    """
    dk[dk < 0.001] = 0.001

    vx, vy, vz = np.real(np.diagonal(v.T))
    v = np.array([vx, vy, vz])
    W = (scaling * dk * v.T).T

    W = np.abs(W)
    W = np.maximum(W[0], W[1], W[2])
    W[W < ewide_min] = ewide_min
    return W

def get_adaptive_ewide_III(v, dk, scaling=1, ewide_min=0.001, ewide_max=None, dk_min=0.001):
    """
      EQ.35 in [PRB 75,195121(2007)]
      for JDOS type calculation
      * dk = (2Pi / L) / Nk
    """
    dk[dk < dk_min] = dk_min
    nw = v.shape[-1]

    vx, vy, vz = np.real(np.diagonal(v.T))

    v = np.array([vx, vy, vz])
    W = (scaling * dk * v.T).T

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
    """
      Slab version of get_adaptive_ewide_II
      recover to get_adaptive_ewide_II when open_boundary = -1

      Algorithm:
      * For open boundary direction
    """
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
        eye_filter = get_eye_filter(nw, win)

        e1, e2 = np.meshgrid(E, E)
        invE = np.real(1 / (e2 - e1 - 1j * 1e-8))
        r_ob = v_ob * invE * eye_filter
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


def get_adaptive_ewide_III_slab(v, E, U, dk, N_ucell, open_boundary=-1, scaling=np.sqrt(2), ewide_min=0.001, inf_CenterW=False, win=None):
    """
      Slab version of get_adaptive_ewide_III
      * recover to get_adaptive_ewide_III when open_boundary = -1.
      * The bulk quantum well states excitation are adapted as
        adaptive broaden.
      * The broadens of surface states are seted as minimal ewide.
      * htb.dk at open_boundary direction affect nothing.

      Algorithm descriptions:
      * For open boundary direction
        idea 1. W_nm = (max_{l} |v_nl| - max_{l} |v_ml|) * dk   (2)
        idea 2.
          w_n = E_n - E_l                           (3)
          l = { l | max( <u_n|partial_k u_l> )}     (4)
          W_nm = |w_n - w_m|                        (5)
    """
    dk[dk < 0.001] = 0.001
    nw = v.shape[-1]
    nw_uc = nw // N_ucell
    if type(win) is int:
        win *= nw
    v_ob = np.imag(v[open_boundary])

    vx, vy, vz = np.real(np.diagonal(v.T))

    vv = np.array([vx, vy, vz])
    W = (scaling * dk * vv.T).T
    is_surf = None

    if open_boundary >= 0:
        eye_filter = get_eye_filter(nw, win)

        e1, e2 = np.meshgrid(E, E)
        invE = np.real(1 / (e2 - e1 - 1j * 1e-8))
        r_ob = v_ob * invE * eye_filter
        r_ob = np.argmax(np.abs(r_ob), axis=0)
        W[open_boundary] = scaling * np.array([E[j] - E[i] for i, j in zip(range(nw), r_ob)])

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

def adapted_gauss_Delta_func(E, ee, v, dk, a=np.sqrt(2), ewide_min=0.001, ewide_max=None):
    D = np.meshgrid(E, E, ee)
    D = D[1] - D[0] - D[2]
    W = get_adaptive_ewide_III(v, dk, scaling=a, ewide_min=ewide_min, ewide_max=ewide_max)
    D = np.exp(-0.5 * (D.T / W.T) ** 2) / W.T / np.sqrt(2 * np.pi)
    D = D.T
    return D

def adapted_gauss_Delta_func_slab(E, ee, v, U, dk, N_ucell, open_boundary=-1, a=np.sqrt(2), ewide_min=0.001, inf_CenterW=False, win=2):
    D = np.meshgrid(E, E, ee)
    D = D[1] - D[0] - D[2]
    W = get_adaptive_ewide_III_slab(v, E, U, dk, N_ucell,
                                     open_boundary=open_boundary, scaling=a, ewide_min=ewide_min, inf_CenterW=inf_CenterW, win=win)
    D = np.exp(-0.5 * (D.T / W.T) ** 2) / W.T / np.sqrt(2 * np.pi)
    D = D.T
    return D

'''
  * Surface
'''
def get_is_surf(U, nw_sc, nw_uc, N_ucell, tolerance=0.3):
    """
    This function returns whether the states are on the surface or not.
    """
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
  * other functions
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

def Rect(e, a, b, T=300):
    return FD(e, T, fermi=b) * (1 - FD(e, T, fermi=a))

def get_eye_filter(N, F):
    eye_filter = np.zeros([N, N])
    if type(F) is int:
        for i in range(-F + 1, F):
            eye_filter += np.eye(N, k=i)
    elif (type(F) is list) or (type(F) is tuple):
        for i in F:
            eye_filter += np.eye(N, N, i)
    else:
        eye_filter = np.ones([N, N])
    return eye_filter

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    smear = 0.1
    E = np.linspace(-1, 1, 1000)
    ismear = 3

    x = E / smear
    delta = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi) / smear
    # smear = 0.05
    # delta1 = smear / np.pi / (smear ** 2 + E ** 2)

    x = E / smear
    delta1 = np.zeros_like(E)
    for n in range(1, ismear + 1):
        An = (-1) ** n / math.factorial(n) / 4 ** n / np.sqrt(np.pi)
        hermite = special.hermite(2 * n)
        delta1 += An * hermite(x) * np.exp(-x ** 2)

    plt.axvline(-smear, color='k', linewidth=0.4)
    plt.axvline(smear, color='k', linewidth=0.4)
    plt.plot(E, delta, 'b')
    plt.plot(E, delta1, 'r')