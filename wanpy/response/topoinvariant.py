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

__date__ = "Otc. 21, 2020"

import os
import numpy as np
from numpy import linalg as LA
from wanpy.core.structure import Htb
import wanpy.response.response as res
from wanpy.core.mesh import make_kpath

__all__ = [
    'cal_wilson_loop_on_2d_kplane',
    'cal_chern_number',
]

'''
  * wilson loop
'''
# def _cal_wilson_loop_on_a_closed_loop(htb, kk, i1, i2, e2=1):
#     '''
#       * calculae Wilson loop
#         W(C) = Int{A(k)*dk[C]}
#         on closed loop C
#     '''  #
#     nk = kk.shape[0]
#
#     # Dky = np.identity(i2-i1)
#     Dky = np.identity(htb.nw)
#     for i in range(nk):
#         hk = res.get_hk(htb, kk[i], tbgauge=True)
#         E, U = LA.eigh(hk)
#         V = U[:, i1:i2]
#         if i + 1 != nk:
#             Dky = LA.multi_dot([Dky, V, V.conj().T])
#         else:
#             V_iGtau = np.diag(np.exp(-1j * htb.lattG.T[e2] @ htb.wcc.T))
#             Dky = LA.multi_dot([V.conj().T, Dky, V_iGtau, V])
#             # Dky = LA.multi_dot([V.conj().T, Dky, V])
#
#     theta = np.sort(np.imag(np.log(LA.eigvals(Dky)))) / 2 / np.pi
#     return theta
#
# def cal_wilson_loop_on_2d_kplane(htb, i1, i2, e1=0, e2=1, e3=2, k3=0, nk1=30, nk2=30):
#     '''
#
#     :param i1, i2: track wannier center for bands index i1-i2
#     :param e1: direction to show wannier center
#     :param e2: direction to integration
#     :param e3: principle direction of plane
#     :param k3: position of the 2D plane
#     '''
#     theta = np.zeros([nk1, i2-i1], dtype='float64')
#     kk1 = np.linspace(0, 1, nk1 + 1)[:-1]
#     for i in range(nk1):
#         _kk = np.zeros([nk2, 3], dtype='float64')
#         _kk.T[e1] = kk1[i]
#         _kk.T[e2] = np.linspace(0, 1, nk2 + 1)[:-1]
#         _kk.T[e3] = k3
#         theta[i] = _cal_wilson_loop_on_a_closed_loop(htb, _kk, i1, i2, e2)
#     return kk1, theta
#
#
# def cal_wilson_loop_on_3d_kplane(htb, i1, i2):
#     pass


def _cal_wilson_loop_on_a_closed_loop(htb, kk, bandindex, e2=1):
    """
      * calculae Wilson loop 
        W(C) = Int{A(k)*dk[C]} 
        on closed loop C
        
      Note. 
        For tb gauge (atomic gauge), e^{iGr} should be used in the last step of product,
        For wannier gauge (lattice gauge), e^{ibr} should be used in each step of products.
    """  #
    nk = kk.shape[0]

    # Dky = np.identity(i2-i1)
    Dky = np.identity(htb.nw)
    for i in range(nk):
        hk = res.get_hk(htb, kk[i], tbgauge=True)
        E, U = LA.eigh(hk)
        V = U[:, bandindex]
        if i + 1 != nk:
            Dky = LA.multi_dot([Dky, V, V.conj().T])
        else:
            V_iGtau = np.diag(np.exp(-1j * htb.lattG.T[e2] @ htb.wcc.T))
            Dky = LA.multi_dot([V.conj().T, Dky, V_iGtau, V])
            # Dky = LA.multi_dot([V.conj().T, Dky, V])

    theta = np.sort(np.imag(np.log(LA.eigvals(Dky)))) / 2 / np.pi
    return theta

def cal_wilson_loop_on_2d_kplane(htb, bandindex, e1=0, e2=1, e3=2, k3=0, nk1=30, nk2=30):
    """

    :param nk2:
    :param nk1:
    :param htb:
    :param bandindex: track wannier center for given bandindex
    :param e1: direction to show wannier center
    :param e2: direction to integration
    :param e3: principle direction of plane
    :param k3: position of the 2D plane
    """
    theta = np.zeros([nk1, bandindex.shape[0]], dtype='float64')
    kk1 = np.linspace(0, 1, nk1 + 1)[:-1]
    for i in range(nk1):
        print('cal wcc at ({}/{})'.format(i + 1, nk1))
        _kk = np.zeros([nk2, 3], dtype='float64')
        _kk.T[e1] = kk1[i]
        _kk.T[e2] = np.linspace(0, 1, nk2 + 1)[:-1]
        _kk.T[e3] = k3
        theta[i] = _cal_wilson_loop_on_a_closed_loop(htb, _kk, bandindex, e2)
    return kk1, theta


'''
  * Chern number
'''
def cal_chern_number(htb, bandindex):
    # effective lattice method
    kpath_HSP = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    kpath = make_kpath(kpath_HSP, nk1=11)[:-1]
    nk = kpath.shape[0]

    Dky = np.identity(htb.nw)
    for i, k in zip(range(nk), kpath):
        hk = res.get_hk(htb, k, tbgauge=False)
        E, U = LA.eigh(hk)
        V = U[:, bandindex]
        if i + 1 != nk:
            Dky = LA.multi_dot([Dky, V, V.conj().T])
        else:
            Dky = LA.multi_dot([V.conj().T, Dky, V])
    berry_phase = np.imag(np.log(np.real(Dky))) / 2 / np.pi
    return berry_phase



wdir = r'/Volumes/jindedata/scidata/test'
'''
  * Job = 
  ** wloop
  ** chern number
'''
Job = 'wloop'

if __name__ == '__main__' and Job == 'wloop':
    os.chdir(wdir)

    htb = Htb()
    htb.load_h5()
    htb.setup()

    '''
      * For PC 
    '''
    # kk1, theta = cal_wilson_loop_on_2d_kplane(htb, i1=0, i2=3, e1=0, e2=1, e3=2, k3=0, nk1=100, nk2=30)

    bandindex = np.array([0, 1, 2])
    kk1, theta = cal_wilson_loop_on_2d_kplane(htb, bandindex, e1=0, e2=1, e3=2, k3=0, nk1=30, nk2=30)

    # if os.environ.get('PYGUI') == 'True':
    #     from wanpy.response.response_plot import plot_wloop
    #     plot_wloop(kk1, theta, ymin=-0.8, ymax=0.8, s=10)

    '''
      * par 
    '''
    # ikk2 = np.arange(nk2)
    # theta = cal_wilson_loop(ikk2, dim)
    #
    # if MPI_RANK == 0:
    #     kk2 = np.linspace(0, 1, nk2 + 1)[:-1]
    #     theta = theta[0]
    #
    #     np.savez_compressed(r'wloop.npz',
    #                         kk2=kk2,
    #                         twist_ang=ham.theta_deg,
    #                         theta=theta,
    #                         )
    #
    #     if os.environ.get('PYGUI') == 'True':
    #         from wanpy.response.response_plot import plot_wloop
    #         data = np.load(r'wloop.npz')
    #         kk2 = data['kk2']
    #         theta = data['theta']
    #
    #         plot_wloop(kk2, theta, ymin=-0.8, ymax=0.8)

    # '''
    #   * multi twisted angles, par
    # '''
    # ikk2 = np.arange(nk2)
    # mm = np.arange(10, 33, 1)
    # # mm = np.arange(10, 18, 1)
    # nmm = mm.shape[0]
    # if MPI_RANK == 0:
    #     thetas = np.zeros([nmm, nk2, 4])
    #
    # for i in range(nmm):
    #     # ham = MacDonald_TMG_wan(m=mm[i], n=mm[i]+1, N=3, w1=0.0797, w2=0.0975, tLu=0, tLd=-1, vac=300, htbfname_u=r'htb_AB_SCAN.npz', htbfname_d=r'htb_AB_SCAN.npz')
    #     ham = MacDonald_TMG_wan(m=mm[i], n=mm[i]+1, N=3, w1=0.0797, w2=0.0975, tLu=0, tLd=-1, vac=300, htbfname_u=r'htb_SL_DFT.npz', htbfname_d=r'htb_SL_DFT.npz', rotk=True)
    #
    #     theta = cal_wilson_loop(ikk2, dim)
    #     if MPI_RANK == 0:
    #         thetas[i] = theta[0]
    #
    # if MPI_RANK == 0:
    #     kk2 = np.linspace(0, 1, nk2 + 1)[:-1]
    #     np.savez_compressed(r'wloops.npz',
    #                         kk2=kk2,
    #                         mm=mm,
    #                         thetas=thetas,
    #                         )
    #
    #     if os.environ.get('PYGUI') == 'True':
    #         from wanpy.response.response_plot import plot_wloop
    #
    #         data = np.load(r'wloops.npz')
    #         kk2 = data['kk2']
    #         mm = data['mm']
    #         thetas = data['thetas']
    #         nsuper, natom, twist_ang = return_twist_par(mm)
    #         data.close()
    #
    #         i = 4
    #         print('m{}n{}, theta={:5.3f} deg'.format(mm[i], mm[i]+1, twist_ang[i]))
    #         plot_wloop(kk2, thetas[i], ymin=-0.8, ymax=0.8, save=False, savefname='wloop.pdf')