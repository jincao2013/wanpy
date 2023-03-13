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

__date__ = "Jan. 16, 2019"

import os
import time
import numpy as np
import numpy.linalg as LA
from numpy.linalg import multi_dot
from wanpy.core.DEL.read_write import Cell, Wout, Htb
from wanpy.core.mesh import make_kpath, make_kmesh_dev001
from wanpy.core.toolkits import kmold


def h_BiH(vppx=1.9298, vpppi=-0.7725, tso=0.65):
    A = 5.5300002098
    B = A * (3 ** 0.5)
    C = 20.0
    # A, B, C = (1, 1, 1)

    nw = 8
    nR = 5

    htb = Htb()
    cell = Cell()
    wout = Wout()
    htb.name = r'BiH From M.Y.Wang'
    htb.nw = nw
    htb.nR = nR
    htb.R = np.array([
        [0, 0, 0], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0],
    ])
    htb.ndegen = np.ones(nR, dtype='int64')
    htb.fermi = 0.0
    htb.hr_Rmn = np.zeros([nR, nw, nw], dtype='complex128')
    htb.r_Ramn = np.zeros([nR, 3, nw, nw], dtype='complex128')


    '''
      R = [0, 0, 0]
    '''
    t0 = np.zeros((nw, nw), dtype='complex128')
    t0[0, 2] = 0.75 * vppx + 0.25 * vpppi
    t0[0, 3] = 3 ** 0.5 / 4 * (vppx - vpppi)
    t0[1, 2] = 3 ** 0.5 / 4 * (vppx - vpppi)
    t0[1, 3] = 0.25 * vppx + 0.75 * vpppi
    t0[2, 4] = vpppi
    t0[3, 5] = vppx
    t0[4, 6] = 0.75 * vppx + 0.25 * vpppi
    t0[4, 7] = -3 ** 0.5 / 4 * (vppx - vpppi)
    t0[5, 6] = -3 ** 0.5 / 4 * (vppx - vpppi)
    t0[5, 7] = 0.25 * vppx + 0.75 * vpppi
    htb.hr_Rmn[0] = t0 + t0.T.conj()

    '''
      R = [0,  1, 0]
          [0, -1, 0]
    '''
    t0 = np.zeros((nw, nw), dtype='complex128')
    t0[0, 6] = vpppi
    t0[1, 7] = vppx
    htb.hr_Rmn[1] = t0
    htb.hr_Rmn[2] = t0.T.conj()

    '''
      R = [ 1, 0, 0]
          [-1, 0, 0]
    '''
    t0 = np.zeros((nw, nw), dtype='complex128')
    t0[0, 2] = 0.75 * vppx + 0.25 * vpppi
    t0[0, 3] = -3 ** 0.5 / 4 * (vppx - vpppi)
    t0[1, 2] = -3 ** 0.5 / 4 * (vppx - vpppi)
    t0[1, 3] = 0.25 * vppx + 0.75 * vpppi

    t0[6, 4] = 0.75 * vppx + 0.25 * vpppi
    t0[7, 4] = 3 ** 0.5 / 4 * (vppx - vpppi)
    t0[6, 5] = 3 ** 0.5 / 4 * (vppx - vpppi)
    t0[7, 5] = 0.25 * vppx + 0.75 * vpppi
    htb.hr_Rmn[3] = t0
    htb.hr_Rmn[4] = t0.T.conj()

    '''
      On site SOC
    '''
    soc = tso * (
        1j * np.diagflat([1, 0, 1, 0, 1, 0, 1], 1) -
        1j * np.diagflat([1, 0, 1, 0, 1, 0, 1], -1)
    )
    htb.hr_Rmn[0] += soc

    '''
      Wcc
    '''
    lattice = np.diag([A, B, C])
    wccf = np.array([
        [0.75, 5/6, 0.5],
        [0.75, 5/6, 0.5],
        [0.25, 4/6, 0.5],
        [0.25, 4/6, 0.5],
        [0.25, 2/6, 0.5],
        [0.25, 2/6, 0.5],
        [0.75, 1/6, 0.5],
        [0.75, 1/6, 0.5],
    ], dtype='float64')
    wcc = multi_dot([lattice, wccf.T]).T

    htb.wcc = wcc
    htb.wccf = wccf

    '''
      Cell
    '''
    cell.name = 'BiH'
    cell.lattice = lattice
    cell.latticeG = 2 * np.pi * LA.inv(lattice.T)
    cell.ions = np.array([
        [0.75, 5/6, 0.5],
        [0.25, 4/6, 0.5],
        [0.25, 2/6, 0.5],
        [0.75, 1/6, 0.5],
    ], dtype='float64')
    cell.ions_car = multi_dot([lattice, cell.ions.T]).T
    cell.N = 4
    cell.spec = ['Bi', 'Bi', 'Bi', 'Bi']

    '''
      r
    '''
    htb.r_Ramn[0, 0] = np.diagflat(wcc.T[0])
    htb.r_Ramn[0, 1] = np.diagflat(wcc.T[1])
    htb.r_Ramn[0, 2] = np.diagflat(wcc.T[2])

    htb.cell = cell

    return htb


def h_QAH(A=1, B=1, M=5):
    nw = 2
    nR = 5

    htb = Htb()
    cell = Cell()
    wout = Wout()
    htb.name = r'Integer Quantum Hall Effect'
    htb.nw = nw
    htb.nR = nR
    htb.R = np.array([
        [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
    ])
    htb.ndegen = np.ones(nR, dtype='int64')
    htb.fermi = 0.0
    htb.hr_Rmn = np.zeros([nR, nw, nw], dtype='complex128')
    htb.r_Ramn = np.zeros([nR, 3, nw, nw], dtype='complex128')


    '''
      hr_Rmn
    '''
    htb.hr_Rmn[0] = (M - 4 * B) * sigma_z
    htb.hr_Rmn[1] = -1j * A / 2 * sigma_x + B * sigma_z
    htb.hr_Rmn[2] = htb.hr_Rmn[1].conj().T
    htb.hr_Rmn[3] = -1j * A / 2 * sigma_y + B * sigma_z
    htb.hr_Rmn[4] = htb.hr_Rmn[3].conj().T

    '''
      Wcc
    '''
    a, b = (1, 1)
    c = 20
    lattice = np.diag([a, b, c])
    wccf = np.array([
        [0, 0, 0],
        [0, 0, 0],
    ], dtype='float64')
    wcc = multi_dot([lattice, wccf.T]).T

    htb.wcc = wcc
    htb.wccf = wccf

    '''
      Cell
    '''
    cell.name = 'BiH'
    cell.lattice = lattice
    cell.latticeG = 2 * np.pi * LA.inv(lattice.T)
    cell.ions = np.array([
        [0, 0, 0],
        [0, 0, 0],
    ], dtype='float64')
    cell.ions_car = multi_dot([lattice, cell.ions.T]).T
    cell.N = 2
    cell.spec = ['C', 'C']

    '''
      r
    '''
    htb.r_Ramn[0, 0] = np.diagflat(wcc.T[0])
    htb.r_Ramn[0, 1] = np.diagflat(wcc.T[1])
    htb.r_Ramn[0, 2] = np.diagflat(wcc.T[2])

    htb.cell = cell

    return htb


def cal_berry_curvature(htb, kmesh):
    nk = kmesh.shape[0]
    BC = np.zeros([nk], dtype='float64')

    for i, k in zip(range(nk), kmesh):
        print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i+1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                k[0], k[1], k[2]
                )
              )

        BC[i] = htb.berry_curvature(k)

    return BC


def cal_band(htb, kpath):
    nk = kpath.shape[0]
    bandE = np.zeros([nk, htb.nw], dtype='float64')
    bandU = np.zeros([nk, htb.nw, htb.nw], dtype='complex128')
    # surfW = np.zeros([nk, nw], dtype='float64')

    for i, k in zip(range(nk), kpath):
        print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
                i+1, nk,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                k[0], k[1], k[2]
                )
              )
        eikr = np.exp(2j * np.pi * np.einsum('a,Ra', k, htb.R, optimize=True))
        hk = np.einsum('R,Rmn->mn', eikr, htb.hr_Rmn, optimize=True)
        E, U = LA.eigh(hk)
        bandE[i] = E - htb.fermi
        bandU[i] = U
        # surfW[i] = multi_dot([arc_weigh, np.real(U * np.conjugate(U))])

    return bandE, bandU


def plot_band(kpath_list, kpath, htb,  bandE,  bandU=None, eemin=-3.0, eemax=3.0, unit='D', plot_surf=False, plot_borden=False, save_csv=False):
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import subplot
    import pandas as pd

    cell = htb.cell
    nk, nw = bandE.shape

    if unit.upper() == 'C':
        kpath = multi_dot([cell.latticeG, kpath.T]).T
    kpath = kmold(kpath)
    nline = kpath_list.shape[0] - 1
    if save_csv:
        col_list = ['Band {}'.format(i+1) for i in range(nw)]
        bandpd = pd.DataFrame(bandE, index=np.arange(nk)+1, columns=col_list)
        bandpd.insert(loc=0, column='|k| (Ans-1)', value=kpath)
        bandpd.to_csv(r'band.csv', float_format='% 10.5f', sep='\t', encoding='utf-8')


    '''
      * plot band
    '''
    G = gridspec.GridSpec(1, 1)
    ax = subplot(G[0, 0])
    ax.axis([kpath.min(), kpath.max(), eemin, eemax])
    plt.axhline(0, color='k', linewidth=0.5)
    for i in range(1, nline):
        plt.axvline(x=kpath[i * nk//nline], linestyle='-', color='k', linewidth=0.5, alpha=1, zorder=2)

    for i in range(nw):
        ax.plot(kpath, bandE[:, i], linewidth=1, linestyle="-", color='k', alpha=0.7)

    # for i in range(76):
    #     ax.plot(kpath, bandE_soc[:, i], linewidth=0.5, linestyle="--", color='red', alpha=1)

    '''
      * plot surf weigh
    '''
    if plot_surf:
        nw_scell = nw
        nw_ucell = nw_scell // Nslab

        bandU = np.abs(bandU) ** 2
        bandU = np.array([
            np.sum(bandU[i].T.reshape(nw_scell, Nslab, nw_ucell), axis=2)
            for i in range(nk)
        ])
        is_surf = np.array([
            [
                np.fft.fft(bandU[ki, wi])[1].real
                for ki in range(nk)
            ]
            for wi in range(nw_scell)
        ]).flatten()
        is_surf = np.array([
            'red' if i > 0.3 else '#b0bec5'
            for i in is_surf
        ])
        X = np.kron(np.ones([nw_scell, 1]), kpath).flatten()
        Y = bandE.T.flatten()
        ax.scatter(X, Y, c=is_surf, s=50, alpha=0.7)
        # ax.clear()

    # '''
    #   * plot borden
    # '''
    # if plot_borden:
    #     k0 = np.array([0.0, 0, 0])
    #     v, vw, E, U, hk, hkk, Awk = get_fft001(k0)
    #     is_surf0 = get_is_surf(U, nw, nw//Nslab, Nslab, tolerance=0.3)
    #     W = get_adaptive_borden_II(v, E, U, dk, Nslab, open_boundary=1, a=1.0, minmalW=0.001) # / np.sqrt(Nslab)
    #
    #     ee = np.linspace(eemin, eemax, 2000)
    #     D = np.meshgrid(E, ee)
    #     D = D[1] - D[0]
    #     delta = (np.exp(-0.5 * (D / W) ** 2)/ np.sqrt(2*np.pi)/ W).T / 40
    #     # is_surf0 = np.ones_like(is_surf0)
    #     for i in range(nw):
    #         if is_surf0[i]:
    #             ax.plot(delta[i], ee, color='blue', linewidth=1, alpha=1)



if __name__ == '__main__':
    wdir = r'C:\Users\Jin\Research\NOR\BiH'
    os.chdir(wdir)

    Nslab = 1
    nk1 = 101
    transM = np.array([
        [1.,  1.,  0.],
        [0.,  2.,  0.],
        [0.,  0.,  1.],
    ])
    kpath_list = np.array([
        [ 0.0, 0.0, 0.0], # G
        [ 0.5, 0.0, 0.0], # M
        [ 1/3, 1/3, 0.0], # K
        [ 0.0, 0.0, 0.0], # G
    ])
    kpath_list = multi_dot([transM.T, kpath_list.T]).T

    kpath_list = np.array([
        [-0.5,  0.0,  0.0],
        [ 0.0,  0.0,  0.0],
        [ 0.5,  0.0,  0.0],
    ])
    kpath = make_kpath(kpath_list, nk1 - 1)

    # htb = h_BiH(vppx=1.9298, vpppi=-0.7725, tso=0.65)
    htb = h_QAH(A=1, B=1, M=5)
    # htb.cell.lattice = np.identity(3)

    bandE, bandU = cal_band(htb, kpath)
    plot_band(kpath_list, kpath, htb, bandE, bandU, eemin=-3.0, eemax=3.0, unit='D',
              plot_surf=False, plot_borden=False, save_csv=False)


    nk1 = 31
    nk2 = 31
    nk3 = 1
    KCUBE = [
        [0,  0,  0],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ]
    sample_method = 'G'
    kmesh = make_kmesh_dev001(nk1, nk2, nk3, sample_method, kcube=KCUBE)
    BC = cal_berry_curvature(htb, kmesh)
    vcell = LA.det(htb.cell.lattice[:2, :2])
    dimcoff = (2 * np.pi) / (nk1 * nk2) / LA.det(htb.cell.lattice[:2, :2])
    hall = dimcoff * np.sum(BC)
    print('Hall conductivity = {} (A.U.)'.format(hall))

