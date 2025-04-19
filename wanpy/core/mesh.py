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

__date__ = "Aug. 11, 2017"

import time
from typing import Union
import numpy as np
from numpy import linalg as LA

__all__ = [
    'make_ws_gridR',
    'make_mesh',
    'make_kpath',
    'kmold',
]

'''
  * R mesh
'''
def make_ws_gridR(ngridR, latt, info=True):
    # ***********
    # init
    # ***********
    a1 = latt.T[0]
    a2 = latt.T[1]
    a3 = latt.T[2]

    # ***********
    # main
    # ***********
    nR = 0  # -1
    ndegen = []
    gridR = []

    g_matrix = np.dot(np.array([a1, a2, a3]),
                      np.array([a1, a2, a3]).T)

    for n1 in range(-ngridR[0], ngridR[0] + 1):
        for n2 in range(-ngridR[1], ngridR[1] + 1):
            for n3 in range(-ngridR[2], ngridR[2] + 1):
                # Loop 125 R
                icnt = -1
                dist = np.zeros((125))
                for i1 in [-2, -1, 0, 1, 2]:
                    for i2 in [-2, -1, 0, 1, 2]:
                        for i3 in [-2, -1, 0, 1, 2]:
                            icnt += 1
                            ndiff = np.array([
                                n1 - i1 * ngridR[0],
                                n2 - i2 * ngridR[1],
                                n3 - i3 * ngridR[2]
                            ])
                            dist[icnt] = ndiff.dot(g_matrix).dot(ndiff)
                # print(dist)

                # dist_min = min(dist.tolist())
                dist_min = np.min(dist)
                if np.abs((dist[62] - dist_min)) < 10 ** -7:
                    # nrpts += 1
                    ndegen.append(0)
                    for i in range(0, 125):
                        if np.abs(dist[i] - dist_min) < 10 ** -7:
                            ndegen[nR] += 1
                    nR += 1

                    # irvec.append(n1 * a1 + n2 * a2 + n3 * a3)
                    gridR.append(np.array([n1, n2, n3]))

    ndegen = np.array(ndegen, dtype='int64')
    gridR = np.array(gridR, dtype='int64')
    # print('nrpts={}'.format(nrpts_s))
    # print('ndegen=\n', ndegen_s)
    # print('irvec=\n')
    # pp.pprint(irvec_s)
    if info:
        print('*=============================================================================*')
        print('|                                   R Grid                                     |')
        print('|    number of R Grid = {:4>}                                                  |'.format(nR))
        print('*=============================================================================*')
        for i in range(nR):
            print('|{: 4}). {: 3} {: 3} {: 3}   *{:2>}  '.format(i + 1, gridR[i, 0], gridR[i, 1], gridR[i, 2], ndegen[i]),
                  end='')
            if (i + 1) % 3 == 0:
                print('|')
        print('')
        print('*--------------------------------------------------------------------------------*')
    return nR, ndegen, gridR

'''
  * k mesh
'''
def make_mesh(
        nmesh,
        basis=np.eye(3),
        mesh_shift=0.,
        dtype='float64',
        mesh_type='Continuous',
        kmesh_type='Gamma',
        centersym=False,
        info=False
    ) -> Union[np.ndarray, str]:
    """
    Generates a mesh grid based on the provided parameters.

    Parameters:
    ----------
    nmesh : np.ndarray of int
        Number of mesh points in each dimension (N1, N2, N3).
    basis : np.ndarray, optional
        A 3x3 matrix defining the basis vectors for the mesh.
        Default is the identity matrix (unit cube).
    mesh_shift : np.ndarray of float, optional
        Shift applied to the mesh points. Default is 0.0.
    dtype : str, optional
        Data type of the mesh points. Default is 'float64'.
    mesh_type : str, optional
        Type of mesh ('continuous' or 'discrete'). Default is 'continuous'.
    kmesh_type : str, optional
        Type of BZ sampling ('Gamma' or 'MP'). Default is 'Gamma'.
    centersym : bool, optional
        If True, ensures the mesh is symmetric around the center.
        Only valid for odd numbers in `nmesh`. Default is False.
    info : bool, optional
        If True, prints timing information. Default is False.

    Returns:
    -------
    np.ndarray
        The generated mesh grid as an Nx3 array, where N = N1 * N2 * N3.
    str
        Error message.

    Notes:
    -----
    * type = continuous, discrete
    * centersym = False, True
    * basis = np.array([f1, f2, f3])

    `basis` is used to get custom shape of BZ,
    the original BZ is defined by `lattG`
        lattG = np.array([
            [b11, b21, b31],
            [b12, b22, b32],
            [b13, b23, b33],
        ])
    the new BZ is defined as
        lattG'[0] = f11 b1 + f12 b2 + f13 b3
        lattG'[1] = f21 b1 + f22 b2 + f23 b3
        lattG'[2] = f31 b1 + f32 b2 + f33 b3
    it is obtained by
        lattG' = lattG @ basis.T                (1)
    where basis is defined as
        basis.T = np.array([
            [f11, f21, f31],
            [f12, f22, f32],
            [f13, f23, f33],
        ])
    or
        basis = np.array([f1, f2, f3])          (2)

    """
    T0 = time.time()

    # Unpack nmesh
    N1, N2, N3 = nmesh
    N = N1 * N2 * N3

    # Create 1D ranges for each dimension
    rangN1 = np.arange(N1, dtype=dtype)
    rangN2 = np.arange(N2, dtype=dtype)
    rangN3 = np.arange(N3, dtype=dtype)
    onesN1 = np.ones(N1, dtype=dtype)
    onesN2 = np.ones(N2, dtype=dtype)
    onesN3 = np.ones(N3, dtype=dtype)

    # Create the full 3D mesh
    mesh = np.zeros([N, 3], dtype=dtype)
    mesh.T[0] = np.kron(rangN1, np.kron(onesN2, onesN3))
    mesh.T[1] = np.kron(onesN1, np.kron(rangN2, onesN3))
    mesh.T[2] = np.kron(onesN1, np.kron(onesN2, rangN3))

    # Handle center symmetry (centersym)
    if centersym:
        if not (np.mod(nmesh, 2) == 1).all():
            print('centersym mesh need odd number of [nmesh]')
            return 'centersym mesh need odd number of [nmesh]'
        else:
            mesh -= mesh[N // 2]

    # 'Gamma-centered mesh' or 'Monkhorst-Pack mesh'
    if kmesh_type[0].upper() == "M":
        mesh[:, 0] += (0 - N1) / 2
        mesh[:, 1] += (0 - N2) / 2
        # mesh[:, 2] += (0 - N3) / 2

    # 'Continuous' or 'Discrete'
    if mesh_type[0].upper() == "C":
        mesh /= np.array(nmesh, dtype=dtype)

    # Scale and shift the mesh
    if np.max(np.abs(basis - np.diag(np.diag(basis)))) < 1e-3:
        mesh *= np.diag(basis)
    else:
        # Full basis transformation
        # This will use large amount of memory
        mesh = (basis.T @ mesh.T).T
        mesh = np.ascontiguousarray(mesh)

    # Apply mesh shift
    mesh += np.array(mesh_shift)

    if info:
        print('Make mesh complited. Time consuming {} s'.format(time.time() - T0))

    return mesh

def make_kpath(kpath_list, nk1):

    # print('\n')
    # print("[from make_kpath] Attention : ")
    # print("[from make_kpath] kpath_list should in unit of b1 b2 b3")

    kpaths = [np.array(i) for i in kpath_list]
    kpaths_delta = [kpaths[i] - kpaths[i-1] for i in range(1,len(kpaths))]

    stage = len(kpaths_delta)

    kmesh_1d = np.zeros((nk1 * stage + 1, 3))
    i = 0
    for g_index in range(stage):   # g is high sym kpoint witch given by kpath_list
        for n1 in range(nk1):
            k = kpaths_delta[g_index] * n1/nk1 + kpaths[g_index]
            kmesh_1d[i] = k
            i = i + 1

    kmesh_1d[-1] = kpaths[-1]

    return kmesh_1d

def make_kpath_dev(kpath_HSP, nk1, keep_boundary=True):

    kpath_HSP = np.array(kpath_HSP, dtype='float64')
    n_HSLine = kpath_HSP.shape[0] - 1
    kpath = np.zeros([n_HSLine, nk1, 3], dtype='float64')

    for i in range(n_HSLine):
        k1 = kpath_HSP[i]
        k2 = kpath_HSP[i+1]
        kpath[i] = np.array([np.linspace(k1[i], k2[i], nk1+1)[:-1] for i in range(3)]).T

    kpath = kpath.reshape(n_HSLine * nk1, 3)
    if keep_boundary:
        kpath = np.vstack([kpath, kpath_HSP[-1]])

    return kpath

def make_kmesh_dev001(nk1, nk2, nk3, sample_method='G', kcube=None, info=True):
    T0 = time.time()
    # print('\n')
    # print("[from make_kmesh] Attention : ")
    # print("[from make_kmesh] kcube_in_bulk should in unit of b1 b2 b3")
    '''

    :param kcube:
            [[Original point of concerned Kcube],
             [1st vector to define this Kcube],
             [2ed vector to define this Kcube],
             [3th vector to define this Kcube]]

           sample_method:
                'MP'  or  'G'
    :return:
            kmesh_3d = np.array([
               [k1x, k1y, k1z],
               ...

            ])

    '''
    if kcube is None:
        kcube = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype='float64')
    else:
        kcube = np.array(kcube)
    if sample_method == 'G':
        pass
    elif sample_method == 'MP':
        kcube[0] += np.array([kcube[1,0] / 2 / nk1, kcube[2,1] / 2 / nk2, kcube[3,2] / 2 / nk3])
    base = kcube[1:]
    D = np.array([
        [n1, n2, n3]
        for n1 in range(nk1)
        for n2 in range(nk2)
        for n3 in range(nk3)
    ], dtype='float64')
    D[:,0] *= 1/nk1
    D[:,1] *= 1/nk2
    D[:,2] *= 1/nk3

    kmesh = np.dot(D, base) + kcube[0]

    if info:
        print('Sample BZ complited. Time consuming {} s'.format(time.time()-T0))
    return kmesh

# def make_gridR(N1, N2, N3):
#     N = N1 * N2 * N3
#     n2, n1, n3 = np.meshgrid(np.arange(N2), np.arange(N1), np.arange(N3))
#     gridR = np.array([n1.reshape(N), n2.reshape(N), n3.reshape(N)]).T
#     return gridR

def kmold(kkc):
    nk = kkc.shape[0]
    kkc = kkc[1:, :] - kkc[:-1, :]
    kkc = np.vstack([np.array([0, 0, 0]), kkc])
    k_mold = np.sqrt(np.einsum('ka,ka->k', kkc, kkc))
    k_mold = LA.multi_dot([np.tri(nk), k_mold])
    return k_mold

if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    import numpy.linalg as LA
    import time


    kplane_in_bulk = [
        [-5, -5, 0],
        [10, 0, 0],
        [0, 10, 0],
    ]
    kcube_in_bulk = np.array([
        [-0.5, -0.5, -0.5],
        [ 1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1],
    ], dtype='float64')
    nk1 = 5
    nk2 = 5
    nk3 = 1
    sample_method = 'MP'
    #ir_kpoints = make_kmesh_2d(kplane_in_bulk, nk1, nk2, sample_method='G')
    #ir_kpoints = make_kmesh_3d(kcube_in_bulk, nk1, nk2, nk3, sample_method='G')

    T0 = time.time()
    # kmesh = make_kmesh_dev001(nk1, nk2, nk3, sample_method, kcube_in_bulk)
    # print('Time consuming {} s'.format(time.time()-T0))
    #
    # plt.figure(figsize=(7, 7))
    # plt.axis([-1, 1, -1, 1])
    # print(kmesh)
    #
    # for k in kmesh:
    #     # if k[2] != 0: continue
    #     plt.scatter(k[0], k[1], s=200, c='#d81b60',
    #                 alpha=0.5,
    #                 marker='o',
    #                 )
    # plt.grid()
    # plt.show()

