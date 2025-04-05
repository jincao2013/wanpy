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
# def make_mesh(nmesh, type='continuous', centersym=False, basis=np.identity(3), mesh_shift=0., info=False):
#     T0 = time.time()
#     '''
#     * type = continuous, discrete
#     * centersym = False, True
#     * basis = np.array([f1, f2, f3])
#
#     `basis` is used to get custom shape of BZ,
#     the original BZ is defined by `lattG`
#         lattG = np.array([
#             [b11, b21, b31],
#             [b12, b22, b32],
#             [b13, b23, b33],
#         ])
#     the new BZ is defined as
#         lattG'[0] = f11 b1 + f12 b2 + f13 b3
#         lattG'[1] = f21 b1 + f22 b2 + f23 b3
#         lattG'[2] = f31 b1 + f32 b2 + f33 b3
#     it is obtained by
#         lattG' = lattG @ basis.T                (1)
#     where basis is defined as
#         basis.T = np.array([
#             [f11, f21, f31],
#             [f12, f22, f32],
#             [f13, f23, f33],
#         ])
#     or
#         basis = np.array([f1, f2, f3])          (2)
#
#     '''  #
#     N1, N2, N3 = nmesh
#     N = N1 * N2 * N3
#     # n1, n2, n3 = np.meshgrid(np.arange(N1), np.arange(N2), np.arange(N3), indexing='ij')
#     n1, n2, n3 = np.mgrid[0:N1:1, 0:N2:1, 0:N3:1]
#     mesh = np.array([n1.flatten(), n2.flatten(), n3.flatten()], dtype='float64').T
#
#     if centersym:
#         if not (np.mod(nmesh, 2) == 1).all():
#             print('centersym mesh need odd number of [nmesh]')
#             return 'centersym mesh need odd number of [nmesh]'
#         else:
#             mesh -= mesh[N // 2]
#     if type[0].lower() == 'c':
#         mesh /= nmesh
#
#     # mesh = LA.multi_dot([basis.T, mesh.T]).T + mesh_shift
#     mesh = (basis.T @ mesh.T).T + mesh_shift
#     mesh = np.ascontiguousarray(mesh)
#     if info:
#         print('Make mesh complited. Time consuming {} s'.format(time.time()-T0))
#
#     return mesh


def make_mesh(nmesh, basis=np.identity(3), diagbasis=True, mesh_shift=0.,
              mesh_dtype='float64', mesh_type='continuous', centersym=False,
              info=False
              ):
    T0 = time.time()
    '''
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

    '''  #
    N1, N2, N3 = nmesh
    N = N1 * N2 * N3
    # mesh_dtype = 'float64'
    rangN1 = np.arange(N1, dtype=mesh_dtype)
    rangN2 = np.arange(N2, dtype=mesh_dtype)
    rangN3 = np.arange(N3, dtype=mesh_dtype)
    onesN1 = np.ones(N1, dtype=mesh_dtype)
    onesN2 = np.ones(N2, dtype=mesh_dtype)
    onesN3 = np.ones(N3, dtype=mesh_dtype)
    mesh = np.zeros([N, 3], dtype=mesh_dtype)
    mesh.T[0] = np.kron(rangN1, np.kron(onesN2, onesN3))
    mesh.T[1] = np.kron(onesN1, np.kron(rangN2, onesN3))
    mesh.T[2] = np.kron(onesN1, np.kron(onesN2, rangN3))

    if centersym:
        if not (np.mod(nmesh, 2) == 1).all():
            print('centersym mesh need odd number of [nmesh]')
            return 'centersym mesh need odd number of [nmesh]'
        else:
            mesh -= mesh[N // 2]
    if mesh_type[0].lower() == 'c':
        mesh /= nmesh

    # scales and shift the mesh
    if diagbasis:
        mesh *= np.diag(basis)
    else:
        # this will use large amount of memory
        mesh = (basis.T @ mesh.T).T
        mesh = np.ascontiguousarray(mesh)
    mesh += mesh_shift

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


def make_kmesh_2d(kplane_in_bulk,nk1,nk2,sample_method='G'):

    # print('\n')
    # print("[from make_kmesh_2d] Attention : ")
    # print("[from make_kmesh_2d] kplane_in_bulk should in unit of b1 b2 b3")
    '''

    :param kplane_in_bulk:
            [[Original point of concerned Kplane],
             [1st vector to define this Kplane],
             [2ed vector to define this Kplane]]

           sample_method:
                'MP'  or  'G'
    :return:
            kmesh_2d = np.array([
               [k1x, k1y, k1z],
               ...

            ])
    '''
    s0 = np.array(kplane_in_bulk[0])
    b1 = np.array(kplane_in_bulk[1])
    b2 = np.array(kplane_in_bulk[2])

    if sample_method == 'MP':
        kmesh_2d = np.zeros((nk1*nk2, 3))
        i = 0
        for n1 in range(nk1):
            for n2 in range(nk2):
                k = b1 * (n1+0.5)/nk1 + b2 * (n2+0.5)/nk2 + s0
                kmesh_2d[i] = k
                i = i + 1
    elif sample_method == 'G':
        kmesh_2d = np.zeros((nk1*nk2, 3))
        i = 0
        for n1 in range(nk1):
            for n2 in range(nk2):
                k = b1 * n1/nk1 + b2 * n2/nk2 + s0
                kmesh_2d[i] = k
                i = i + 1

    print('kmesh_2d complited')
    return kmesh_2d

def make_kmesh_3d(kcube_in_bulk,nk1,nk2,nk3,sample_method='G'):

    # print('\n')
    # print("[from make_kmesh_3d] Attention : ")
    # print("[from make_kmesh_3d] kcube_in_bulk should in unit of b1 b2 b3")
    '''

    :param kcube_in_bulk:
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
    s0 = np.array(kcube_in_bulk[0])
    base1 = np.array(kcube_in_bulk[1])
    base2 = np.array(kcube_in_bulk[2])
    base3 = np.array(kcube_in_bulk[3])

    if sample_method == 'MP':
        kmesh_3d = np.zeros((nk1*nk2*nk3,3))
        i = 0
        for n1 in range(nk1):
            for n2 in range(nk2):
                for n3 in range(nk3):
                    k = base1 * (n1+0.5)/nk1 + base2 * (n2+0.5)/nk2 + base3 * (n3+0.5)/nk3 + s0
                    kmesh_3d[i] = k
                    i = i + 1
    elif sample_method == 'G':
        kmesh_3d = np.zeros((nk1*nk2*nk3,3 ))
        i = 0
        for n1 in range(nk1):
            for n2 in range(nk2):
                for n3 in range(nk3):
                    k = base1 * n1/nk1 + base2 * n2/nk2 + base3 * n3/nk3 + s0
                    kmesh_3d[i] = k
                    i = i + 1
    print('kmesh_3d complited')
    return kmesh_3d


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


def make_kmesh_dev002(nk1, nk2, nk3, sample_method='G', kcube=None, info=True):
    T0 = time.time()
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
    Nk = nk1 * nk2 * nk3
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

    n2, n1, n3 = np.meshgrid(np.arange(nk2), np.arange(nk1), np.arange(nk3))
    D = np.array([n1.reshape(Nk), n2.reshape(Nk), n3.reshape(Nk)], dtype='float64').T
    D[:,0] /= nk1
    D[:,1] /= nk2
    D[:,2] /= nk3

    kmesh = np.dot(D, base) + kcube[0]

    if info:
        print('Sample BZ complited. Time consuming {} s'.format(time.time()-T0))
    return kmesh


def make_kmesh(nk1, nk2, nk3, sample_method='G', kcube=None, info=True):
    return make_kmesh_dev002(nk1, nk2, nk3, sample_method, kcube, info)


def make_kmesh_3dmgrid(kcube_in_bulk,nk1,nk2,nk3,sample_method='G'):

    # print("[from make_kmesh_3d] Attention : ")
    # print("[from make_kmesh_3d] kplane_in_bulk should in unit of b1 b2 b3")
    '''

    :param kcube_in_bulk:
            [[Original point of concerned Kcube],
             [1st vector to define this Kcube],
             [2ed vector to define this Kcube],
             [3th vector to define this Kcube]]

           sample_method:
                'MP'  or  'G'
    :return:
            kmesh_3d = np.mgrid format
    '''
    s0 = np.array(kcube_in_bulk[0])
    base1 = np.array(kcube_in_bulk[1])
    base2 = np.array(kcube_in_bulk[2])
    base3 = np.array(kcube_in_bulk[3])

    base1 = base1 - s0
    base2 = base2 - s0
    base3 = base3 - s0

    if sample_method == 'G':
        kmesh_3d_i = np.zeros((nk1, nk2, nk3))
        kmesh_3d_j = np.zeros((nk1, nk2, nk3))
        kmesh_3d_k = np.zeros((nk1, nk2, nk3))

        i = 0
        for n1 in range(nk1):
            for n2 in range(nk2):
                for n3 in range(nk3):
                    k = base1 * (n1+0.5)/nk1 + \
                        base2 * (n2+0.5)/nk2 + \
                        base3 * (n3+0.5)/nk3
                    kmesh_3d_i[n1][n2][n3] = k[0]
                    kmesh_3d_j[n1][n2][n3] = k[1]
                    kmesh_3d_k[n1][n2][n3] = k[2]
                    i = i + 1
    elif sample_method == 'MP':
        kmesh_3d_i = np.zeros((nk1+1, nk2+1, nk3+1))
        kmesh_3d_j = np.zeros((nk1+1, nk2+1, nk3+1))
        kmesh_3d_k = np.zeros((nk1+1, nk2+1, nk3+1))
        i = 0
        for n1 in range(nk1 + 1):
            for n2 in range(nk2 + 1):
                for n3 in range(nk3 + 1):
                    k = base1 * n1/nk1 + \
                        base2 * n2/nk2 + \
                        base3 * n3/nk3
                    kmesh_3d_i[n1][n2][n3] = k[0]
                    kmesh_3d_j[n1][n2][n3] = k[1]
                    kmesh_3d_k[n1][n2][n3] = k[2]
                    i = i + 1

    return kmesh_3d_i, kmesh_3d_j, kmesh_3d_k

def mgrid_CUBE(n1, n2, n3, lattice, sample_method='MP', cube=None):
    if cube is None:
        cube = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype='float64')
    else:
        cube = np.array(cube)
    if sample_method == 'MP':
        cube[0] += np.array([cube[1,0] / 2 / n1, cube[2,1] / 2 / n2, cube[3,2] / 2 / n3])
    D = np.array([
        [_n1, _n2, _n3]
        for _n1 in range(n1)
        for _n2 in range(n2)
        for _n3 in range(n3)
    ], dtype='float64')
    D[:,0] *= 1/n1
    D[:,1] *= 1/n2
    D[:,2] *= 1/n3

    base = cube[1:]
    mesh = np.dot(D, base) + cube[0]

    meshc = np.einsum('ij,nj->ni', lattice, mesh, optimize=True)
    x = meshc[:,0].reshape(n1, n2, n3)
    y = meshc[:,1].reshape(n1, n2, n3)
    z = meshc[:,2].reshape(n1, n2, n3)

    return x, y, z


def make_gridR(N1, N2, N3):
    N = N1 * N2 * N3
    n2, n1, n3 = np.meshgrid(np.arange(N2), np.arange(N1), np.arange(N3))
    gridR = np.array([n1.reshape(N), n2.reshape(N), n3.reshape(N)]).T
    return gridR


def kmold(kkc):
    nk = kkc.shape[0]
    kkc = kkc[1:, :] - kkc[:-1, :]
    kkc = np.vstack([np.array([0, 0, 0]), kkc])
    k_mold = np.sqrt(np.einsum('ka,ka->k', kkc, kkc))
    k_mold = LA.multi_dot([np.tri(nk), k_mold])
    return k_mold

if __name__ == '__main__':
    import matplotlib.pyplot as plt
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
    kmesh = make_kmesh_dev001(nk1, nk2, nk3, sample_method, kcube_in_bulk)
    print('Time consuming {} s'.format(time.time()-T0))

    plt.figure(figsize=(7, 7))
    plt.axis([-1, 1, -1, 1])
    print(kmesh)

    for k in kmesh:
        # if k[2] != 0: continue
        plt.scatter(k[0], k[1], s=200, c='#d81b60',
                    alpha=0.5,
                    marker='o',
                    )
    plt.grid()
    plt.show()

