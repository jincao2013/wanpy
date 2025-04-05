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

import re
import math
import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as scipy_rot

__all__ = [
    # Matrix operations
    'commdot',
    'anticommdot',
    'esum',
    # Symmetry functions
    'get_op_cartesian',
    'get_ntheta_from_rotmatrix',
    'print_symmops',
    # I/O
    'wannier90_read_rr',
    'wannier90_read_hr',
    'wannier90_load_wcc',
    'wannier90_load_wsvec',
    'wannier90_read_spin',
    'wannier90_read_rr_v2x',
    # Criterion
    'check_valid_symmops',
    'wanpy_check_if_uudd_amn',
    'get_random_unitary_matrix',
]

"""
  * Matrix operations
"""
commdot = lambda A, B: A @ B - B @ A
anticommdot = lambda A, B: A @ B + B @ A

def esum(subscripts, *operands, optimize=True, **kwargs):
    return np.einsum(subscripts, *operands, optimize=optimize, **kwargs)

"""
  Symmetry functions
"""
def get_op_cartesian(symmop):
    """
    Get O(3) rotation matrix from symmop
    """
    TR, det, theta, nx, ny, nz, taux, tauy, tauz = symmop
    axis = np.array([nx, ny, nz]) / LA.norm(np.array([nx, ny, nz]))
    rot = scipy_rot.from_rotvec(theta * axis)
    op_cartesian = det * rot.as_matrix()
    op_cartesian[np.abs(op_cartesian)<1e-10] = 0
    return op_cartesian

def get_ntheta_from_rotmatrix(TR, tau, rot_car, atol=1e-5):
    det = LA.det(rot_car)
    u, v = LA.eig(det * rot_car)

    axis = np.real(v.T[np.argwhere(np.isclose(u, 1))[0,0]])
    _theta = np.arccos((np.trace(det * rot_car) - 1)/2) # [0, pi]
    nx, ny, nz = axis
    taux, tauy, tauz = tau

    symmop = None
    for theta in [_theta, 2*np.pi-_theta]:
        rot_car_test = det * scipy_rot.from_rotvec(theta * axis).as_matrix()
        rot_car_test[np.abs(rot_car_test) < 1e-10] = 0
        if np.isclose(rot_car, rot_car_test, atol=atol).all():
            symmop = [TR, det, theta, nx, ny, nz, taux, tauy, tauz]
            break

    # if symmop is None:
    #     print('error in get_ntheta_from_rotmatrix', det, theta/np.pi*180, axis)
    # else:
    #     print('get_ntheta_from_rotmatrix', det, theta/np.pi*180, axis)
    # print(symmop)

    return symmop

def print_symmops(symmops):
    n_operations = symmops.shape[0]
    print('In axis-angle form:\n')
    print('   TR  det      alpha   alpha(deg)         nx        ny         nz       taux       tauy       tauz')
    for i in range(n_operations):
        _TR, _det, _theta, _nx, _ny, _nz, _taux, _tauy, _tauz = symmops[i]
        print('{:>5d}{:>5d}{:11.6f} ({:6.0f} deg){:11.6f}{:11.6f}{:11.6f}{:11.6f}{:11.6f}{:11.6f}'.format(
            int(_TR), int(np.rint(_det)), _theta, _theta/np.pi*180, _nx, _ny, _nz, _taux, _tauy, _tauz
        ))
    print('')
    print('* TR = 0 (without TR) or TR = 1 (with TR)')

"""
  I/O functions
"""
def wannier90_read_hr(fname):
    with open(fname, 'r') as hr_file:
        hr_file.readline()

        nw = int(hr_file.readline().strip())
        nR = int(hr_file.readline().strip())

        ndegen = np.zeros(nR, dtype='int64')
        index = 0
        for i in range(math.ceil(nR / 15)):
            for j in hr_file.readline().split():
                ndegen[index] = int(j)
                index += 1

        R = np.zeros((nR, 3), dtype='int64')
        hr_Rmn = np.zeros((nR, nw, nw), dtype='complex128')
        for nrpts_i in range(nR):

            for m in range(nw):
                for n in range(nw):
                    inline = hr_file.readline().split()
                    inline_m = int(inline[3]) - 1
                    inline_n = int(inline[4]) - 1
                    hr_Rmn[nrpts_i, inline_m, inline_n] = complex(float(inline[5]), float(inline[6]))
            R[nrpts_i] = np.array(inline[:3], dtype='int64')

    return nw, nR, ndegen, R, hr_Rmn

def wannier90_read_rr(fname):
    with open(fname, 'r') as r_file:
        r_file.readline()
        nw = int(r_file.readline().strip())
        nR = int(r_file.readline().strip())

        R = np.zeros((nR, 3), dtype='int64')
        r_Ramn = np.zeros((nR, 3, nw, nw), dtype='complex128')
        for nrpts_i in range(nR):

            for m in range(nw):
                for n in range(nw):
                    inline = r_file.readline().split()
                    inline_m = int(inline[3]) - 1
                    inline_n = int(inline[4]) - 1
                    r_Ramn[nrpts_i, 0, inline_m, inline_n] = complex(float(inline[5]), float(inline[6]))
                    r_Ramn[nrpts_i, 1, inline_m, inline_n] = complex(float(inline[7]), float(inline[8]))
                    r_Ramn[nrpts_i, 2, inline_m, inline_n] = complex(float(inline[9]), float(inline[10]))

            R[nrpts_i] = np.array(inline[:3], dtype='int64')

    return r_Ramn

def wannier90_read_spin(fname):
    with open(fname, 'r') as f:
        f.readline()
        nw = int(f.readline().strip())
        nR = int(f.readline().strip())

        R = np.zeros([nR, 3], dtype='int64')
        spin0_Rmn = np.zeros([nR, nw, nw], dtype='complex128')
        spin_Ramn = np.zeros([nR, 3, nw, nw], dtype='complex128')
        for nrpts_i in range(nR):
            for m in range(nw):
                for n in range(nw):
                    inline = f.readline().split()
                    inline_m = int(inline[3]) - 1
                    inline_n = int(inline[4]) - 1
                    spin0_Rmn[nrpts_i, inline_m, inline_n] = complex(float(inline[5]), float(inline[6]))
                    spin_Ramn[nrpts_i, 0, inline_m, inline_n] = complex(float(inline[7]), float(inline[8]))
                    spin_Ramn[nrpts_i, 1, inline_m, inline_n] = complex(float(inline[9]), float(inline[10]))
                    spin_Ramn[nrpts_i, 2, inline_m, inline_n] = complex(float(inline[11]), float(inline[12]))

            R[nrpts_i] = np.array(inline[:3], dtype='int64')

    return spin0_Rmn, spin_Ramn

def wannier90_read_rr_v2x(fname, nR):
    """
     interface with wannier90 v2.x
    """
    with open(fname, 'r') as r_file:
        r_file.readline()
        nw = int(r_file.readline().strip())

        R = np.zeros((nR, 3), dtype='int64')
        r_Ramn = np.zeros((nR, 3, nw, nw), dtype='complex128')
        for nrpts_i in range(nR):

            for m in range(nw):
                for n in range(nw):
                    inline = r_file.readline().split()
                    inline_m = int(inline[3]) - 1
                    inline_n = int(inline[4]) - 1
                    r_Ramn[nrpts_i, 0, inline_m, inline_n] = complex(float(inline[5]), float(inline[6]))
                    r_Ramn[nrpts_i, 1, inline_m, inline_n] = complex(float(inline[7]), float(inline[8]))
                    r_Ramn[nrpts_i, 2, inline_m, inline_n] = complex(float(inline[9]), float(inline[10]))

            R[nrpts_i] = np.array(inline[:3], dtype='int64')

    return r_Ramn

def wannier90_load_wcc(fname, shiftincell=False, check_if_uudd_amn=True):
    lattice = np.zeros((3, 3), dtype='float64')
    with open(fname, 'r') as f:
        inline = f.readline()
        while 'Lattice Vectors' not in inline:
            inline = f.readline()
        lattice[:, 0] = np.array(re.findall(r'.\d+\.\d+', f.readline()), dtype='float64')
        lattice[:, 1] = np.array(re.findall(r'.\d+\.\d+', f.readline()), dtype='float64')
        lattice[:, 2] = np.array(re.findall(r'.\d+\.\d+', f.readline()), dtype='float64')
        while 'Number of Wannier Functions' not in inline:
            inline = f.readline()
        nw = int(re.findall(r'\d+', inline)[0])
        wcc = np.zeros((nw, 3), dtype='float64')
        wbroaden = np.zeros(nw, dtype='float64')
        while inline != ' Final State\n':
            inline = f.readline()
        for i in range(nw):
            inline = np.array(re.findall(r'.\d+\.\d+', f.readline()), dtype='float64')
            wcc[i] = inline[:3]
            wbroaden[i] = inline[-1]
    f.close()

    wccf = LA.multi_dot([LA.inv(lattice), wcc.T]).T
    if shiftincell:
        wccf = np.remainder(wccf, np.array([1, 1, 1]))
        wcc = LA.multi_dot([lattice, wccf.T]).T

    if check_if_uudd_amn:
        if_uudd_amn = wanpy_check_if_uudd_amn(wcc, wbroaden)
        if not if_uudd_amn:
            print('\033[0;31m.amn is not in uudd order \033[0m')
            print('\033[0;31mone may use twist_amn to reorganize the .amn file into the uudd order, and then perform the disentanglement process again.  \033[0m')
        else:
            print("\033[92m.amn is in uudd order\033[0m")
    return wcc, wccf, wbroaden

def wannier90_load_wsvec(fname, nw, nR):
    max_ndegenT = 8  # max number of unit cells that can touch

    R = np.zeros([nR, 3], dtype='int64')
    ndegenT = np.zeros([nR, nw, nw], dtype='int64')
    invndegenT = np.zeros([max_ndegenT, nR, nw, nw], dtype='float64')
    wsvecT = np.zeros([max_ndegenT, nR, nw, nw, 3], dtype='int64')

    with open(fname, 'r') as f:
        f.readline()
        for iR in range(nR):
            for wm in range(nw):
                for wn in range(nw):
                    inline = np.array(f.readline().split(), dtype='int64')
                    m, n = inline[3:] - 1
                    R[iR] = inline[:3]
                    _ndegenT = int(f.readline())
                    ndegenT[iR, m, n] = _ndegenT
                    invndegenT[:_ndegenT, iR, m, n] = 1 / _ndegenT
                    for j in range(_ndegenT):
                        wsvecT[j, iR, m, n] = np.array(f.readline().split(), dtype='int64')

    max_ndegenT = np.max(ndegenT)
    wsvecT = wsvecT[:max_ndegenT]
    invndegenT = invndegenT[:max_ndegenT]

    return invndegenT, wsvecT

"""
  Criterion
"""
def check_valid_symmops(symmops):
    assert set([int(i) for i in symmops.T[0]]) <= {0, 1}
    assert set([int(np.rint(i)) for i in symmops.T[1]]) <= {-1, 1}

def wanpy_check_if_uudd_amn(wcc, wbroaden, info=False):
    nw = wcc.shape[0]

    # test uudd
    _wcc = wcc.reshape([2, nw // 2, 3])
    _wbroaden = wbroaden.reshape([2, nw // 2])
    distance_uudd = np.sum((_wcc[1] - _wcc[0]) ** 2) / nw
    maxdiff_broden_uudd = np.max(np.abs(_wbroaden[1] - _wbroaden[0]))

    # test udud
    _wcc = wcc.reshape([nw // 2, 2, 3])
    _wbroaden = wbroaden.reshape([nw // 2, 2])
    distance_udud = np.sum((_wcc[:,1,:] - _wcc[:,0,:]) ** 2) / nw
    maxdiff_broden_udud = np.max(np.abs(_wbroaden[:,1] - _wbroaden[:,0]))

    if info:
        print('distance_uudd:{:14.6f},    maxdiff_broden_uudd:{:14.6f}'.format(distance_uudd, maxdiff_broden_uudd))
        print('distance_udud:{:14.6f},    maxdiff_broden_udud:{:14.6f}'.format(distance_udud, maxdiff_broden_udud))

    if_uudd_amn = (distance_udud > 0.1 > distance_uudd) or (maxdiff_broden_udud > 0.1 > maxdiff_broden_uudd)
    return if_uudd_amn

def get_random_unitary_matrix(N):
    h = np.random.random([N, N]) + 1j * np.random.random([N, N])
    h = h + h.conj().T
    w, v = LA.eigh(h)
    return v