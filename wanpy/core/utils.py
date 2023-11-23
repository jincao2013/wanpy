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

import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as scipy_rot

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

def get_ntheta_from_rotmatrix(TR, tau, rot_car):
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
        if np.isclose(rot_car, rot_car_test, atol=1e-05).all():
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