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

__date__ = "May. 10, 2021"

import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation as scipy_rot
from scipy.linalg import block_diag

from wanpy.core.units import *
from wanpy.core.bz import FD_zero, FD
import wanpy.response.response as res
from wanpy.response.response import commdot


# def get_pg_op(det, theta, nx, ny, nz):
#     n = LA.norm([nx, ny, nz])
#     Jx = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
#     Jy = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
#     Jz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
#     rot = np.exp(theta * (nx * Jx + ny * Jy + nz * Jz) / n)
#     rot = np.array([
#         []
#     ])
#     scipy_rot.from_rotvec(theta * )
#     return det * rot


class MPointGroup(object):

    def __init__(self):
        self.name = None
        self.latt = None
        self.lattG = None
        self.n_op = None
        self.elements = None
        self.op_axis = None
        self.op_fraction = None
        self.op_cartesian = None
        self.TR = None

    def get_op_cartesian(self):
        self.n_op = len(self.op_axis)
        self.op_cartesian = np.zeros([self.n_op, 3, 3], dtype='float64')
        self.TR = np.zeros([self.n_op], dtype='float64')
        for i in range(self.n_op):
            _TR, det, theta, nx, ny, nz = self.op_axis[i]
            n = np.array([nx, ny, nz]) / LA.norm(np.array([nx, ny, nz]))
            rot = scipy_rot.from_rotvec(theta * n)
            self.op_cartesian[i] = det * rot.as_matrix()
            self.TR[i] = _TR


