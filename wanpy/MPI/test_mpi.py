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

__date__ = "Nov.3, 2019"

import os
import sys
sys.path.append(os.environ.get('PYTHONPATH'))

import numpy as np
from wanpy.core.mesh import make_kmesh_dev001

from mpi4py import MPI
from wanpy.MPI.MPI import MPI_Reduce, MPI_Gather

'''
  * Calculators
'''
@MPI_Reduce(MPI, iterprint=1)
def calculator_reduce(k, dim):
    array1 = np.linspace(0, 1, ne)
    array2 = np.linspace(1, 2, ne)
    return array1, array2


@MPI_Gather(MPI, iterprint=1)
def calculator_gather(k, dim):
    array1 = np.linspace(0, 1, ne)
    array2 = np.linspace(1, 2, ne)
    return array1, array2


if __name__ == '__main__':
    MPI_COMM = MPI.COMM_WORLD
    MPI_RANK = MPI_COMM.Get_rank()
    MPI_NCORE = MPI_COMM.Get_size()

    ne = 10
    dim = [2, ne]

    kmesh = make_kmesh_dev001(12, 12, 1, info=False)
    NK = kmesh.shape[0]

    ARRAY1, ARRAY2 = calculator_reduce(kmesh, dim)
    ARRAY3, ARRAY4 = calculator_gather(kmesh, dim)


    if MPI_RANK == 0:
        print(ARRAY2)

