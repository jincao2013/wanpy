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

__date__ = "Nov. 9, 2019"

import sys
import numpy as np
import numpy.linalg as LA
from wanpy.core.structure import Htb
from wanpy.core.mesh import make_mesh

__all__ = [
    'init_kmesh',
    'init_htb_response_data',
]

def init_kmesh(MPI, nkmesh, random_k=False, type='continuous', centersym=False,
               diagbasis=True, kcube=np.identity(3), kmesh_shift=0.,
               mesh_dtype='float64'
               ):
    COMM = MPI.COMM_WORLD
    MPI_rank = COMM.Get_rank()
    if MPI_rank == 0:
        kmesh = make_mesh(nkmesh, basis=kcube, diagbasis=diagbasis, mesh_shift=kmesh_shift,
                          mesh_dtype=mesh_dtype, mesh_type=type, centersym=centersym,
                          info=True)
        if random_k:
            np.random.shuffle(kmesh)
            print('complite shuffle kmesh. (Speed up --> valley type <-- band)')
    else:
        kmesh = None

    return kmesh

def init_htb_response_data(MPI, htb, tmin_h=-0.1, tmin_r=-0.1, open_boundary=-1, istb=False, use_wcc=False, atomic_wcc=False):
    COMM = MPI.COMM_WORLD
    MPI_rank = COMM.Get_rank()
    MPI_ncore = COMM.Get_size()
    '''
      * bcast 
        nw, fermi, cell
        nR_hr, nR_r, R_hr, R_r, hr_Rmn, r_Ramn
        Rc_hr, Rc_r
        kps
    ''' #
    if MPI_rank == 0:

        # htb = Htb()
        # htb.load_htb(htb_fname)

        htb.setup()

        # this will replace htb.wcc and htb.wccf with exact atomic positions
        if atomic_wcc: htb.use_atomic_wcc()

        htb.printer()

        # by setting use_wcc=True will replace diagonal part of r_Ramn[nR//2] with htb.wcc
        nR_hr, nR_r, R_hr, R_r, hr_Rmn, r_Ramn = htb.reduce_htb(tmin=tmin_h, tmin_r=tmin_r, tb=istb, use_wcc=use_wcc, open_boundary=open_boundary)
        nw = htb.nw

        Rc_hr = np.zeros([nR_hr, 3], dtype='float64')
        Rc_r = np.zeros([nR_r, 3], dtype='float64')
        Rc_hr += LA.multi_dot([htb.cell.lattice, R_hr.T]).T
        Rc_r += LA.multi_dot([htb.cell.lattice, R_r.T]).T

        print('')
        print('                               -----------------------')
        print('                               Reduced R Grid (hr_Rmn)')
        print('                               -----------------------')
        print('')
        htb.print_RGrid(R_hr, np.ones(R_hr.shape[0]))

        # object larger than 4Gb cannot bcast
        htb.hr_Rmn = None
        htb.r_Ramn = None
        htb.R_hr = None
        htb.R_r = None
        htb.Rc_hr = None
        htb.Rc_r = None
    else:
        nw, nR_hr, nR_r = None, None, None
        htb = Htb()
        R_hr, R_r, Rc_hr, Rc_r = None, None, None, None
        hr_Rmn, r_Ramn = None, None

    nw = COMM.bcast(nw, root=0)
    htb = COMM.bcast(htb, root=0)
    nR_hr = COMM.bcast(nR_hr, root=0)
    nR_r = COMM.bcast(nR_r, root=0)

    if MPI_rank != 0:
        R_hr = np.zeros([nR_hr, 3], dtype='int64')
        R_r = np.zeros([nR_r, 3], dtype='int64')
        Rc_hr = np.zeros([nR_hr, 3], dtype='float64')
        Rc_r = np.zeros([nR_r, 3], dtype='float64')
        hr_Rmn = np.zeros([nR_hr, nw, nw], dtype='complex128')
        r_Ramn = np.zeros([nR_r, 3, nw, nw], dtype='complex128')

    '''
      * htb_response object
    '''
    COMM.Bcast(R_hr, root=0)
    COMM.Bcast(R_r, root=0)
    COMM.Bcast(Rc_hr, root=0)
    COMM.Bcast(Rc_r, root=0)
    COMM.Bcast(hr_Rmn, root=0)
    COMM.Bcast(r_Ramn, root=0)
    COMM.Barrier()

    htb.hr_Rmn = hr_Rmn
    htb.r_Ramn = r_Ramn
    htb.R_hr = R_hr
    htb.R_r = R_r
    htb.Rc_hr = Rc_hr
    htb.Rc_r = Rc_r

    COMM.Barrier()
    sys.stdout.flush()

    return htb
