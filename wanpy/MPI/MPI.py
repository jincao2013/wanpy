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

__date__ = "Nov.3, 2019"

import time
import os
import sys
sys.path.append(os.environ.get('PYTHONPATH'))

import functools
import numpy as np
import numpy.linalg as LA

__all__ = [
    'MPI_Reduce',
    'MPI_Gather',
    'get_kgroups',
    'parprint',
]

'''
  * KPar decorator
'''
def MPI_Reduce(MPI, iterprint=100, dtype='float64', mpinote=False):
    def decorator(func):
        @functools.wraps(func)
        def par_cal(kps, dim, *args, **kwargs):
            COMM = MPI.COMM_WORLD
            MPI_rank = COMM.Get_rank()
            MPI_ncore = COMM.Get_size()
            MPI_main = not MPI_rank

            if MPI_main:
                kgps = get_kgroups(kps, MPI_ncore, mode='distributeF')
                nk_list = [i.shape[0] for i in kgps]
                # print(nk_list)
            else:
                kgps, nk_list = None, None

            # list_nk_per_core = COMM.bcast(nk_list, root=0)
            nk_list = COMM.bcast(nk_list, root=0)

            COMM.Barrier()

            '''
              * Send kpoints to all cores
            '''
            if MPI_main:
                kps_rank_i = np.ascontiguousarray(kgps[0])
                for i in range(1, MPI_ncore):
                    COMM.Send(np.ascontiguousarray(kgps[i]), dest=i)
            else:
                kps_rank_i = np.zeros([nk_list[MPI_rank], 3], dtype='float64')
                COMM.Recv(kps_rank_i, source=0)

            COMM.Barrier()
            sys.stdout.flush()

            if MPI_main:
                kgps = None
                print('MPI_Reduce: Distributed kmesh to all cores')

            sys.stdout.flush()

            '''
              * kpar run 
            '''
            result_kgp = np.zeros(dim, dtype=dtype)
            i = 0
            for k in kps_rank_i:
                result_kgp += np.array(func(k, dim, *args, **kwargs))
                i += 1
                if mpinote:
                    pass
                if i % iterprint != 0: continue
                print('[Rank {:<4d} {:>6d}/{:<6d}] {} Calculated k at ({:.5f} {:.5f} {:.5f})'.format(
                    MPI_rank, i, nk_list[MPI_rank],
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    k[0], k[1], k[2]
                    )
                )

            result = np.zeros(dim, dtype=dtype) if MPI_main else None
            COMM.Reduce(sendbuf=result_kgp, recvbuf=result, op=MPI.SUM, root=0)
            COMM.Barrier()
            return result if MPI_main else None

        return par_cal
    return decorator

def MPI_Gather(MPI, iterprint=100, dtype='float64', mpinote=False):
    def decorator(func):
        @functools.wraps(func)
        def par_cal(kps, dim, *args, **kwargs):
            COMM = MPI.COMM_WORLD
            MPI_rank = COMM.Get_rank()
            MPI_ncore = COMM.Get_size()
            MPI_main = not MPI_rank

            if MPI_main:
                kgps = get_kgroups(kps, MPI_ncore, mode='distributeC')
                nk_list = [i.shape[0] for i in kgps]     # num_of_k_per_core
                sendcounts = [_nk * np.prod(dim) for _nk in nk_list]     # num_of_k_per_core
                # print(nk_list)
            else:
                kgps, nk_list, sendcounts = None, None, None

            nk_list = COMM.bcast(nk_list, root=0)
            sendcounts = COMM.bcast(sendcounts, root=0)

            COMM.Barrier()

            '''
              * Send kpoints to all cores
            '''
            if MPI_main:
                kps_rank_i = np.ascontiguousarray(kgps[0])
                for i in range(1, MPI_ncore):
                    COMM.Send(buf=np.ascontiguousarray(kgps[i]), dest=i)
            else:
                kps_rank_i = np.zeros([nk_list[MPI_rank], 3], dtype='float64')
                COMM.Recv(buf=kps_rank_i, source=0)

            COMM.Barrier()
            sys.stdout.flush()

            if MPI_main:
                kgps = None
                print('MPI_Gather: Distributed kmesh to all cores')

            sys.stdout.flush()

            '''
              * kpar run 
            '''
            sendbuf = np.zeros([nk_list[MPI_rank]] + dim, dtype=dtype)
            i = 0
            for k in kps_rank_i:
                sendbuf[i] = np.array(func(k, dim, *args, **kwargs))
                i += 1
                if mpinote:
                    pass
                if i % iterprint != 0: continue
                print('[Rank {:<4d} {:>6d}/{:<6d}] {} Calculated k at ({:.5f} {:.5f} {:.5f})'.format(
                    MPI_rank, i, nk_list[MPI_rank],
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    k[0], k[1], k[2]
                    )
                )

            recvbuf = np.zeros([sum(nk_list)] + dim, dtype=dtype) if MPI_main else None
            COMM.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=0)
            COMM.Barrier()
            return recvbuf if MPI_main else None

        return par_cal
    return decorator

'''
  * Par toolkits
'''
def get_kgroups(kps, Ncore, mode='distributeF'):
    """
      :param kps: Jobs list in shape of (Nk, ...)
      :param Ncore: Number of cores to excute jobs
      :return kgps: a group of job list

      * distributeF mode (default)
        distribute jobs along core index (Fortran like)
        e.g.
        [1, 3, 5, 7, 9]
        [2, 4, 6, 8]
        This mode have better resouce balance.

      * distributeC mode
        distribute jobs along job index (C like)
        e.g.
        [1, 2, 3, 4, 5]
        [6, 7, 8, 9]
        Choose this mode if you wish the cpu excute your
        jobs in original seqence.

      * cut mode
        In this mode, the jobs list(kps) are cut into Ncore number
        of kpoints groups in the same seqence of kps.
        The jobs will be excuted in original seqence, however, the
        last core may have few number of jobs.
    """
    Nk = kps.shape[0]

    if mode == 'distributeF':
        kgps = [kps[i::Ncore] for i in range(Ncore)]
        return kgps
    elif mode == 'distributeC':
        njobs = [0] + [np.arange(Nk)[i::Ncore].shape[0] for i in range(Ncore)]
        index = np.array(LA.multi_dot([np.tri(Ncore + 1), np.array(njobs)]), dtype='int')
        kgps = [kps[index[i]:index[i + 1]] for i in range(Ncore)]
        return kgps
    elif mode == 'cut':
        njob_on_cores = Nk // Ncore + 1
        njob_on_last_core = Nk % njob_on_cores
        kgps = [kps[i:i + njob_on_cores] for i in range(0, len(kps), njob_on_cores)]
        # print([i.shape for i in kgps])
        return kgps
    else:
        pass

def parprint(*args, **kwargs):
    print(*args, **kwargs)

