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

__date__ = "Jan.16, 2022"

import time
import os
import sys
sys.path.append(os.environ.get('PYTHONPATH'))

import functools
import numpy as np
import numpy.linalg as LA

from mpi4py import MPI
from wanpy.MPI.MPI import get_kgroups
from wanpy.core.mesh import make_mesh

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
                print('[Rank {:<2d} {:>6d}/{:<6d}] {} Calculated k at ({:.5f} {:.5f} {:.5f})'.format(
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
                sendcounts = [np.product(i.shape) for i in kgps]     # num_of_k_per_core
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
                print('[Rank {:<2d} {:>6d}/{:<6d}] {} Calculated k at ({:.5f} {:.5f} {:.5f})'.format(
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

# def MPI_Gather(MPI, iterprint=100, dtype='float64'):
#     def decorator(func):
#         @functools.wraps(func)
#         def par_cal(kps, dim, **kwargs):
#             COMM = MPI.COMM_WORLD
#             MPI_rank = COMM.Get_rank()
#             MPI_ncore = COMM.Get_size()
#
#             if MPI_rank == 0:
#                 NK = kps.shape[0]
#             else:
#                 NK = None
#             NK = COMM.bcast(NK, root=0)
#
#             nout = dim[0]
#             dim_Gather = [MPI_ncore, NK // MPI_ncore] + dim
#
#             if MPI_rank == 0:
#                 #v kgps, njobempty = get_kpar_gps(kps, MPI_ncore, del_empty=True)
#                 kgps = get_kgroups(kps, MPI_ncore, mode='distributeC')
#                 I = np.zeros(dim_Gather, dtype=dtype)
#             else:
#                 kgps, I = None, None
#
#             kgps = COMM.bcast(kgps, root=0)
#
#             I_kgp = np.zeros(dim_Gather[1:], dtype=dtype)
#             nkgp = kgps[MPI_rank].shape[0]
#             i = 0
#             for k in kgps[MPI_rank]:
#                 I_kgp[i] = np.array(func(k, dim, **kwargs))
#
#                 i += 1
#                 if i % iterprint != 0: continue
#                 print('[Rank {:<2d} {:>6d}/{:<6d}] {} Calculated k at ({:.5f} {:.5f} {:.5f})'.format(
#                     MPI_rank, i, nkgp,
#                     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
#                     k[0], k[1], k[2]
#                 )
#                 )
#
#             COMM.Gather(I_kgp, I, root=0)
#
#             if MPI_rank == 0:
#                 I = I.reshape([NK] + dim)
#                 I = tuple(I[:, i, :] for i in range(nout))
#             else:
#                 I = (None for i in range(nout))
#
#             return I
#         return par_cal
#     return decorator


def _cal_on_k(k):
    # y = LA.norm(k)
    # y = 10 * k
    y = (10 + 10j) * k
    return y


@MPI_Reduce(MPI, iterprint=1, dtype='complex128')
def mpirun_reduce_on_kmesh(k, dim):
    return _cal_on_k(k)


@MPI_Gather(MPI, iterprint=1, dtype='complex128')
def mpirun_gather_on_kmesh(k, dim):
    return _cal_on_k(k)



if __name__ == '__main__':
    MPI_COMM = MPI.COMM_WORLD
    MPI_RANK = MPI_COMM.Get_rank()
    MPI_NCORE = MPI_COMM.Get_size()

    kmesh = make_mesh([10, 10, 1])

    dim = [3]
    RS1 = mpirun_reduce_on_kmesh(kmesh, dim)
    if MPI_RANK == 0:
        print(RS1)

    # dim = [3]
    # RS2 = mpirun_gather_on_kmesh(kmesh, dim)
    #
    # if MPI_RANK == 0:
    #     print(RS2)
    #     assert (kmesh*(10+10j) == RS2).all()









    # kgps = get_kgroups(kmesh, MPI_NCORE, mode='distributeC')

    # if MPI_RANK == 0:
    #     list_nk = [i.shape[0] for i in kgps]
    #     sendcounts = [np.product(i.shape) for i in kgps]
    # else:
    #     sendcounts = None
    # MPI_COMM.bcast(sendcounts)
    #
    # # sendcounts = np.array([[4, 3], [3, 3], [3, 3]])
    #
    # kps_rank_i = kgps[MPI_RANK]
    # kps_rank_i *= 10
    #
    # print(MPI_RANK, kps_rank_i)
    #
    # if MPI_RANK == 0:
    #     recvbuf = np.zeros([10, 3], dtype='float64')
    #     print('list_nk:', list_nk)
    #     print('sendcounts:', sendcounts)
    # else:
    #     recvbuf = None
    # MPI_COMM.Gatherv(sendbuf=kps_rank_i, recvbuf=(recvbuf, sendcounts), root=0)
    #
    # if MPI_RANK == 0:
    #     print('=================')
    #     print(recvbuf)

    # import numpy as np
    # from mpi4py import MPI
    # import random
    #
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # root = 0
    #
    # # local_array = np.kron([rank] * random.randint(2, 5), np.ones([3, 1])).T
    # local_array = np.kron([rank] * (rank+1), np.ones([3, 1])).T
    # print("rank: {}, local_array: {}".format(rank, local_array))
    #
    # sendbuf = np.array(local_array, dtype='float64')
    #
    # # Collect local array sizes using the high-level mpi4py gather
    # # sendcounts = np.array(comm.gather(len(sendbuf.flatten()), root))
    # # sendcounts = np.array([[1,3], [2,3]])
    # sendcounts = np.array([1, 2, 3, 4]) * 3
    # displ = np.array([0, 1, 3, 6]) * 3
    #
    # if rank == root:
    #     print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
    #     recvbuf = np.empty([10, 3], dtype='float64')
    # else:
    #     recvbuf = None
    #
    # comm.Gatherv(sendbuf=sendbuf, recvbuf=[recvbuf, sendcounts], root=root)
    #
    # if rank == root:
    #     print("Gathered array: {}".format(recvbuf))
