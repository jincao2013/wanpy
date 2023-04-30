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

import matplotlib.pyplot as plt
from wanpy.core.bz import delta_func

sys.path.append(os.environ.get('PYTHONPATH'))

import functools
import numpy as np
import numpy.linalg as LA

from mpi4py import MPI
from wanpy.MPI.MPI import get_kgroups
from wanpy.core.mesh import make_mesh

def MPI_Reduce_adaptive_kmesh(MPI, iterprint=100, dtype='float64', mpinote=False):
    def decorator(func):
        @functools.wraps(func)
        def par_cal(kps, dim, nkmesh, adaptive=False, nkmesh_adaptive=np.ones(3), *args, **kwargs):
            COMM = MPI.COMM_WORLD
            MPI_rank = COMM.Get_rank()
            MPI_ncore = COMM.Get_size()
            MPI_main = not MPI_rank

            if MPI_main:
                kgps = get_kgroups(kps, MPI_ncore, mode='distributeF')
                nk_list = [i.shape[0] for i in kgps]
            else:
                kgps, nk_list = None, None

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
            index_kps_exceed_threshold = []
            i = 0
            for k in kps_rank_i:
                result_of_k, exceed_threshold = func(k, dim, *args, **kwargs)
                if exceed_threshold and adaptive:
                    print(MPI_rank, k, result_of_k, 'add more k here')
                    index_kps_exceed_threshold.append(i)
                else:
                    print(MPI_rank, k, result_of_k)
                    result_kgp += np.array(result_of_k)
                i += 1
                if mpinote:
                    pass
                if i % iterprint != 0: continue
                # print('[Rank {:<2d} {:>6d}/{:<6d}] {} Calculated k at ({:.5f} {:.5f} {:.5f})'.format(
                #     MPI_rank, i, nk_list[MPI_rank],
                #     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                #     k[0], k[1], k[2]
                #     )
                # )

            result = np.zeros(dim, dtype=dtype) if MPI_main else None
            COMM.Reduce(sendbuf=result_kgp, recvbuf=result, op=MPI.SUM, root=0)
            COMM.Barrier()

            if adaptive:
                if MPI_main: print('adaptive kmesh will used.')
                '''
                  * gather k-points that exceed the threshold 
                '''
                if not MPI_main:
                    MPI_COMM.send(kps_rank_i[index_kps_exceed_threshold], dest=0, tag=MPI_rank)
                else:
                    recv = [kps_rank_i[index_kps_exceed_threshold]]
                    for i in range(1, MPI_ncore):
                        recv.append(MPI_COMM.recv(source=i, tag=i))
                    kps_exceed_threshold = np.vstack(recv)
                    print('there are {} k-points that exceed the threshold.'.format(kps_exceed_threshold.shape[0]))
                    print('the adaptive kmesh is {}*{}*{}'.format(nkmesh_adaptive[0], nkmesh_adaptive[1], nkmesh_adaptive[2]))

                if MPI_main:
                    kps_adaptive = np.vstack([
                        make_mesh(nkmesh_adaptive) / nkmesh + k
                        for k in kps_exceed_threshold
                    ])
                    # kps_adaptive = kps_exceed_threshold  # revise here
                    print('{} new k-points will be calculated.'.format(kps_adaptive.shape[0]))

                '''
                  * =========================================================
                  *  run on adaptive kmesh
                  * =========================================================
                '''
                # if MPI_main: print('calculating on adaptive kmesh ...')
                if MPI_main:
                    kgps = get_kgroups(kps_adaptive, MPI_ncore, mode='distributeF')
                    nk_list = [i.shape[0] for i in kgps]
                else:
                    kgps, nk_list = None, None

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
                index_kps_exceed_threshold = []
                i = 0
                for k in kps_rank_i:
                    result_of_k, exceed_threshold = func(k, dim, *args, **kwargs)
                    print(MPI_rank, k, result_of_k)
                    result_kgp += np.array(result_of_k)
                    i += 1
                    if mpinote:
                        pass
                    if i % iterprint != 0: continue
                    # print('[Rank {:<2d} {:>6d}/{:<6d}] {} Calculated k at ({:.5f} {:.5f} {:.5f})'.format(
                    #     MPI_rank, i, nk_list[MPI_rank],
                    #     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    #     k[0], k[1], k[2]
                    #     )
                    # )

                result_adaptive = np.zeros(dim, dtype=dtype) if MPI_main else None
                COMM.Reduce(sendbuf=result_kgp, recvbuf=result_adaptive, op=MPI.SUM, root=0)
                COMM.Barrier()

            if MPI_main:
                nk = kps.shape[0]
                if adaptive:
                    nk_adaptive = np.product(nkmesh_adaptive)
                    print(result / nk)
                    print(result_adaptive / nk / nk_adaptive)
                    result = (result + result_adaptive / nk_adaptive) / nk
                else:
                    result = result / nk

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

def _cal_on_k(k):
    # y = LA.norm(k)
    # y = 10 * k
    # y = (10 + 10j) * k
    kk = LA.norm(k)
    y = delta_func(kk - 0.12, smear=0.01) + \
         delta_func(kk - 0.34, smear=0.015) + \
         delta_func(kk - 0.42, smear=0.015) + \
         delta_func(kk - 0.55, smear=0.015) + \
         delta_func(kk - 0.61, smear=0.015) + \
         delta_func(kk - 0.89, smear=0.01)

    return y, y > 15


@MPI_Reduce_adaptive_kmesh(MPI, iterprint=1, dtype='float64')
def mpirun_reduce_on_kmesh(k, dim):
    return _cal_on_k(k)


@MPI_Gather(MPI, iterprint=1, dtype='complex128')
def mpirun_gather_on_kmesh(k, dim):
    return _cal_on_k(k)


def get_adaptive_kmesh(nkmesh, nkmesh_adaptive, kmesh_exceed_threshold):
    kcube = None
    kmesh = make_mesh(nkmesh)

    kmesh_adaptive = np.vstack([
        make_mesh(nkmesh) / nkmesh + k
        for k in kmesh_exceed_threshold
    ])
    return kmesh_adaptive


if __name__ == '__main__':
    MPI_COMM = MPI.COMM_WORLD
    MPI_RANK = MPI_COMM.Get_rank()
    MPI_NCORE = MPI_COMM.Get_size()

    nk = 200
    nk_adaptive = 2
    nkmesh = np.array([nk, 1, 1])
    kmesh = make_mesh(nkmesh)
    adaptive = True
    nkmesh_adaptive = np.array([nk_adaptive, 1, 1])

    dim = [1]
    RS1 = mpirun_reduce_on_kmesh(kmesh, dim, nkmesh, adaptive=adaptive, nkmesh_adaptive=nkmesh_adaptive)
    if MPI_RANK == 0:
        print(RS1)

    # kk = np.linspace(0, 1, 100)
    # yy = delta_func(kk - 0.12, smear=0.015) + \
    #      delta_func(kk - 0.34, smear=0.015) + \
    #      delta_func(kk - 0.42, smear=0.015) + \
    #      delta_func(kk - 0.55, smear=0.015) + \
    #      delta_func(kk - 0.61, smear=0.015) + \
    #      delta_func(kk - 0.89, smear=0.015)
    # plt.plot(kk, yy, marker='*')
    # plt.axhline(16)

    # if MPI_RANK == 0:
    #     print(MPI_RANK)
    #     kps_need_more_k = np.array([
    #         [0, 0, 0]
    #     ])
    #     # MPI_COMM.send(kps_need_more_k, dest=0, tag=0)
    # elif MPI_RANK == 1:
    #     print(MPI_RANK)
    #     kps_need_more_k = np.array([
    #         [1, 0, 0]
    #     ])
    #     MPI_COMM.send(kps_need_more_k, dest=0, tag=1)
    # elif MPI_RANK == 2:
    #     print(MPI_RANK)
    #     kps_need_more_k = np.array([
    #         [2, 0, 0],
    #         [2, 1, 0]
    #     ])
    #     MPI_COMM.send(kps_need_more_k, dest=0, tag=2)
    #
    # if MPI_RANK == 0:
    #     print(MPI_RANK)
    #     # recv0 = MPI_COMM.recv(source=0, tag=0)
    #     # recv1 = MPI_COMM.recv(source=1, tag=1)
    #     # recv2 = MPI_COMM.recv(source=2, tag=2)
    #
    #     recv = [
    #         kps_need_more_k,
    #         MPI_COMM.recv(source=1, tag=1),
    #         MPI_COMM.recv(source=2, tag=2)
    #     ]
    #     recv = np.vstack(recv)
    #     print(recv)
    #
    # nkmesh = np.array([5, 1, 1])
    # kmesh = make_mesh(nkmesh)
    # kmesh_exceed_threshold = kmesh[np.array([1, 2])]
    #
    # nkmesh_adaptive = np.array([4, 1, 1])
    # kmesh_adaptive = np.vstack([
    #     make_mesh(nkmesh_adaptive) / nkmesh + k
    #     for k in kmesh_exceed_threshold
    # ])