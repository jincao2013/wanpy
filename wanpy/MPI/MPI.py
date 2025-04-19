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
import sys
import functools
import numpy as np
import numpy.linalg as LA
from mpi4py import MPI

__all__ = [
    'MPI_Reduce_Fine_Grained',
    'MPI_Reduce',
    'MPI_Gather',
    'get_kgroups',
    'parprint',
]

'''
  * KPar decorator
'''
class MPI_Reduce_Fine_Grained:
    def __init__(self, config, dim, dtype='float64'):
        comm = config.comm
        self.comm = comm
        self.rank = comm.Get_rank()
        self.ncore = comm.Get_size()
        self.is_main = (self.rank == 0)

        self.dim = dim
        self.dtype = dtype

        # read info from config
        self.config = config
        self.iterprint = config.iterprint
        self.kmesh = config.kmesh
        self.nk = config.nk
        self.kcube = config.kcube
        self.volume_ucell = config.volume_ucell
        self.refine_kmesh = config.refine_kmesh
        self.kmesh_fine = config.kmesh_fine
        self.nk_fine = config.nk_fine
        self.nkmesh_fine = config.nkmesh_fine

    def __call__(self, calculator_single_k):
        @functools.wraps(calculator_single_k)
        def par_cal(*args, **kwargs):
            comm = self.comm

            # Extract `kmesh` and `dim` from args or kwargs
            # dim = kwargs.get('dim', args[0] if len(args) > 0 else None)
            # kmesh = kwargs.get('kmesh', args[0] if len(args) > 1 else None)
            # reture_str = kwargs.get('reture_str', args[2] if len(args) > 2 else None)

            # bcast kmesh to cores
            kmesh_local = self.bcast_kpoints(self.kmesh)

            # Main Loop
            if self.is_main:
                print()
                print('Enter main loop')
            kpts_index_need_refine = []
            result_local = np.zeros(self.dim, dtype=self.dtype)
            for i, k in enumerate(kmesh_local, start=1):
                if self.refine_kmesh:
                    result_at_k = calculator_single_k(k, reture_str=True)
                    if type(result_at_k) is np.ndarray:
                        result_local += np.asarray(result_at_k)
                    elif result_at_k == 'refine':
                        kpts_index_need_refine.append(i-1)
                    else:
                        raise TypeError("result_at_k must be np.ndarray or str")
                else:
                    result_local += np.asarray(calculator_single_k(k, reture_str=False))

                if i % self.iterprint == 0:
                    print('[Rank {:<4d} {:>6d}/{:<6d}] {} Calculated k at ({:.5f} {:.5f} {:.5f})'.format(
                        self.rank, i, kmesh_local.shape[0],
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        k[0], k[1], k[2]
                    ))

            # Reduce results to main rank
            result_global = np.zeros(self.dim, dtype=self.dtype) if self.is_main else None
            comm.Reduce(sendbuf=result_local, recvbuf=result_global, op=MPI.SUM, root=0)
            comm.Barrier()
            if self.is_main:
                result_global = result_global / self.nk / self.volume_ucell * LA.det(self.kcube)

            # Get kpts need be refined and count its number
            if self.refine_kmesh:
                kpts_need_refine = self.gather_variable_kpts(kmesh_local[kpts_index_need_refine], root=0)
                nk_coarse2fine = kpts_need_refine.shape[0] if self.is_main else None
                nk_coarse2fine = comm.bcast(nk_coarse2fine, root=0)
                self.config.nk_coarse2fine = nk_coarse2fine
                if self.is_main:
                    print()
                    print(f'The refined {self.nkmesh_fine[0]}*{self.nkmesh_fine[1]}*{self.nkmesh_fine[2]} '
                          f'grids are applied for {nk_coarse2fine} out of {self.nk} kpoints ({100*nk_coarse2fine/self.nk:.6f}%).')
                    # self.config.kpts_need_refine = kpts_need_refine  # for debug

            # Enter loop on fine grained kpoints if there are kpts need be refined
            if self.refine_kmesh and nk_coarse2fine > 0:
                if self.is_main:
                    print('Enter loop on fine-grained kmesh')
                    fine_grained_kpoints = kpts_need_refine[:, np.newaxis, :] + self.kmesh_fine[np.newaxis, :, :]
                    fine_grained_kpoints = fine_grained_kpoints.reshape(-1, 3)
                else:
                    fine_grained_kpoints = None
                fine_grained_kpoints_local = self.bcast_kpoints(fine_grained_kpoints)
                result_refine_local = np.zeros(self.dim, dtype=self.dtype)
                for i, k in enumerate(fine_grained_kpoints_local, start=1):
                    result_refine_local += np.asarray(calculator_single_k(k, reture_str=False))
                    if i % self.iterprint == 0:
                        print('[Fine-grained][Rank {:<4d} {:>6d}/{:<6d}] {} Calculated k at ({:.5f} {:.5f} {:.5f})'.format(
                            self.rank, i, fine_grained_kpoints_local.shape[0],
                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            k[0], k[1], k[2]
                        ))
                # Reduce results to main rank
                result_refine_global = np.zeros(self.dim, dtype=self.dtype) if self.is_main else None
                comm.Reduce(sendbuf=result_refine_local, recvbuf=result_refine_global, op=MPI.SUM, root=0)
                comm.Barrier()
                if self.is_main:
                    result_global = result_global + result_refine_global / self.nk / self.nk_fine / self.volume_ucell * LA.det(self.kcube)

            return result_global if self.is_main else None
        return par_cal

    def bcast_kpoints(self, kpoints):
        comm = self.comm
        # Distribute k-points
        if self.is_main:
            kgroups = get_kgroups(kpoints, self.ncore, mode='distributeF')
            nk_list = [group.shape[0] for group in kgroups]
        else:
            kgroups, nk_list = None, None

        nk_list = comm.bcast(nk_list, root=0)
        comm.Barrier()

        # Send kpoints to all cores
        if self.is_main:
            kmesh_local = np.ascontiguousarray(kgroups[0])
            for i in range(1, self.ncore):
                comm.Send(np.ascontiguousarray(kgroups[i]), dest=i)
        else:
            kmesh_local = np.zeros([nk_list[self.rank], 3], dtype='float64')
            comm.Recv(kmesh_local, source=0)

        comm.Barrier()
        sys.stdout.flush()

        if self.is_main:
            del kgroups
            print('MPI_Reduce: Distributed kmesh to all cores')

        sys.stdout.flush()
        return kmesh_local

    def gather_variable_kpts(self, kpts_need_refine, root=0):
        """
        Gather numpy arrays of shape (nk, 3) with variable nk from all ranks to root.

        Parameters:
        -----------
        kpts_need_refine : np.ndarray
            Local array of shape (nk, 3) on each rank.
        comm : MPI.Comm
            The MPI communicator. Default is MPI.COMM_WORLD.
        root : int
            The rank to gather data to. Default is 0.

        Returns:
        --------
        gathered_array : np.ndarray or None
            Stacked array of shape (sum(nk), 3) on root, None on other ranks.
        """
        comm = self.comm
        rank = comm.Get_rank()
        MPI_ncore = comm.Get_size()
        MPI_main = (rank == 0)

        # Flatten the local array
        sendbuf = kpts_need_refine.flatten()
        sendcount = sendbuf.size

        # Gather sizes
        recvcounts = comm.gather(sendcount, root=root)

        if rank == root:
            displs = np.insert(np.cumsum(recvcounts), 0, 0)[0:-1]
            recvbuf = np.empty(sum(recvcounts), dtype='d')
        else:
            recvbuf = None
            displs = None

        # Gather the data
        comm.Gatherv(sendbuf, [recvbuf, recvcounts, displs, MPI.DOUBLE], root=root)

        if rank == root:
            nk_list = [count // 3 for count in recvcounts]
            result = []
            idx = 0
            for nk in nk_list:
                chunk = recvbuf[idx:idx + nk * 3].reshape((nk, 3))
                result.append(chunk)
                idx += nk * 3
            return np.vstack(result)
        else:
            return None

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

