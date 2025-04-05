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

__date__ = "Aug. 12, 2021"

import os
import errno
import numpy as np
from wanpy.core.structure import Htb, Cell, Worbi
from wanpy.core.symmetry import Symmetrize_Htb_kspace, Symmetrize_Htb_rspace, get_proj_info
from wanpy.core.utils import wannier90_load_wcc
from wanpy.core.units import *
from wanpy.interface.wannier90 import *

__all__ = [
    'WannierInterpolation'
]

class WannierInterpolation(object):
    """
    Input:
        fermi_level
        wannier90.nnkp, wannier90.wout, wannier90.chk, wannier90.eig
        WAVECAR (if cal_spin)
    Output:
        wannier90_hr.dat, wannier90_r.dat,
        wannier90_spin.dat (if cal_spin)
        and a single .h5 file `htb.h5` containing all Wannier tight-binding information.

    Usage:
        1). perform dft calculation by vasp, and generating:
            WAVECAR (needed by WannierInterpolation)
            .eig (needed by WannierInterpolation) .amn .mmn

            N.B. Currently, one needs turn off the symmetry in vasp (ISYM=0)
                 for calculating spin matrix.

        2). generate .nnkp (we need bk) and .wout (we need wb) file by:
            wannier90.x -pp wannier90.win

        3). do disentanglement and get wannier90.chk by:
            wannier90.x wannier90.win

        4). wanrun = WannierInterpolation()
            wanrun.run()
    """

    def __init__(self, fermi=0., poscar_fname='POSCAR', seedname='wannier90',
                 symmetric_htb=False, symmetric_method='kspace', rspace_use_ngridR=False,
                 wannier_center_def=None, ngridR_symmhtb=None, symmops=None,
                 check_if_uudd_amn=True):
        self.fermi = fermi
        self.poscar_fname = poscar_fname
        self.seedname = seedname
        self.wannier_center_def = wannier_center_def

        # POSCAR
        self.cell = Cell()
        print('reading {}'.format(poscar_fname))
        if os.path.exists(poscar_fname):
            self.cell.load_poscar(fname=poscar_fname)
            self.name = self.cell.name
            self.latt = self.cell.lattice
            self.lattG = self.cell.latticeG
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), poscar_fname)

        # To read bk and wb, this needs .nnkp .wout
        w90_nnkp = W90_nnkp()
        w90_nnkp.load_from_w90(seedname)
        # self.w90_nnkp = w90_nnkp
        self.bk = w90_nnkp.bk
        self.wb = w90_nnkp.wb
        del w90_nnkp

        self.worbi = Worbi()
        self.worbi.load_from_nnkp(convert_to_uudd=True, wannier_center_def=wannier_center_def, seedname=seedname)
        self.soc = self.worbi.soc

        # read wcc and check whether uudd_amn in .wout
        if self.soc and not check_if_uudd_amn:
            print('please check carefully whether the .amn is in uudd order.')
        if not self.soc: check_if_uudd_amn = False
        self.wcc, self.wccf, wbroaden = wannier90_load_wcc(seedname+'.wout', shiftincell=False, check_if_uudd_amn=check_if_uudd_amn)

        # .chk
        w90_chk = W90_chk()
        w90_chk.load_from_w90(seedname, verbose=False)
        self.w90_chk = w90_chk

        self.nk = w90_chk.nk
        self.nb = w90_chk.nb
        self.nw = w90_chk.nw
        self.meshkf = w90_chk.kpt_latt
        self.nR = w90_chk.nR
        self.gridR = w90_chk.gridR
        self.ndegen = w90_chk.ndegen
        # self.umat_dis = w90_chk.umat_dis
        # self.umat = w90_chk.umat
        self.vmat = w90_chk.vmat
        self.m_matrix = w90_chk.m_matrix # Mmn(k,b)=<um,k|un,k+b>, index=(ik,nn,n,m)

        # # .umn
        # w90_umat = W90_umat()
        # w90_umat.load_from_w90(seedname)
        # umat = w90_umat.umat
        # w90_umat_dis = W90_umat_dis()
        # w90_umat_dis.load_from_w90(seedname)
        # umat_dis = w90_umat_dis.umat_dis
        # self.vmat = np.einsum('kmi,kin->kmn', umat_dis, umat) # the line is definitely right

        # .eig
        w90_eig = W90_eig(self.nb, self.nk)
        w90_eig.load_from_w90()
        self.eig = w90_eig.eig
        # eigenvalues in outer windows, in same dimension os eig, but start from dis_win_min to dis_win_max
        # the others is set at zero
        self.eig_win = np.zeros_like(self.eig)
        for ik in range(self.nk):
            _index = np.where(w90_chk.lwindow[ik] == -1)[0]
            self.eig_win[ik, :w90_chk.ndimwin[ik]] = self.eig[ik, _index]

        # symmetry
        self.symmetric_htb, self.symmetric_method, self.rspace_use_ngridR = \
            symmetric_htb, symmetric_method, rspace_use_ngridR
        self.ngridR_symmhtb, self.symmops = ngridR_symmhtb, symmops

        print('calculating F.T. matrix')
        self.eikR = np.exp(-2j * np.pi * np.einsum('ka,Ra->kR', self.meshkf, self.gridR))

        # output
        self.hr_Rmn = None
        self.r_Ramn = None
        self.spin0_Rmn, self.spin_Ramn = None, None
        self.htb = Htb(fermi)

    def run(self, cal_r=False, cal_spin=False, write_h5=True, write_dat=False,
            write_spn=False, hermite_r=True, iprint=2,
            h5decimals=16, fmt='12.6'):
        # write_h5: if write htb.h5
        # write_dat: if write .dat files
        # write_spn: if write .spn file

        print('interpolating hr_Rmn')
        self.hr_Rmn = self.get_hr_Rmn()

        if cal_r:
            print('interpolating r_Ramn')
            self.r_Ramn = self.get_r_Ramn(hermite=hermite_r)
        else:
            self.r_Ramn = np.zeros([self.nR, 3, self.nw, self.nw], dtype='complex128')
            self.r_Ramn[self.nR//2, 0] = np.diag(self.wcc.T[0])
            self.r_Ramn[self.nR//2, 1] = np.diag(self.wcc.T[1])
            self.r_Ramn[self.nR//2, 2] = np.diag(self.wcc.T[2])

        if cal_spin:
            print('interpolating spin_Ramn')
            self.spin0_Rmn, self.spin_Ramn = self.get_spin_Ramn(write_spn)

        Rc = (self.latt @ self.gridR.T).T
        self.htb.load(cell=self.cell, worbi=self.worbi,
                      name=self.name, fermi=self.fermi, latt=self.latt, lattG=self.lattG, wcc=self.wcc, wccf=self.wccf,
                      nw=self.nw, nR=self.nR, ndegen=self.ndegen, R=self.gridR, Rc=Rc,
                      hr_Rmn=self.hr_Rmn, r_Ramn=self.r_Ramn,
                      spin0_Rmn=self.spin0_Rmn, spin_Ramn=self.spin_Ramn
                      )

        if self.symmetric_htb:
            print('symmetrizing Wannier TB model')
            atoms_pos, atoms_orbi = get_proj_info(htb=self.htb, wannier_center_def=self.wannier_center_def)

            if self.symmetric_method[0] == 'k':
                symmhtb = Symmetrize_Htb_kspace(ngridR=self.ngridR_symmhtb,
                                                htb=self.htb,
                                                symmops=self.symmops,
                                                atoms_pos=atoms_pos,
                                                atoms_orbi=atoms_orbi,
                                                soc=self.htb.worbi.soc,
                                                iprint=iprint,
                                                )
                symmhtb.run(tmin=1e-6)
                self.htb = symmhtb.htb
            elif self.symmetric_method[0] == 'r':
                symmhtb = Symmetrize_Htb_rspace(htb=self.htb,
                                                symmops=self.symmops,
                                                atoms_pos=atoms_pos,
                                                atoms_orbi=atoms_orbi,
                                                soc=self.htb.worbi.soc,
                                                iprint=iprint,
                                                )
                if self.rspace_use_ngridR:
                    symmhtb.use_ngridR(self.ngridR_symmhtb)
                symmhtb.run()
                self.htb = symmhtb.htb

        if write_dat:
            print('write hr matrix')
            self.htb.save_wannier90_hr_dat(self.seedname, fmt=fmt)
            if cal_r:
                print('write r matrix')
                self.htb.save_wannier90_r_dat(self.seedname, fmt=fmt)
            if cal_spin:
                print('write spin matrix')
                self.htb.save_wannier90_spin_dat(self.seedname, fmt=fmt)

        if write_h5:
            if h5decimals is not None:
                print('write htb.h5 with rounded double precision (decimals={})'.format(h5decimals))
            else:
                print('write htb.h5 with fully double precision (larger size of .h5)')
            # to reduce the size of .h5
            # one can use htb.hr_Rmn = np.around(htb.hr_Rmn, 7) which is
            # in line with the numerical precision of wannier90_hr.dat.
            self.htb.save_h5('htb.h5', decimals=h5decimals)

    def get_hr_Rmn(self, hermite=True):
        hwk = np.einsum('kim,ki,kin->kmn', self.vmat.conj(), self.eig_win, self.vmat, optimize=True)
        hwk = 0.5 * (hwk + np.einsum('kmn->knm', hwk.conj()))
        hr_Rmn = np.einsum('kR,kmn->Rmn', self.eikR, hwk) / self.nk
        return hr_Rmn

    def get_r_Ramn(self, hermite=True):
        Awkmn = 1j * np.einsum('b,kba,kbnm->akmn', self.wb, self.bk, self.m_matrix, optimize=True)
        Awknn = -1 * np.einsum('b,kba,kbnn->akn', self.wb, self.bk, np.imag(np.log(self.m_matrix)), optimize=True)
        np.einsum('aknn->akn', Awkmn)[:] = Awknn
        if hermite:
            Awkmn = 0.5 * (Awkmn + np.einsum('akmn->aknm', Awkmn.conj()))
        r_Ramn = np.einsum('kR,akmn->Ramn', self.eikR, Awkmn) / self.nk
        return r_Ramn

    def get_spin_Ramn(self, write_spn=False):
        overlapk, spink = self.cal_vasp_spin_matrix()

        spin0wk = np.einsum('kim,kij,kjn->kmn', self.vmat.conj(), overlapk, self.vmat, optimize=True)
        spin0wk = 0.5 * (spin0wk + np.einsum('kmn->knm', spin0wk.conj()))
        spin0_Rmn = np.einsum('kR,kmn->Rmn', self.eikR, spin0wk) / self.nk

        spinwk = np.einsum('kim,akij,kjn->akmn', self.vmat.conj(), spink, self.vmat, optimize=True)
        spinwk = 0.5 * (spinwk + np.einsum('akmn->aknm', spinwk.conj()))
        spin_Ramn = np.einsum('kR,akmn->Ramn', self.eikR, spinwk) / self.nk

        if write_spn:
            fname = self.seedname + '.spn'
            if os.path.exists(fname):
                os.remove(fname)
            with open(fname, 'a') as f:
                f.write('formated .spn file written by wanpy \n')
                f.write('{:>12}{:>12}'.format(self.nb, self.nk))
                for ik in range(self.nk):
                    for m in range(self.nb):
                        for n in range(m):
                            for a in range(3):
                                inline = spink[a, ik, m, n]
                                f.write('{:+26.16E}{:+26.16E}'.format(inline.real, inline.imag))

        return spin0_Rmn, spin_Ramn

    # @staticmethod
    def cal_vasp_spin_matrix(self, verbose=False):
        from wanpy.interface.vasp import VASP_wavecar
        print('reading WAVECAR')
        wavecar = VASP_wavecar('WAVECAR', verbose=False, vasp_type='ncl')

        nk, nb = wavecar.nk, wavecar.nb
        wfs = wavecar.wfs

        overlapk = np.zeros([nk, nb, nb], dtype='complex128')
        spink = np.zeros([3, nk, nb, nb], dtype='complex128')
        print('calculating spin matrix in basis of pseudo wavefunctions (vasp)')
        for ik in range(nk):
            if verbose: print('cal spin matrix on {}/{}'.format(ik+1, nk))
            wf = np.array(wfs[ik], dtype='complex128')
            norm = np.einsum('nsG->n', np.abs(wf)**2)
            wf = np.einsum('n,nsG->nsG', norm, wf)
            overlapk[ik] = np.einsum('nsG,msG->nm', wf.conj(), wf)
            spink[0, ik] = np.einsum('naG,ab,mbG->nm', wf.conj(), sigmax, wf, optimize=True)
            spink[1, ik] = np.einsum('naG,ab,mbG->nm', wf.conj(), sigmay, wf, optimize=True)
            spink[2, ik] = np.einsum('naG,ab,mbG->nm', wf.conj(), sigmaz, wf, optimize=True)

        return overlapk, spink

    def write_tb(self):
        pass


def plot_error(x, y):
    import matplotlib.pyplot as plt
    # check if consistant with wannier90 results
    dataset_orig1 = np.abs(x).flatten()
    dataset_orig2 = np.abs(y).flatten()
    dataset_error = np.abs(y - x).flatten()
    xx = np.arange(dataset_orig1.shape[0]) + 1

    plt.clf()
    plt.scatter(xx, dataset_orig1, s=60, color='b', marker='^')
    plt.scatter(xx, dataset_orig2, s=60, color='r', marker='v')
    # plt.scatter(xx, dataset_error, s=2, color='g')
    plt.yscale('log')
    # wannier90 only write up to 1e-6
    # so the compare of the values smaller than 1e-5 is meanless
    plt.axis([0, xx.max(), 1e-5, 1])

'''
  tools
'''
def check_htb_equ(htb1, htb2):
    assert (htb1.R == htb2.R).all()
    assert (htb1.Rc == htb2.Rc).all()
    assert (htb1.ndegen == htb2.ndegen).all()
    assert (np.abs(htb1.hr_Rmn - htb2.hr_Rmn) < 1e-6).all()
    assert (np.abs(htb1.r_Ramn - htb2.r_Ramn) < 1e-6).all()
    assert (np.abs(htb1.spin0_Rmn - htb2.spin0_Rmn) < 1e-6).all()
    assert (np.abs(htb1.spin_Ramn - htb2.spin_Ramn) < 1e-6).all()


if __name__ == "__main__":
    pass
    # import matplotlib.pyplot as plt
    # from wanpy.core.plot import *
    # from wanpy.response.response_plot import *
    # from wanpy.env import ROOT_WDIR
    # wdir = os.path.join(ROOT_WDIR, r'wanpy_debug/buildwannier/MnPd')
    # input_dir = os.path.join(ROOT_WDIR, r'wanpy_debug/buildwannier/MnPd')
    # os.chdir(wdir)
    # 
    # htb1 = Htb()
    # # htb1.load_wannier90_dat()
    # htb1.load_h5('htb.h5')
    #
    # # htb3 = Htb()
    # # # htb1.load_wannier90_dat()
    # # htb3.load_h5('htb.soc.mx.U3.wanpy.h5')
    #
    # wanrun = WannierInterpolation(symmetric_htb=True)
    # # wanrun.run(cal_r=False, cal_spin=False, write_h5=False, write_spn=False)
    # wanrun.run(write_h5=False)
    # htb = wanrun.htb

    # htb2.save_h5('test.d9.h5', decimals=9)

    # plt.plot([6, 7, 8, 9, 10, 11], [8.6, 12.5, 17.6, 23.5, 29.2, 36.3])
    # assert (htb1.R == htb2.R).all()
    # assert (htb1.Rc == htb2.Rc).all()
    # assert (htb1.ndegen == htb2.ndegen).all()
    # assert (np.abs(htb1.hr_Rmn - htb2.hr_Rmn) < 1e-6).all()
    # assert (np.abs(htb1.r_Ramn - htb2.r_Ramn) < 5e-6).all()
    # assert (np.abs(htb1.spin0_Rmn - htb2.spin0_Rmn) < 1e-6).all()
    # assert (np.abs(htb1.spin_Ramn - htb2.spin_Ramn) < 1e-6).all()

    # # check if the wcc is right
    # wcc = np.array([
    #     np.diag(htb2.r_Ramn[htb2.nR//2, 0]),
    #     np.diag(htb2.r_Ramn[htb2.nR//2, 1]),
    #     np.diag(htb2.r_Ramn[htb2.nR//2, 2]),
    # ]).T.real

    # hr_Rmn = np.random.random([200, 100, 100]) + 1j * np.random.random([200, 100, 100])
    # # os.remove('test2.h5')
    # f = h5py.File('test2.h5', "w")
    # f.create_group('htb')
    # htb = f['htb']
    # htb.create_dataset('hr_Rmn', data=np.around(hr_Rmn, decimals=6), dtype='complex128', compression="gzip")
    # f.close()




    # plot_matrix(spinwk[2, 1].real, cmap='seismic')

    # plot_error(htb1.hr_Rmn, wanrun.hr_Rmn)

    # plot_error(htb.r_Ramn.real, wanrun.r_Ramn.real)
    # plot_error(htb.r_Ramn.imag, wanrun.r_Ramn.imag)
    # plot_error(htb.r_Ramn, wanrun.r_Ramn)


    '''
      * debug
    '''
    # wavecar = VASP_wavecar('WAVECAR', verbose=False, vasp_type='ncl')

    # wavecar = Wavecar('WAVECAR', verbose=False, vasp_type='ncl')
    #
    # latt = wavecar.a.T
    # lattG = wavecar.b.T
    # fermi = wavecar.efermi
    # bandE = np.array(wavecar.band_energy, dtype='float64')[:,:,0]       # nk, nb
    # bandEocc = np.array(wavecar.band_energy, dtype='float64')[:,:,2]    # nk, nb
    # nk, nb = wavecar.nk, wavecar.nb
    # kpts = np.array(wavecar.kpoints, dtype='float64')
    # wfs = wavecar.coeffs        # nk, nb, 2(spin), nG(not uniform)
    # Gpoints = wavecar.Gpoints   # nk, nG(not uniform)
    # encut = wavecar.encut

    # overlapk = np.zeros([nk, nb, nb], dtype='complex128')
    # spink = np.zeros([3, nk, nb, nb], dtype='complex128')
    # for ik in range(nk):
    #     print('cal spin matrix on {}/{}'.format(ik+1, nk))
    #     wf = np.array(wfs[ik], dtype='complex128')
    #     norm = np.einsum('nsG->n', np.abs(wf)**2)
    #     wf = np.einsum('n,nsG->nsG', norm, wf)
    #     overlapk = np.einsum('nsG,msG->nm', wf.conj(), wf)
    #     spink[0, ik] = np.einsum('naG,ab,mbG->nm', wf.conj(), sigmax, wf, optimize=True)
    #     spink[1, ik] = np.einsum('naG,ab,mbG->nm', wf.conj(), sigmay, wf, optimize=True)
    #     spink[2, ik] = np.einsum('naG,ab,mbG->nm', wf.conj(), sigmaz, wf, optimize=True)


    # plot_matrix(np.abs(spink[2, -1]))




    # w90_eig = W90_eig(nb, nk)
    # w90_eig.load_from_w90()
    # eig = w90_eig.eig
    #
    # w90_nnkp = W90_nnkp()
    # w90_nnkp.load_from_w90()
    # bk = w90_nnkp.bk
    #
    #
    # w90_umat = W90_umat()
    # w90_umat.load_from_w90()
    # umat = w90_umat.umat
    #
    # w90_umat_dis = W90_umat_dis()
    # w90_umat_dis.load_from_w90()
    # umat_dis = w90_umat_dis.umat_dis
    #
    # meshkf = w90_umat.meshkf
    # gridR = htb.R
    #
    # vmat = np.einsum('kmi,kin->kmn', umat_dis, umat)
    # eikR = np.exp(-2j * np.pi * np.einsum('ka,Ra->kR', meshkf, gridR))
    #
    #
    # hwk = np.einsum('kim,ki,kin->kmn', vmat.conj(), eig, vmat, optimize=True)
    # hwk = 0.5 * (hwk + np.einsum('kmn->knm', hwk.conj()))
    # hr_Rmn = np.einsum('kR,kmn->Rmn', eikR, hwk) / nk
    #
    # spinwk = np.einsum('kim,akij,kjn->akmn', vmat.conj(), spink, vmat, optimize=True)
    # spinwk = 0.5 * (spinwk + np.einsum('akmn->aknm', spinwk.conj()))
    # spinRmn = np.einsum('kR,akmn->aRmn', eikR, spinwk) / nk
    #
    #
    # hrerror = np.abs(htb.hr_Rmn - hr_Rmn)
    # plot_matrix(hrerror)
