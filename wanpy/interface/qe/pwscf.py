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

__date__ = "Nov. 4, 2017"

import re
import os
import numpy as np
import numpy.linalg as LA
from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.core import Orbital as Ob
import matplotlib.pyplot as plt


def kmold(kkc):
    nk = kkc.shape[0]
    kkc = kkc[1:, :] - kkc[:-1, :]
    kkc = np.vstack([np.array([0, 0, 0]), kkc])
    k_mold = np.sqrt(np.einsum('ka,ka->k', kkc, kkc))
    k_mold = LA.multi_dot([np.tri(nk), k_mold])
    return k_mold


def read_vasp_eigenval(spin=Spin.up, efermi=None):
    print('Reading vasprun.xml and POTCAR...')
    vasp = Vasprun(r'./vasprun.xml', parse_projected_eigen=False)

    if efermi == None:
        efermi = vasp.tdos.efermi
    atomic_symbols = vasp.atomic_symbols

    lattice = vasp.lattice.matrix
    latiiceG = 2 * np.pi * np.linalg.inv(lattice)
    kk = np.array(vasp.actual_kpoints)
    kkc = LA.multi_dot([latiiceG, kk.T]).T
    kpath = kmold(kkc)

    band = vasp.eigenvalues[spin][:,:,0] - efermi # shape = (nk, nband)
    band = band.T  # shape = (nband, nk)

    return kpath, band


def read_qe_eigenval(efermi):
    nk = -1
    with open(r'QE_bands.out.gnu', 'r') as f:
        inline = 'initial'
        while inline != '':
            inline = f.readline().strip()
            nk += 1
    f.close()

    f = open(r'QE_bands.out.rap', 'r')
    rap = f.readlines()
    f.close()
    nband, nk = np.array(re.findall(r'\d+', rap[0]), dtype='int')
    mark = []
    i = 0
    for inline in rap:
        _isHSP = inline.split()[-1]
        if _isHSP in ['T', 'F']:
            if _isHSP is 'T':
                mark.append(i)
            i += 1

    data = np.loadtxt(qefname)
    nband = data.shape[0] // nk
    data = np.loadtxt(qefname).reshape(nband, nk, 2)
    kpath = data[0,:,0]
    band = data[:,:,1] - efermi
    return kpath, band, mark


def compare_qe_vasp_band(yy=[-10, 10]):
    kpath_qe, band_qe, mark_qe = read_qe_eigenval(efermi=efermi_qe)
    kpath_vasp, band_vasp = read_vasp_eigenval(spin=Spin.up, efermi=efermi_vasp)
    kpath_qe = kpath_qe / kpath_qe[-1]
    kpath_vasp = kpath_vasp / kpath_vasp[-1]

    nband, nk_qe = band_qe.shape

    plt.axis([0, 1, yy[0], yy[1]])
    plt.axhline(0, color='k', linewidth=0.5, linestyle='--')
    for i in mark_qe:
        plt.axvline(kpath_qe[i], color='k', linewidth=0.5, linestyle='--')

    for i in range(nband):
        plt.plot(kpath_qe, band_qe[i],
                 color='red', linewidth=1, linestyle='-', alpha=1)
    for i in range(nband):
        plt.plot(kpath_vasp, band_vasp[i],
                 color='k', linewidth=1, linestyle='-', alpha=1)




def plot_qeband(fname=r'QE_bands.out.gnu'):
    data = np.loadtxt(fname).reshape(nband, nk, 2) - efermi_qe
    kk = data[:,:,0]
    ee = data[:,:,1]

    plt.axis([kk.min(), kk.max(), ee.min(), ee.max()])

    plt.axhline(0, color='k', linewidth=0.5, linestyle='--')
    for i in range(nband):
        plt.plot(data[i, :, 0], data[i, :, 1],
                 color='k', linewidth=1, linestyle='-', marker='*')





wdir = r''
qefname = r'QE_bands.out.gnu'


efermi_vasp = 0 #7.2925
efermi_qe = 0 # 10.4429


if __name__ == '__main__':
   os.chdir(wdir)
   compare_qe_vasp_band(yy=[-10, 10])

