#!/usr/bin/env python

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

import os
import sys
import re
from optparse import OptionParser
import numpy as np
from numpy import linalg as LA
from wanpy.interface.wannier90 import W90_band
from wanpy.core.structure import Cell
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.outputs import Eigenval
from pymatgen.electronic_structure.core import Spin
import matplotlib
import matplotlib.pyplot as plt

def kmold(kkc):
    nk = kkc.shape[0]
    kkc = kkc[1:, :] - kkc[:-1, :]
    kkc = np.vstack([np.array([0, 0, 0]), kkc])
    k_mold = np.sqrt(np.einsum('ka,ka->k', kkc, kkc))
    k_mold = LA.multi_dot([np.tri(nk), k_mold])
    return k_mold

def get_wannier_win_kpoints(fname='KPOINTS'):
    kpoints = Kpoints.from_file(fname)
    kpts = kpoints.kpts
    labels = kpoints.labels
    nline = len(kpoints.labels) // 2
    for i in range(nline):
        print('{:>3} {:>7.4f} {:>7.4f} {:>7.4f} {:>3} {:>7.4f} {:>7.4f} {:>7.4f}'.format(
            labels[2*i], kpts[2*i][0], kpts[2*i][1], kpts[2*i][2],
            labels[2*i+1], kpts[2*i+1][0], kpts[2*i+1][1], kpts[2*i+1][2]))

def load_HSP_from_wout(seedname=r'wannier90'):
    HSP = []
    fname = seedname + '.wout'
    with open(fname, 'r') as f:

        inline = f.readline()
        while 'K-space path sections' not in inline:
            inline = f.readline()

        inline = f.readline()
        HSP.append(inline.split()[2])
        HSP.append(inline.split()[7])

        inline = f.readline()
        while 'From' in inline:
            HSP.append(inline.split()[7])
            inline = f.readline()
    f.close()
    HSP = '-'.join(HSP)
    return HSP

def load_HSP_KPOINTS(fname='KPOINTS'):
    kpoints = Kpoints.from_file(fname)
    labels = kpoints.labels
    labels = labels[::2] + [labels[-1]]
    HSP = '-'.join(labels)
    return HSP

def plot_compare(cell, vaspeig, w90_band, yy=[-1,1], HSP='G-G', fermi_dft=0, fermi_wan=0, spin='all', savefig=False):
    ispin = vaspeig.ispin

    nk, nb, nele = vaspeig.nkpt, vaspeig.nbands, vaspeig.nelect
    band_gap, cbm, vbm, is_band_gap_direct = vaspeig.eigenvalue_band_properties

    kk = np.array(vaspeig.kpoints, dtype='float64')
    kkc = LA.multi_dot([cell.latticeG, kk.T]).T
    kpath = kmold(kkc)
    kk_weights = np.array(vaspeig.kpoints_weights, dtype='float64')

    band_1 = vaspeig.eigenvalues.get(Spin.up)[:,:,0]
    band_2 = vaspeig.eigenvalues.get(Spin.down)[:,:,0] if ispin == 2 else 0

    nline = len(HSP.split('-')) - 1

    print('Info from plot_dft_band_dev:')
    print('ispin = ', ispin)
    print('band_gap, cbm, vbm, is_band_gap_direct = ', vaspeig.eigenvalue_band_properties)
    print('fermi level = ', fermi_dft)

    '''
      * Plot
    '''
    plt.figure(figsize=[4, 3], dpi=300)

    plt.axis([kpath.min(), kpath.max(), yy[0], yy[1]])

    ax = plt.gca()
    ax.set_ylabel('$E-E_F$ (eV)')

    plt.axhline(0, linestyle='--', color='k', linewidth=0.3, alpha=1, zorder=2)
    for i in range(1, nline):
        plt.axvline(x=kpath[i * int(nk / nline)], linestyle='--', color='k', linewidth=0.3, alpha=1, zorder=2)

    # plot HSP
    i_HSP = np.arange(nline+1) * (nk // nline)
    i_HSP[-1] = -1
    plt.xticks(kpath[i_HSP], HSP.split('-'))

    # bloch band
    if ispin == 2:
        if spin == 'all':
            plt.plot(kpath, band_1-fermi_dft, linewidth=0.5, linestyle="-", color='red', alpha=1.0, zorder=20)
            plt.plot(kpath, band_2-fermi_dft, linewidth=0.5, linestyle="-", color='blue', alpha=1.0, zorder=20)
        elif spin == 'up':
            plt.plot(kpath, band_1-fermi_dft, linewidth=0.5, linestyle="-", color='black', alpha=1.0, zorder=20)
        elif spin == 'dn':
            plt.plot(kpath, band_2-fermi_dft, linewidth=0.5, linestyle="-", color='black', alpha=1.0, zorder=20)
        else:
            pass
    else:
        plt.plot(kpath, band_1-fermi_dft, linewidth=0.5, linestyle="-", color='black', alpha=1.0, zorder=20)

    # wannier band
    for i in range(w90_band.nw):
        plt.plot(w90_band.kpath, w90_band.eig.T[i]-fermi_wan, linewidth=1, linestyle="-", color='green', alpha=0.6, zorder=10)

    plt.tight_layout()

    if savefig:
        plt.savefig('compareband.pdf')
    else:
        plt.show()

def plot_dft_band(cell, vaspeig, yy=[-1,1], HSP='G-G', fermi_dft=0, spin='all', savefig=False):
    print('Info from plot_dft_band:')
    ispin = vaspeig.ispin

    nk, nb, nele = vaspeig.nkpt, vaspeig.nbands, vaspeig.nelect
    band_gap, cbm, vbm, is_band_gap_direct = vaspeig.eigenvalue_band_properties
    kk_weights = np.array(vaspeig.kpoints_weights, dtype='float64')
    bandindex = ...

    if np.sum(np.isclose(kk_weights, 0)) >= 1:
        print('Found HSE type EIGENVAL, plot kpoint with zero weight.')
        bandindex = np.where(np.isclose(kk_weights, 0))[0]
        nk = bandindex.shape[0]

    kk = np.array(vaspeig.kpoints, dtype='float64')[bandindex]
    kkc = LA.multi_dot([cell.latticeG, kk.T]).T
    kpath = kmold(kkc)

    band_1 = vaspeig.eigenvalues.get(Spin.up)[bandindex,:,0]
    band_2 = vaspeig.eigenvalues.get(Spin.down)[bandindex,:,0] if ispin == 2 else 0

    nline = len(HSP.split('-')) - 1

    print('ispin: ', ispin)
    print('band_gap, cbm, vbm, is_band_gap_direct: ', vaspeig.eigenvalue_band_properties)
    print('fermi level in plot: ', fermi_dft)

    '''
      * Plot
    '''
    plt.figure(figsize=[4, 3], dpi=150)

    plt.axis([kpath.min(), kpath.max(), yy[0], yy[1]])

    ax = plt.gca()
    ax.set_ylabel('$E$ (eV)')

    plt.axhline(0, linestyle='--', color='k', linewidth=1, alpha=1, zorder=2)
    for i in range(1, nline):
        plt.axvline(x=kpath[i * int(nk / nline)], linestyle='--', color='k', linewidth=1, alpha=1, zorder=2)

    # plot HSP
    i_HSP = np.arange(nline+1) * (nk // nline)
    i_HSP[-1] = -1
    plt.xticks(kpath[i_HSP], HSP.split('-'))

    # bloch band
    if ispin == 2:
        if spin == 'all':
            print('plotting both spin up (red) and dn (blue) band. ')
            plt.plot(kpath, band_1-fermi_dft, linewidth=1., linestyle="-", color='red', alpha=1.0)
            plt.plot(kpath, band_2-fermi_dft, linewidth=1., linestyle="-", color='blue', alpha=1.0)
            # plt.legend(loc='best', fontsize=10, frameon=True, facecolor='white', edgecolor='k', framealpha=0)
        elif spin == 'up':
            print('plotting spin up (red) band. ')
            plt.plot(kpath, band_1-fermi_dft, linewidth=1., linestyle="-", color='red', alpha=1.0)
        elif spin == 'dn':
            print('plotting spin dn (blue) band. ')
            plt.plot(kpath, band_2-fermi_dft, linewidth=1., linestyle="-", color='blue', alpha=1.0)
        else:
            pass
    else:
        plt.plot(kpath, band_1-fermi_dft, linewidth=1., linestyle="-", color='black', alpha=1.0)

    # plt.show()
    plt.tight_layout()

    if savefig:
        plt.savefig('dftband.pdf')
    else:
        plt.show()

def plot_wan_band(w90_band, HSP, fermi_wan, yy=[-1,1], savefig=False):
    nline = len(HSP.split('-')) - 1

    kpath = w90_band.kpath
    nk = kpath.shape[0]

    plt.figure(figsize=[4, 3], dpi=300)

    plt.axis([kpath.min(), kpath.max(), yy[0], yy[1]])

    ax = plt.gca()
    ax.set_ylabel('$E$ (eV)')

    plt.axhline(0, linestyle='--', color='k', linewidth=1, alpha=1, zorder=2)
    for i in range(1, nline):
        plt.axvline(x=kpath[i * int(nk / nline)], linestyle='--', color='k', linewidth=1, alpha=1, zorder=2)

    # plot HSP
    i_HSP = np.arange(nline+1) * (nk // nline)
    i_HSP[-1] = -1
    plt.xticks(kpath[i_HSP], HSP.split('-'))

    # wannier band
    for i in range(w90_band.nw):
        plt.plot(kpath, w90_band.eig.T[i]-fermi_wan, linewidth=0.6, linestyle="-", color='green', alpha=0.6, zorder=20)

    # plt.show()
    plt.tight_layout()

    if savefig:
        plt.savefig('band_wannier.pdf')
    else:
        plt.show()

def main():
    wdir = os.getcwd()
    os.chdir(wdir)

    if os.path.exists(r'vasprun.xml'):
        print('fermi level read from vasprun.xml')
        fermi_dft = float(re.findall('.\d+.\d+', os.popen('grep fermi vasprun.xml').readline())[0])
        fermi_wan = fermi_dft
    else:
        fermi_dft = None
        fermi_wan = None
    print('dft fermi level: {}'.format(fermi_dft))

    argv = sys.argv[1:]

    usage = "wpyplotband -j [compare|dft|wannier] --spin [1|up|dn] --seedname wannier90 -y '-1 1' -e 0 -w 0 -K 'G-G' "
    parser = OptionParser(usage)
    parser.add_option("--seedname", action="store", dest="seedname", default=None)
    parser.add_option("-j", "--job", action="store", choices=['dft', 'wannier', 'compare'], dest="job", default='dft')
    parser.add_option("-y", "--yy", action="store", type="string", dest="yy", default=r'-5 5')
    parser.add_option("-e", "--fermi", action="store", type="float", dest="fermi_dft", default=fermi_dft)
    parser.add_option("-w", "--fermiwan", action="store", type="float", dest="fermi_wan", default=fermi_wan)
    parser.add_option("-K", "--hsp", action="store", type="string", dest="HSP", default="G-G")
    parser.add_option("--spin", action="store", choices=['all', 'up', 'dn'], dest="spin", default="all")
    parser.add_option("-o", "--save", action="store_true", dest="savefig", default=False)

    options, args = parser.parse_args(argv)

    Job = options.job
    yy = np.array(options.yy.split(), dtype='float64')
    HSP = options.HSP
    nline = len(HSP.split('-')) - 1
    fermi_dft = options.fermi_dft
    fermi_wan = options.fermi_wan
    spin = options.spin
    seedname = options.seedname
    if seedname is None:
        seedname = {'all': 'wannier90', 'up': 'wannier90.up', 'dn': 'wannier90.dn'}.get(spin)

    savefig = options.savefig

    if os.path.exists(seedname+'.wout'):
        HSP = load_HSP_from_wout(seedname)

    if savefig:
        matplotlib.use('Agg')

    if Job == "wannier":
        cell = Cell()
        cell.load_poscar()

        w90_band = W90_band(cell.lattice)
        w90_band.load_from_w90(seedname)

        plot_wan_band(w90_band, HSP, fermi_wan, yy=yy, savefig=savefig)
    elif Job == "dft":
        cell = Cell()
        cell.load_poscar()
        vaspeig = Eigenval(r'EIGENVAL.HSP')

        plot_dft_band(cell, vaspeig, HSP=HSP, yy=yy, fermi_dft=fermi_dft, spin=spin, savefig=savefig)
    elif Job == "compare":
        cell = Cell()
        cell.load_poscar()
        vaspeig = Eigenval(r'EIGENVAL.HSP')

        w90_band = W90_band(cell.lattice)
        w90_band.load_from_w90(seedname)

        plot_compare(cell, vaspeig, w90_band, yy=yy, HSP=HSP, fermi_dft=fermi_dft, fermi_wan=fermi_wan, spin=spin, savefig=savefig)

if __name__ == '__main__':
    main()