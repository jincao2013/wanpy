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

import time
import os
import numpy as np
import argparse
import sys
# sys.path.append(os.environ.get('PYTHONPATH'))

def handle_wannier(args):
    import re
    from wanpy import Cell, parse_symmetry_inputfile
    from wanpy.interface import WannierInterpolation

    if args.write_temp:
        if os.path.exists('symmetry.in'):
            print('file symmetry.in exist.')
            sys.exit()
        with open('symmetry.in', 'w') as f:
            f.write(temp_input_file)
        f.close()
        return

    efermi = args.efermi
    if args.efermi is None:
        efermi = 0.0
        if os.path.exists('vasprun.xml'):
            efermi = float(re.findall(r'.\d+.\d+', os.popen('grep efermi vasprun.xml').readline())[0])
            print('fermi level is {} readed from vasprun.xml'.format(efermi))

    if args.cal_spin:
        print('WAVECAR will be used for interpolating spin operator.')

    T0 = time.time()
    dat_fmt = '12.6'
    # symmetric_method, rspace_use_ngridR, ngridR_symmhtb, symmops = None, None, None, None
    win = parse_symmetry_inputfile('symmetry.in')
    symmops = win.symmops

    if args.symmetry:
        dat_fmt = '16.10'
        if win.parse_symmetry == 'auto':
            print('Using magmoms to determine msg, the following msg is found: ')
            cell = Cell()
            cell.load_poscar('POSCAR')
            symmops = cell.get_msg(win.magmoms, win.symprec,
                                   wannier_center_def=win.wannier_center_def, info=True)

    wanrun = WannierInterpolation(efermi, seedname=args.seedname,
                                  symmetric_htb=args.symmetry, symmetric_method=win.symmetric_method,
                                  wannier_center_def=win.wannier_center_def,
                                  rspace_use_ngridR=win.rspace_use_ngridR,
                                  ngridR_symmhtb=win.ngridR, symmops=symmops,
                                  check_if_uudd_amn=True
                                  )
    wanrun.run(cal_r=args.cal_r, cal_spin=args.cal_spin,
               write_h5=True, write_dat=args.write_dat, write_spn=args.write_spn,
               h5decimals=args.h5decimals, fmt=dat_fmt, iprint=args.verbose,
               )
    T1 = time.time()

    print('Wannier interpolation completed, time consuming {:.3f} s'.format(T1-T0))

def handle_htb(args):
    import re
    from wanpy import Htb, Cell, Symmetrize_Htb_rspace, Symmetrize_Htb_kspace, get_proj_info, parse_symmetry_inputfile

    efermi = args.efermi
    if args.efermi is None:
        efermi = 0.0
        if os.path.exists('vasprun.xml'):
            efermi = float(re.findall(r'.\d+.\d+', os.popen('grep efermi vasprun.xml').readline())[0])
            print('fermi level is {} readed from vasprun.xml'.format(efermi))

    # print instructions
    print('check the wcc before calculations.')
    print('Make sure the following setting in wannier90: ')
    print('guiding_centres      =  T')
    print('translate_home_cell  =  F (it may shift the wcc within a lattice, which is far from the initial projections.)')
    print('use_ws_distance      =  F (see J. Phys. Cond. Matt. 32, 165902 (2020))')
    # print('transl_inv           =  T')    # postw90 tag

    htb = Htb(efermi)

    T0 = time.time()
    win = parse_symmetry_inputfile('symmetry.in')
    htb.load_wannier90_dat(seedname=args.seedname, load_wsvec=args.load_wsvec, wannier_center_def=win.wannier_center_def)
    T1 = time.time()
    print('complete loading hamiltonian from {}, time consuming {:.3f} s'.format(args.seedname, T1-T0))

    symmops = win.symmops
    # symmetric wannier htb
    if args.symmetry:
        print('symmetrizing Wannier TB model')

        if win.parse_symmetry == 'auto':
            print('Using magmoms to determine msg, the following msg is found: ')
            cell = Cell()
            cell.load_poscar('POSCAR')
            symmops = cell.get_msg(win.magmoms, win.symprec, info=True)
        atoms_pos, atoms_orbi = get_proj_info(htb=htb, wannier_center_def=win.wannier_center_def)


        if win.symmetric_method[0] == 'k':
            symmhtb = Symmetrize_Htb_kspace(ngridR=win.ngridR,
                                            htb=htb,
                                            symmops=symmops,
                                            atoms_pos=atoms_pos,
                                            atoms_orbi=atoms_orbi,
                                            soc=htb.worbi.soc
                                            # iprint=iprint,
                                            )
            symmhtb.run(tmin=1e-6)
            htb = symmhtb.htb
        elif win.symmetric_method[0] == 'r':
            symmhtb = Symmetrize_Htb_rspace(htb=htb,
                                            symmops=symmops,
                                            atoms_pos=atoms_pos,
                                            atoms_orbi=atoms_orbi,
                                            soc=htb.worbi.soc,
                                            # iprint=iprint,
                                            )
            if win.rspace_use_ngridR:
                symmhtb.use_ngridR(win.ngridR)
            symmhtb.run()
            htb = symmhtb.htb

    print('Fermi level = {} eV'.format(htb.fermi))

    print('The following items are saved in .h5 format')
    print(htb._contents)
    print(htb._contents_large)

    htb.save_htb()
    T2 = time.time()
    print('h5htb complete, time consuming {:.3f} s'.format(T2-T0))

def handle_write_dat(args):
    from wanpy import Htb

    fmt = str(6+args.decimals) + '.' + str(args.decimals)
    htb = Htb()
    htb.load_h5(args.htb_fname)
    htb.save_wannier90_hr_dat(fmt=fmt)
    if args.save_r:
        htb.save_wannier90_r_dat(fmt=fmt)

def handle_twist_amn(args):
    from wanpy import wannier90_load_wcc, wanpy_check_if_uudd_amn
    from wanpy.interface import W90_amn

    is_uudd_amn, is_udud_amn = None, None
    if not args.skip_check:
        wcc, wccf, wbroaden = wannier90_load_wcc('wannier90.wout', shiftincell=False, check_if_uudd_amn=False)
        is_uudd_amn = wanpy_check_if_uudd_amn(wcc, wbroaden)
        is_udud_amn = not is_uudd_amn

    if args.twist_type == 'udud_to_uudd':
        if is_uudd_amn:
            print('\033[91mThe .amn file is in udud order. No action has been taken.\033[0m')
            return
        os.rename('wannier90.amn', 'wannier90_udud.amn')
        print('found wannier90.amn')
        print('this will twist .amn order from udud to uudd')
        AMN = W90_amn()
        AMN.load_from_w90('wannier90_udud.amn')
        AMN.amn = np.einsum('knis->knsi', AMN.amn.reshape([AMN.nk, AMN.nb, AMN.nw // 2, 2])).reshape([AMN.nk, AMN.nb, -1])
        AMN.save_w90('wannier90_uudd.amn')
        os.system('ln -s wannier90_uudd.amn wannier90.amn')
    elif args.twist_type == 'uudd_to_udud':
        if is_udud_amn:
            print('\033[91mThe .amn file is in uudd order. No action has been taken.\033[0m')
            return
        os.rename('wannier90.amn', 'wannier90_uudd.amn')
        print('found wannier90.amn')
        print('this will twist .amn order from uudd to udud')
        AMN = W90_amn()
        AMN.load_from_w90('wannier90_uudd.amn')
        AMN.amn = np.einsum('knsi->knis', AMN.amn.reshape([AMN.nk, AMN.nb, 2, AMN.nw//2])).reshape([AMN.nk, AMN.nb, -1])
        AMN.save_w90('wannier90_udud.amn')
        os.system('ln -s wannier90_uudd.amn wannier90.amn')
    else:
        print('unknown twist_type')

def main():
    # fermi_vasprun = 0

    parser = argparse.ArgumentParser(description='wanpy code')
    subparsers = parser.add_subparsers(dest='command')

    ''' Subparser for the wannier command '''
    wannier_parser = subparsers.add_parser('wannier', help='wanpy wannier [options]', epilog='')
    wannier_parser.add_argument('-e', '--efermi', action='store', type=float, default=None,
                                help='fermi level, default is readed from vasprun.xml, or zero if there is no vasprun.xml')
    wannier_parser.add_argument('--seedname', action='store', type=str, default='wannier90',
                                help='default is wannier90')
    wannier_parser.add_argument('--symmetry', action='store_true', dest='symmetry',
                                help='if symmetrizing wannier tb, default is False')
    wannier_parser.add_argument('-r', '--ramn', action='store_true', dest='cal_r',
                                help='if interpolat r operator, default is False')
    wannier_parser.add_argument('--verbose', action='store', type=int, dest="verbose", default=1,
                                help='default is 1, set 2 to be more verbos.')
    wannier_parser.add_argument('--spin', action='store_true', dest='cal_spin',
                                help='if interpolat spin operator, default is False')
    wannier_parser.add_argument('--spn', action='store_true', dest='write_spn',
                                help='if write .spn file (formated), default is False')
    wannier_parser.add_argument('--dat', action='store_true', dest='write_dat',
                                help='if write .dat file, default is False. It will save with 6 decimals by default, and 10 decimals for symmetric case')
    wannier_parser.add_argument('--h5decimals', action='store', type=int, dest='h5decimals', default=16,
                                help='save .h5 with rounded double precision, default decimals is 16')
    wannier_parser.add_argument('--temp', action='store_true', dest='write_temp',
                                help='write a templet symmetry.in file, default is False')

    ''' Subparser for the htb command '''
    htb_parser = subparsers.add_parser('htb', help='wanpy htb [options]')
    htb_parser.add_argument('-e', '--efermi', action='store', type=float, default=None,
                            help='fermi level, default is readed from vasprun.xml, or zero if there is no vasprun.xml')
    htb_parser.add_argument('--seedname', action='store', type=str, default='wannier90')
    htb_parser.add_argument('--symmetry', action='store_true', dest='symmetry',
                            help='if symmetrizing wannier tb, default is False')
    htb_parser.add_argument('--wsvec', action='store_true', dest='load_wsvec',
                            help='if load wannier90_wsvec.dat, default is False')

    ''' Subparser for the write_dat command '''
    write_dat_parser = subparsers.add_parser('write_dat', help='wanpy write_dat [options]')
    write_dat_parser.add_argument('-n', '--name', action='store', type=str, dest='htb_fname', default='htb.h5',
                                  help='fname of .h5, default is htb.h5')
    write_dat_parser.add_argument('--decimals', action='store', type=int, dest='decimals', default=10,
                                  help='save .dat with rounded values, default decimals is 10')
    write_dat_parser.add_argument('-r', '--ramn', action='store_true', dest='save_r',
                                  help='if write _r.dat, default is False')

    ''' Subparser for the twist_amn command '''
    twist_amn_parser = subparsers.add_parser('twist_amn', help='wanpy twist_amn [options]')
    twist_amn_parser.add_argument('-t', '--type', action='store', type=str, dest='twist_type', default='udud_to_uudd',
                                  help='udud_to_uudd or uudd_to_udud, default is udud_to_uudd')
    twist_amn_parser.add_argument('--skip_check', action='store_true', dest='skip_check',
                                  help='skip check the amn order from .wout, default is False')

    args = parser.parse_args()

    if args.command == 'wannier':
        handle_wannier(args)
    elif args.command == 'htb':
        handle_htb(args)
    elif args.command == 'write_dat':
        handle_write_dat(args)
    elif args.command == 'twist_amn':
        handle_twist_amn(args)
    else:
        print("Unknown command")

temp_input_file = """# Input file for building symmetric Wannier TB model

# Choose method, default is kspace
symmetric_method = rspace   # Options: {rspace, kspace}
rspace_use_ngridR = F       # use ngridR to resample the TB model, default is False

# Choose between manually setting symmops or automatically detecting symmops from magmoms
parse_symmetry = auto       # Options: {man, auto}

# Choose how wannier centers are defined in calculating .amn
# Options: {poscar, ws}
#   poscar: the same as poscar. Use this if amn is generated by VASP 5.4.4 (see mlwf.F for details). 
#   ws: refined in range of [-0.5, 0.5). Use this if amn is generated by VASP 6.4.3 (see mlwf.F for details). 
wannier_center_def = ws 

ngridR = 12 12 1            # in kspace method, use a slightly larger value than the original TB model 

# Parameters used when parse_symmetry = auto
symprec = 1e-5
&magmoms
0 0 1
0 0 -1
/

# Parameters used when parse_symmetry = man
&symmops
# TR  det  alpha  nx  ny  nz  taux  tauy  tauz
  0   1    0      0   0   1   0     0     0   # e
  0   1    180    0   0   1   0.5   0.5   0   # c2z
# Anti-unitary symmetry operations
  1   1    0      0   0   1   0.5   0.5   0   # T
  1   1    180    0   0   1   0     0     0   # Tc2z
/
"""

if __name__ == '__main__':
    main()
