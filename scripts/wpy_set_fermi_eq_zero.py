#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Jin Cao'
__copyright__ = "Copyright 2017, Quantum Functional Materials Design and Application Laboratory"
__version__ = "0.99"
__maintainer__ = "Jin Cao"
__email__ = "caojin.phy@gmail.com"
__date__ = "Dec. 4, 2019"

import time
import os
import sys
import re
from optparse import OptionParser
sys.path.append(os.environ.get('PYTHONPATH'))
import numpy as np
from wanpy.interface.wannier90 import W90_eig

def read_par():
    with open('wannier90.amn', 'r') as f:
        f.readline()
        nb, nk, nw = np.array(f.readline().split(), dtype='int64')
    return nb, nk, nw

def main():
    fermi_vasp = float(re.findall('.\d+.\d+', os.popen('grep fermi vasprun.xml').readline())[0])

    argv = sys.argv[1:]

    usage = "set_eig_fermi_eq_zero [ -f <wannier90.eig> -e <efermi>] arg1[,arg2..]"
    parser = OptionParser(usage)
    parser.add_option("-e", "--efermi", action="store", type="float", dest="fermi", default=fermi_vasp)
    parser.add_option("-f", "--fname", action="store", type="string", dest="fname", default='wannier90.eig')

    options, args = parser.parse_args(argv)

    fermi = options.fermi
    fname = options.fname

    savefname = r'wannier90.eig.zero'
    nb, nk, nw = read_par()
    EIG = W90_eig(nb, nk)
    EIG.load_from_w90(fname)
    EIG.eig -= fermi
    EIG.save_w90(savefname)

    print('Fermi level have been set at 0 eV by adjusting')
    print('E(wannier) = E(DFT) - ({}) eV'.format(fermi))


if __name__ == '__main__':
    work_dir = os.getcwd()
    os.chdir(work_dir)
    main()
