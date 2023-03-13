#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Jin Cao'
__copyright__ = "Copyright 2017, Quantum Functional Materials Design and Application Laboratory"
__version__ = "1.01"
__maintainer__ = "Jin Cao"
__email__ = "cao.jin.phy@gmail.com"
__date__ = "Jan. 3, 2019"

import time
import os
import sys
import getpass
from optparse import OptionParser
sys.path.append(os.environ.get('PYTHONPATH'))
from wanpy.core.DEL.read_write import Wout


if __name__ == '__main__':
    wdir = os.getcwd()
    os.chdir(wdir)

    wout = Wout()
    wout.load_wc(fname=r'wannier90.wout', shiftincell=False)
    wout.save_wcc()


