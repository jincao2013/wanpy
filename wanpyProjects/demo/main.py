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

__date__ = "Mar. 31, 2025"

import os
import argparse
import numpy as np
from numpy import linalg as LA
from mpi4py import MPI
import wanpy as wp
import wanpy.response as res
from wanpy.MPI import Config
from wanpy.MPI import MPI_Reduce, MPI_Gather, init_kmesh, init_htb_response_data
from wanpyProjects.demo.lib import libwtb

# if wp.PYGUI:
#     import matplotlib.pyplot as plt

if wp.PYGUI:
    wdir = os.path.join(wp.ROOT_WDIR, r'demo')
    config_dir = os.path.join(wp.ROOT_WDIR, r'demo/configs')
    config_path = os.path.join(config_dir, "config.toml")
else:
    wdir = os.getcwd()
    # load config_path from command-line argument
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("-t", "--toml", required=True, help="Path to config.toml")
    args = parser.parse_args()
    config_path = args.toml

'''
  * main
'''
if __name__ == '__main__':
    # Load config.toml
    cf = Config(MPI)
    cf.load_config(config_path)

    # Initialize htb object and loads htb data from .h5 for main MPI rank
    htb = wp.Htb()
    if cf.MPI_main: htb.load_h5(os.path.join(cf.htb_dir, cf.htb_fname))
    os.chdir(wdir)

    ''' Custom parameters: '''
    ''' End of custom parameters  '''

    cf.print_config()
    htb = init_htb_response_data(MPI, htb, tmin_h=cf.tmin_h, tmin_r=cf.tmin_r, open_boundary=cf.open_boundary, istb=cf.istb, use_wcc=cf.use_wcc, atomic_wcc=cf.atomic_wcc)
    kmesh = init_kmesh(MPI, cf.nkmesh, random_k=cf.random_k, kcube=cf.kcube, kmesh_shift=cf.kmesh_shift)

    cf.nk = kmesh.shape[0] if cf.MPI_main else None
    cf.start_timer()

'''
  * MPI calculator
'''
@MPI_Gather(MPI, iterprint=cf.iterprint, dtype='float64')
def cal_Fermi_surface(k, dim):
    return libwtb.cal_Fermi_surface(htb, k, cf.omega, eta=cf.ewidth_imag)

if __name__ == '__main__' and cf.job == 'dos':
    dim = [1]
    RS = cal_Fermi_surface(kmesh, dim)
    if cf.MPI_main:
        dos = RS
        kmesh_car = (htb.lattG @ kmesh.T).T
        np.savez_compressed(r'dos.npz', cf=cf.dict_serializable(),
                            lattG=htb.lattG, dos=dos)

if __name__ == '__main__' and cf.job == 'debug':
    pass

'''
  * End 
'''
if __name__ == '__main__' and cf.job is not None:
    cf.end_timer()