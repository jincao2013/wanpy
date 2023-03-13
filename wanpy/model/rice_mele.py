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

__date__ = "Jan. 16, 2019"

import os
from wanpy.core.DEL.read_write import Cell, Wout, Htb
from wanpy.core.DEL._calband import TBband


def rdt(nn):
    t = np.random.uniform(0.0, 2.0) * np.exp(-nn)
    return round(t, 4)


def get_RiceMele_model(niu=1.0, t=2.0, dt=1.0):
    '''
        Rice Mele Model
        band width = t - dt
    '''
    '''
      * Cell
    '''
    a = 2
    cell = Cell()
    cell.name = r'Rice Mele TB Model'
    cell.lattice = np.diag(np.array([a, 2, 2]))
    cell.latticeG = cell.get_latticeG()
    cell.N = 2
    cell.spec = ['C', 'H']
    cell.ions = np.array([
        [0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ])
    cell.ions_car = cell.get_ions_car()

    '''
      * Wout
    '''
    wout = Wout()
    wout.lattice = cell.lattice
    wout.wcc = cell.ions_car
    wout.wccf = cell.ions
    wout.wborden = np.array([1., 1.])

    '''
      * Htb
    '''
    htb = Htb(fermi=0.)
    htb.name = r'Rice Mele TB Model (niu={}, t={}, dt={})'.format(niu, t, dt)
    htb.nw = 2
    htb.nR = 3
    htb.R = np.array([
        [ 0, 0, 0],
        [-1, 0, 0],
        [ 1, 0, 0],
    ])
    htb.ndegen = np.array([1, 1, 1])
    htb.hr_Rmn = np.zeros([htb.nR, htb.nw, htb.nw], dtype='complex128')
    htb.r_Ramn = np.zeros([htb.nR, 3, htb.nw, htb.nw], dtype='complex128')
    htb.cell = cell
    htb.wout = wout
    htb.wcc = cell.ions_car
    htb.wccf = cell.ions

    t1 = 0.5 * (t + dt)
    t2 = 0.5 * (t - dt)
    htb.hr_Rmn[0] = niu * np.diagflat([1, -1], 0) + np.diagflat([t1], 1) + np.diagflat([t1], -1)
    htb.hr_Rmn[1] = np.diagflat([t2], 1)
    htb.hr_Rmn[2] = np.diagflat([t2], -1)

    htb.r_Ramn[0, 0] = np.diagflat(htb.cell.ions_car.T[0])
    htb.r_Ramn[0, 1] = np.diagflat(htb.cell.ions_car.T[1])
    htb.r_Ramn[0, 2] = np.diagflat(htb.cell.ions_car.T[2])

    return htb


def get_RiceMele_4band_model(niu=1.0, t=2.0, dt=1.0):
    '''
        Modified Rice Mele Model
    '''
    '''
      * Cell
    '''
    a = 4
    cell = Cell()
    cell.name = r'4Band Rice Mele TB Model'
    cell.lattice = np.diag(np.array([a, 2, 2]))
    cell.latticeG = cell.get_latticeG()
    cell.N = 4
    cell.spec = ['C', 'C', 'C', 'C']
    cell.ions = np.array([
        [0, 0.5, 0.5],
        [1/4, 0.5, 0.5],
        [2/4, 0.5, 0.5],
        [3/4, 0.5, 0.5],
    ])
    cell.ions_car = cell.get_ions_car()

    '''
      * Wout
    '''
    wout = Wout()
    wout.lattice = cell.lattice
    wout.wcc = cell.ions_car
    wout.wccf = cell.ions
    wout.wborden = np.array([1., 1., 1., 1.])

    '''
      * Htb
    '''
    htb = Htb(fermi=0.)
    htb.name = r'4Band Rice Mele TB Model'
    htb.nw = 4
    htb.nR = 3
    htb.R = np.array([
        [ 0, 0, 0],
        [-1, 0, 0],
        [ 1, 0, 0],
    ])
    htb.ndegen = np.array([1, 1, 1])
    htb.hr_Rmn = np.zeros([htb.nR, htb.nw, htb.nw], dtype='complex128')
    htb.r_Ramn = np.zeros([htb.nR, 3, htb.nw, htb.nw], dtype='complex128')
    htb.cell = cell
    htb.wout = wout
    htb.wcc = cell.ions_car
    htb.wccf = cell.ions

    t1 = 0.5 * (t + dt)
    t2 = 0.5 * (t - dt)
    # s1 = 0.5 * (s + ds)
    # s2 = 0.5 * (s - ds)
    # r1 = 0.5 * (r + dr)
    # r2 = 0.5 * (r - dr)

    # htb.hr_Rmn[0] = niu * np.diagflat([1, -1, 1, -1], 0) + \
    #                 np.diagflat([t1, t2, t1], 1) + np.diagflat([t1, t2, t1], -1) + \
    #                 np.diagflat([rdt(1), rdt(1)], 2) + np.diagflat([rdt(1), rdt(1)], -2) + \
    #                 np.diagflat([rdt(1)], 3) + np.diagflat([rdt(1)], -3)
    # htb.hr_Rmn[0] = 0.5 * (htb.hr_Rmn[0] + htb.hr_Rmn[0].T)
    # htb.hr_Rmn[1] = np.diagflat([t2], 3) + \
    #                 np.diagflat([rdt(1), rdt(1), rdt(1)], 1) + np.diagflat([rdt(1), rdt(1)], 2)
    # htb.hr_Rmn[2] = htb.hr_Rmn[1].T

    htb.hr_Rmn[0] = niu * np.diagflat([1, -1, 1, -1], 0) + \
                    np.diagflat([t1, t2, t1], 1) + np.diagflat([t1, t2, t1], -1) + \
                    np.diagflat([0.6, 0.5], 2) + np.diagflat([0.6, 0.5], -2) + \
                    np.diagflat([0.4], 3) + np.diagflat([0.4], -3)
    htb.hr_Rmn[0] = 0.5 * (htb.hr_Rmn[0] + htb.hr_Rmn[0].T)
    htb.hr_Rmn[1] = np.diagflat([0.3], 3) + \
                    np.diagflat([0.5, 0.8, 0.6], 1) + np.diagflat([0.6, 0.7], 2)
    htb.hr_Rmn[2] = htb.hr_Rmn[1].T

    htb.r_Ramn[0, 0] = np.diagflat(htb.cell.ions_car.T[0])
    htb.r_Ramn[0, 1] = np.diagflat(htb.cell.ions_car.T[1])
    htb.r_Ramn[0, 2] = np.diagflat(htb.cell.ions_car.T[2])

    return htb


def get_Random_4band_model(niu=1.0, t=2.0, dt=1.0):
    '''
        Modified Rice Mele Model
    '''
    '''
      * Cell
    '''
    a = 4
    cell = Cell()
    cell.name = r'4Band Rice Mele TB Model'
    cell.lattice = np.diag(np.array([a, 2, 2]))
    cell.latticeG = cell.get_latticeG()
    cell.N = 4
    cell.spec = ['C', 'C', 'C', 'C']
    cell.ions = np.array([
        [0, 0.5, 0.5],
        [1/4, 0.5, 0.5],
        [2/4, 0.5, 0.5],
        [3/4, 0.5, 0.5],
    ])
    cell.ions_car = cell.get_ions_car()

    '''
      * Wout
    '''
    wout = Wout()
    wout.lattice = cell.lattice
    wout.wcc = cell.ions_car
    wout.wccf = cell.ions
    wout.wborden = np.array([1., 1., 1., 1.])

    '''
      * Htb
    '''
    htb = Htb(fermi=0.)
    htb.name = r'4Band Rice Mele TB Model'
    htb.nw = 4
    htb.nR = 3
    htb.R = np.array([
        [ 0, 0, 0],
        [-1, 0, 0],
        [ 1, 0, 0],
    ])
    htb.ndegen = np.array([1, 1, 1])
    htb.hr_Rmn = np.zeros([htb.nR, htb.nw, htb.nw], dtype='complex128')
    htb.r_Ramn = np.zeros([htb.nR, 3, htb.nw, htb.nw], dtype='complex128')
    htb.cell = cell
    htb.wout = wout
    htb.wcc = cell.ions_car
    htb.wccf = cell.ions

    t1 = 0.5 * (t + dt)
    t2 = 0.5 * (t - dt)
    htb.hr_Rmn[0] = niu * np.diagflat([1, -1, 1, -1], 0) + \
                    np.diagflat([t1, t2, t1], 1) + np.diagflat([t1, t2, t1], -1) + \
                    np.diagflat([rdt(0), rdt(0), rdt(0), rdt(0)], 0) + \
                    np.diagflat([rdt(1), rdt(1), rdt(1)], 1) + np.diagflat([rdt(1), rdt(1), rdt(1)], -1) + \
                    np.diagflat([rdt(2), rdt(2)], 2) + np.diagflat([rdt(2), rdt(2)], -2) + \
                    np.diagflat([rdt(3)], 3) + np.diagflat([rdt(3)], -3)
    htb.hr_Rmn[0] = 0.5 * (htb.hr_Rmn[0] + htb.hr_Rmn[0].T)
    htb.hr_Rmn[1] = np.diagflat([t2], 3) + \
                    np.diagflat([rdt(1)], 3) + \
                    np.diagflat([rdt(3), rdt(3), rdt(3)], 1) + np.diagflat([rdt(2), rdt(2)], 2)
    htb.hr_Rmn[2] = htb.hr_Rmn[1].T

    htb.r_Ramn[0, 0] = np.diagflat(htb.cell.ions_car.T[0])
    htb.r_Ramn[0, 1] = np.diagflat(htb.cell.ions_car.T[1])
    htb.r_Ramn[0, 2] = np.diagflat(htb.cell.ions_car.T[2])

    return htb


if __name__ == '__main__':
    wdir = r'F:\##Research_Data_BIT\WORK_11_NOR\4_Rice_Mele\1_2band'
    wdir = r'F:\##Research_Data_BIT\WORK_11_NOR\4_Rice_Mele\2_4band'
    os.chdir(wdir)

    htb_fname = r'htb.npz'

    Nslab = 1
    nk1 = 101
    kpath = np.array([
        [-0.5, 0.0, 0.0], # G
        [ 0.0, 0.0, 0.0], # G
        [ 0.5, 0.0, 0.0], # M
    ])

    # htb = get_RiceMele_model(niu=0.5, t=1.5, dt=0.2)
    htb = get_RiceMele_4band_model(niu=1.0, t=1.6, dt=1.0)
    # htb = get_Random_4band_model(niu=1.0, t=1.5, dt=0.2)
    htb.save_htb()
    htb.cell.save_poscar()
    htb.save_wcc()

    tbband = TBband(kpath, nk1)
    tbband.load_htb(htb_fname)
    tbband.cal_band(unit='D')
    tbband.plot_band(eemin=-3.0, eemax=5.0)

