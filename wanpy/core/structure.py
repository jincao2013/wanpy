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

__date__ = "May. 23, 2020"

import os
import errno
import sys

from wanpy.core.errorhandler import WanpyInputError

sys.path.append(os.environ.get('PYTHONPATH'))
import spglib
from enum import Enum, unique
import numpy as np
from numpy import linalg as LA
from wanpy.core.utils import get_ntheta_from_rotmatrix, print_symmops, wanpy_check_if_uudd_amn
from wanpy.core.utils import wannier90_read_hr, wannier90_read_rr, wannier90_load_wcc, \
    wannier90_read_spin, wannier90_load_wsvec
import h5py

__all__ = [
    'periodic_table',
    'Cell', 'Worbi', 'Htb'
]

'''
  Basic object
'''
periodic_table = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
    'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}
periodic_table_inv = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
    19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
    41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
    61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
    71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
    91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
    101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt',
    110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'
}


@unique
class OrbitalType(Enum):
    """
    Enum type for orbital type. Indices are basically the azimuthal quantum
    number, l.
    """

    s = 0
    p = 1
    d = 2
    f = 3

    def __str__(self):
        return str(self.name)


@unique
class Orbital(Enum):
    """
    Enum type for specific orbitals. The indices are basically the order in
    which the orbitals are reported in VASP and has no special meaning.
    """

    s = 0
    py = 1
    pz = 2
    px = 3
    dxy = 4
    dyz = 5
    dz2 = 6
    dxz = 7
    dx2 = 8
    f_3 = 9
    f_2 = 10
    f_1 = 11
    f0 = 12
    f1 = 13
    f2 = 14
    f3 = 15

    def __int__(self):
        return self.value

    def __str__(self):
        return str(self.name)

    @property
    def orbital_type(self):
        """
        Returns OrbitalType of an orbital.
        """
        # pylint: disable=E1136
        return OrbitalType[self.name[0]]


class Cell(object):

    def __init__(self):
        self._container = ['name', 'lattice', 'latticeG', 'N', 'spec', 'ions', 'ions_car']
        self.name = None
        self.lattice = None
        self.latticeG = None
        self.N = None
        self._spec = None
        self.ions = None
        self.ions_car = None

    @property
    def spec(self):
        return [i.decode('utf-8') for i in self._spec]

    @spec.setter
    def spec(self, value):
        self._spec = [i if type(i) is bytes else i.encode('utf-8') for i in value]

    def get_latticeG(self):
        return 2 * np.pi * LA.inv(self.lattice.T)

    def get_ions(self):
        if self.ions_car is None:
            print('self.ions_car is None')
            return None
        return LA.multi_dot([LA.inv(self.lattice), self.ions_car.T]).T

    def get_ions_car(self):
        if self.ions is None:
            print('self.ions is None')
            return None
        return LA.multi_dot([self.lattice, self.ions.T]).T

    def get_vcell(self):
        return LA.det(self.lattice)

    def load_poscar(self, fname=r'POSCAR'):
        with open(fname, 'r') as poscar:
            name = poscar.readline().strip()
            t = float(poscar.readline())
            a1 = t * np.array(poscar.readline().split(), dtype='float64')
            a2 = t * np.array(poscar.readline().split(), dtype='float64')
            a3 = t * np.array(poscar.readline().split(), dtype='float64')
            lattice = np.array([a1, a2, a3]).T
            latticeG = 2 * np.pi * LA.inv(lattice.T)

            spec_name = poscar.readline().split()
            spec_num = np.array(poscar.readline().split(), dtype='int64')
            num_ion = spec_num.sum()
            ion_spec = []
            for i in range(len(spec_name)):
                for num in range(spec_num[i]):
                    ion_spec.append(spec_name[i])

            ion_unit = poscar.readline().strip().upper()[0]

            ions = np.zeros((num_ion, 3))
            for i in range(num_ion):
                inline = poscar.readline().split()
                ions[i] = np.array(inline[:3], dtype='float64')
        poscar.close()

        self.name = name
        self.lattice = lattice
        self.latticeG = latticeG
        self.N = num_ion
        self.spec = ion_spec

        if ion_unit == 'D':
            self.ions = ions
            self.ions_car = LA.multi_dot([lattice, ions.T]).T
        elif ion_unit == 'C':
            self.ions = LA.multi_dot([LA.inv(lattice), ions.T]).T
            self.ions_car = ions
        else:
            print('ION_UNIT error in reading poscar')
            sys.exit(1)

    def save_poscar(self, fname='POSCAR.vasp', cartesian=False):
        if os.path.exists(fname):
            os.remove(fname)

        ions_unit = 'D'
        ions = self.ions
        spec_name, spec_index, spec_num = np.unique(self.spec, return_index=True, return_counts=True)
        spec_name = spec_name[np.argsort(spec_index)]
        spec_num = spec_num[np.argsort(spec_index)]

        if cartesian:
            ions_unit = 'C'
            ions = self.ions_car

        with open(fname, 'a') as poscar:
            poscar.write('writen by wanpy\n')
            poscar.write('   1.0\n')
            for i in self.lattice.T: # crystal_data['lattice'].T:
                poscar.write('   {: 2.16f}    {: 2.16f}    {: 2.16f}\n'.format(i[0], i[1], i[2]))

            poscar.write('    '.join(spec_name))
            poscar.write('\n')
            poscar.write('    '.join(np.array(spec_num, dtype='U')))
            poscar.write('\n')

            poscar.write('{}\n'.format(ions_unit))

            for i in ions:
                poscar.write('  {: 2.16f}  {: 2.16f}  {: 2.16f}\n'.format(i[0], i[1], i[2]))
            poscar.write('\n')
        poscar.close()

    def save_h5(self, fname='cell.h5'):
        with h5py.File(fname, "a") as f:
            f.create_group('cell')
            cell = f['cell']
            h5st = h5py.string_dtype(encoding='utf-8')
            hdf5_create_dataset(cell, 'name', data=self.name, dtype=h5st)
            hdf5_create_dataset(cell, 'lattice', data=self.lattice, dtype='float64')
            hdf5_create_dataset(cell, 'latticeG', data=self.latticeG, dtype='float64')
            hdf5_create_dataset(cell, 'N', data=self.N, dtype='int64')
            hdf5_create_dataset(cell, 'spec', data=self._spec, dtype=h5st)
            hdf5_create_dataset(cell, 'ions', data=self.ions, dtype='float64')
            hdf5_create_dataset(cell, 'ions_car', data=self.ions_car, dtype='float64')
            f.close()

    def load_h5(self, fname='cell.h5'):
        f = h5py.File(fname, "r")
        cell = f.get('cell')
        if cell is None:
            f.close()
            return
        for i in self._container:
            item = cell.get(i)
            if item is not None:
                self.__dict__[i] = item[()]
        self.spec = hdf5_read_dataset(cell, 'spec', default=[])
        f.close()

    def get_spglib_cell(self, magmoms=None, wannier_center_def='ws'):
        if wannier_center_def.lower() == 'ws':
            # refined in range of [-0.5, 0.5) to keep in line with the wannier center
            # used in calculating amn in VASP 6.4.3.
            ions = np.remainder(self.ions + 100.5, 1) - 0.5
        elif wannier_center_def.lower() == 'poscar':
            # proj_wccf origins from wannier_setup, and are in line with POSCAR,
            # if wannier_center_def = poscar, do nothing here.
            ions = self.ions
        else:
            WanpyInputError('wannier_center_def should be poscar or ws')

        if magmoms is None:
            magmoms = np.zeros_like(self.ions)

        cell = (self.lattice.T, ions, [periodic_table.get(i) for i in self.spec], magmoms)
        return cell

    def get_msg(self, magmoms, symprec=1e-5, wannier_center_def='ws', info=False):
        cell_mag = self.get_spglib_cell(magmoms, wannier_center_def)
        latt = self.lattice

        info_mag = spglib.get_magnetic_symmetry(cell_mag, symprec=symprec, angle_tolerance=-1.0, mag_symprec=-1.0)
        info_mag_dataset = spglib.get_magnetic_symmetry_dataset(cell_mag, symprec=symprec)
        msg_symbol = spglib.get_magnetic_spacegroup_type(info_mag_dataset.uni_number)
        info_standard_msg = spglib.get_magnetic_symmetry_from_database(info_mag_dataset.uni_number)

        n_operations = info_mag_dataset.n_operations
        msg_type = info_mag_dataset.msg_type
        bns_number = msg_symbol.bns_number
        rot = info_mag_dataset.rotations
        tau = info_mag_dataset.translations
        tau[np.abs(tau) < 1e-5] = 0
        TR = info_mag_dataset.time_reversals
        symmops = np.array(
            [get_ntheta_from_rotmatrix(int(TR[i]), tau[i], latt @ rot[i] @ LA.inv(latt), atol=symprec)
             for i in range(n_operations)
        ])

        if info:
            print('\n\nMagnetic space group for magnetic structure (symprec:{:10.7f} Angstrom)'.format(symprec))
            print('-' * 100)
            # print('  Magnetic space group')
            print('  msg_type: ', msg_type)
            print('  bns_number: ', bns_number)
            print('  n_operations: ', n_operations)
            print('')
            print_symmops(symmops)
            print('-' * 100)
            print('')

        return symmops

    def printer(self, site=True):
        lattice = self.lattice
        latticeG = self.latticeG
        spec = self.spec

        print('')
        print('                                    ------')
        print('                                    SYSTEM')
        print('                                    ------')
        print('')
        print('                              Lattice Vectors (Ang)')
        print('                    a_1  {: 11.6f}{: 11.6f}{: 11.6f}'.format(lattice[0, 0], lattice[1, 0], lattice[2, 0]))
        print('                    a_2  {: 11.6f}{: 11.6f}{: 11.6f}'.format(lattice[0, 1], lattice[1, 1], lattice[2, 1]))
        print('                    a_3  {: 11.6f}{: 11.6f}{: 11.6f}'.format(lattice[0, 2], lattice[1, 2], lattice[2, 2]))
        print('')
        print('                   Unit Cell Volume: {: 13.5f}  (Ang^3)'.format(LA.det(lattice)))
        print('')
        print('                        Reciprocal-Space Vectors (Ang^-1)')
        print('                    b_1  {: 11.6f}{: 11.6f}{: 11.6f}'.format(latticeG[0, 0], latticeG[1, 0], latticeG[2, 0]))
        print('                    b_2  {: 11.6f}{: 11.6f}{: 11.6f}'.format(latticeG[0, 1], latticeG[1, 1], latticeG[2, 1]))
        print('                    b_3  {: 11.6f}{: 11.6f}{: 11.6f}'.format(latticeG[0, 2], latticeG[1, 2], latticeG[2, 2]))
        print('')
        if site:
            print('  +-----------------------------------------------------------------------+')
            print('  | Site      Fractional Coordinate        Cartesian Coordinate (Ang)     |')
            print('  +-----------------------------------------------------------------------+')
            for i in range(self.ions.shape[0]):
                print('  | {} {: 10.5f}{: 10.5f}{: 10.5f}   | {: 10.5f}{: 10.5f}{: 10.5f}   |'.format(
                    spec[i],
                    self.ions[i, 0], self.ions[i, 1], self.ions[i, 2],
                    self.ions_car[i, 0], self.ions_car[i, 1], self.ions_car[i, 2],
                ))
            print('  +-----------------------------------------------------------------------+')
        print('')


class Worbi(object):

    """
      * Note
        The wannier90 interface v1.2 (interface with vasp) writes projections .amn in this order:
        {for spins: for atoms: for atomic orbitals: ...} or uudd for short
        But in the interface v2.x and v3.x, it is
        {for atoms: for atomic orbitals: for spins: ...} or udud for short
        wanpy use uudd order, and will transform nnkp in uudd order.
    """

    def __init__(self):
        self._container = ['latt', 'lattG', 'nw',
                           'proj_wcc', 'proj_wccf', 'proj_lmr', 'proj_zona',
                           'proj_z', 'proj_x',
                           'proj_spin', 'proj_spin_qaxis',
                           'soc'
                           ]
        self.latt = np.zeros([3, 3], dtype='float64')
        self.lattG = np.zeros([3, 3], dtype='float64')
        self.nw = None
        self.proj_wcc = None
        self.proj_wccf = None
        self.proj_lmr = None
        self.proj_z = None
        self.proj_x = None
        self.proj_zona = None
        self.proj_spin = None
        self.proj_spin_qaxis = None
        self.soc = None

    def load_from_nnkp(self, convert_to_uudd, wannier_center_def, seedname='wannier90'):
        print(f'reading {seedname}.nnkp into htb.worbi')
        fname = seedname + '.nnkp'
        with open(fname, 'r') as f:
            inline = f.readline()
            while 'begin real_lattice' not in inline:
                inline = f.readline()

            self.latt.T[0] = np.array(f.readline().split(), dtype='float64')
            self.latt.T[1] = np.array(f.readline().split(), dtype='float64')
            self.latt.T[2] = np.array(f.readline().split(), dtype='float64')

            inline = f.readline()
            while 'begin recip_lattice' not in inline:
                inline = f.readline()

            self.lattG.T[0] = np.array(f.readline().split(), dtype='float64')
            self.lattG.T[1] = np.array(f.readline().split(), dtype='float64')
            self.lattG.T[2] = np.array(f.readline().split(), dtype='float64')

            # inline = f.readline()
            # while 'begin kpoints' not in inline:
            #     inline = f.readline()
            #
            # nk = int(f.readline())
            # kk = np.zeros([nk, 3], dtype='float64')
            # for i in range(nk):
            #     kk[i] = np.array(f.readline().split(), dtype='float64')

            inline = f.readline()
            while 'projections' not in inline:
                inline = f.readline()
                if 'begin spinor_projections' in inline:
                    self.soc = True
                    self.nw = int(f.readline())
                    self.proj_wccf = np.zeros([self.nw, 3], dtype='float64')
                    self.proj_lmr = np.zeros([self.nw, 3], dtype='int64')
                    self.proj_z = np.zeros([self.nw, 3], dtype='float64')
                    self.proj_x = np.zeros([self.nw, 3], dtype='float64')
                    self.proj_zona = np.zeros([self.nw], dtype='float64')
                    self.proj_spin = np.zeros([self.nw], dtype='int64')
                    self.proj_spin_qaxis = np.zeros([self.nw, 3], dtype='float64')
                    for i in range(self.nw):
                        inline = f.readline().split()
                        self.proj_wccf[i] = np.array(inline[:3], dtype='float64')
                        self.proj_lmr[i] = np.array(inline[3:], dtype='int64')
                        inline = f.readline().split()
                        self.proj_z[i] = np.array(inline[:3], dtype='float64')
                        self.proj_x[i] = np.array(inline[3:-1], dtype='float64')
                        self.proj_zona[i] = np.array(inline[-1], dtype='float64')
                        inline = f.readline().split()
                        self.proj_spin[i] = np.array(inline[0], dtype='int64')
                        self.proj_spin_qaxis[i] = np.array(inline[1:], dtype='float64')

                    if convert_to_uudd:
                        # transform nnkp from udud to uudd order
                        nw = self.nw
                        self.proj_wccf = np.einsum('nsa->sna', self.proj_wccf.reshape([nw//2, 2, -1])).reshape([nw, 3])
                        self.proj_lmr = np.einsum('nsa->sna', self.proj_lmr.reshape([nw//2, 2, -1])).reshape([nw, 3])
                        self.proj_x = np.einsum('nsa->sna', self.proj_x.reshape([nw//2, 2, -1])).reshape([nw, 3])
                        self.proj_z = np.einsum('nsa->sna', self.proj_z.reshape([nw//2, 2, -1])).reshape([nw, 3])
                        self.proj_zona = np.einsum('ns->sn', self.proj_zona.reshape([nw//2, 2])).reshape([nw])
                        self.proj_spin = np.einsum('ns->sn', self.proj_spin.reshape([nw//2, 2])).reshape([nw])
                        self.proj_spin_qaxis = np.einsum('nsa->sna', self.proj_spin_qaxis.reshape([nw//2, 2, -1])).reshape([nw, 3])

                elif 'begin projections' in inline:
                    self.soc = False
                    self.nw = int(f.readline())
                    self.proj_wccf = np.zeros([self.nw, 3], dtype='float64')
                    self.proj_lmr = np.zeros([self.nw, 3], dtype='int64')
                    self.proj_z = np.zeros([self.nw, 3], dtype='float64')
                    self.proj_x = np.zeros([self.nw, 3], dtype='float64')
                    self.proj_zona = np.zeros([self.nw], dtype='float64')
                    for i in range(self.nw):
                        inline = f.readline().split()
                        self.proj_wccf[i] = np.array(inline[:3], dtype='float64')
                        self.proj_lmr[i] = np.array(inline[3:], dtype='int64')
                        inline = f.readline().split()
                        self.proj_z[i] = np.array(inline[:3], dtype='float64')
                        self.proj_x[i] = np.array(inline[3:-1], dtype='float64')
                        self.proj_zona[i] = np.array(inline[-1], dtype='float64')
                else:
                    pass

        if wannier_center_def.lower() == 'ws':
            print('\033[92muse ws definition of wannier center\033[0m in htb.worib')
            # refined in range of [-0.5, 0.5) to keep in line with the wannier center
            # used in calculating amn in VASP 6.4.3.
            self.proj_wccf = np.remainder(self.proj_wccf + 100.5, 1) - 0.5
        elif wannier_center_def.lower() == 'poscar':
            print('\033[92muse poscar definition of wannier center\033[0m in htb.worib')
            # proj_wccf origins from wannier_setup, and are in line with POSCAR,
            # if wannier_center_def = poscar, do nothing here.
            pass
        else:
            WanpyInputError('\033[0;31mwannier_center_def should be poscar or ws \033[0m')
        self.proj_wcc = (self.latt @ self.proj_wccf.T).T
    
    def save_h5(self, fname='worbi.h5'):
        with h5py.File(fname, "a") as f:
            f.create_group('worbi')
            worbi = f['worbi']
            h5st = h5py.string_dtype(encoding='utf-8')
            hdf5_create_dataset(worbi, 'latt', data=self.latt, dtype='float64')
            hdf5_create_dataset(worbi, 'lattG', data=self.lattG, dtype='float64')
            hdf5_create_dataset(worbi, 'nw', data=self.nw, dtype='int64')
            hdf5_create_dataset(worbi, 'proj_wcc', data=self.proj_wcc, dtype='float64')
            hdf5_create_dataset(worbi, 'proj_wccf', data=self.proj_wccf, dtype='float64')
            hdf5_create_dataset(worbi, 'proj_lmr', data=self.proj_lmr, dtype='int64')
            hdf5_create_dataset(worbi, 'proj_z', data=self.proj_z, dtype='float64')
            hdf5_create_dataset(worbi, 'proj_x', data=self.proj_x, dtype='float64')
            hdf5_create_dataset(worbi, 'proj_zona', data=self.proj_zona, dtype='float64')
            hdf5_create_dataset(worbi, 'proj_spin', data=self.proj_spin, dtype='int64')
            hdf5_create_dataset(worbi, 'proj_spin_qaxis', data=self.proj_spin_qaxis, dtype='float64')
            hdf5_create_dataset(worbi, 'soc', data=self.soc, dtype='bool')
            f.close()

    def load_h5(self, fname='worbi.h5'):
        f = h5py.File(fname, "r")
        worbi = f.get('worbi')
        if worbi is None:
            f.close()
            return
        for i in self._container:
            item = worbi.get(i)
            if item is not None:
                self.__dict__[i] = item[()]
        f.close()

    def get_wcc(self):
        return self.proj_wcc

    def get_wccf(self):
        return self.proj_wccf

    def printer(self):
        if self.proj_wcc is None:
            print('                           -------------------')
            print('                           htb.worbi NOT FOUND')
            print('                           -------------------')
            return

        print('                                    ---------')
        print('                                    htb.worbi')
        print('                                    ---------')
        print('')
        print('  +-----------------------------------------------------------------------------+')
        print('  |            Frac. Coord.         l mr r      z-axis         x-axis      Z/a  |')
        print('  +-----------------------------------------------------------------------------+')
        for i in range(self.nw):
            print('  |{:4}{: 9.4f}{: 9.4f}{: 9.4f}{:3}{:3}{:3}{: 5.1f}{: 5.1f}{: 5.1f}{: 5.1f}{: 5.1f}{: 5.1f}{: 6.2f} |'.format(
                i+1,
                self.proj_wccf[i, 0], self.proj_wccf[i, 1], self.proj_wccf[i, 2],
                self.proj_lmr[i, 0], self.proj_lmr[i, 1], self.proj_lmr[i, 2],
                self.proj_z[i, 0], self.proj_z[i, 1], self.proj_z[i, 2],
                self.proj_x[i, 0], self.proj_x[i, 1], self.proj_x[i, 2],
                self.proj_zona[i]
            ))
        print('  +-----------------------------------------------------------------------------+')
        print('  |            Cart. Coord.         l mr r      z-axis         x-axis      Z/a  |')
        print('  +-----------------------------------------------------------------------------+')
        for i in range(self.nw):
            print('  |{:4}{: 9.4f}{: 9.4f}{: 9.4f}{:3}{:3}{:3}{: 5.1f}{: 5.1f}{: 5.1f}{: 5.1f}{: 5.1f}{: 5.1f}{: 6.2f} |'.format(
                i+1,
                self.proj_wcc[i, 0], self.proj_wcc[i, 1], self.proj_wcc[i, 2],
                self.proj_lmr[i, 0], self.proj_lmr[i, 1], self.proj_lmr[i, 2],
                self.proj_z[i, 0], self.proj_z[i, 1], self.proj_z[i, 2],
                self.proj_x[i, 0], self.proj_x[i, 1], self.proj_x[i, 2],
                self.proj_zona[i]
            ))
        print('  +-----------------------------------------------------------------------------+')


class Htb(object):
    """
    Usage:
        htb = Htb()
        htb.load_from_w90()
        htb.load_h5()
        htb.load(**kwargs)
    """

    def __init__(self, fermi=0.):
        self._contents = [
            'name', 'fermi', 'nw', 'nR', 'R', 'Rc', 'ndegen', 'N_ucell',
            'latt', 'lattG',
            'wcc', 'wccf',
            'symmops'
            # 'nD', 'D_namelist',
        ]
        self._contents_large = ['hr_Rmn', 'r_Ramn',
                                'spin0_Rmn', 'spin_Ramn',
                                # 'D_iRmn',
                                'wsvecT', 'invndegenT'
                                ]

        self.name = None
        self.fermi = fermi
        self.nw = None
        self.nR = None
        self.R = None
        self.Rc = None
        self.ndegen = None
        self.N_ucell = 1

        self.latt = None
        self.lattG = None

        self.cell = Cell()
        self.worbi = Worbi()
        # self.wout = Wout()
        self.wcc = None
        self.wccf = None

        self.hr_Rmn = None
        self.r_Ramn = None
        self.spin0_Rmn = None
        self.spin_Ramn = None

        # symmetry - tb way
        self.symmops = None

        # symmetric operators {D} - wannier way
        self.nD = None
        self._D_namelist = None
        self.D_iRmn = None

        # IJ-dependent shift
        self.wsvecT = None
        self.invndegenT = None

        # etc.
        self.dk = None
        self._R_hr = None
        self._R_r = None
        self._Rc_hr = None
        self._Rc_r = None

    # @property
    # def D_namelist(self):
    #     return [i.decode('utf-8') for i in self._D_namelist]
    #
    # @D_namelist.setter
    # def D_namelist(self, value):
    #     self._D_namelist = [i if type(i) is bytes else i.encode('utf-8') for i in value]

    @property
    def R_hr(self):
        if self._R_hr is not None:
            return self._R_hr
        else:
            return self.R

    @R_hr.setter
    def R_hr(self, value):
        self._R_hr = value

    @property
    def R_r(self):
        if self._R_r is not None:
            return self._R_r
        else:
            return self.R

    @R_r.setter
    def R_r(self, value):
        self._R_r = value

    @property
    def Rc_hr(self):
        if self._Rc_hr is not None:
            return self._Rc_hr
        else:
            return self.Rc

    @Rc_hr.setter
    def Rc_hr(self, value):
        self._Rc_hr = value

    @property
    def Rc_r(self):
        if self._Rc_r is not None:
            return self._Rc_r
        else:
            return self.Rc

    @Rc_r.setter
    def Rc_r(self, value):
        self._Rc_r = value


    '''
      * I/O level 1
    '''
    def load_htb(self, htb_fname=r'htb.h5'):
        self.load_h5(htb_fname)

    def save_htb(self, fname='htb.h5', decimals=16):
        self.save_h5(fname=fname, decimals=decimals)

    def save_h5(self, fname='htb.h5', decimals=None):
        self.Rc = LA.multi_dot([self.latt, self.R.T]).T

        if os.path.exists(fname):
            os.remove(fname)

        with h5py.File(fname, "w") as f:
            f.create_group('htb')
            htb = f['htb']
            h5st = h5py.string_dtype(encoding='utf-8')
            hdf5_create_dataset(htb, 'name', data=self.name, dtype=h5st)
            hdf5_create_dataset(htb, 'fermi', data=self.fermi, dtype='float64')
            hdf5_create_dataset(htb, 'nw', data=self.nw, dtype='int64')
            hdf5_create_dataset(htb, 'nR', data=self.nR, dtype='int64')
            hdf5_create_dataset(htb, 'R', data=self.R, dtype='int64')
            hdf5_create_dataset(htb, 'Rc', data=self.Rc, dtype='float64')
            hdf5_create_dataset(htb, 'ndegen', data=self.ndegen, dtype='int64')
            hdf5_create_dataset(htb, 'N_ucell', data=self.N_ucell, dtype='int64')

            hdf5_create_dataset(htb, 'latt', data=self.latt, dtype='float64')
            hdf5_create_dataset(htb, 'lattG', data=self.lattG, dtype='float64')
            hdf5_create_dataset(htb, 'wcc', data=self.wcc, dtype='float64')
            hdf5_create_dataset(htb, 'wccf', data=self.wccf, dtype='float64')

            hdf5_create_dataset(htb, 'symmops', data=self.symmops, dtype='float64')

            # hdf5_create_dataset(htb, 'nD', data=self.nD, dtype='int64')
            # hdf5_create_dataset(htb, 'D_namelist', data=self._D_namelist, dtype=h5st)

            hdf5_create_dataset(htb, 'wsvecT', data=self.wsvecT, dtype='int32', compression="gzip")
            hdf5_create_dataset_around(htb, 'invndegenT', data=self.invndegenT, dtype='float64', decimals=decimals, compression="gzip")

            hdf5_create_dataset_around(htb, 'hr_Rmn', data=self.hr_Rmn, dtype='complex128', decimals=decimals, compression="gzip")
            hdf5_create_dataset_around(htb, 'r_Ramn', data=self.r_Ramn, dtype='complex128', decimals=decimals, compression="gzip")
            hdf5_create_dataset_around(htb, 'spin0_Rmn', data=self.spin0_Rmn, dtype='complex128', decimals=decimals, compression="gzip")
            hdf5_create_dataset_around(htb, 'spin_Ramn', data=self.spin_Ramn, dtype='complex128', decimals=decimals, compression="gzip")
            # hdf5_create_dataset_around(htb, 'D_iRmn', data=self.D_iRmn, dtype='complex128', decimals=decimals, compression="gzip")
            f.close()

        self.cell.save_h5(fname)
        self.worbi.save_h5(fname)

    def load_h5(self, fname='htb.h5'):
        f = h5py.File(fname, 'r')
        htb = f.get('htb')
        for i in self._contents + self._contents_large:
            item = htb.get(i)
            if item is not None:
                self.__dict__[i] = item[()]
        # if htb.get('D_namelist') is not None:                                       # WANNING!!!!!! Dec.2 2020
        #     self.D_namelist = hdf5_read_dataset(htb, 'D_namelist', default=[])
        f.close()
        self.cell.load_h5(fname)
        self.worbi.load_h5(fname)

    def load(self, **kwargs):
        for i in self._contents + self._contents_large:
            item = kwargs.get(i)
            if item is not None:
                self.__dict__[i] = item
        self.cell = kwargs.get('cell')
        self.worbi = kwargs.get('worbi')

    def load_wannier90_dat(self,
                           poscar_fname=r'POSCAR',
                           seedname='wannier90',
                           load_wsvec=False,
                           wannier_center_def='poscar',
                           ):
        wout_fname = seedname + '.wout'
        nnkp_fname = seedname + '.nnkp'
        hr_fname = seedname + '_hr.dat'
        r_fname = seedname + '_r.dat'
        spin_fname = seedname + '_spin.dat'
        wsvec_fname = seedname + '_wsvec.dat'
        if os.path.exists(poscar_fname):
            self.cell.load_poscar(fname=poscar_fname)
            self.name = self.cell.name
            self.latt = self.cell.lattice
            self.lattG = self.cell.latticeG
            print('loaded from {}'.format(poscar_fname))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), poscar_fname)

        if os.path.exists(wout_fname):
            self.wcc, self.wccf, wborden = wannier90_load_wcc(fname=wout_fname, shiftincell=False, check_if_uudd_amn=True)
            print('loaded from {}'.format(wout_fname))
        else:
            print('\033[0;31mfile not found: {} \033[0m'.format(wout_fname))

        nnkp_wanning = False
        if os.path.exists(nnkp_fname):
            nnkp_wanning = True
            self.worbi.load_from_nnkp(convert_to_uudd=True, wannier_center_def=wannier_center_def, seedname=seedname)
            print('loaded from {}'.format(nnkp_fname))
        else:
            print('\033[0;31mfile not found: {} \033[0m'.format(nnkp_fname))

        if os.path.exists(hr_fname):
            self.nw, self.nR, self.ndegen, self.R, self.hr_Rmn = wannier90_read_hr(fname=hr_fname)
            self.Rc = (self.latt @ self.R.T).T
            # self.r_Ramn = self._read_rr_v2x(fname=r_fname, nR=self.nR)
            print('loaded from {}'.format(hr_fname))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), hr_fname)
            # print('\033[1;31m[WANNING] \033[0m' + '{} file not found'.format(hr_fname))

        if os.path.exists(r_fname):
            self.r_Ramn = wannier90_read_rr(fname=r_fname)
            print('loaded from {}'.format(r_fname))
        else:
            print('\033[0;31mfile not found: {} \033[0m'.format(r_fname))

        if os.path.exists(spin_fname):
            self.spin0_Rmn, self.spin_Ramn = wannier90_read_spin(fname=spin_fname)
            print('loaded from {}'.format(spin_fname))
        else:
            print('\033[0;31mfile not found: {} \033[0m'.format(spin_fname))

        if os.path.exists(wsvec_fname) and load_wsvec:
            self.invndegenT, self.wsvecT = wannier90_load_wsvec(fname=wsvec_fname, nw=self.nw, nR=self.nR)
            print('loaded from {}'.format(wsvec_fname))
        else:
            print('{} not loaded'.format(wsvec_fname))

        if nnkp_wanning:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('Important note:')
            print('The .nnkp was loaded. Please ensure ')
            print('1. It is generated by v3.x version of wannier90 (maybe v2.x is also OK);')
            print('2. The .amn should be in uudd order. ')
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    def save_wannier90_dat(self, seedname='wannier90'):
        self.save_wannier90_hr_dat(seedname)
        self.save_wannier90_r_dat(seedname)
        self.save_wannier90_spin_dat(seedname)

    def save_wannier90_hr_dat(self, seedname='wannier90', fmt='12.6'):
        hr_fname = seedname + '_hr.dat'

        hr_Rmn = self.hr_Rmn

        if os.path.exists(hr_fname):
            os.remove(hr_fname)

        with open(hr_fname, 'a') as f:
            f.write('written by wanpy\n')
            f.write('          {}\n'.format(self.nw))
            f.write('          {}\n'.format(self.nR))

            countor = 0
            for i in range(self.nR):
                if countor == 15:
                    countor = 0
                    f.write(' \n')
                f.write('    ')
                f.write(str(self.ndegen[i]))
                countor += 1
            f.write(' \n')

            for i, _R in zip(range(self.nR), self.R):
                for wi in range(self.nw):
                    for wj in range(self.nw):
                        values = [hr_Rmn[i, wj, wi].real, hr_Rmn[i, wj, wi].imag]
                        fmt_str = '{: >' + fmt + 'f}'
                        formatted_values = [fmt_str.format(value) for value in values]
                        formatted_values.insert(0, '{: >5d}{: >5d}'.format(wj + 1, wi + 1, ))
                        formatted_values.insert(0, '{: >5d}{: >5d}{: >5d}'.format(_R[0], _R[1], _R[2]))
                        f.write(''.join(formatted_values) + '\n')
        f.close()

    def save_wannier90_r_dat(self, seedname='wannier90', fmt=12.6):
        r_fname = seedname + '_r.dat'
        if os.path.exists(r_fname):
            os.remove(r_fname)

        r_Ramn = self.r_Ramn
        with open(r_fname, 'a') as f:
            f.write('written by wanpy\n')
            f.write('          {}\n'.format(self.nw))
            f.write('          {}\n'.format(self.nR))  # Wannier90 v3x added

            for i, _R in zip(range(self.nR), self.R):
                for wi in range(self.nw):
                    for wj in range(self.nw):
                        values = [
                            r_Ramn[i, 0, wj, wi].real, r_Ramn[i, 0, wj, wi].imag,
                            r_Ramn[i, 1, wj, wi].real, r_Ramn[i, 1, wj, wi].imag,
                            r_Ramn[i, 2, wj, wi].real, r_Ramn[i, 2, wj, wi].imag,
                        ]
                        fmt_str = '{: >' + fmt + 'f}'
                        formatted_values = [fmt_str.format(value) for value in values]
                        formatted_values.insert(0, '{: >5d}{: >5d}'.format(wj + 1, wi + 1, ))
                        formatted_values.insert(0, '{: >5d}{: >5d}{: >5d}'.format(_R[0], _R[1], _R[2]))
                        f.write(''.join(formatted_values) + '\n')
        f.close()

    def save_wannier90_spin_dat(self, seedname='wannier90', fmt='12.6'):
        fname = seedname + '_spin.dat'
        if os.path.exists(fname):
            os.remove(fname)

        spin0_Rmn = self.spin0_Rmn
        spin_Ramn = self.spin_Ramn
        with open(fname, 'a') as f:
            f.write('written by wanpy\n')
            f.write('          {}\n'.format(self.nw))
            f.write('          {}\n'.format(self.nR))  # Wannier90 v3x added

            for i, _R in zip(range(self.nR), self.R):
                for wi in range(self.nw):
                    for wj in range(self.nw):
                        values = [
                            spin0_Rmn[i, wj, wi].real, spin0_Rmn[i, wj, wi].imag,
                            spin_Ramn[i, 0, wj, wi].real, spin_Ramn[i, 0, wj, wi].imag,
                            spin_Ramn[i, 1, wj, wi].real, spin_Ramn[i, 1, wj, wi].imag,
                            spin_Ramn[i, 2, wj, wi].real, spin_Ramn[i, 2, wj, wi].imag,
                        ]
                        fmt_str = '{: >' + fmt + 'f}'
                        formatted_values = [fmt_str.format(value) for value in values]
                        formatted_values.insert(0, '{: >5d}{: >5d}'.format(wj + 1, wi + 1, ))
                        formatted_values.insert(0, '{: >5d}{: >5d}{: >5d}'.format(_R[0], _R[1], _R[2]))
                        f.write(''.join(formatted_values) + '\n')
        f.close()

    def save_wcc(self, fname=r'wcc.vasp', cartesian=False):
        if os.path.exists(fname):
            os.remove(fname)

        wcc_unit = 'D'
        wcc = self.wccf

        if cartesian:
            wcc_unit = 'C'
            wcc = self.wcc

        with open(fname, 'a') as poscar:
            poscar.write('writen by wanpy\n')
            poscar.write('   1.0\n')
            for i in self.cell.lattice.T:
                poscar.write('   {: 2.16f}    {: 2.16f}    {: 2.16f}\n'.format(i[0], i[1], i[2]))

            poscar.write('H   \n')
            poscar.write('{}    \n'.format(self.nw))

            poscar.write('{}\n'.format(wcc_unit))

            for i in wcc:
                poscar.write('  {: 2.16f}  {: 2.16f}  {: 2.16f}\n'.format(i[0], i[1], i[2]))
            poscar.write('\n')
        poscar.close()

    '''
      * operation
    '''
    def use_atomic_wcc(self):
        self.wcc = self.worbi.get_wcc()
        self.wccf = self.worbi.get_wccf()

    def set_ndegen_ones(self):
        self.hr_Rmn = np.einsum('R,Rmn->Rmn', 1/self.ndegen, self.hr_Rmn)
        self.r_Ramn = np.einsum('R,Ramn->Ramn', 1/self.ndegen, self.r_Ramn)
        if self.D_iRmn is not None:
            self.D_iRmn = np.einsum('R,iRmn->iRmn', 1/self.ndegen, self.D_iRmn)
        self.ndegen = np.ones(self.nR, dtype='int64')
        print('[FROM Htb] htb.ndegen have seted to ones')

    def reduce_htb(self, tmin=-1.0, tmin_r=-1.0, open_boundary=-1, tb=False, use_wcc=False):
        """
          this function do not modify the variable in htb,
          alernatively it returns the reduced htb
        """
        nR = self.nR
        if not np.isclose(self.ndegen, 1.0).all():
            print('*********************************************************************************')
            print('*****  [WARNING IN REDUCE HTB] np.isclose(self.ndegen, 1.0).all() = False.  *****')
            print('*********************************************************************************')

        hr_remain = np.array([1 if np.abs(self.hr_Rmn[i]).max() > tmin else 0 for i in range(nR)])
        r_remain = np.array([1 if np.abs(self.r_Ramn[i]).max() > tmin_r else 0 for i in range(nR)])
        # if tmin_r > 0:
        #     r_remain = np.array([1 if np.abs(self.r_Ramn[i]).max() > tmin_r else 0 for i in range(nR)])
        # elif tmin_r < 0:
        #     r_remain = np.array([1 if (self.R[i] == 0).all() else 0 for i in range(nR)])

        if tb:
            r_remain = np.array([1 if (self.R[i] == 0).all() else 0 for i in range(nR)])

        '''
          * handle open boundary case
            this is used for slab or 2D material
        '''
        if open_boundary in [0, 1, 2]:
            ob_remain = np.zeros(nR, dtype='int')
            for i in range(nR):
                if self.R[i, open_boundary] == 0:
                    ob_remain[i] = 1
                else:
                    ob_remain[i] = 0

            hr_remain *= ob_remain
            r_remain *= ob_remain

        '''
          * get reduced hr_Rmn, R_hr
        '''
        if (hr_remain == 1).all():
            hr_Rmn = np.array(self.hr_Rmn, dtype='complex128').copy()
            R_hr = np.array(self.R, dtype='int64').copy()
        else:
            hr_Rmn = np.array([self.hr_Rmn[i] for i in range(nR) if hr_remain[i]], dtype='complex128').copy()
            R_hr = np.array([self.R[i] for i in range(nR) if hr_remain[i]], dtype='int64').copy()

        '''
          * get reduced r_Ramn, R_r
        '''
        if not np.all(self.R[self.nR//2] == 0):
            print('[ERROR IN REDUCE HTB] htb.R[htb.nR//2] != 0')
            sys.exit()

        r_Ramn = np.array(self.r_Ramn, dtype='complex128').copy()
        if use_wcc:
            for i in range(3):
                r_Ramn[self.nR//2, i] *= 1 - np.eye(self.nw)
                r_Ramn[self.nR//2, i] += np.diag(self.wcc.T[i])

        if tb:
            for i in range(self.nR):
                if (self.R[i] == 0).all():
                    r_Ramn = np.array([[
                        np.diag(np.diag(r_Ramn[i, 0])),
                        np.diag(np.diag(r_Ramn[i, 1])),
                        np.diag(np.diag(r_Ramn[i, 2])),
                    ]], dtype='complex128')
            R_r = np.array([[0, 0, 0]], dtype='int64')
        else:
            if (r_remain == 1).all():
                r_Ramn = np.array(r_Ramn, dtype='complex128').copy()
                R_r = np.array(self.R, dtype='int64').copy()
            else:
                r_Ramn = np.array([self.r_Ramn[i] for i in range(nR) if r_remain[i]], dtype='complex128').copy()
                R_r = np.array([self.R[i] for i in range(nR) if r_remain[i]], dtype='int64').copy()

        nR_hr = R_hr.shape[0]
        nR_r = R_r.shape[0]
        print('[hr reduction] nR has reduced from {} to {}'.format(nR, nR_hr))
        print('[r reduction] nR has reduced from {} to {}'.format(nR, nR_r))

        return nR_hr, nR_r, R_hr, R_r, hr_Rmn, r_Ramn

    def get_dk(self, nkmesh):
        self.dk = LA.norm(self.cell.latticeG.T, axis=1) / nkmesh
        return self.dk

    def setup(self):
        if self.r_Ramn is None:
            print('[WANNING] htb.r_Ramn is None, setting proper values according to htb.wcc.')
            r_Ramn = np.zeros([self.nR, 3, self.nw, self.nw], dtype='complex128')
            r_Ramn[self.nR//2, 0] += np.diag(self.wcc.T[0])
            r_Ramn[self.nR//2, 1] += np.diag(self.wcc.T[1])
            r_Ramn[self.nR//2, 2] += np.diag(self.wcc.T[2])
            self.r_Ramn = r_Ramn

        self.set_ndegen_ones()
        self.Rc_hr = LA.multi_dot([self.latt, self.R_hr.T]).T
        self.Rc_r = LA.multi_dot([self.latt, self.R_r.T]).T

    def get_hk(self, k):
        eikr = np.exp(2j * np.pi * np.einsum('a,Ra', k, self.R)) / self.ndegen
        hk = np.einsum('R,Rmn->mn', eikr, self.hr_Rmn)
        return hk

    '''
      * printer
    '''
    def printer(self):
        self.cell.printer()
        self.worbi.printer()
        self.print_wcc()
        self.print_RGrid()

    def print_RGrid(self, gridR=None, ndegen=None):
        if gridR is None:
            gridR = self.R
            ndegen = self.ndegen
        nR = gridR.shape[0]
        print('  +-----------------------------------------------------------------------------+')
        print('  |                                     R Grid                                  |')
        print('  |    number of R Grid = {:4}                                                  |'.format(nR))
        print('  +-----------------------------------------------------------------------------+')
        for i in range(nR):
            print('  |{: 4}). {: 3} {: 3} {: 3}  *{:2>} '.format(i + 1, gridR[i, 0], gridR[i, 1], gridR[i, 2], ndegen[i]), end='')
            if (i + 1) % 3 == 0:
                print('  |')
        if nR % 3 != 0: print('                                                      |')
        print('  +-----------------------------------------------------------------------------+')
        print('')

    def print_wcc(self):
        if self.wcc is None:
            print('                                    -----------------')
            print('                                    htb.wcc NOT FOUND')
            print('                                    -----------------')
            return

        print('')
        print('                               ------------------')
        print('                               htb.wccf & htb.wcc')
        print('                               ------------------')
        print('')
        print('  +-----------------------------------------------------------------------------+')
        print('  |              Frac. Coord.          |               Cart. Coord.             |')
        print('  +-----------------------------------------------------------------------------+')
        for i in range(self.nw):
            print('  |{:4}{: 10.5f}{: 10.5f}{: 10.5f}  |    {: 10.5f}{: 10.5f}{: 10.5f}      |'.format(
                i+1,
                self.wccf[i, 0], self.wccf[i, 1], self.wccf[i, 2],
                self.wcc[i, 0], self.wcc[i, 1], self.wcc[i, 2],
            ))
        print('  +-----------------------------------------------------------------------------+')
        print('')


'''
  h5 adaptor
'''
def hdf5_create_dataset(group, name, data, dtype, **kwds):
    if data is not None:
        group.create_dataset(name, data=data, dtype=dtype, **kwds)

def hdf5_create_dataset_around(group, name, data, dtype, decimals=None, **kwds):
    if data is not None:
        if decimals is not None:
            group.create_dataset(name, data=np.around(data, decimals=decimals), dtype=dtype, **kwds)
        else:
            group.create_dataset(name, data=data, dtype=dtype, **kwds)

def hdf5_read_dataset(group, name, default=None):
    data = group.get(name)
    if data is not None:
        return data[()]
    else:
        return default

