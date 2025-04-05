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

__date__ = "Dec. 6, 2019"

import os
import numpy as np
import numpy.linalg as LA
from wanpy.core.toolkits import kmold
try:
    from pymatgen.io.vasp.outputs import Wavecar
except ModuleNotFoundError:
    pass
else:
    pass

__all__ = [
    'VASP',
    'VASP_EIGENVAL_HSP',
    'VASP_wavecar',
]

class VASP(object):

    def __init__(self):
        pass


class VASP_EIGENVAL_HSP(VASP):

    def __init__(self, latt=np.identity(3)):
        VASP.__init__(self)
        self.name = None
        self.nele = None
        self.nk = None
        self.nb = None
        self.kpt = None
        self.kpath = None
        self.eig = None
        self.occ = None
        self.latt = latt
        self.lattG = 2 * np.pi * LA.inv(latt.T)

    def load_from_vasp(self, fname='EIGENVAL'):
        with open(fname, 'r') as f:
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            self.name = f.readline()
            self.nele, self.nk, self.nb = np.array(f.readline().split(), dtype='int')

            self.kpt = np.zeros((self.nk, 3), dtype='float64')
            self.eig = np.zeros((self.nk, self.nb), dtype='float64')
            self.occ = np.zeros((self.nk, self.nb), dtype='float64')

            f.readline()

            for ik in range(self.nk):
                inline = np.array(f.readline().split(), dtype='float64')  # kx ky kz weigh
                self.kpt[ik] = inline[:3]
                for ib in range(self.nb):
                    inline = np.array(f.readline().split()[1:], dtype='float64')
                    self.eig[ik, ib] = inline[0]
                    self.occ[ik, ib] = inline[1]
                f.readline()  # block line
        f.close()

        kkc = LA.multi_dot([self.lattG, self.kpt.T]).T
        self.kpath = kmold(kkc)


class VASP_wavecar(object):

    def __init__(self, fname='WAVECAR', verbose=False, vasp_type='ncl'):
        wavecar = Wavecar(fname, verbose, vasp_type)

        self.latt = wavecar.a.T
        self.lattG = wavecar.b.T
        self.fermi = wavecar.efermi
        self.bandE = np.array(wavecar.band_energy, dtype='float64')[:,:,0]       # nk, nb
        self.bandEocc = np.array(wavecar.band_energy, dtype='float64')[:,:,2]    # nk, nb
        self.nk, self.nb = wavecar.nk, wavecar.nb
        self.kpts = np.array(wavecar.kpoints, dtype='float64')
        self.wfs = wavecar.coeffs        # nk, nb, 2(spin), nG(not uniform)
        self.Gpoints = wavecar.Gpoints   # nk, nG(not uniform)
        self.encut = wavecar.encut


if __name__ == '__main__':
    os.chdir(r'')

    eig = VASP_EIGENVAL_HSP()
    eig.load_from_vasp()

    from wanpy.env import ROOT_WDIR
    os.chdir(os.path.join(ROOT_WDIR, r'wanpy_debug/buildwannier/graphene'))
    wavecar = Wavecar('WAVECAR', verbose=False, vasp_type='ncl')

    wavecar.write_unks('dftwfs')
