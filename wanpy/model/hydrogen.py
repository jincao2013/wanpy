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

__date__ = "Sep. 2, 2019"


import time
import os
import sys
import getpass
sys.path.append(os.environ.get('PYTHONPATH'))

import numpy as np
from numpy import linalg as LA
from numpy.linalg import multi_dot
import scipy.special as sc

if getpass.getuser() == 'Jin':
    from pylab import *
else:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import subplot

from wanpy.core.mesh import make_kpath, make_kmesh
from wanpy.core.units import *
from wanpy.interface.wannier90 import W90_nnkp, W90_mmn, W90_eig, W90_amn


def orbital_intergration():

    def get_gs(rr):
        a0 = 0.529
        a = a0 * 0.7
        r = LA.norm(rr, axis=1)

        Rn = 2 * a**-1.5 * np.exp(-r/a)
        Ylm = np.sqrt(1/4/np.pi)
        g = Rn * Ylm
        return g

    def get_gpz(rr):
        a0 = 0.529
        a = a0 * 0.7
        z = rr[:, 2]
        r = LA.norm(rr, axis=1)
        invr = np.real(1 / (r - 0.1j))

        Rn = 2 * a**-1.5 * np.exp(-r/a)
        Ylm = np.sqrt(3/4/np.pi) * z * invr
        g = Rn * Ylm
        return g

    def get_gpx(rr):
        a0 = 0.529
        a = a0 * 0.7
        x = rr[:, 0]
        r = LA.norm(rr, axis=1)
        invr = np.real(1 / (r - 0.1j))

        Rn = 2 * a**-1.5 * np.exp(-r/a)
        Ylm = np.sqrt(3/4/np.pi) * x * invr
        g = Rn * Ylm
        return g

    def get_gpy(rr):
        a0 = 0.529
        a = a0 * 0.7
        y = rr[:, 1]
        r = LA.norm(rr, axis=1)
        invr = np.real(1 / (r - 0.1j))

        Rn = 2 * a**-1.5 * np.exp(-r/a)
        Ylm = np.sqrt(3/4/np.pi) * y * invr
        g = Rn * Ylm
        return g

    kcube = np.array([
        [-0.5, -0.5, -0.5],
        [1, 0, 0 ],
        [0, 1, 0 ],
        [0, 0, 1 ],
    ], dtype='float64')
    latt = np.diag([100., 100., 10.])
    vcell = LA.det(latt)
    meshR = make_kmesh(100, 100, 100, sample_method='G', kcube=kcube)
    meshR = make_kmesh(50, 50, 50, sample_method='G', kcube=kcube)

    meshRc = LA.multi_dot([latt, meshR.T]).T
    meshRc[:, 0] /= 10
    meshRc[:, 1] /= 10
    nrr = meshR.shape[0]

    gs = get_gs(meshRc)
    gpx = get_gpx(meshRc)
    gpy = get_gpy(meshRc)
    gpz = get_gpz(meshRc)
    gn = np.array([gs, gpz, gpx, gpy]).T
    gmn = np.einsum('Rm,Rn->mn', gn, gn) * vcell / nrr

    print('sun of gs = {}'.format(np.sum(gs**2) * vcell / nrr))
    print('sun of gpz = {}'.format(np.sum(gpz**2) * vcell / nrr))
    print('sun of gpx = {}'.format(np.sum(gpx**2) * vcell / nrr))
    print('sun of gpy = {}'.format(np.sum(gpy**2) * vcell / nrr))


    print('<gm|gn> =\n')
    print(gmn)


def test_intergration():

    def get_2pz(rr):
        a0 = 0.529 * 0.3
        z = rr[:, 2]
        r = LA.norm(rr, axis=1)
        gpz = (32*np.pi)**-0.5 * (a0**-1.5) * (z/a0) * np.exp(-r/2/a0)
        return gpz

    def get_gauss_pz(rr):
        '''
        ----------------
        borden=0.7~1.0 rr=(10,10,24)
        ----------------
        '''#
        a0 = 0.529
        borden = a0 * 0.8
        z = rr[:, 2]
        r = LA.norm(rr, axis=1)
        gpz = (z/borden) * np.exp(-0.5 * (r/borden)**2)
        return gpz

    def get_g(rr):
        return get_gauss_pz(rr)


    lattA_graphene = 2.46
    lattC_graphene = 3.015
    latt = np.zeros([3, 3], dtype='float64')
    latt[:2, :2] = lattA_graphene * np.array([
        [3 ** 0.5 / 2, -0.5],
        [3 ** 0.5 / 2, 0.5],
    ]).T
    latt[2, 2] = 3 * lattC_graphene
    vcell = LA.det(latt)
    ions = np.array([
        [0, 0, 0],
        [1/3, 1/3, 0],
        [0, 0, 0.5],
        [1/3, 1/3, 0.5],
    ])
    Rcs = LA.multi_dot([latt, ions.T]).T
    Rcs[0, 2] = 1 * lattC_graphene
    Rcs[1, 2] = 1 * lattC_graphene
    Rcs[2, 2] = 2 * lattC_graphene
    Rcs[3, 2] = 2 * lattC_graphene

    rr = make_kmesh(10, 10, 24, sample_method='MP')
    gridR = make_kmesh(3, 3, 1) * 3 - np.array([1, 1, 0])
    rr = LA.multi_dot([latt, rr.T]).T
    gridR = LA.multi_dot([latt, gridR.T]).T
    Nrr = rr.shape[0]
    nnR = gridR.shape[0]
    r = LA.norm(rr, axis=1)

    rr_ext = np.kron(np.ones([nnR, 1]), rr) - np.kron(gridR, np.ones([Nrr, 1]))

    gn = np.zeros([4, Nrr], dtype='float64')
    gn_ext = np.zeros([4, nnR * Nrr], dtype='float64')
    for i in range(4):
        gn_ext[i] = get_g(rr_ext - Rcs[i])
        gn_ext[i] /= (np.sum(gn_ext[i]**2) * (vcell / Nrr)) ** 0.5
        # _gn = get_gauss(rr_nnWS)
        gn[i] = np.sum(gn_ext[i].reshape(nnR, Nrr), axis=0)

    # umn = np.einsum('mx,nx->mn', gn, gn) * (vcell / Nrr)
    umn = np.einsum('mx,nx->mn', gn_ext, gn_ext) * (vcell / Nrr)
    xn = np.einsum('ia,mi,mi->ma', rr_ext, gn_ext, gn_ext) * (vcell / Nrr)

    return umn, xn



if __name__ == '__main__':
    wdir = r''
    os.chdir(wdir)

    # umn, xn = test_intergration()
    # print(umn)
    # print(xn)

    orbital_intergration()

