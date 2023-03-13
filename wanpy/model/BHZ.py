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

__date__ = "Jan. 3, 2019"

import time
import os
import sys
sys.path.append(os.environ.get('PYTHONPATH'))

import numpy as np
from numpy import linalg as LA
from wanpy.core.units import *

from wanpy.core.structure import BandstructureHSP
from wanpy.core.mesh import make_kpath_dev

class BHZ_model(object):

    def __init__(self):
        self.nb = 4
        self.A = -13.68
        self.B = -16.9
        self.C = -0.0263
        self.D = -0.514
        self.M = -2.058

    def get_hk(self, kc):
        kx, ky, kz = kc
        d0 = self.C - 2 * self.D * (2 - np.cos(kx) - np.cos(ky))
        d1 = self.A * np.sin(kx)
        d2 = self.A * np.sin(ky)
        d3 = -2 * self.B * (2 - self.M/2/self.B - np.cos(kx) - np.cos(ky))
        h = d0 * sigma_0 + d1 * sigma_x + d2 * sigma_y + d3 * sigma_z
        hC = d0 * sigma_0 - d1 * sigma_x + d2 * sigma_y + d3 * sigma_z
        zero = np.zeros([2, 2])
        hk = np.block([
            [h, zero],
            [zero, hC],
        ])
        return hk

    def cal_wilson_loop(self):
        nb = self.nb
        nw = 2
        nkx = 200
        nky = 50
        kkx = np.linspace(0, 2 * np.pi, nkx+1)[:-1]
        kky = np.linspace(0, np.pi, nky+1)[:-1]

        theta = np.zeros([nky, nw], dtype='float64')
        for yi in range(nky):
            print('cal wcc at ({}/{}) ky={:.3f}'.format(yi + 1, nky, kky[yi]))
            Dky = np.identity(nb)
            for xi in range(nkx):
                kc = np.array([kkx[xi], kky[yi], 0])
                E, U = LA.eigh(self.get_hk(kc))
                V = U[:, :2]
                if xi+1 != nkx:
                    Dky = LA.multi_dot([Dky, V, V.conj().T])
                else:
                    Dky = LA.multi_dot([V.conj().T, Dky, V])
            theta[yi] = np.sort(np.imag(np.log(LA.eigvals(Dky))))
        theta /= np.pi * 2
        return kky, theta


if __name__ == "__main__":
    Job = 'wilson_loop'


if __name__ == "__main__" and Job == 'band':
    bhz = BHZ_model()

    band = BandstructureHSP()
    band.HSP_list = np.array([
        [-0.5, -0.5, 0],
        [0, 0, 0],
        [0.5, 0, 0],
    ])
    band.HSP_path_frac = make_kpath_dev(band.HSP_list, 100)
    band.HSP_path_car = 2 * np.pi * band.HSP_path_frac
    band.nk = band.HSP_path_frac.shape[0]
    band.nb = 4
    band.eig = np.zeros([band.nk, band.nb])
    for i, _kc in zip(range(band.nk), band.HSP_path_car):
        E, U = LA.eigh(bhz.get_hk(_kc))
        band.eig[i] = E

    band.plot_band()


if __name__ == "__main__" and Job == 'wilson_loop':
    import matplotlib.pyplot as plt
    bhz = BHZ_model()
    kky, theta = bhz.cal_wilson_loop()
    for i in range(2):
        plt.plot(kky, theta.T[i])
        plt.scatter(kky, theta.T[i])