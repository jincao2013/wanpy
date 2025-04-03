# Copyright (C) 2024 Jin Cao
#
# This file is distributed as part of the wanpy code and
# under the terms of the GNU General Public License. See the
# file LICENSE in the root directory of the wanpy
# distribution, or http://www.gnu.org/licenses/gpl-3.0.txt
#
# The wanpy code is hosted on GitHub:
#
# https://github.com/jincao2013/wanpy

__date__ = "Sep. 26, 2024"

import numpy as np
from numpy import linalg as LA
import wanpy as wp
from wanpy import response as res

__all__ = [
    'cal_Fermi_surface',
]

def cal_Fermi_surface(htb, k, omega, eta=1e-6):
    hk = res.get_hk(htb, k)
    GR = LA.inv((omega+htb.fermi-1j*eta)*np.eye(htb.nw)-hk)
    dos = 1/np.pi * np.trace(GR).imag
    return dos
