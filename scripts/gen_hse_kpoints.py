#!/usr/bin/env python3
import os
import numpy as np
from pymatgen.io.vasp.outputs import Eigenval

vaspeig = Eigenval(r'EIGENVAL.HSP')
nk, nb, nele = vaspeig.nkpt, vaspeig.nbands, vaspeig.nelect
kk = np.array(vaspeig.kpoints, dtype='float64')
kk_plus_zero_weight = np.vstack([kk.T, np.zeros(nk)]).T
np.savetxt('IBZKPT.HSP', kk_plus_zero_weight, fmt='%12.8f')

os.system('cat IBZKPT IBZKPT.HSP > KPOINTS.HSE')