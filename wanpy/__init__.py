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

__author__ = 'Jin Cao'
__email__ = "caojin.phy@gmail.com"
__version__ = "0.16.0"

import os
# from .env import __author__, __email__, __version__, ROOT_WDIR, PYGUI
# from .env import ROOT_WDIR, PYGUI

ROOT_WDIR = r''
PYGUI = False

if os.getenv('WANPY_ROOT_DIR') is not None:
    ROOT_WDIR = os.getenv('WANPY_ROOT_DIR')

if os.getenv('PYGUI') in ['True', '1']:
    PYGUI = True

from .core.bz import fermi_dirac_func, delta_func
from .core.greenfunc import self_energy
from .core.mesh import make_ws_gridR, make_mesh, make_kpath, kmold
from .core.structure import Cell, Worbi, Htb
from .core.symmetry import Symmetrize_Htb_kspace, Symmetrize_Htb_rspace, get_proj_info, parse_symmetry_inputfile
from .core.trans_hr import Supercell_Htb, FT_htb
from .core.utils import *
from .core.units import *
import wanpy.interface
import wanpy.MPI
import wanpy.response

import warnings
# This is to ignore SyntaxWarning globally
# In wanpy, there are some latex note in the docstrings, and they are recognized as
# invalid escape sequence by python 3.12+.
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Remove symbols imported for internal use
del os
