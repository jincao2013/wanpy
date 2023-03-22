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

__date__ = "Mar. 20, 2023"

import os

def load_input_index_from_path(wdir, MPI_main):
    rank2dict = {'x': 0, 'y': 1, 'z': 2}
    index = list(os.path.split(wdir)[1][-2:])
    index = tuple(rank2dict[index[i]] for i in range(2))
    if MPI_main:
        print('[load_input_index_from_path] sucess parse index = {}'.format(index))
    return index