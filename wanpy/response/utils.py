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

__date__ = "Mar. 20, 2023"

import os
import time

__all__ = [
    'printk',
    'load_input_index_from_path',
]

'''
  * print
'''
def printk(i, nk, k, sep=1):
    if i % sep == 0:
        print('[{:>3d}/{:<3d}] {} Cal k at ({:.3f} {:.3f} {:.3f})'.format(
            i+1, nk,
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            k[0], k[1], k[2]
            )
        )

def load_input_index_from_path(wdir, MPI_main):
    rank2dict = {'x': 0, 'y': 1, 'z': 2}
    index = list(os.path.split(wdir)[1][-2:])
    index = tuple(rank2dict[index[i]] for i in range(2))
    if MPI_main:
        print('[load_input_index_from_path] sucess parse index = {}'.format(index))
    return index