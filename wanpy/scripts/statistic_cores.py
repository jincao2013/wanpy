#!/usr/bin/env python3

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

__date__ = "Jan. 3, 2019"

import sys
import re
import datetime
import numpy as np

def main():
    argv = sys.argv[1:]
    fname = argv[0]
    print('load file from', fname)

    with open(fname, 'r', errors='replace') as f:
        inline = f.readline()
        while 'Running on' not in inline:
            inline = f.readline()
        ncores = int(inline.split()[2])

        while '[Rank' not in inline:
            inline = f.readline()
        # inline = inline.split()
        # time_str = inline[3][2:] + ' ' + inline[4]
        day = re.search(r'\d\d-\d\d-\d\d', inline).group()
        time = re.search(r'\d\d:\d\d:\d\d', inline).group()
        time_str = ' '.join([day, time])
        t0 = datetime.datetime.strptime(time_str, "%y-%m-%d %H:%M:%S")

        data = f.readlines()[::-1]
    f.close()

    t1 = datetime.datetime.now()
    dt = t1 - t0

    persentage = np.zeros([ncores], dtype='float64')

    for i in range(ncores):
        for j in data:
            if '[Rank {}'.format(i) in j:
                # inline = np.array(j.split()[2].split('/'), dtype='float64')
                inline = np.array(re.findall('\d+', j)[1:3], dtype='float64')
                persentage[i] = inline[0] / inline[1]
                break

    seconds_in_total = dt.total_seconds()/persentage
    t2 = t0 + datetime.timedelta(seconds=np.max(seconds_in_total))

    for i in range(ncores):
        print('[Rank {:3}] [{:20}] {:3.1f}%'.format(i, '*'*int(20*persentage[i]), persentage[i]*100))
    print('')
    print(' Predicted time in total: {:4.1f} h'.format(np.max(seconds_in_total)/3600))
    print('                Time now: {}'.format(t1.strftime('%y-%m-%d %H:%M:%S')))
    print('   Predicted finish time: {}'.format(t2.strftime('%y-%m-%d %H:%M:%S')))

if __name__ == "__main__":
    main()
