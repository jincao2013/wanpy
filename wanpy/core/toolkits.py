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

__date__ = "Aug. 11, 2017"

__all__ = [
    'trans001', 'trans002', 'kmold',
    'kpar', 'kgps_Gather', 'get_kpar_gps', 'par_Gather', 'reduce_open_boundary',
    'max_value', 'eye_filter'
]

import time
import numpy as np
import numpy.linalg as LA

'''
   Tools
'''
def trans001(astr, asarray=False):
    '''
      astr = '1-3;5-7'
      return [0,1,2, 4,5,6]
      astr = '1-3;5-7;9'
      return [0,1,2, 4,5,6, 8]
    '''
    alist = []

    astr1 = astr.split(';')   # = ['1-3', '5-7']
    for _astr1 in astr1:
        astr2 = _astr1.split('-') # = ['1', '3']
        alist.extend([i for i in range(int(astr2[0])-1, int(astr2[-1]))])
    if asarray:
        alist = np.array(alist, dtype='int64')
    return alist

def trans002(spec, astr):
    '''
      spec = "Fe"
      astr = ["Li", "Fe", "Fe", "P", "P", "P"]
      return [1,2]
    '''
    alist = [i for i, _spec in enumerate(astr) if _spec == spec]
    return alist

def kmold(kkc):
    nk = kkc.shape[0]
    kkc = kkc[1:, :] - kkc[:-1, :]
    kkc = np.vstack([np.array([0, 0, 0]), kkc])
    k_mold = np.sqrt(np.einsum('ka,ka->k', kkc, kkc))
    k_mold = LA.multi_dot([np.tri(nk), k_mold])
    return k_mold

def distance(vec_1,vec_2):

    vec_1 = np.array(vec_1,dtype='float')
    vec_2 = np.array(vec_2,dtype='float')

    return LA.norm(vec_1 - vec_2)

def k_frac_to_cart(b1,b2,b3,k_vectors_frac):

    b1 = np.array(b1,dtype='float')
    b2 = np.array(b2,dtype='float')
    b3 = np.array(b3,dtype='float')
    k_vectors_frac = np.array(k_vectors_frac,dtype='float').T

    G_Vectors_matrix = np.array([b1,b2,b3]).T
    k_vectors_cart = np.dot(G_Vectors_matrix, k_vectors_frac)

    return k_vectors_cart

def k_cart_to_frac(b1,b2,b3,k_vectors_cart):

    b1 = np.array(b1,dtype='float')
    b2 = np.array(b2,dtype='float')
    b3 = np.array(b3,dtype='float')
    k_vectors_cart = np.array(k_vectors_cart,dtype='float').T

    G_Vectors_matrix_inv = LA.inv(np.array([b1,b2,b3]).T)
    k_vectors_frac = np.dot(G_Vectors_matrix_inv, k_vectors_cart)

    return k_vectors_frac

def r_frac_to_cart(a1,a2,a3,r_vectors_frac):
    return k_frac_to_cart(a1,a2,a3,r_vectors_frac)

def r_cart_to_frac(a1,a2,a3,r_vectors_cart):
    return k_cart_to_frac(a1,a2,a3,r_vectors_cart)

def abslog(x):
    y1 = np.log(np.abs(x))
    y2 = np.sign(x)
    return y1 * y2

def abslog10(x):
    x[x==0.0] = 1.0
    y1 = np.log10(np.abs(x))
    y2 = np.sign(x)
    return y1 * y2

'''
  * reduce hr r procedrue
  * * reduce_hr
  * * reduce_r
'''
def reduce_hr(hr, maxR):
    degeneracy, _R, hr_mn = hr
    _nrpts, num_wann = hr_mn.shape[:2]
    if type(maxR) is int:
        a, b, c = (maxR, maxR, maxR)
    else:
        a, b, c = maxR

    if (np.array([a, b, c]) > 100).all():
        return hr

    degeneracy = np.array([
        degeneracy[i] for i in range(_nrpts)
        if np.abs(_R[i,0]) <= a and np.abs(_R[i,1]) <= b and np.abs(_R[i,2]) <= c
    ])

    hr_mn = np.array([
        hr_mn[i] for i in range(_nrpts)
        if np.abs(_R[i,0]) <= a and np.abs(_R[i,1]) <= b and np.abs(_R[i,2]) <= c
    ])

    R = np.array([
        _R[i] for i in range(_nrpts)
        if np.abs(_R[i,0]) <= a and np.abs(_R[i,1]) <= b and np.abs(_R[i,2]) <= c
    ])
    hr = [degeneracy, R, hr_mn]
    print('[reduce hr procedure] maxR=({},{},{}), nrpts have reduced from {} to {}'.format(a, b, c, _nrpts, hr_mn.shape[0]))

    return hr

def reduce_r(r, maxR):
    _R, r_mn = r
    _nrpts, num_wann = (r_mn.shape[0], r_mn.shape[2])
    if type(maxR) is int:
        a, b, c = (maxR, maxR, maxR)
    else:
        a, b, c = maxR

    if (np.array([a, b, c]) > 100).all():
        return r

    r_mn = np.array([
        r_mn[i] for i in range(_nrpts)
        if np.abs(_R[i,0]) <= a and np.abs(_R[i,1]) <= b and np.abs(_R[i,2]) <= c
    ])

    R = np.array([
        _R[i] for i in range(_nrpts)
        if np.abs(_R[i,0]) <= a and np.abs(_R[i,1]) <= b and np.abs(_R[i,2]) <= c
    ])
    r = [R, r_mn]
    print('[reduce r procedure] maxR=({},{},{}), nrpts have reduced from {} to {}'.format(a, b, c, _nrpts, r_mn.shape[0]))

    return r

def reduce_hopping_tb(hr, r, tmin, tb=False):
    degeneracy, _R, hr_mn = hr
    _nrpts, num_wann = hr_mn.shape[:2]

    hr_remain = np.array([
        1
        if np.abs(hr_mn[i]).max() > tmin else 0
        for i in range(_nrpts)
    ])
    degeneracy_hr = np.array([degeneracy[i] for i in range(_nrpts) if hr_remain[i]])
    hr_mn = np.array([hr_mn[i] for i in range(_nrpts) if hr_remain[i]])
    R_hr = np.array([_R[i] for i in range(_nrpts) if hr_remain[i]])
    hr = [degeneracy_hr, R_hr, hr_mn]
    print('[reduce hr procedure] nrpts have reduced from {} to {}'.format(_nrpts, hr_mn.shape[0]))

    try:
        _R, r_mn = r
    except ValueError:
        _R, r_mn, _degeneracy_r = r

    r_remain = np.array([
        1
        if np.abs(r_mn[i]).max() > tmin else 0
        for i in range(_nrpts)
    ])
    if tb:
        r_remain = np.array([
            1
            if (_R[i] == 0).all() else 0
            for i in range(_nrpts)
        ])
    degeneracy_r = np.array([degeneracy[i] for i in range(_nrpts) if r_remain[i]])
    r_mn = np.array([r_mn[i] for i in range(_nrpts) if r_remain[i]])
    R_r = np.array([_R[i] for i in range(_nrpts) if r_remain[i]])
    if tb:
        r_mn = np.array([[
            np.diag(np.diag(r_mn[0, 0])),
            np.diag(np.diag(r_mn[0, 1])),
            np.diag(np.diag(r_mn[0, 2])),
        ]])
    r = [R_r, r_mn, degeneracy_r]
    print('[reduce r procedure] nrpts have reduced from {} to {}'.format(_nrpts, r_mn.shape[0]))

    return hr, r

def reduce_htb(htb, tb=False, open_boundary=-1):
    _nR = htb['head']['nR']
    nw = htb['head']['nw']
    _R = htb['head']['R']
    hr_Rmn = htb['hr_Rmn']
    r_Ramn = htb['r_Ramn']

    hr_remain = np.zeros(_nR, dtype='i')
    r_remain = np.zeros(_nR, dtype='i')
    for i in range(_nR):
        if np.real(hr_Rmn[i] * hr_Rmn[i].conj()).max() > tmin:
            if open_boundary == -1:
                hr_remain[i] = 1
            elif (open_boundary != -1) and (_R[i, open_boundary] == 0):
                hr_remain[i] = 1
            else:
                hr_remain[i] = 0
        else:
            hr_remain[i] = 0

    for i in range(_nR):
        if np.abs(r_Ramn[i]).max() > tmin:
            if open_boundary == -1:
                r_remain[i] = 1
            elif (open_boundary != -1) and (_R[i, open_boundary] == 0):
                r_remain[i] = 1
            else:
                r_remain[i] = 0
        else:
            r_remain[i] = 0

    if tb:
        r_remain = np.array([
            1
            if (_R[i] == 0).all() else 0
            for i in range(_nR)
        ])

    hr_Rmn = np.array([hr_Rmn[i] for i in range(_nR) if hr_remain[i]])
    R_hr = np.array([_R[i] for i in range(_nR) if hr_remain[i]])

    r_Ramn = np.array([r_Ramn[i] for i in range(_nR) if r_remain[i]])
    R_r = np.array([_R[i] for i in range(_nR) if r_remain[i]])
    if tb:
        r_Ramn = np.array([[
            np.diag(np.diag(r_Ramn[0, 0])),
            np.diag(np.diag(r_Ramn[0, 1])),
            np.diag(np.diag(r_Ramn[0, 2])),
        ]])

    print('[reduce hr procedure] nrpts have reduced from {} to {}'.format(_nR, hr_Rmn.shape[0]))
    print('[reduce r procedure] nrpts have reduced from {} to {}'.format(_nR, r_Ramn.shape[0]))

    return hr_Rmn, r_Ramn, R_hr, R_r


def reduce_open_boundary(nR_hr, nR_r, R_hr, R_r, hr_Rmn, r_Ramn, open_boundary=0):
    slab_remain_hr = np.zeros(nR_hr, dtype='int')
    slab_remain_r = np.zeros(nR_r, dtype='int')

    for i in range(nR_hr):
        if R_hr[i, open_boundary] == 0:
            slab_remain_hr[i] = 1
        else:
            slab_remain_hr[i] = 0

    for i in range(nR_r):
        if R_r[i, open_boundary] == 0:
            slab_remain_r[i] = 1
        else:
            slab_remain_r[i] = 0

    hr_Rmn = np.array([hr_Rmn[i] for i in range(nR_hr) if slab_remain_hr[i]], dtype='complex128')
    R_hr = np.array([R_hr[i] for i in range(nR_hr) if slab_remain_hr[i]], dtype='int64')
    r_Ramn = np.array([r_Ramn[i] for i in range(nR_r) if slab_remain_r[i]], dtype='complex128')
    R_r = np.array([R_r[i] for i in range(nR_r) if slab_remain_r[i]], dtype='int64')

    nR_hr = np.count_nonzero(slab_remain_hr)
    nR_r = np.count_nonzero(slab_remain_r)

    return nR_hr, nR_r, R_hr, R_r, hr_Rmn, r_Ramn


'''
  * reduce wannier orbital procedrue
  * * reduce_Orbital_hr
  * * reduce_Orbital_r
'''
def reduce_Orbital_hr(hr, orbitals):
    degeneracy, R, _hr_mn = hr
    nrpts, _num_wann = _hr_mn.shape[:2]
    num_wann = len(orbitals)

    t = np.zeros((num_wann, _num_wann))
    hr_mn = np.zeros((nrpts, num_wann, num_wann), dtype='complex128')
    for i in range(num_wann): t[i, orbitals[i]-1] = 1

    for i in range(nrpts): hr_mn[i] = LA.multi_dot([t, _hr_mn[i], t.T])

    hr = [degeneracy, R, hr_mn]
    print('[reduce hr wannier orbital procedrue] selected orbitals are {}'.format(orbitals))

    return hr

'''
  * Slab H
  * * 
'''
def slab_hr(hr, open_boundary):
    if open_boundary == -1:
        return hr
    ob_dict = {0: 'x', 1: 'y', 2: 'z'}
    _degeneracy, _R, _hr_mn = hr
    _nrpts, num_wann = _hr_mn.shape[:2]
    remain = np.array([
        1
        if _R[i, open_boundary] == 0 else 0
        for i in range(_nrpts)
    ])

    degeneracy = np.array([_degeneracy[i] for i in range(_nrpts) if remain[i]])
    R = np.array([_R[i] for i in range(_nrpts) if remain[i]])
    hr_mn = np.array([_hr_mn[i] for i in range(_nrpts) if remain[i]])

    nrpts, num_wann = hr_mn.shape[:2]
    hr = [degeneracy, R, hr_mn]
    print('[Build slab hr in {} direction] nrpts = {}'.format(ob_dict[open_boundary], nrpts))

    return hr

def slab_rr(rr, open_boundary):
    if open_boundary == -1:
        return rr
    ob_dict = {0: 'x', 1: 'y', 2: 'z'}
    _R, _r_mn, _degeneracy = rr
    _nrpts, num_wann = _r_mn.shape[:2]
    remain = np.array([
        1
        if _R[i, open_boundary] == 0 else 0
        for i in range(_nrpts)
    ])

    degeneracy = np.array([_degeneracy[i] for i in range(_nrpts) if remain[i]])
    R = np.array([_R[i] for i in range(_nrpts) if remain[i]])
    r_mn = np.array([_r_mn[i] for i in range(_nrpts) if remain[i]])

    nrpts, num_wann = r_mn.shape[:2]
    rr = [R, r_mn, degeneracy]
    print('[Build slab rr in {} direction] nrpts = {}'.format(ob_dict[open_boundary], nrpts))

    return rr

'''
  * Parallel 
  * * kpar kgps_Gather get_kpar_gps par_Gather
'''

def kpar(kps, Ncore):
    njob = int(np.ceil(len(kps) / Ncore))
    kgps = [kps[i:i + njob] for i in range(0, len(kps), njob)]
    return kgps

def kgps_Gather(kps, Ncore, del_empty=False):
    '''
      * kgps[-1][:-njobempty]
    '''
    njob = int(np.ceil(len(kps) / Ncore))
    njobLast = len(kps) % njob
    njobempty = njob - njobLast

    kgps = [kps[i:i + njob] for i in range(0, len(kps), njob)]

    if del_empty:
        return kgps, njobempty

    if njobLast != 0:
        emptyk = np.zeros((njobempty, 3))
        kgps[-1] = np.vstack((kgps[-1], emptyk))

    return kgps, njobempty

def get_kpar_gps(kps, Ncore, del_empty=False):
    '''
      * kgps[-1][:-njobempty]
    '''
    njob = int(np.ceil(len(kps) / Ncore))
    njobLast = len(kps) % njob
    njobempty = njob - njobLast

    kgps = [kps[i:i + njob] for i in range(0, len(kps), njob)]

    if del_empty:
        return kgps, njobempty

    if njobLast != 0:
        emptyk = np.zeros((njobempty, 3))
        kgps[-1] = np.vstack((kgps[-1], emptyk))

    return kgps, njobempty

def par_Gather(kps, Ncore, sameshape=True):
    '''
      * gps[-1][:-njobempty]
    '''
    njob = int(np.ceil(len(kps) / Ncore))
    njobLast = len(kps) % njob
    njobempty = njob - njobLast

    kgps = [kps[i:i + njob] for i in range(0, len(kps), njob)]

    if sameshape:
        if njobLast != 0:
            kgps_1 = np.zeros_like(kgps[0])
            kgps_1[:-njobempty] = kgps[-1]
            kgps[-1] = kgps_1
            return kgps, njobempty
        else:
            return kgps, njobempty
    else:
        return kgps, njobempty

'''
  * Time 
'''
def timefunc(func, *args, **kwargs):
    def wrapper():
        start = time.clock()
        func(*args, **kwargs)
        end =time.clock()
        print('used: {:.2f} ms'.format((end - start) * 1000))
    return wrapper

'''
  * Check
'''
def check_hermition(H, n_print=None):
    B = 0.5 * (H - H.conj().T)
    B0 = B / np.abs(B)
    return B0


def plot_h(h):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.mlab import griddata

    ax = plt.gca()

    G = gridspec.GridSpec(1, 2)

    ax1 = plt.subplot(G[0, 0])
    ax1.matshow(h[0].real)
    plt.colorbar()

    ax2 = plt.subplot(G[0, 1])
    ax2.matshow(h[0].imag)
    cbar2 = plt.colorbar()

    subplot(1, 2, 1)
    plt.matshow(h[0].real)
    plt.colorbar()

    subplot(1, 2, 2)
    plt.matshow(h[0].imag)
    plt.colorbar()

    plt.show()

'''
  * Plot
'''

def plot_grid(grid):
    import matplotlib.gridspec as gridspec
    from matplotlib.pyplot import subplot
    import matplotlib.pyplot as plt

    G = gridspec.GridSpec(1, 1)

    ax = subplot(G[0, 0])
    plt.title('grid_K')
    ax.scatter(grid[:,0], grid[:,1], alpha=0.5)
    ax.axis('equal')


def plot_orbital(rr, un, grid):
    import matplotlib.gridspec as gridspec
    from matplotlib.pyplot import subplot
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    import seaborn as sns

    un = np.abs(un)

    rr = rr.reshape(grid[0], grid[1], grid[2], 3)[:, :, 0, :2]
    un = un.reshape(grid[0], grid[1], grid[2])[:, :, 0]
    vmax = max_value(un)
    vmax = np.max(np.abs(un))

    G = gridspec.GridSpec(1, 1)

    ax = subplot(G[0, 0])
    ax.axis([0, 134.238, -201.357, 0])

    params = {
        'figure.figsize': '6, 8'  # set figure size
    }
    pylab.rcParams.update(params)

    cmap = 'Reds'
    # cmap = sns.diverging_palette(127, 255, s=99, l=57, n=100, as_cmap=True)
    levels = np.linspace(0, vmax, 500)
    CS = plt.contourf(rr[:,:,0], rr[:,:,1], un, levels, vmax=vmax, vmin=0, cmap=cmap)

    plt.xlabel('$X (\mathring{A})$')
    plt.ylabel('$Y (\mathring{A})$')

    cbar = plt.colorbar(CS)
    cbar.set_label('$|g_n>$')
    cbar.set_ticks(np.linspace(-vmax, vmax, 5))

    plt.tight_layout()
    ax.axis('equal')


def plot_matrix(A):
    import matplotlib.pyplot as plt
    cmap = 'Reds'
    plt.imshow(A, cmap)
    plt.colorbar()
    plt.show()

# def max_value(x):
#     y = format(np.max(np.abs(x)), '.1g')
#     t = y.count('0')
#     y = float(y) + 0.1 ** t
#     return y

def max_value(xx):
    x = np.max(np.abs(xx))
    for i in range(64):
        y = round(x, i)
        if y != 0:
            break
    y += 10**-i
    return y

'''
  * Matrix
'''
# def eye_filter(N, F):
#     if type(F) is int:
#         filter = np.array([np.eye(N, k=i) for i in np.arange(-F + 1, F)])
#         filter = np.sum(filter, axis=0)
#     elif (type(F) is list) or (type(F) is tuple):
#         filter = np.array([
#             np.eye(N, N, i)
#             for i in F
#         ])
#         filter = np.sum(filter, axis=0)
#     else:
#         filter = np.ones([N, N])
#     return filter

def eye_filter(N, F):
    filter = np.zeros([N, N])
    if type(F) is int:
        for i in range(-F + 1, F):
            filter += np.eye(N, k=i)
    elif (type(F) is list) or (type(F) is tuple):
        for i in F:
            filter += np.eye(N, N, i)
    else:
        filter = np.ones([N, N])
    return filter

'''
  * is odd or even list
'''

def sign_list(y, n):
    x = np.arange(1, n+1)
    sign = np.zeros([n, n], dtype='int')
    for i, j in zip(x, y):
        sign[i-1, j-1] = 1
    sign = LA.det(sign)
    if sign == 0:
        print('ERROR sign = 0')
    return sign


'''
  * eye
'''
def eye_matrix(r_Rmn, nw, nR):
    H = np.zeros([nw*nR, nw*nR], dtype='complex128')
    for i in range(-nR//2+1, nR//2+1):
        H += np.kron(np.eye(nR, nR, i), r_Rmn[i])
    return H


# def scissor_operator(H):
#     return H