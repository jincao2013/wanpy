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

__date__ = "Aug. 11, 2017"

from wanpy.core.mesh import kmold

__all__ = [
    'kmold'
]

# def kmold(kkc):
#     nk = kkc.shape[0]
#     kkc = kkc[1:, :] - kkc[:-1, :]
#     kkc = np.vstack([np.array([0, 0, 0]), kkc])
#     k_mold = np.sqrt(np.einsum('ka,ka->k', kkc, kkc))
#     k_mold = LA.multi_dot([np.tri(nk), k_mold])
#     return k_mold

# def distance(vec_1,vec_2):
#
#     vec_1 = np.array(vec_1,dtype='float')
#     vec_2 = np.array(vec_2,dtype='float')
#
#     return LA.norm(vec_1 - vec_2)

# def reduce_hr(hr, maxR):
#     degeneracy, _R, hr_mn = hr
#     _nrpts, num_wann = hr_mn.shape[:2]
#     if type(maxR) is int:
#         a, b, c = (maxR, maxR, maxR)
#     else:
#         a, b, c = maxR
#
#     if (np.array([a, b, c]) > 100).all():
#         return hr
#
#     degeneracy = np.array([
#         degeneracy[i] for i in range(_nrpts)
#         if np.abs(_R[i,0]) <= a and np.abs(_R[i,1]) <= b and np.abs(_R[i,2]) <= c
#     ])
#
#     hr_mn = np.array([
#         hr_mn[i] for i in range(_nrpts)
#         if np.abs(_R[i,0]) <= a and np.abs(_R[i,1]) <= b and np.abs(_R[i,2]) <= c
#     ])
#
#     R = np.array([
#         _R[i] for i in range(_nrpts)
#         if np.abs(_R[i,0]) <= a and np.abs(_R[i,1]) <= b and np.abs(_R[i,2]) <= c
#     ])
#     hr = [degeneracy, R, hr_mn]
#     print('[reduce hr procedure] maxR=({},{},{}), nrpts have reduced from {} to {}'.format(a, b, c, _nrpts, hr_mn.shape[0]))
#
#     return hr

# def reduce_r(r, maxR):
#     _R, r_mn = r
#     _nrpts, num_wann = (r_mn.shape[0], r_mn.shape[2])
#     if type(maxR) is int:
#         a, b, c = (maxR, maxR, maxR)
#     else:
#         a, b, c = maxR
#
#     if (np.array([a, b, c]) > 100).all():
#         return r
#
#     r_mn = np.array([
#         r_mn[i] for i in range(_nrpts)
#         if np.abs(_R[i,0]) <= a and np.abs(_R[i,1]) <= b and np.abs(_R[i,2]) <= c
#     ])
#
#     R = np.array([
#         _R[i] for i in range(_nrpts)
#         if np.abs(_R[i,0]) <= a and np.abs(_R[i,1]) <= b and np.abs(_R[i,2]) <= c
#     ])
#     r = [R, r_mn]
#     print('[reduce r procedure] maxR=({},{},{}), nrpts have reduced from {} to {}'.format(a, b, c, _nrpts, r_mn.shape[0]))
#
#     return r

# def reduce_hopping_tb(hr, r, tmin, tb=False):
#     degeneracy, _R, hr_mn = hr
#     _nrpts, num_wann = hr_mn.shape[:2]
#
#     hr_remain = np.array([
#         1
#         if np.abs(hr_mn[i]).max() > tmin else 0
#         for i in range(_nrpts)
#     ])
#     degeneracy_hr = np.array([degeneracy[i] for i in range(_nrpts) if hr_remain[i]])
#     hr_mn = np.array([hr_mn[i] for i in range(_nrpts) if hr_remain[i]])
#     R_hr = np.array([_R[i] for i in range(_nrpts) if hr_remain[i]])
#     hr = [degeneracy_hr, R_hr, hr_mn]
#     print('[reduce hr procedure] nrpts have reduced from {} to {}'.format(_nrpts, hr_mn.shape[0]))
#
#     try:
#         _R, r_mn = r
#     except ValueError:
#         _R, r_mn, _degeneracy_r = r
#
#     r_remain = np.array([
#         1
#         if np.abs(r_mn[i]).max() > tmin else 0
#         for i in range(_nrpts)
#     ])
#     if tb:
#         r_remain = np.array([
#             1
#             if (_R[i] == 0).all() else 0
#             for i in range(_nrpts)
#         ])
#     degeneracy_r = np.array([degeneracy[i] for i in range(_nrpts) if r_remain[i]])
#     r_mn = np.array([r_mn[i] for i in range(_nrpts) if r_remain[i]])
#     R_r = np.array([_R[i] for i in range(_nrpts) if r_remain[i]])
#     if tb:
#         r_mn = np.array([[
#             np.diag(np.diag(r_mn[0, 0])),
#             np.diag(np.diag(r_mn[0, 1])),
#             np.diag(np.diag(r_mn[0, 2])),
#         ]])
#     r = [R_r, r_mn, degeneracy_r]
#     print('[reduce r procedure] nrpts have reduced from {} to {}'.format(_nrpts, r_mn.shape[0]))
#
#     return hr, r

# def reduce_htb(htb, tb=False, open_boundary=-1):
#     _nR = htb['head']['nR']
#     nw = htb['head']['nw']
#     _R = htb['head']['R']
#     hr_Rmn = htb['hr_Rmn']
#     r_Ramn = htb['r_Ramn']
#
#     hr_remain = np.zeros(_nR, dtype='i')
#     r_remain = np.zeros(_nR, dtype='i')
#     for i in range(_nR):
#         if np.real(hr_Rmn[i] * hr_Rmn[i].conj()).max() > tmin:
#             if open_boundary == -1:
#                 hr_remain[i] = 1
#             elif (open_boundary != -1) and (_R[i, open_boundary] == 0):
#                 hr_remain[i] = 1
#             else:
#                 hr_remain[i] = 0
#         else:
#             hr_remain[i] = 0
#
#     for i in range(_nR):
#         if np.abs(r_Ramn[i]).max() > tmin:
#             if open_boundary == -1:
#                 r_remain[i] = 1
#             elif (open_boundary != -1) and (_R[i, open_boundary] == 0):
#                 r_remain[i] = 1
#             else:
#                 r_remain[i] = 0
#         else:
#             r_remain[i] = 0
#
#     if tb:
#         r_remain = np.array([
#             1
#             if (_R[i] == 0).all() else 0
#             for i in range(_nR)
#         ])
#
#     hr_Rmn = np.array([hr_Rmn[i] for i in range(_nR) if hr_remain[i]])
#     R_hr = np.array([_R[i] for i in range(_nR) if hr_remain[i]])
#
#     r_Ramn = np.array([r_Ramn[i] for i in range(_nR) if r_remain[i]])
#     R_r = np.array([_R[i] for i in range(_nR) if r_remain[i]])
#     if tb:
#         r_Ramn = np.array([[
#             np.diag(np.diag(r_Ramn[0, 0])),
#             np.diag(np.diag(r_Ramn[0, 1])),
#             np.diag(np.diag(r_Ramn[0, 2])),
#         ]])
#
#     print('[reduce hr procedure] nrpts have reduced from {} to {}'.format(_nR, hr_Rmn.shape[0]))
#     print('[reduce r procedure] nrpts have reduced from {} to {}'.format(_nR, r_Ramn.shape[0]))
#
#     return hr_Rmn, r_Ramn, R_hr, R_r


# def reduce_open_boundary(nR_hr, nR_r, R_hr, R_r, hr_Rmn, r_Ramn, open_boundary=0):
#     slab_remain_hr = np.zeros(nR_hr, dtype='int')
#     slab_remain_r = np.zeros(nR_r, dtype='int')
#
#     for i in range(nR_hr):
#         if R_hr[i, open_boundary] == 0:
#             slab_remain_hr[i] = 1
#         else:
#             slab_remain_hr[i] = 0
#
#     for i in range(nR_r):
#         if R_r[i, open_boundary] == 0:
#             slab_remain_r[i] = 1
#         else:
#             slab_remain_r[i] = 0
#
#     hr_Rmn = np.array([hr_Rmn[i] for i in range(nR_hr) if slab_remain_hr[i]], dtype='complex128')
#     R_hr = np.array([R_hr[i] for i in range(nR_hr) if slab_remain_hr[i]], dtype='int64')
#     r_Ramn = np.array([r_Ramn[i] for i in range(nR_r) if slab_remain_r[i]], dtype='complex128')
#     R_r = np.array([R_r[i] for i in range(nR_r) if slab_remain_r[i]], dtype='int64')
#
#     nR_hr = np.count_nonzero(slab_remain_hr)
#     nR_r = np.count_nonzero(slab_remain_r)
#
#     return nR_hr, nR_r, R_hr, R_r, hr_Rmn, r_Ramn


'''
  * reduce wannier orbital procedrue
  * * reduce_Orbital_hr
  * * reduce_Orbital_r
'''
# def reduce_Orbital_hr(hr, orbitals):
#     degeneracy, R, _hr_mn = hr
#     nrpts, _num_wann = _hr_mn.shape[:2]
#     num_wann = len(orbitals)
#
#     t = np.zeros((num_wann, _num_wann))
#     hr_mn = np.zeros((nrpts, num_wann, num_wann), dtype='complex128')
#     for i in range(num_wann): t[i, orbitals[i]-1] = 1
#
#     for i in range(nrpts): hr_mn[i] = LA.multi_dot([t, _hr_mn[i], t.T])
#
#     hr = [degeneracy, R, hr_mn]
#     print('[reduce hr wannier orbital procedrue] selected orbitals are {}'.format(orbitals))
#
#     return hr

'''
  * Slab H
  * * 
'''
# def slab_hr(hr, open_boundary):
#     if open_boundary == -1:
#         return hr
#     ob_dict = {0: 'x', 1: 'y', 2: 'z'}
#     _degeneracy, _R, _hr_mn = hr
#     _nrpts, num_wann = _hr_mn.shape[:2]
#     remain = np.array([
#         1
#         if _R[i, open_boundary] == 0 else 0
#         for i in range(_nrpts)
#     ])
#
#     degeneracy = np.array([_degeneracy[i] for i in range(_nrpts) if remain[i]])
#     R = np.array([_R[i] for i in range(_nrpts) if remain[i]])
#     hr_mn = np.array([_hr_mn[i] for i in range(_nrpts) if remain[i]])
#
#     nrpts, num_wann = hr_mn.shape[:2]
#     hr = [degeneracy, R, hr_mn]
#     print('[Build slab hr in {} direction] nrpts = {}'.format(ob_dict[open_boundary], nrpts))
#
#     return hr
#
# def slab_rr(rr, open_boundary):
#     if open_boundary == -1:
#         return rr
#     ob_dict = {0: 'x', 1: 'y', 2: 'z'}
#     _R, _r_mn, _degeneracy = rr
#     _nrpts, num_wann = _r_mn.shape[:2]
#     remain = np.array([
#         1
#         if _R[i, open_boundary] == 0 else 0
#         for i in range(_nrpts)
#     ])
#
#     degeneracy = np.array([_degeneracy[i] for i in range(_nrpts) if remain[i]])
#     R = np.array([_R[i] for i in range(_nrpts) if remain[i]])
#     r_mn = np.array([_r_mn[i] for i in range(_nrpts) if remain[i]])
#
#     nrpts, num_wann = r_mn.shape[:2]
#     rr = [R, r_mn, degeneracy]
#     print('[Build slab rr in {} direction] nrpts = {}'.format(ob_dict[open_boundary], nrpts))
#
#     return rr

'''
  * Parallel 
  * * kpar kgps_Gather get_kpar_gps par_Gather
'''
# def kpar(kps, Ncore):
#     njob = int(np.ceil(len(kps) / Ncore))
#     kgps = [kps[i:i + njob] for i in range(0, len(kps), njob)]
#     return kgps
#
# def kgps_Gather(kps, Ncore, del_empty=False):
#     '''
#       * kgps[-1][:-njobempty]
#     '''
#     njob = int(np.ceil(len(kps) / Ncore))
#     njobLast = len(kps) % njob
#     njobempty = njob - njobLast
#
#     kgps = [kps[i:i + njob] for i in range(0, len(kps), njob)]
#
#     if del_empty:
#         return kgps, njobempty
#
#     if njobLast != 0:
#         emptyk = np.zeros((njobempty, 3))
#         kgps[-1] = np.vstack((kgps[-1], emptyk))
#
#     return kgps, njobempty
#
# def get_kpar_gps(kps, Ncore, del_empty=False):
#     '''
#       * kgps[-1][:-njobempty]
#     '''
#     njob = int(np.ceil(len(kps) / Ncore))
#     njobLast = len(kps) % njob
#     njobempty = njob - njobLast
#
#     kgps = [kps[i:i + njob] for i in range(0, len(kps), njob)]
#
#     if del_empty:
#         return kgps, njobempty
#
#     if njobLast != 0:
#         emptyk = np.zeros((njobempty, 3))
#         kgps[-1] = np.vstack((kgps[-1], emptyk))
#
#     return kgps, njobempty
#
# def par_Gather(kps, Ncore, sameshape=True):
#     '''
#       * gps[-1][:-njobempty]
#     '''
#     njob = int(np.ceil(len(kps) / Ncore))
#     njobLast = len(kps) % njob
#     njobempty = njob - njobLast
#
#     kgps = [kps[i:i + njob] for i in range(0, len(kps), njob)]
#
#     if sameshape:
#         if njobLast != 0:
#             kgps_1 = np.zeros_like(kgps[0])
#             kgps_1[:-njobempty] = kgps[-1]
#             kgps[-1] = kgps_1
#             return kgps, njobempty
#         else:
#             return kgps, njobempty
#     else:
#         return kgps, njobempty

# def timefunc(func, *args, **kwargs):
#     def wrapper():
#         start = time.clock()
#         func(*args, **kwargs)
#         end =time.clock()
#         print('used: {:.2f} ms'.format((end - start) * 1000))
#     return wrapper
