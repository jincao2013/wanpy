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

__date__ = "Nov. 5, 2019"

import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_matrix',
           'plot_matrix2',
           'plot_grid',
           'plot_color_on_grid',
           'plot_orbital',
           ]

def plot_matrix(A, cmap='Reds'):
    # cmap = 'seismic'
    plt.clf()
    plt.imshow(A, cmap)
    plt.colorbar()
    plt.show()

def plot_matrix2(A, setnorm=False):
    plt.clf()
    cmap = 'Reds'
    if setnorm:
        A /= np.max(A)
    plt.matshow(A)
    plt.colorbar()
    plt.show()

def plot_grid(grid, lattice=None, XX=None, YY=None):
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches

    plt.figure('grid')
    plt.clf()

    plt.scatter(grid[:, 0], grid[:, 1], alpha=0.5)

    if XX is not None and YY is not None:
        plt.axis([XX[0], XX[1], YY[0], YY[1]])
    plt.axis('equal')

    if lattice is not None:
        cube = lattice[:2, :2].T
        path_data = [
            (mpath.Path.MOVETO, [0, 0]),
            (mpath.Path.LINETO, cube[0]),
            (mpath.Path.LINETO, cube[0] + cube[1]),
            (mpath.Path.LINETO, cube[1]),
            (mpath.Path.LINETO, [0, 0]),
        ]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor='#607d8b', alpha=0.3)
        plt.gca().add_patch(patch)

    plt.show()

def plot_color_on_grid(grid, c, s=20, latt=None, vv=None, XX=None, YY=None, cmap='seismic'):
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches

    plt.figure('color_on_grid', figsize=[4,3], dpi=150)
    plt.clf()
    # cmap = 'seismic'

    if vv is None:
        vmin, vmax = np.min(c), np.max(c)
    levels = np.linspace(vmin, vmax, 500)

    # plt.title('grid_K')
    cs = plt.scatter(grid[:,0], grid[:,1], c=c, s=s, cmap=cmap, alpha=1, vmax=vmax, vmin=vmin)

    if XX is not None and YY is not None:
        plt.axis([XX[0], XX[1], YY[0], YY[1]])
    plt.axis('equal')

    if latt is not None:
        cube = latt[:2, :2].T
        path_data = [
            (mpath.Path.MOVETO, [0, 0]),
            (mpath.Path.LINETO, cube[0]),
            (mpath.Path.LINETO, cube[0] + cube[1]),
            (mpath.Path.LINETO, cube[1]),
            (mpath.Path.LINETO, [0, 0]),
        ]
        codes, verts = zip(*path_data)
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor='#ffff00', alpha=0.3)
        plt.gca().add_patch(patch)

    cbar = plt.colorbar(cs)
    cbar.set_ticks(np.linspace(vmin, vmax, 5))

    plt.tick_params(direction="in", pad=8)
    plt.show()

def plot_grid2_complex(grid, c, s=200, vmax=None):
    from matplotlib.pyplot import subplot
    import matplotlib.gridspec as gridspec

    cmap = 'seismic'
    G = gridspec.GridSpec(1, 1)

    norm = np.abs(c)
    angle = np.angle(c)

    if vmax == None:
        vmax = np.max(np.abs(norm))
    levels = np.linspace(-vmax, vmax, 500)

    ax = subplot(G[0, 0])
    plt.title('grid_K')
    cs = ax.scatter(grid[:,0], grid[:,1], c=norm, s=s, cmap=cmap, alpha=1, vmax=vmax, vmin=0)
    ax.axis('equal')

    cbar = plt.colorbar(cs)
    cbar.set_ticks(np.linspace(0, vmax, 5))

    ax.quiver(grid[:, 0], grid[:, 1], np.cos(angle), np.sin(angle), scale=20, pivot='middle', color='grey')

def plot_orbital(rr, un, grid, latt=None, cmap='Reds'):
    from matplotlib.pyplot import subplot
    import matplotlib.gridspec as gridspec
    import pylab
    # un = np.log(np.abs(un))
    un = np.abs(un)

    rr = rr.reshape(grid[0], grid[1], grid[2], 3)[:, :, 0, :2]
    un = un.reshape(grid[0], grid[1], grid[2])[:, :, 0]
    # vmax = max_value(un)
    vmax = np.max(un)
    vmin = 0 # np.min(un)

    G = gridspec.GridSpec(1, 1)

    ax = subplot(G[0, 0])
    ax.axis([0, 134.238, -201.357, 0])

    params = {
        'figure.figsize': '6, 8'  # set figure size
    }
    pylab.rcParams.update(params)

    # cmap = 'Reds'
    # cmap = sns.diverging_palette(127, 255, s=99, l=57, n=100, as_cmap=True)
    levels = np.linspace(vmin, vmax, 500)
    CS = plt.contourf(rr[:,:,0], rr[:,:,1], un, levels, vmax=vmax, vmin=vmin, cmap=cmap)

    plt.xlabel('$X$')
    plt.ylabel('$Y$')

    cbar = plt.colorbar(CS)
    cbar.set_label('$|g_n>$')
    cbar.set_ticks(np.linspace(-vmax, vmax, 5))

    plt.tight_layout()
    ax.axis('equal')

# def plot_band(band, eemin=-3.0, eemax=3.0, unit='D', xlabel=None, save_csv=False):
#     bandE = band['bandE']
#     bandU = band['bandU']
#     kpath = band['kpath']
#     kpath_list = band['kpath_HSP']
#     tbg = band['tbg']
#     BC = band['BC'] * 0.001
#     cmap = 'seismic'
#
#     nk, nw = bandE.shape
#     nline = kpath_list.shape[0] - 1
#     if unit.upper() == 'C':
#         kpath = multi_dot([tbg.lattG, kpath.T]).T
#     kpath = kmold(kpath)
#
#     if save_csv:
#         col_list = ['Band {}'.format(i+1) for i in range(nw)]
#         bandpd = pd.DataFrame(bandE, index=np.arange(nk)+1, columns=col_list)
#         bandpd.insert(loc=0, column='|k| (Ans-1)', value=kpath)
#         bandpd.to_csv(r'band.csv', float_format='% 10.5f', sep='\t', encoding='utf-8')
#
#
#     '''
#       * plot band
#     '''
#     G = gridspec.GridSpec(1, 1)
#     ax = subplot(G[0, 0])
#     ax.axis([kpath.min(), kpath.max(), eemin, eemax])
#     plt.axhline(0, color='k', linewidth=0.5)
#
#     for i in range(nw):
#         ax.plot(kpath, bandE[:, i], linewidth=1, linestyle="-", color='k', alpha=0.7)
#
#     for i in range(1, nline):
#         plt.axvline(x=kpath[i * nk//nline], linestyle='-', color='k', linewidth=0.5, alpha=1, zorder=2)
#
#     if xlabel is not None:
#         # xlabel = ['K', 'G', '-K', '-M', 'G', 'M', 'K']
#         num_xlabel = len(xlabel)
#         plt.xticks(kpath[np.arange(num_xlabel) * (nk // nline)],
#                    xlabel)
#
#     plt.tight_layout()
#     plt.show
#