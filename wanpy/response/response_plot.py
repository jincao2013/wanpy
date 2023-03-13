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

__date__ = "Nov. 5, 2019"


import time
import os
import sys
import getpass
sys.path.append(os.environ.get('PYTHONPATH'))

import numpy as np
from numpy import linalg as LA
from numpy.linalg import multi_dot
from wanpy.core.toolkits import kmold

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pylab as pylab

from matplotlib.pyplot import subplot
from pylab import *
import seaborn as sns

from wanpy.core.toolkits import *

import pandas as pd


def plot001_GRID21_LINE1(fname, yy=None, div_e2_1=False, div_e2_2=False):
    '''
      * plot ORMap(wannier) response function
      * in one fig
    '''
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import matplotlib.pylab as pylab

    datas = np.load(fname, allow_pickle=True)
    ee = datas['ee']
    gate = datas['gate']
    rank2Index = datas['rank2Index']
    rank3Index = datas['rank3Index']
    RS1 = datas['RS1']
    RS2 = datas['RS2']
    cell = datas['cell']
    RSunit = datas['RSunit']
    wcc = datas['wcc']
    title_RS1 = datas['title_RS1']
    title_RS2 = datas['title_RS2']

    ne = ee.shape[0]
    ngate = gate.shape[0]
    datas.close()

    if div_e2_1:
        RS1 = np.real(RS1 / (ee ** 2 + 1j * 0.01))
    if div_e2_2:
        RS2 = np.real(RS2 / (ee ** 2 + 1j * 0.01))

    RS1max = max_value(RS1)
    RS2max = max_value(RS2)

    loc = r'upper right'

    params = {
        # 'axes.labelsize': '16',
        # 'xtick.labelsize': '16',
        # 'ytick.labelsize': '13',
        # 'lines.linewidth': '2',
        # 'legend.fontsize': '20',
        'figure.figsize': '5, 8'  # set figure size
    }
    pylab.rcParams.update(params)

    G = gridspec.GridSpec(2, 1)

    ax = subplot(G[0, 0])
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel(r'$RS1 {}$'.format(RSunit))
    ax.set_xlabel(r'$\omega(eV)$')
    ax.axis([ee.min(), ee.max(), -RS1max, RS1max])
    if yy != None:
        ax.axis([ee.min(), ee.max(), yy[0], yy[1]])
    ax.plot(ee, RS1, color='#263238', linewidth=1.5, alpha=1, label=title_RS1)
    legend = ax.legend(loc=loc, shadow=True, fontsize=10)
    legend.get_frame().set_facecolor('#eceff1')

    ax = subplot(G[1, 0])
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel(r'$RS2 {}$'.format(RSunit))
    ax.set_xlabel(r'$\omega(eV)$')
    ax.axis([ee.min(), ee.max(), -RS2max, RS2max])
    if yy != None:
        ax.axis([ee.min(), ee.max(), yy[0], yy[1]])
    ax.plot(ee, RS2, color='#263238', linewidth=1.5, alpha=1, label=title_RS2)
    legend = ax.legend(loc=loc, shadow=True, fontsize=10)
    legend.get_frame().set_facecolor('#eceff1')

    plt.tight_layout()
    plt.show()


def plot002_GRID11_LINE_ngate_Rank3(fname, yy=None):
    '''
      * plot ORMap(wannier) response function
      * in one fig
    '''
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import matplotlib.pylab as pylab

    datas = np.load(fname, allow_pickle=True)
    ee = datas['ee']
    gate = datas['gate']
    rank2Index = datas['rank2Index']
    rank3Index = datas['rank3Index']
    RS = datas['RS'] / 1e2
    cell = datas['cell']
    RSunit = datas['RSunit']
    wcc = datas['wcc']
    title_RS = datas['title_RS']
    ne = ee.shape[0]
    ngate = gate.shape[0]
    datas.close()

    RSmax = max_value(RS)
    loc = r'upper right'
    tensordict = {0: 'x', 1: 'y', 2: 'z'}
    tensorstring = '{}{}{}'.format(tensordict[rank3Index[0]], tensordict[rank3Index[1]], tensordict[rank3Index[2]])
    cm = sns.diverging_palette(10, 220, sep=1, n=ngate)

    params = {
        # 'axes.labelsize': '16',
        # 'xtick.labelsize': '16',
        # 'ytick.labelsize': '13',
        # 'lines.linewidth': '2',
        # 'legend.fontsize': '20',
        'figure.figsize': '6, 5'  # set figure size
    }
    pylab.rcParams.update(params)

    G = gridspec.GridSpec(1, 1)

    ax = subplot(G[0, 0])
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel(r'$R^{} {}$'.format('{'+tensorstring+'}', RSunit))
    ax.set_xlabel(r'$\omega(eV)$')
    ax.axis([ee.min(), ee.max(), -RSmax, RSmax])
    if yy is not None:
        ax.axis([ee.min(), ee.max(), yy[0], yy[1]])
    for i in range(ngate):
        # ax.plot(ee, RS[i], color=cm[i], linewidth=1.5, alpha=1)
        ax.plot(ee, RS[i], color='blue', linewidth=1.5, alpha=1)
    # ax.plot(ee, OR1, color='#263238', linewidth=1.5, alpha=1, label='Shift current $\sigma_{}$ \n ewide={} meV'.format('{'+tensorstring+'}', ewide))
    # ax.plot(ee, 3.13e-4/np.sqrt(ee-0.2), color='#d32f2f', linewidth=1.5, alpha=1) #  THE jdos of 1d-free-ele-gas
    # legend = ax.legend(loc=loc, shadow=True, fontsize=10)
    # legend.get_frame().set_facecolor('#eceff1')

    plt.tight_layout()
    plt.show()


'''
  * unamed
'''

def plot001_OR_Bulk(fname=r'SC.npz', yy=None, div_e2_1=False, div_e2_2=False):
    '''
      * plot ORMap(wannier) response function
      * in one fig
    '''
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import matplotlib.pylab as pylab

    datas = np.load(fname)
    ee = datas['ee']
    ne = ee.shape[0]
    tensorIndex = datas['tensorIndex']
    OR1 = datas['RS1']
    ORi = datas['RS2']
    datas.close()

    if div_e2_1:
        OR1 = np.real(OR1 / (ee ** 2 + 1j * 0.01))
    if div_e2_2:
        ORi = np.real(ORi / (ee ** 2 + 1j * 0.01))

    OR1max = max_value(OR1)
    ORimax = max_value(ORi)
    loc = r'upper right'
    tensordict = {0: 'x', 1: 'y', 2: 'z'}
    tensorstring = '{}{}{}'.format(tensordict[tensorIndex[0]], tensordict[tensorIndex[1]], tensordict[tensorIndex[2]])
    # cm = sns.diverging_palette(10, 220, sep=1, n=nmap)

    params = {
        # 'axes.labelsize': '16',
        # 'xtick.labelsize': '16',
        # 'ytick.labelsize': '13',
        # 'lines.linewidth': '2',
        # 'legend.fontsize': '20',
        'figure.figsize': '5, 8'  # set figure size
    }
    pylab.rcParams.update(params)

    G = gridspec.GridSpec(2, 1)

    ax = subplot(G[0, 0])
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel(r'$\sigma_{}^{} (A.U.)$'.format('{sc}','{'+tensorstring+'}'))
    ax.set_xlabel(r'$\omega(eV)$')
    ax.axis([ee.min(), ee.max(), -OR1max, OR1max])
    if yy is not None:
        ax.axis([ee.min(), ee.max(), yy[0], yy[1]])
    ax.plot(ee, OR1, color='#263238', linewidth=1.5, alpha=1, label='$\sigma_{}$'.format('{'+tensorstring+'}'))
    # ax.plot(ee, 3.13e-4 / np.sqrt(ee - 0.2), color='#d32f2f', linewidth=1.5, alpha=1) #  THE jdos of 1d-free-ele-gas
    legend = ax.legend(loc=loc, shadow=True, fontsize=10)
    legend.get_frame().set_facecolor('#eceff1')

    ax = subplot(G[1, 0])
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel(r'$\sigma_{}^{} (A.U.)$'.format('{ic}','{'+tensorstring+'}'))
    ax.set_xlabel(r'$\omega(eV)$')
    ax.axis([ee.min(), ee.max(), -ORimax, ORimax])
    if yy is not None:
        ax.axis([ee.min(), ee.max(), yy[0], yy[1]])
    ax.plot(ee, ORi, color='#263238', linewidth=1.5, alpha=1, label='$\sigma_{}$, AD ewide'.format('{'+tensorstring+'}'))
    # ax.plot(ee, 3.13e-4/np.sqrt(ee-0.2), color='#d32f2f', linewidth=1.5, alpha=1) #  THE jdos of 1d-free-ele-gas
    legend = ax.legend(loc=loc, shadow=True, fontsize=10)
    legend.get_frame().set_facecolor('#eceff1')

    plt.tight_layout()
    plt.show()


def plot002_rank2(fname, yy=None):
    '''
      * plot ORMap(wannier) response function
      * in one fig
    '''
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import matplotlib.pylab as pylab

    datas = np.load(fname)
    ee = datas['ee']
    ne = ee.shape[0]
    rank2Index = datas['rank2Index']
    OR1 = datas['RS1']
    ORi = datas['RS2']
    datas.close()

    OR1max = max_value(OR1)
    ORimax = max_value(ORi)
    loc = r'upper right'
    tensordict = {0: 'x', 1: 'y', 2: 'z'}
    tensorstring = '{}{}'.format(tensordict[rank2Index[0]], tensordict[rank2Index[1]])
    # cm = sns.diverging_palette(10, 220, sep=1, n=nmap)

    params = {
        # 'axes.labelsize': '16',
        # 'xtick.labelsize': '16',
        # 'ytick.labelsize': '13',
        # 'lines.linewidth': '2',
        # 'legend.fontsize': '20',
        'figure.figsize': '5, 8'  # set figure size
    }
    pylab.rcParams.update(params)

    G = gridspec.GridSpec(2, 1)

    ax = subplot(G[0, 0])
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel(r'$Re[\epsilon^{{RPA}}_{}]$'.format('{'+tensorstring+'}'))
    ax.set_xlabel(r'Frequency of incident light(eV)')
    ax.axis([ee.min(), ee.max(), -OR1max, OR1max])
    if yy is not None:
        ax.axis([ee.min(), ee.max(), yy[0], yy[1]])
    ax.plot(ee, OR1, color='#263238', linewidth=1.5, alpha=1, label=r'$Re[\epsilon^{{RPA}}_{}]$'.format('{'+tensorstring+'}'))
    # ax.plot(ee, 3.13e-4 / np.sqrt(ee - 0.2), color='#d32f2f', linewidth=1.5, alpha=1) #  THE jdos of 1d-free-ele-gas
    legend = ax.legend(loc=loc, shadow=True, fontsize=10)
    legend.get_frame().set_facecolor('#eceff1')

    ax = subplot(G[1, 0])
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel(r'$Im[\epsilon^{{RPA}}_{}]$'.format('{'+tensorstring+'}'))
    ax.set_xlabel(r'Frequency of incident light (eV)')
    ax.axis([ee.min(), ee.max(), -ORimax, ORimax])
    if yy is not None:
        ax.axis([ee.min(), ee.max(), yy[0], yy[1]])
    ax.plot(ee, ORi, color='#263238', linewidth=1.5, alpha=1, label=r'$Im[\epsilon^{{RPA}}_{}]$'.format('{'+tensorstring+'}'))
    # ax.plot(ee, 3.13e-4/np.sqrt(ee-0.2), color='#d32f2f', linewidth=1.5, alpha=1) #  THE jdos of 1d-free-ele-gas
    legend = ax.legend(loc=loc, shadow=True, fontsize=10)
    legend.get_frame().set_facecolor('#eceff1')

    plt.tight_layout()
    plt.show()


def plot003_kerr(fname=r'OR_Bulk.npz', yy=None):
    '''
      * plot ORMap(wannier) response function
      * in one fig
    '''
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import matplotlib.pylab as pylab

    datas = np.load(fname)
    ee = datas['ee']
    ne = ee.shape[0]
    rank2Index = datas['rank2Index']
    phi_p = np.real(datas['phi_p'])
    phi_s = np.real(datas['phi_s'])
    datas.close()

    OR1max = max_value(phi_p)
    ORimax = max_value(phi_s)
    loc = r'upper right'

    params = {
        'figure.figsize': '5, 8'
    }
    pylab.rcParams.update(params)

    G = gridspec.GridSpec(2, 1)

    ax = subplot(G[0, 0])
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel(r'$\phi_{K}^{p} (degree)$')
    ax.set_xlabel(r'Frequency of incident light(eV)')
    ax.axis([ee.min(), ee.max(), -OR1max, OR1max])
    if yy is not None:
        ax.axis([ee.min(), ee.max(), yy[0], yy[1]])
    ax.plot(ee, phi_p, color='#263238', linewidth=1.5, alpha=1, label=r'$\phi_{K}^{p} (degree)$')
    # ax.plot(ee, 3.13e-4 / np.sqrt(ee - 0.2), color='#d32f2f', linewidth=1.5, alpha=1) #  THE jdos of 1d-free-ele-gas
    legend = ax.legend(loc=loc, shadow=True, fontsize=10)
    legend.get_frame().set_facecolor('#eceff1')

    ax = subplot(G[1, 0])
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel(r'$\phi_{K}^{s} (degree)$')
    ax.set_xlabel(r'Frequency of incident light (eV)')
    ax.axis([ee.min(), ee.max(), -ORimax, ORimax])
    if yy is not None:
        ax.axis([ee.min(), ee.max(), yy[0], yy[1]])
    ax.plot(ee, phi_s, color='#263238', linewidth=1.5, alpha=1, label=r'$\phi_{K}^{s} (degree)$')
    # ax.plot(ee, 3.13e-4/np.sqrt(ee-0.2), color='#d32f2f', linewidth=1.5, alpha=1) #  THE jdos of 1d-free-ele-gas
    legend = ax.legend(loc=loc, shadow=True, fontsize=10)
    legend.get_frame().set_facecolor('#eceff1')

    plt.tight_layout()
    plt.show()


def plot004_intra_dipole(fname=r'BC_dipole.npz', yy=None):
    '''
      * plot ORMap(wannier) response function
      * in one fig
    '''
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import matplotlib.pylab as pylab

    datas = np.load(fname)
    gate = datas['gate']
    ngate = gate.shape[0]
    Dx = datas['RS1']
    Dy = datas['RS2']
    datas.close()

    loc = r'upper right'

    params = {
        'figure.figsize': '6, 8'
    }
    pylab.rcParams.update(params)


    D = np.sqrt(Dx ** 2 + Dy ** 2)
    theta = angle(Dx + 1j * Dy) * 360 / np.pi / 2

    maxY = np.max(D)

    G = gridspec.GridSpec(2, 1)

    ax = subplot(G[0, 0])
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel(r'BC dipole $(\mathring{A})$')
    ax.set_xlabel(r'Fermi level(eV)')
    ax.axis([gate.min(), gate.max(), 0, maxY])
    ax.plot(gate, D, color='#263238', linewidth=1.5, alpha=1, label=r'BC dipole $(\mathring{A})$')
    # ax.plot(ee, 3.13e-4 / np.sqrt(ee - 0.2), color='#d32f2f', linewidth=1.5, alpha=1) #  THE jdos of 1d-free-ele-gas
    legend = ax.legend(loc=loc, shadow=True, fontsize=10)
    legend.get_frame().set_facecolor('#eceff1')

    ax = subplot(G[1, 0])
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylabel(r'$Degree$')
    ax.set_xlabel(r'Fermi level(eV)')
    ax.axis([gate.min(), gate.max(), -200, 200])
    ax.plot(gate, theta, color='#263238', linewidth=1.5, alpha=1, label=r'$\theta$')
    ax.scatter(gate, theta, color='#304ffe')
    # ax.plot(ee, 3.13e-4/np.sqrt(ee-0.2), color='#d32f2f', linewidth=1.5, alpha=1) #  THE jdos of 1d-free-ele-gas

    plt.tight_layout()
    plt.show()


def plot005_SC(fname=r'SC.npz', yy=None):
    '''
      * plot ORMap(wannier) response function
      * in one fig
    '''
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import matplotlib.pylab as pylab

    datas = np.load(fname)
    ee = datas['ee']
    ne = ee.shape[0]
    gate = datas['gate']
    ngate = gate.shape[0]
    tensorIndex = datas['tensorIndex']
    ewide = datas['ewide'] * 1e3
    RS = datas['RS1'] / 1e2
    datas.close()

    RSmax = max_value(RS)
    loc = r'upper right'
    tensordict = {0: 'x', 1: 'y', 2: 'z'}
    tensorstring = '{}{}{}'.format(tensordict[tensorIndex[0]], tensordict[tensorIndex[1]], tensordict[tensorIndex[2]])
    cm = sns.diverging_palette(10, 220, sep=1, n=ngate)

    params = {
        # 'axes.labelsize': '16',
        # 'xtick.labelsize': '16',
        # 'ytick.labelsize': '13',
        # 'lines.linewidth': '2',
        # 'legend.fontsize': '20',
        'figure.figsize': '6, 5'  # set figure size
    }
    pylab.rcParams.update(params)

    G = gridspec.GridSpec(1, 1)

    ax = subplot(G[0, 0])
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    unit = r'\times\mathrm{10}^{2}\left(\mu\mathrm{A}\cdot\mathrm{\mathring{A}}\cdot\mathrm{V}^{-2}\right)'
    ax.set_ylabel(r'$\sigma_{}^{} {}$'.format('{sc}', '{'+tensorstring+'}', '{'+unit+'}'))
    ax.set_xlabel(r'$\omega(eV)$')
    ax.axis([ee.min(), ee.max(), -RSmax, RSmax])
    if yy is not None:
        ax.axis([ee.min(), ee.max(), yy[0], yy[1]])
    for i in range(ngate):
        ax.plot(ee, RS[i], color=cm[i], linewidth=1.5, alpha=1)
    # ax.plot(ee, OR1, color='#263238', linewidth=1.5, alpha=1, label='Shift current $\sigma_{}$ \n ewide={} meV'.format('{'+tensorstring+'}', ewide))
    # ax.plot(ee, 3.13e-4/np.sqrt(ee-0.2), color='#d32f2f', linewidth=1.5, alpha=1) #  THE jdos of 1d-free-ele-gas
    # legend = ax.legend(loc=loc, shadow=True, fontsize=10)
    # legend.get_frame().set_facecolor('#eceff1')

    plt.tight_layout()
    plt.show()


def plot006_CPGE(fname=r'CPGE.npz', yy=None):
    '''
      * plot ORMap(wannier) response function
      * in one fig
    '''
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import matplotlib.pylab as pylab

    datas = np.load(fname)
    ee = datas['ee']
    ne = ee.shape[0]
    gate = datas['gate']
    ngate = gate.shape[0]
    tensorIndex = datas['tensorIndex']
    ewide = datas['ewide'] * 1e3
    RS = datas['RS1'] / 1e2
    datas.close()

    RSmax = max_value(RS)
    loc = r'upper right'
    tensordict = {0: 'x', 1: 'y', 2: 'z'}
    tensorstring = '{}{}{}'.format(tensordict[tensorIndex[0]], tensordict[tensorIndex[1]], tensordict[tensorIndex[2]])
    cm = sns.diverging_palette(10, 220, sep=1, n=ngate)

    params = {
        # 'axes.labelsize': '16',
        # 'xtick.labelsize': '16',
        # 'ytick.labelsize': '13',
        # 'lines.linewidth': '2',
        # 'legend.fontsize': '20',
        'figure.figsize': '6, 5'  # set figure size
    }
    pylab.rcParams.update(params)

    G = gridspec.GridSpec(1, 1)

    ax = subplot(G[0, 0])
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    unit = r'\times\mathrm{10}^{2}\left(\mu\mathrm{A}\cdot\mathrm{\mathring{A}}\cdot\mathrm{V}^{-2}\right)'
    ax.set_ylabel(r'$\sigma_{}^{} {}$'.format('{CPGE}', '{'+tensorstring+'}', '{'+unit+'}'))
    ax.set_xlabel(r'$\omega(eV)$')
    ax.axis([ee.min(), ee.max(), -RSmax, RSmax])
    if yy is not None:
        ax.axis([ee.min(), ee.max(), yy[0], yy[1]])

    for i in range(ngate):
        ax.plot(ee, RS[i], color=cm[i], linewidth=1.5, alpha=1)
    # ax.plot(ee, OR1, color='#263238', linewidth=1.5, alpha=1, label='CPGE $\sigma_{}$ \n ewide={} meV'.format('{'+tensorstring+'}', ewide))
    # ax.plot(ee, 3.13e-4/np.sqrt(ee-0.2), color='#d32f2f', linewidth=1.5, alpha=1) #  THE jdos of 1d-free-ele-gas
    # legend = ax.legend(loc=loc, shadow=True, fontsize=10)
    # legend.get_frame().set_facecolor('#eceff1')

    plt.tight_layout()
    plt.show()


def plot007_band_BC_distri(band, unit='C', eemin=-0.23, eemax=0.23, xlabel=None):

    tbg = band['tbg']
    bandE = band['bandE']
    kpath = band['kpath']
    kpath_list = band['kpath_HSP']
    BC = band['BC'] * 0.01
    cmap = 'seismic'

    nk, nw = bandE.shape
    nline = kpath_list.shape[0] - 1
    if unit.upper() == 'C':
        kpath = multi_dot([tbg.lattG, kpath.T]).T
    kpath = kmold(kpath)

    '''
      * plot band
    '''
    plt.figure(figsize=(8, 6))
    G = gridspec.GridSpec(1, 1)

    ax = subplot(G[0, 0])
    ax.axis([kpath.min(), kpath.max(), eemin, eemax])
    plt.axhline(0, color='k', linewidth=0.5, linestyle="--", alpha=1)
    for i in range(1, nline):
        plt.axvline(x=kpath[i * nk//nline], linestyle='--', color='k', linewidth=0.5, alpha=1, zorder=2)

    ax.plot(np.kron(kpath, np.ones([nw, 1])).T, bandE, linewidth=1, linestyle="-", color='k', alpha=1)
    ax.scatter(np.kron(kpath, np.ones([nw, 1])).T, bandE, c=BC, cmap=cmap, s=np.abs(BC), alpha=1)

    if xlabel is not None:
        # xlabel = ['K', 'G', '-K', '-M', 'G', 'M', 'K']
        num_xlabel = len(xlabel)
        plt.xticks(kpath[np.arange(num_xlabel) * (nk // nline)],
                   xlabel)

    plt.tight_layout()


def plot_wloop(kk2, theta, plot_line=False, ymin=-0.5, ymax=0.5, s=25, save=False, savefname='wloop.png'):
    theta_period = np.hstack([theta - 1, theta, theta + 1])
    nk2, nw = theta.shape

    # matplotlib.rcdefaults()
    # params = {
    #     'axes.labelsize': '12',
    #     'xtick.labelsize': '12',
    #     'ytick.labelsize': '12',
    #     'legend.fontsize': '12',
    #     'xtick.direction': 'in',
    #     'ytick.direction': 'in',
    #     'ytick.minor.visible': True,
    #     'xtick.minor.visible': True,
    #     'xtick.top': True,
    #     'ytick.right': True,
    #     # 'figure.figsize': '5, 4',  # set figure size
    #     # 'figure.dpi': '100',  # set figure size
    #     'pdf.fonttype': '42',  # set figure size
    #     'font.family': 'sans-serif',  # set figure size
    #     # 'font.serif': 'Times',  # set figure size
    #     'font.sans-serif': 'Arial',  # set figure size
    # }
    # pylab.rcParams.update(params)

    fig = plt.figure('wloop', figsize=[4, 3], dpi=150)
    fig.clf()
    ax = fig.add_subplot(111)
    # ax.tick_params()

    ax.axis([kk2.min(), kk2.max(), ymin, ymax])
    # ax.minorticks_on()
    ax.axhline(0, color='k', linewidth=0.5, linestyle="--", zorder=101)
    ax.axhline(0.5, color='k', linewidth=0.5, linestyle="-", zorder=101)
    ax.axhline(-0.5, color='k', linewidth=0.5, linestyle="-", zorder=101)
    ax.axvline(0.5, color='k', linewidth=0.5, linestyle="--", zorder=101)

    for i in range(nw):
        ax.scatter(kk2, theta.T[i], color='red', s=s, zorder=3)

    for i in range(nw*3):
        ax.scatter(kk2, theta_period.T[i], color='#b0bec5', s=s, zorder=2)

    if plot_line:
        for i in range(nw):
            ax.plot(kk2, theta.T[i], color='k')

    # ax.set_xticks(np.linspace(kk2.min(), kk2.max(), 5))
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_yticks(np.linspace(ymin, ymax, 5))
    ax.set_xlabel('$k_c/(2\pi)$')
    ax.set_ylabel('$Eig[W(k_c)]/(2\pi)$')

    fig.tight_layout()
    fig.show()


def plot009_fatband(kpath_norm, EIG, AMN, nline=3, eemin=-0.23, eemax=0.23):
    import pylab

    nk = EIG.nk
    nb = EIG.nb
    norbi = AMN.nw

    eig = EIG.eig # nk, nband
    amn = AMN.amn
    projected_eigenvalues = np.abs(amn) # nk, nband, norbi

    params = {
        # 'axes.labelsize': '16',
        # 'xtick.labelsize': '16',
        # 'ytick.labelsize': '13',
        # 'lines.linewidth': '2',
        # 'legend.fontsize': '20',
        'figure.figsize': '6, 5'  # set figure size
    }
    pylab.rcParams.update(params)

    G = gridspec.GridSpec(1, 1)
    ax = subplot(G[0, 0])
    ax.axis([kpath_norm.min(), kpath_norm.max(), eemin, eemax])
    plt.axhline(0, color='k', linewidth=0.5)
    for i in range(1, nline):
        plt.axvline(x=kpath_norm[i * nk//nline], linestyle='-', color='k', linewidth=0.5, alpha=1, zorder=2)

    for i in range(nb):
        ax.plot(kpath_norm, eig[:, i], linewidth=1, linestyle="-", color='#263238', alpha=1)

    C = [
        'red', 'red', 'red',
        'green', 'green', 'green', 'green',
        'blue','blue','blue',
    ]
    Z = [
        2, 2, 2,
        3, 3, 3, 3,
        1, 1, 1
    ]
    X = np.kron(np.ones([norbi, nb, 1]), kpath_norm).T
    Y = np.kron(np.ones([norbi, 1, 1]), eig.T).T
    S = projected_eigenvalues * 400

    for i in range(norbi):
        # ax.scatter(X[:, :, i], Y[:, :, i], c=C[i], s=S[:, :, i], alpha=0.7)
        ax.scatter(X.T[i], Y.T[i], c=C[i], s=S.T[i], zorder=Z[i], alpha=0.4, label='#{}'.format(i+1))
    ax.legend(loc='upper right')
    plt.tight_layout()


