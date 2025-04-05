__date__ = "Dec. 22, 2017"

import numpy as np
from numpy import linalg as LA

__all__ = [
    'self_energy'
]

def self_energy(hs, h0, h1, h2):
    """
    Fast inverse tri-diagonal (infinite large) matrix of the following form:
    H = [
        [ hs h1            ]
        [ h2 h0 h1         ]
        [    h2 h0 h1      ]
        [       h2 h0 h1   ]
        [          ******* ]
    ]

    Denote the infinite matrix by H, the result will be the (1,1) block of Inverse(H).
    N is the total iteration times.

    The algorithm has been documented in details in
    J. Phys. F: Met. Phys. 14(1984) 1205-1215.
    Quick iterative scheme for the calculation of transfer matrices: application to MO( 100)
    M P Lopez Sancho, J M Lopez Sancho and J Rubio

    Essentially, the idea is to solve the surface Green function of a half-infinite chain.
    As the first step, one can establish the recursive relation among G(1,1), G(1,2), G(1,3) ...
    Secondly, the recursive relation can be established among G(1,1), G(1,2), G(1,4), ...
    Thirdly, among G(1,1), G(1,4), G(1,16) ...
    Analogous to the idea of 'renormalization group', the recursive relation can generalized to
    larger and larger 'cells', with adaptive coupling constants.
    The algorithm is very efficient that n step iteration takes into account of 2^n sites.

    code below is the implementation of the following recursive relation:
    (1) a(n)=a(n-1)*g(n-1)*a(n-1)
    (2) b(n)=b(n-1)*g(n-1)*a(n-1)
    (3) g(n)=Inverse(1-g(n-1)*a(n-1)*g(n-1)*b(n-1)-g(n-1)*b(n-1)*g(n-1)*a(n-1))*g(n-1)
    (4) gs(n)=Inverse(1-gs(n-1)*a(n-1)*g(n-1)*b(n-1))*gs(n-1)
    the intial conditions are:
    (5) a(0)=-h1, b(0)=-h2, g(0)=Inverse(h0), gs(0)=Inverse(hs)
    after N step interation, result=gs(N).
    """
    N = 8000

    # e = np.matlib.identity(h0.shape[0], dtype='complex128')
    e = np.identity(h0.shape[0], dtype='complex128')

    a = -h1
    b = -h2
    g = LA.inv(h0)
    gs = LA.inv(hs)

    for i in range(N):
        agb = a @ g @ b # multi_dot([a, g, b])
        bga = b @ g @ a # multi_dot([b, g, a])
        gs_new = LA.inv(e - gs @ agb) @ gs # inv(e - multi_dot([gs, agb])).dot(gs)

        if np.abs(gs_new - gs).max() < 1e-8:
            # print('iterative procedure reach accuracy at {}th iterations'.format(i))
            break

        gs = gs_new
        b = b @ g @ b # multi_dot([b, g, b]) # or multi_dot([b, g, b]) ?
        a = a @ g @ a # multi_dot([a, g, a])
        g = LA.inv(e - g @ agb - g @ bga) @ g # inv(e - multi_dot([g, agb]) - multi_dot([g, bga])).dot(g)

        # print('iter.',i)

        if i == N-1:
            print('Warning ! iterative procedure not reach accuracy !')
    selfen = h1 @ gs @ h2 # multi_dot([h1, gs, h2])
    return selfen

def fermi_dirac_dis_0T(e, fermi=0.0):
    return (1 - np.sign(e-fermi)) / 2

class GreenWannier(object):
    """
      * GreenWannier Object
    """
    def __init__(self, e, k, hr, fermi=0., c=2, eps=1e-3):
        self.e = e + fermi
        self.k = k
        self.hr = hr
        self.nrpts, self.nw = hr[2].shape[:2]
        self.fermi = fermi
        self.c = c
        self.eps = eps
        self.f = np.array([fermi_dirac_dis_0T(i, self.fermi) for i in self.e])
        self.er = np.array([
            (i + 1j * eps) * np.identity(self.nw)
            for i in self.e
        ])
        self.ea = np.array([
            (i - 1j * eps) * np.identity(self.nw)
            for i in self.e
        ])
        self.h0, self.h1l, self.h1r = self._get_htb()
        self.selfenLr = np.array([
            self_energy(e - self.h0, e - self.h0, self.h1l, self.h1l.H)
            for e in self.er
        ])
        self.selfenRr = np.array([
            self_energy(e - self.h0, e - self.h0, self.h1r, self.h1r.H)
            for e in self.er
        ])

    def _get_htb(self):
        h0 = np.matlib.zeros((self.nw, self.nw), dtype='complex128')
        h1l = np.matlib.zeros((self.nw, self.nw), dtype='complex128')
        h1r = np.matlib.zeros((self.nw, self.nw), dtype='complex128')
        for degen, Rf, hr_mn in zip(self.hr[0], self.hr[1], self.hr[2]):
            if Rf[self.c] == 0:
                h0 += np.exp(2j * np.pi * np.dot(self.k, Rf)) * hr_mn / degen
            elif Rf[self.c] == 1:
                h1r += np.exp(2j * np.pi * np.dot(self.k, Rf)) * hr_mn / degen
            elif Rf[self.c] == -1:
                h1l += np.exp(2j * np.pi * np.dot(self.k, Rf)) * hr_mn / degen
            else:
                continue
        return h0, h1l, h1r

    def get_GLr(self):
        return np.array([
            LA.inv(e - self.h0 - selfenLr)
            for e, selfenLr in zip(self.er, self.selfenLr)
        ])

    def get_GRr(self):
        return np.array([
            LA.inv(e - self.h0 - selfenRr)
            for e, selfenRr in zip(self.er, self.selfenRr)
        ])

    def get_GBr(self):
        return np.array([
            LA.inv(e - self.h0 - selfenLr - selfenRr)
            for e, selfenLr, selfenRr in zip(self.er, self.selfenLr, self.selfenRr)
        ])

    def get_AL(self, GLr):
        return -2 * np.imag(GLr)

    def get_AR(self, GRr):
        return -2 * np.imag(GRr)

    def get_AB(self, GBr):
        return -2 * np.imag(GBr)

    def get_GLlesser(self, GLr):
        AL = self.get_AL(GLr)
        return 1j * np.einsum('e,emn->emn', self.f, AL)

    def get_GRlesser(self, GRr):
        AR = self.get_AL(GRr)
        return 1j * np.einsum('e,emn->emn', self.f, AR)

    def get_GBlesser(self, GBr):
        AB = self.get_AB(GBr)
        return 1j * np.einsum('e,emn->emn', self.f, AB)


if __name__ == '__main__':
    from numpy.linalg import inv
    '''
    H = [
        [ e-1  -0.5               ]
        [ -0.5  e-1  -0.5         ]
        [      -0.5   e-1  -0.5   ]
        [            -0.5   e-1   ]
        [                 ******* ]
    ]
    '''
    eta = 0.01
    e = (1.1 + 1j * eta) * np.identity(2)
    hs = np.array([
        [0.74471898+0.j,  0.04991671+0.j],
        [0.04991671+0.j,  0.84463567+0.j]
    ])
    h0 = hs
    h1 = np.matrix([
        [2.50+0.j, -0.25+0.j],
        [0.25+0.j, -2.50+0.j],
    ])
    h2 = h1.H
    selfen = self_energy(e-hs, e-h0, h1, h2)
    G = LA.inv(e - h0 - selfen)
    dos = -np.trace(G).imag
    print('selfen=', selfen)
    print('G=', G)
    print('dos=', dos)
