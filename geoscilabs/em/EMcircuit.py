from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import numpy as np
from scipy.constants import mu_0, pi, epsilon_0
from scipy.special import erf


def Qfun(R, L, f, alpha=None):
    if alpha is None:
        omega = np.pi * 2 * f
        tau = L / R
        alpha = omega * tau
    Q = (alpha ** 2 + 1j * alpha) / (1 + alpha ** 2)
    return alpha, Q


def Mijfun(x, y, z, incl, decl, x1, y1, z1, incl1, decl1, area=1.0, area0=1.0):
    """
        Compute mutual inductance between two loops

        This

        Parameters
        ----------
        x : array
            x location of the Tx loop
        y : array
            y location of the Tx loop
        z : array
            z location of the Tx loop
        incl:
            XXX
        decl:
            XXX
        x1 : array
            XXX
        y1 : array
            XXX
        z1 : array
            XXX
        incl1:
            XXX
        decl1:
            XXX
    """

    # Pretty sure below assumes dipole
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    x1 = np.array(x1, dtype=float)
    y1 = np.array(y1, dtype=float)
    z1 = np.array(z1, dtype=float)
    incl = np.array(incl, dtype=float)
    decl = np.array(decl, dtype=float)
    incl1 = np.array(incl1, dtype=float)
    decl1 = np.array(decl1, dtype=float)

    di = np.pi * incl / 180.0
    dd = np.pi * decl / 180.0

    cx = np.cos(di) * np.cos(dd)
    cy = np.cos(di) * np.sin(dd)
    cz = np.sin(di)

    ai = np.pi * incl1 / 180.0
    ad = np.pi * decl1 / 180.0

    ax = np.cos(ai) * np.cos(ad)
    ay = np.cos(ai) * np.sin(ad)
    az = np.sin(ai)

    # begin the calculation
    a = x - x1
    b = y - y1
    h = z - z1

    rt = np.sqrt(a ** 2.0 + b ** 2.0 + h ** 2.0) ** 5.0

    txy = 3.0 * a * b / rt
    txz = 3.0 * a * h / rt
    tyz = 3.0 * b * h / rt

    txx = (2.0 * a ** 2.0 - b ** 2.0 - h ** 2.0) / rt
    tyy = (2.0 * b ** 2.0 - a ** 2.0 - h ** 2.0) / rt
    tzz = -(txx + tyy)

    scale = mu_0 * np.pi * area * area0 / 4
    # scale = 1.

    bx = txx * cx + txy * cy + txz * cz
    by = txy * cx + tyy * cy + tyz * cz
    bz = txz * cx + tyz * cy + tzz * cz

    return scale * (bx * ax + by * ay + bz * az)


def Cfun(L, R, xc, yc, zc, incl, decl, S, ht, f, xyz):
    """
        Compute coupling coefficients

        .. math::
            - \frac{M_{12} M_{23}}{M_{13}L_2}

        Parameters
        ----------

    """
    L = np.array(L, dtype=float)
    R = np.array(R, dtype=float)
    xc = np.array(xc, dtype=float)
    yc = np.array(yc, dtype=float)
    zc = np.array(zc, dtype=float)
    incl = np.array(incl, dtype=float)
    decl = np.array(decl, dtype=float)
    S = np.array(S, dtype=float)
    f = np.array(f, dtype=float)

    # This is a bug, hence needs to be fixed later
    x = xyz[:, 1]
    y = xyz[:, 0]
    z = xyz[:, 2]

    # simulate anomalies
    yt = y - S / 2.0
    yr = y + S / 2.0

    dm = -S / 2.0
    dp = S / 2.0

    # Computes mutual inducances
    # Mijfun(x,y,z,incl,decl,x1,y1,z1,incl1,decl1)
    M13 = Mijfun(0.0, dm, 0.0, 90.0, 0.0, 0.0, dp, 0.0, 90.0, 0.0)
    M12 = Mijfun(x, yt, z, 90.0, 0.0, xc, yc, zc, incl, decl, area=1.0, area0=3.0)
    M23 = Mijfun(xc, yc, zc, incl, decl, x, yr, z, 90.0, 0.0, area=3.0, area0=1.0)

    C = -M12 * M23 / (M13 * L)
    return C, M12, M23, M13 * np.ones_like(C)


if __name__ == "__main__":
    out = Mijfun(0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0, 0.0, 0.0, 0.0)
    anal = mu_0 * np.pi / (2 * 10 ** 3)
    err = abs(out - anal)
    print(err)
    showIt = False
    import matplotlib.pyplot as plt

    f = np.logspace(-3, 3, 61)
    alpha, Q = Qfun(1.0, 0.1, f)
    if showIt:
        plt.semilogx(alpha, Q.real)
        plt.semilogx(alpha, Q.imag)
        plt.show()

    L = 1.0
    R = 2000.0
    xc = 0.0
    yc = 0.0
    zc = 2.0
    incl = 0.0
    decl = 90.0
    S = 4.0
    ht = 0.0
    f = 10000.0
    xmin = -10.0
    xmax = 10.0
    dx = 0.25

    xp = np.linspace(xmin, xmax, 101)
    yp = xp.copy()
    zp = np.r_[-ht]
    [Y, X] = np.meshgrid(yp, xp)
    xyz = np.c_[X.flatten(), Y.flatten(), np.ones_like(X.flatten()) * ht]
    C, M12, M23, M13 = Cfun(L, R, xc, yc, zc, incl, decl, S, ht, f, xyz)
    [Xp, Yp] = np.meshgrid(xp, yp)
    if showIt:
        plt.contourf(X, Y, C.reshape(X.shape), 100)
        plt.show()

    # xyz = np.c_[xp, np.zeros_like(yp), np.zeros_like(yp)]
    # C, M12, M23, M13 = Cfun(L,R,xc,yc,zc,incl,decl,S,ht,f,xyz)
    # plt.plot(xp, C, 'k')
    # plt.plot(xp, M12, 'b')
    # plt.plot(xp, M23, 'g')
    # plt.plot(xp, M13, 'r')
    # plt.show()
