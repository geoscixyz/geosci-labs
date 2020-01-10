from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib.path import Path
import matplotlib.patches as patches


##############################################
#   PLOTTING FUNCTIONS FOR WIDGETS
##############################################


def fcn_TDEM_InductionSpherePlaneWidget(
    xtx, ytx, ztx, m, orient, x0, y0, z0, a, sig, mur, xrx, yrx, zrx, logt, Comp, Type
):

    sig = 10 ** sig
    t = 10 ** logt

    if Type == "B":
        Type = "b"
    elif Type == "dB/dt":
        Type = "dbdt"

    tvec = np.logspace(-6, 0, 31)

    xmin, xmax, dx, ymin, ymax, dy = -30.0, 30.0, 0.3, -30.0, 30.0, 0.4
    X, Y = np.mgrid[xmin : xmax + dx : dx, ymin : ymax + dy : dy]
    X = np.transpose(X)
    Y = np.transpose(Y)

    Obj = SphereTEM(m, orient, xtx, ytx, ztx)

    Bx, By, Bz, Babs = Obj.fcn_ComputeTimeResponse(
        t, sig, mur, a, x0, y0, z0, X, Y, zrx, Type
    )
    Bxi, Byi, Bzi, Babsi = Obj.fcn_ComputeTimeResponse(
        tvec, sig, mur, a, x0, y0, z0, xrx, yrx, zrx, Type
    )

    fig1 = plt.figure(figsize=(17, 6))
    Ax1 = fig1.add_axes([0.04, 0, 0.43, 1])
    Ax2 = fig1.add_axes([0.6, 0, 0.4, 1])

    if Comp == "x":
        Ax1 = plotAnomalyXYplane(Ax1, t, X, Y, ztx, Bx, Comp, Type)
        Ax1 = plotPlaceTxRxSphereXY(Ax1, xtx, ytx, xrx, yrx, x0, y0, a)
        Ax2 = plotResponseTEM(Ax2, t, tvec, Bxi, Comp, Type)
    elif Comp == "y":
        Ax1 = plotAnomalyXYplane(Ax1, t, X, Y, ztx, By, Comp, Type)
        Ax1 = plotPlaceTxRxSphereXY(Ax1, xtx, ytx, xrx, yrx, x0, y0, a)
        Ax2 = plotResponseTEM(Ax2, t, tvec, Byi, Comp, Type)
    elif Comp == "z":
        Ax1 = plotAnomalyXYplane(Ax1, t, X, Y, ztx, Bz, Comp, Type)
        Ax1 = plotPlaceTxRxSphereXY(Ax1, xtx, ytx, xrx, yrx, x0, y0, a)
        Ax2 = plotResponseTEM(Ax2, t, tvec, Bzi, Comp, Type)
    elif Comp == "abs":
        Ax1 = plotAnomalyXYplane(Ax1, t, X, Y, ztx, Babs, Comp, Type)
        Ax1 = plotPlaceTxRxSphereXY(Ax1, xtx, ytx, xrx, yrx, x0, y0, a)
        Ax2 = plotResponseTEM(Ax2, t, tvec, Babsi, Comp, Type)

    plt.show(fig1)


def fcn_TDEM_InductionSphereProfileWidget(
    xtx, ztx, m, orient, x0, z0, a, sig, mur, xrx, zrx, logt, Flag
):

    sig = 10 ** sig
    t = 10 ** logt

    if orient == "Vert. Coaxial":
        orient = "x"
    elif orient == "Horiz. Coplanar":
        orient = "z"

    if Flag == "dBs/dt":
        Type = "dbdt"
    else:
        Type = "b"

    # Same global functions can be used but with ytx, y0, yrx, Y = 0.

    tvec = np.logspace(-6, 0, 31)

    xmin, xmax, dx, zmin, zmax, dz = -30.0, 30.0, 0.3, -40.0, 20.0, 0.4
    X, Z = np.mgrid[xmin : xmax + dx : dx, zmin : zmax + dz : dz]
    X = np.transpose(X)
    Z = np.transpose(Z)

    Obj = SphereTEM(m, orient, xtx, 0.0, ztx)

    Bxi, Byi, Bzi, Babsi = Obj.fcn_ComputeTimeResponse(
        tvec, sig, mur, a, x0, 0.0, z0, xrx, 0.0, zrx, Type
    )
    Hxt, Hyt, Hzt = fcn_ComputePrimary(m, orient, xtx, 0.0, ztx, x0, 0.0, z0)

    fig1 = plt.figure(figsize=(17, 6))
    Ax1 = fig1.add_axes([0.04, 0, 0.38, 1])
    Ax2 = fig1.add_axes([0.6, 0, 0.4, 1])

    Ax1 = plotProfileTxRxSphere(Ax1, xtx, ztx, x0, z0, a, xrx, zrx, X, Z, orient)

    if Flag == "Bp":
        Hpx, Hpy, Hpz = fcn_ComputePrimary(m, orient, xtx, 0.0, ztx, X, 0.0, Z)
        Ax1 = plotProfileTxRxArrow(Ax1, x0, z0, Hxt, Hzt, Flag)
        Ax1 = plotProfileXZplane(Ax1, X, Z, Hpx, Hpz, Flag)
    elif Flag == "Bs":
        Bx, By, Bz, Babs = Obj.fcn_ComputeTimeResponse(
            t, sig, mur, a, x0, 0.0, z0, X, 0.0, Z, Type
        )
        Chi = fcn_ComputeExcitation_TEM(t, sig, mur, a)
        Ax1 = plotProfileTxRxArrow(Ax1, x0, z0, Chi * Hxt, Chi * Hzt, Type)
        Ax1 = plotProfileXZplane(Ax1, X, Z, Bx, Bz, Flag)
    elif Flag == "dBs/dt":
        Bx, By, Bz, Babs = Obj.fcn_ComputeTimeResponse(
            t, sig, mur, a, x0, 0.0, z0, X, 0.0, Z, Type
        )
        Chi = fcn_ComputeExcitation_TEM(t, sig, mur, a)
        Ax1 = plotProfileTxRxArrow(Ax1, x0, z0, Chi * Hxt, Chi * Hzt, Type)
        Ax1 = plotProfileXZplane(Ax1, X, Z, Bx, Bz, Flag)

    if (orient == "x") & (Flag == "Bp"):
        Type = "b"
        Ax2 = plotResponseTEM(Ax2, t, tvec, Bxi, orient, Type)
    elif (orient == "z") & (Flag == "Bp"):
        Type = "b"
        Ax2 = plotResponseTEM(Ax2, t, tvec, Bzi, orient, Type)
    elif (orient == "x") & (Flag == "Bs"):
        Type = "b"
        Ax2 = plotResponseTEM(Ax2, t, tvec, Bxi, orient, Type)
    elif (orient == "z") & (Flag == "Bs"):
        Type = "b"
        Ax2 = plotResponseTEM(Ax2, t, tvec, Bzi, orient, Type)
    elif (orient == "x") & (Flag == "dBs/dt"):
        Type = "dbdt"
        Ax2 = plotResponseTEM(Ax2, t, tvec, Bxi, orient, Type)
    elif (orient == "z") & (Flag == "dBs/dt"):
        Type = "dbdt"
        Ax2 = plotResponseTEM(Ax2, t, tvec, Bzi, orient, Type)

    plt.show(fig1)


def fcn_TDEM_InductionSphereProfileEM61Widget(
    xtx, ztx, L, m, orient, x0, z0, a, sig, mur, logt, Flag
):

    xtx = xtx - L / 2
    xrx = xtx + L
    zrx = ztx
    sig = 10 ** sig
    t = 10 ** logt

    if orient == "Vert. Coaxial":
        orient = "x"
    elif orient == "Horiz. Coplanar":
        orient = "z"

    if Flag == "dBs/dt":
        Type = "dbdt"
    else:
        Type = "b"

    # Same global functions can be used but with ytx, y0, yrx, Y = 0.

    tvec = np.logspace(-6, 0, 31)

    xmin, xmax, dx, zmin, zmax, dz = -30.0, 30.0, 0.3, -40.0, 20.0, 0.4
    X, Z = np.mgrid[xmin : xmax + dx : dx, zmin : zmax + dz : dz]
    X = np.transpose(X)
    Z = np.transpose(Z)

    Obj = SphereTEM(m, orient, xtx, 0.0, ztx)

    Bxi, Byi, Bzi, Babsi = Obj.fcn_ComputeTimeResponse(
        tvec, sig, mur, a, x0, 0.0, z0, xrx, 0.0, zrx, Type
    )
    Hxt, Hyt, Hzt = fcn_ComputePrimary(m, orient, xtx, 0.0, ztx, x0, 0.0, z0)

    fig1 = plt.figure(figsize=(17, 6))
    Ax1 = fig1.add_axes([0.04, 0, 0.38, 1])
    Ax2 = fig1.add_axes([0.6, 0, 0.4, 1])

    Ax1 = plotProfileTxRxSphere(Ax1, xtx, ztx, x0, z0, a, xrx, zrx, X, Z, orient)

    if Flag == "Bp":
        Hpx, Hpy, Hpz = fcn_ComputePrimary(m, orient, xtx, 0.0, ztx, X, 0.0, Z)
        Ax1 = plotProfileTxRxArrow(Ax1, x0, z0, Hxt, Hzt, Flag)
        Ax1 = plotProfileXZplane(Ax1, X, Z, Hpx, Hpz, Flag)
    elif Flag == "Bs":
        Bx, By, Bz, Babs = Obj.fcn_ComputeTimeResponse(
            t, sig, mur, a, x0, 0.0, z0, X, 0.0, Z, "b"
        )
        Chi = fcn_ComputeExcitation_TEM(t, sig, mur, a, Type)
        Ax1 = plotProfileTxRxArrow(Ax1, x0, z0, Chi * Hxt, Chi * Hzt, Flag)
        Ax1 = plotProfileXZplane(Ax1, X, Z, Bx, Bz, Flag)
    elif Flag == "dBs/dt":
        Bx, By, Bz, Babs = Obj.fcn_ComputeTimeResponse(
            t, sig, mur, a, x0, 0.0, z0, X, 0.0, Z, "dbdt"
        )
        Chi = fcn_ComputeExcitation_TEM(t, sig, mur, a, Type)
        Ax1 = plotProfileTxRxArrow(Ax1, x0, z0, Chi * Hxt, Chi * Hzt, Flag)
        Ax1 = plotProfileXZplane(Ax1, X, Z, Bx, Bz, Flag)

    if (orient == "x") & (Flag == "Bp"):
        Type = "b"
        Ax2 = plotResponseTEM(Ax2, t, tvec, Bxi, orient, Type)
    elif (orient == "z") & (Flag == "Bp"):
        Type = "b"
        Ax2 = plotResponseTEM(Ax2, t, tvec, Bzi, orient, Type)
    elif (orient == "x") & (Flag == "Bs"):
        Type = "b"
        Ax2 = plotResponseTEM(Ax2, t, tvec, Bxi, orient, Type)
    elif (orient == "z") & (Flag == "Bs"):
        Type = "b"
        Ax2 = plotResponseTEM(Ax2, t, tvec, Bzi, orient, Type)
    elif (orient == "x") & (Flag == "dBs/dt"):
        Type = "dbdt"
        Ax2 = plotResponseTEM(Ax2, t, tvec, Bxi, orient, Type)
    elif (orient == "z") & (Flag == "dBs/dt"):
        Type = "dbdt"
        Ax2 = plotResponseTEM(Ax2, t, tvec, Bzi, orient, Type)

    plt.show(fig1)


##############################################
#   GLOBAL FUNTIONS
##############################################


def fcn_ComputeExcitation_TEM(t, sig, mur, a, Type):
    """Compute Excitation Factor (TEM)"""

    beta = np.sqrt(4 * np.pi * 1e-7 * sig) * a
    N = 2000
    nvec = np.linspace(1, N, N)

    if mur < 1.01:

        chi = np.zeros(np.size(t))

        if Type == "b":

            if np.size(t) == 1:
                SUM_1 = np.sum(np.exp(-((nvec * beta) ** 2) / t))
                SUM_2 = np.sum(nvec * sp.special.erfc(nvec * beta / np.sqrt(t)))
                chi = (9 / 2) * (
                    1 / 3
                    + t / beta ** 2
                    - (2 / beta) * np.sqrt(t / np.pi) * (1 + 2 * SUM_1)
                    + 4 * SUM_2
                )

            else:
                for tt in range(0, np.size(t)):
                    SUM_1 = np.sum(np.exp(-((nvec * beta) ** 2) / t[tt]))
                    SUM_2 = np.sum(nvec * sp.special.erfc(nvec * beta / np.sqrt(t[tt])))
                    chi[tt] = (9 / 2) * (
                        1 / 3
                        + t[tt] / beta ** 2
                        - (2 / beta) * np.sqrt(t[tt] / np.pi) * (1 + 2 * SUM_1)
                        + 4 * SUM_2
                    )

        elif Type == "dbdt":

            if np.size(t) == 1:
                SUM = np.sum(np.exp(-((nvec * beta) ** 2) / t))
                chi = (9 / 2) * (
                    1 / beta ** 2 - (1 / (beta * np.sqrt(np.pi * t))) * (1 + 2 * SUM)
                )

            else:
                for tt in range(0, np.size(t)):
                    SUM = np.sum(np.exp(-((nvec * beta) ** 2) / t[tt]))
                    chi[tt] = (9 / 2) * (
                        1 / beta ** 2
                        - (1 / (beta * np.sqrt(np.pi * t[tt]))) * (1 + 2 * SUM)
                    )

    else:

        N = 2000  # Coefficients

        eta = np.pi * (np.linspace(1, N, N) + 1 / 4)
        eta0 = np.pi * np.linspace(1, N, N)

        # Converge eta coefficients
        for pp in range(0, 10):
            eta = eta0 + np.arctan((mur - 1) * eta / (mur - 1 + eta ** 2))

        chi = np.zeros(np.size(t))

        # Get Excitation Factor
        if Type == "b":

            if np.size(t) == 1:
                chi = (9 * mur) * np.sum(
                    np.exp(-t * (eta / beta) ** 2) / ((mur + 2) * (mur - 1) + eta ** 2)
                )

            else:
                for tt in range(0, np.size(t)):
                    chi[tt] = (9 * mur) * np.sum(
                        np.exp(-t[tt] * (eta / beta) ** 2)
                        / ((mur + 2) * (mur - 1) + eta ** 2)
                    )

        elif Type == "dbdt":

            if np.size(t) == 1:
                chi = -(9 * mur) * np.sum(
                    eta ** 2
                    * np.exp(-t * (eta / beta) ** 2)
                    / (beta ** 2 * ((mur + 2) * (mur - 1) + eta ** 2))
                )

            else:
                for tt in range(0, np.size(t)):
                    chi[tt] = -(9 * mur) * np.sum(
                        eta ** 2
                        * np.exp(-t[tt] * (eta / beta) ** 2)
                        / (beta ** 2 * ((mur + 2) * (mur - 1) + eta ** 2))
                    )

    return chi


def fcn_ComputePrimary(m, orient, xtx, ytx, ztx, X, Y, Z):
    """Computes Inducing Field at Sphere"""

    R = np.sqrt((X - xtx) ** 2 + (Y - ytx) ** 2 + (Z - ztx) ** 2)

    if orient == "x":
        Hpx = (1 / (4 * np.pi)) * (3 * m * (X - xtx) * (X - xtx) / R ** 5 - m / R ** 3)
        Hpy = (1 / (4 * np.pi)) * (3 * m * (Y - ytx) * (X - xtx) / R ** 5)
        Hpz = (1 / (4 * np.pi)) * (3 * m * (Z - ztx) * (X - xtx) / R ** 5)
    elif orient == "y":
        Hpx = (1 / (4 * np.pi)) * (3 * m * (X - xtx) * (Y - ytx) / R ** 5)
        Hpy = (1 / (4 * np.pi)) * (3 * m * (Y - ytx) * (Y - ytx) / R ** 5 - m / R ** 3)
        Hpz = (1 / (4 * np.pi)) * (3 * m * (Z - ztx) * (Y - ytx) / R ** 5)
    elif orient == "z":
        Hpx = (1 / (4 * np.pi)) * (3 * m * (X - xtx) * (Z - ztx) / R ** 5)
        Hpy = (1 / (4 * np.pi)) * (3 * m * (Y - ytx) * (Z - ztx) / R ** 5)
        Hpz = (1 / (4 * np.pi)) * (3 * m * (Z - ztx) * (Z - ztx) / R ** 5 - m / R ** 3)

    return Hpx, Hpy, Hpz


##############################################
#   GLOBAL PLOTTING FUNTIONS
##############################################


def plotAnomalyXYplane(Ax, t, X, Y, Z, B, Comp, Type):

    FS = 20

    tol = 1e5

    Sign = np.sign(B)
    B = 1e9 * np.abs(B)  # convert to nT or nT/s
    MAX = np.max(B)

    B = np.log10(tol * B / MAX)

    Sign[B < 0] = 0.0
    B[B < 0] = 0.0

    Cmap = "RdYlBu"
    # Cmap = 'seismic_r'

    if Comp == "abs":
        TickLabels = MAX * np.array(
            [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 0.0, -1e-4, -1e-3, -1e-2, -1e-1, -1]
        )
        TickLabels = ["%.1e" % x for x in TickLabels]
        Cplot = Ax.contourf(X, Y, Sign * B, 50, cmap=Cmap, vmin=-5, vmax=5)
        cbar = plt.colorbar(Cplot, ax=Ax, pad=0.02, ticks=-np.linspace(-5, 5, 11))
    else:
        TickLabels = MAX * np.array(
            [-1.0, -1e-1, -1e-2, -1e-3, -1e-4, 0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        )
        TickLabels = ["%.1e" % x for x in TickLabels]
        Cplot = Ax.contourf(X, Y, Sign * B, 50, cmap=Cmap, vmin=-5, vmax=5)
        cbar = plt.colorbar(Cplot, ax=Ax, pad=0.02, ticks=np.linspace(-5, 5, 11))

    if Comp == "x" and Type == "b":
        cbar.set_label("[nT]", rotation=270, labelpad=25, size=FS + 4)
        Ax.set_title("$\mathbf{Bx}$", fontsize=FS + 6)
    elif Comp == "y" and Type == "b":
        cbar.set_label("[nT]", rotation=270, labelpad=25, size=FS + 4)
        Ax.set_title("$\mathbf{By}$", fontsize=FS + 6)
    elif Comp == "z" and Type == "b":
        cbar.set_label("[nT]", rotation=270, labelpad=25, size=FS + 4)
        Ax.set_title("$\mathbf{Bz}$", fontsize=FS + 6)
    elif Comp == "x" and Type == "dbdt":
        cbar.set_label("[nT/s]", rotation=270, labelpad=25, size=FS + 4)
        Ax.set_title("$\mathbf{dBx/dt}$", fontsize=FS + 6)
    elif Comp == "y" and Type == "dbdt":
        cbar.set_label("[nT/s]", rotation=270, labelpad=25, size=FS + 4)
        Ax.set_title("$\mathbf{dBy/dt}$", fontsize=FS + 6)
    elif Comp == "z" and Type == "dbdt":
        cbar.set_label("[nT/s]", rotation=270, labelpad=25, size=FS + 4)
        Ax.set_title("$\mathbf{dBz/dt}$", fontsize=FS + 6)

    cbar.set_ticklabels(TickLabels)
    cbar.ax.tick_params(labelsize=FS - 2)

    Ax.set_xbound(np.min(X), np.max(X))
    Ax.set_ybound(np.min(Y), np.max(Y))
    Ax.set_xlabel("X [m]", fontsize=FS + 2)
    Ax.set_ylabel("Y [m]", fontsize=FS + 2, labelpad=-10)
    Ax.tick_params(labelsize=FS - 2)

    return Ax


def plotPlaceTxRxSphereXY(Ax, xtx, ytx, xrx, yrx, x0, y0, a):

    Xlim = Ax.get_xlim()
    Ylim = Ax.get_ylim()

    FS = 20

    Ax.scatter(xtx, ytx, s=100, color="k")
    Ax.text(xtx - 0.75, ytx + 1.5, "$\mathbf{Tx}$", fontsize=FS + 6)
    Ax.scatter(xrx, yrx, s=100, color="k")
    Ax.text(xrx - 0.75, yrx - 4, "$\mathbf{Rx}$", fontsize=FS + 6)

    xs = x0 + a * np.cos(np.linspace(0, 2 * np.pi, 41))
    ys = y0 + a * np.sin(np.linspace(0, 2 * np.pi, 41))

    Ax.plot(xs, ys, ls=":", color="k", linewidth=3)

    Ax.set_xbound(Xlim)
    Ax.set_ybound(Ylim)

    return Ax


def plotResponseTEM(Ax, ti, t, B, Comp, Type):

    FS = 20

    B = 1e9 * np.abs(B)  # turn to nT or nT/s and python can`t loglog negative values!

    if Type == "b":
        Ylim = np.array([B[0] / 1e3, B[0]])
    elif Type == "dbdt":
        Ylim = np.array([B[0] / 1e6, B[0]])

    B[B < Ylim[0]] = 0.1 * Ylim[0]

    Ax.grid("both", linestyle="-", linewidth=0.8, color=[0.8, 0.8, 0.8])
    Ax.loglog(t, 0 * t, color="k", linewidth=2)
    Ax.loglog(t, B, color="k", linewidth=4)
    Ax.loglog(np.array([ti, ti]), 1.1 * Ylim, linewidth=3, color="r")
    Ax.set_xbound(np.min(t), np.max(t))
    Ax.set_ybound(1.1 * Ylim)
    Ax.set_xlabel("Times [s]", fontsize=FS + 2)
    Ax.tick_params(labelsize=FS - 2)
    Ax.yaxis.set_major_formatter(FormatStrFormatter("%.1e"))

    if Comp == "x" and Type == "b":
        Ax.set_ylabel("$\mathbf{|Bx|}$ [nT]", fontsize=FS + 4, labelpad=-5)
        Ax.set_title("$\mathbf{Bx}$ Response at $\mathbf{Rx}$", fontsize=FS + 6)
    elif Comp == "z" and Type == "b":
        Ax.set_ylabel("$\mathbf{|Bz|}$ [nT]", fontsize=FS + 4, labelpad=-5)
        Ax.set_title("$\mathbf{Bz}$ Response at $\mathbf{Rx}$", fontsize=FS + 6)
    elif Comp == "x" and Type == "dbdt":
        Ax.set_ylabel("$\mathbf{|dBx/dt|}$ [nT/s]", fontsize=FS + 4, labelpad=-5)
        Ax.set_title("$\mathbf{dBx/dt}$ Response at $\mathbf{Rx}$", fontsize=FS + 6)
    elif Comp == "z" and Type == "dbdt":
        Ax.set_ylabel("$\mathbf{|dBz/dt|}$ [nT/s]", fontsize=FS + 4, labelpad=-5)
        Ax.set_title("$\mathbf{dBz/dt}$ Response at $\mathbf{Rx}$", fontsize=FS + 6)

    return Ax


def plotProfileTxRxSphere(Ax, xtx, ztx, x0, z0, a, xrx, zrx, X, Z, orient):

    FS = 22

    phi = np.linspace(0, 2 * np.pi, 41)
    # psi = np.linspace(0, np.pi, 21)

    if orient == "x":
        Xtx = xtx + 0.5 * np.cos(phi)
        Ztx = ztx + 2 * np.sin(phi)
        Xrx = xrx + 0.5 * np.cos(phi)
        Zrx = zrx + 2 * np.sin(phi)
    elif orient == "z":
        Xtx = xtx + 2 * np.cos(phi)
        Ztx = ztx + 0.5 * np.sin(phi)
        Xrx = xrx + 2 * np.cos(phi)
        Zrx = zrx + 0.5 * np.sin(phi)

    # Xs = x0 + a*np.cos(psi)
    # Zs1 = z0 + a*np.sin(psi)
    # Zs2 = z0 - a*np.sin(psi)

    XS = x0 + a * np.cos(phi)
    ZS = z0 + a * np.sin(phi)

    Ax.fill_between(
        np.array([np.min(X), np.max(X)]),
        np.array([0.0, 0.0]),
        np.array([np.max(Z), np.max(Z)]),
        facecolor=(0.9, 0.9, 0.9),
    )
    Ax.fill_between(
        np.array([np.min(X), np.max(X)]),
        np.array([0.0, 0.0]),
        np.array([np.min(Z), np.min(Z)]),
        facecolor=(0.6, 0.6, 0.6),
        linewidth=2,
    )
    # Ax.fill_between(Xs,Zs1,Zs2,facecolor=(0.4,0.4,0.4),linewidth=4)

    polyObj = plt.Polygon(
        np.c_[XS, ZS],
        closed=True,
        facecolor=((0.4, 0.4, 0.4)),
        edgecolor="k",
        linewidth=2,
    )
    Ax.add_patch(polyObj)

    Ax.plot(Xtx, Ztx, "k", linewidth=4)
    Ax.plot(Xrx, Zrx, "k", linewidth=4)
    # Ax.plot(x0+a*np.cos(phi),z0+a*np.sin(phi),'k',linewidth=2)

    Ax.set_xbound(np.min(X), np.max(X))
    Ax.set_ybound(np.min(Z), np.max(Z))

    Ax.text(xtx - 4, ztx + 2, "$\mathbf{Tx}$", fontsize=FS)
    Ax.text(xrx, zrx + 2, "$\mathbf{Rx}$", fontsize=FS)

    return Ax


def plotProfileXZplane(Ax, X, Z, Bx, Bz, Flag):

    FS = 20

    if Flag == "Bp":
        Ax.streamplot(X, Z, Bx, Bz, color="b", linewidth=3.5, arrowsize=2)
        Ax.set_title("Primary Field", fontsize=FS + 6)
    elif Flag == "Bs":
        Ax.streamplot(X, Z, Bx, Bz, color="r", linewidth=3.5, arrowsize=2)
        Ax.set_title("Secondary Field", fontsize=FS + 6)
    elif Flag == "dBs/dt":
        Ax.streamplot(X, Z, Bx, Bz, color="r", linewidth=3.5, arrowsize=2)
        Ax.set_title("Secondary Time Derivative", fontsize=FS + 6)

    Ax.set_xbound(np.min(X), np.max(X))
    Ax.set_ybound(np.min(Z), np.max(Z))
    Ax.set_xlabel("X [m]", fontsize=FS + 2)
    Ax.set_ylabel("Z [m]", fontsize=FS + 2, labelpad=-10)
    Ax.tick_params(labelsize=FS - 2)


def plotProfileTxRxArrow(Ax, x0, z0, Bxt, Bzt, Flag):

    Babst = np.sqrt(Bxt ** 2 + Bzt ** 2)
    dx = Bxt / Babst
    dz = Bzt / Babst

    if Flag == "Bp":
        Ax.arrow(
            x0 - 2.5 * dx,
            z0 - 2.75 * dz,
            3 * dx,
            3 * dz,
            fc=(0.0, 0.0, 0.8),
            ec="k",
            head_width=2.5,
            head_length=2.5,
            width=1,
            linewidth=2,
        )
    elif Flag == "Bs":
        Ax.arrow(
            x0 - 2.5 * dx,
            z0 - 2.75 * dz,
            3 * dx,
            3 * dz,
            fc=(0.8, 0.0, 0.0),
            ec="k",
            head_width=2.5,
            head_length=2.5,
            width=1,
            linewidth=2,
        )
    elif Flag == "dBs/dt":
        Ax.arrow(
            x0 - 2.5 * dx,
            z0 - 2.75 * dz,
            3 * dx,
            3 * dz,
            fc=(0.8, 0.0, 0.0),
            ec="k",
            head_width=2.5,
            head_length=2.5,
            width=1,
            linewidth=2,
        )

    return Ax


############################################
#   CLASS: SPHERE TOP VIEW
############################################

############################################
#   DEFINE CLASS


class SphereTEM:
    """Fucntionwhcihdf
    Input variables:

        Output variables:
    """

    def __init__(self, m, orient, xtx, ytx, ztx):
        """Defines Initial Attributes"""

        # INITIALIZES OBJECT

        # m: Transmitter dipole moment
        # orient: Transmitter dipole orentation 'x', 'y' or 'z'
        # xtx: Transmitter x location
        # ytx: Transmitter y location
        # ztx: Transmitter z location

        self.m = m
        self.orient = orient
        self.xtx = xtx
        self.ytx = ytx
        self.ztx = ztx

    ############################################
    #   DEFINE METHODS

    def fcn_ComputeTimeResponse(self, t, sig, mur, a, x0, y0, z0, X, Y, Z, Type):
        """Compute Single Frequency Response at (X,Y,Z) in T or T/s"""

        m = self.m
        orient = self.orient
        xtx = self.xtx
        ytx = self.ytx
        ztx = self.ztx

        chi = fcn_ComputeExcitation_TEM(t, sig, mur, a, Type)
        Hpx, Hpy, Hpz = fcn_ComputePrimary(m, orient, xtx, ytx, ztx, x0, y0, z0)

        mx = 4 * np.pi * a ** 3 * chi * Hpx / 3
        my = 4 * np.pi * a ** 3 * chi * Hpy / 3
        mz = 4 * np.pi * a ** 3 * chi * Hpz / 3
        R = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2)

        Bx = (1e-9) * (
            3 * (X - x0) * (mx * (X - x0) + my * (Y - y0) + mz * (Z - z0)) / R ** 5
            - mx / R ** 3
        )
        By = (1e-9) * (
            3 * (Y - y0) * (mx * (X - x0) + my * (Y - y0) + mz * (Z - z0)) / R ** 5
            - my / R ** 3
        )
        Bz = (1e-9) * (
            3 * (Z - z0) * (mx * (X - x0) + my * (Y - y0) + mz * (Z - z0)) / R ** 5
            - mz / R ** 3
        )
        Babs = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

        return Bx, By, Bz, Babs
