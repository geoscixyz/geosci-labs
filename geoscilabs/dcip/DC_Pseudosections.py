from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator

from ipywidgets import (
    interactive,
    IntSlider,
    FloatSlider,
    FloatText,
    ToggleButtons,
    VBox,
)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
from matplotlib.path import Path
import matplotlib.patches as patches

from discretize import TensorMesh
from pymatsolver import Pardiso

from SimPEG import maps, SolverLU, utils
from SimPEG.electromagnetics.static import resistivity as DC
from SimPEG.maps import IdentityMap
from SimPEG.electromagnetics.static.utils import static_utils

from ..base import widgetify


class ParametricCircleLayerMap(IdentityMap):

    slope = 1e-1

    def __init__(self, mesh, logSigma=True):
        assert mesh.dim == 2, (
            "Working for a 2D mesh only right now. "
            "But it isn't that hard to change.. :)"
        )
        IdentityMap.__init__(self, mesh)
        # TODO: this should be done through a composition with and ExpMap
        self.logSigma = logSigma

    @property
    def nP(self):
        return 7

    def _transform(self, m):
        # a = self.slope
        sig1, sig2, sig3, x, zc, r, zh = m[0], m[1], m[2], m[3], m[4], m[5], m[6]
        if self.logSigma:
            sig1, sig2, sig3 = np.exp(sig1), np.exp(sig2), np.exp(sig3)
        sigma = np.ones(mesh.nC) * sig1
        sigma[mesh.gridCC[:, 1] < zh] = sig2
        blkind = utils.ModelBuilder.getIndicesSphere(np.r_[x, zc], r, mesh.gridCC)
        sigma[blkind] = sig3
        return sigma


# Mesh, mapping can be globals
npad = 15
cs = 1.25
hx = [(cs, npad, -1.3), (cs, 100), (cs, npad, 1.3)]
hy = [(cs, npad, -1.3), (cs, 50)]
mesh = TensorMesh([hx, hy], "CN")
circmap = ParametricCircleLayerMap(mesh)
circmap.slope = 1e5
mapping = circmap
dx = 5
xr = np.arange(-40, 41, dx)
dxr = np.diff(xr)
xmin = -40.0
xmax = 40.0
ymin = -40.0
ymax = 5.0
xylim = np.c_[[xmin, ymin], [xmax, ymax]]
indCC, meshcore = utils.ExtractCoreMesh(xylim, mesh)
indx = (
    (mesh.gridFx[:, 0] >= xmin)
    & (mesh.gridFx[:, 0] <= xmax)
    & (mesh.gridFx[:, 1] >= ymin)
    & (mesh.gridFx[:, 1] <= ymax)
)
indy = (
    (mesh.gridFy[:, 0] >= xmin)
    & (mesh.gridFy[:, 0] <= xmax)
    & (mesh.gridFy[:, 1] >= ymin)
    & (mesh.gridFy[:, 1] <= ymax)
)
indF = np.concatenate((indx, indy))


def DC2Dsurvey(flag="PolePole"):
    """
    Function that define a surface DC survey
    :param str flag: Survey Type 'PoleDipole', 'DipoleDipole', 'DipolePole', 'PolePole'
    """
    if flag == "PoleDipole":
        ntx, nmax = xr.size - 2, 8
    elif flag == "DipolePole":
        ntx, nmax = xr.size - 2, 8
    elif flag == "DipoleDipole":
        ntx, nmax = xr.size - 3, 8
    elif flag == "PolePole":
        ntx, nmax = xr.size - 2, 8
    else:
        raise Exception("Not Implemented")
    xzlocs = getPseudoLocs(xr, ntx, nmax, flag)

    txList = []
    zloc = -2.5
    for i in range(ntx):
        if flag == "PoleDipole":
            A = np.r_[xr[i], zloc]
            B = np.r_[mesh.vectorCCx.min(), zloc]
            if i < ntx - nmax + 1:
                M = np.c_[xr[i + 1 : i + 1 + nmax], np.ones(nmax) * zloc]
                N = np.c_[xr[i + 2 : i + 2 + nmax], np.ones(nmax) * zloc]
            else:
                M = np.c_[xr[i + 1 : ntx + 1], np.ones(ntx - i) * zloc]
                N = np.c_[xr[i + 2 : i + 2 + nmax], np.ones(ntx - i) * zloc]
        elif flag == "DipolePole":
            A = np.r_[xr[i], zloc]
            B = np.r_[xr[i + 1], zloc]
            if i < ntx - nmax + 1:
                M = np.c_[xr[i + 2 : i + 2 + nmax], np.ones(nmax) * zloc]
                N = np.c_[np.ones(nmax) * mesh.vectorCCx.max(), np.ones(nmax) * zloc]
            else:
                M = np.c_[xr[i + 2 : ntx + 2], np.ones(ntx - i) * zloc]
                N = np.c_[
                    np.ones(ntx - i) * mesh.vectorCCx.max(), np.ones(ntx - i) * zloc
                ]
        elif flag == "DipoleDipole":
            A = np.r_[xr[i], zloc]
            B = np.r_[xr[i + 1], zloc]
            if i < ntx - nmax:
                M = np.c_[
                    xr[i + 2 : i + 2 + nmax],
                    np.ones(len(xr[i + 2 : i + 2 + nmax])) * zloc,
                ]
                N = np.c_[
                    xr[i + 3 : i + 3 + nmax],
                    np.ones(len(xr[i + 3 : i + 3 + nmax])) * zloc,
                ]
            else:
                M = np.c_[
                    xr[i + 2 : len(xr) - 1],
                    np.ones(len(xr[i + 2 : len(xr) - 1])) * zloc,
                ]
                N = np.c_[xr[i + 3 : len(xr)], np.ones(len(xr[i + 3 : len(xr)])) * zloc]
        elif flag == "PolePole":
            A = np.r_[xr[i], zloc]
            B = np.r_[mesh.vectorCCx.min(), zloc]

            if i < ntx - nmax + 1:
                M = np.c_[xr[i + 2 : i + 2 + nmax], np.ones(nmax) * zloc]
                N = np.c_[np.ones(nmax) * mesh.vectorCCx.max(), np.ones(nmax) * zloc]
            else:
                M = np.c_[xr[i + 2 : ntx + 2], np.ones(ntx - i) * zloc]
                N = np.c_[
                    np.ones(ntx - i) * mesh.vectorCCx.max(), np.ones(ntx - i) * zloc
                ]

        rx = DC.receivers.Dipole(M, N)
        src = DC.sources.Dipole([rx], A, B)
        txList.append(src)

    survey = DC.Survey(txList)
    simulation = DC.Simulation2DCellCentered(
        mesh, survey=survey, sigmaMap=mapping, solver=Pardiso
    )

    sigblk, sighalf, siglayer = 2e-2, 2e-3, 1e-3
    xc, yc, r, zh = -15, -8, 4, -5
    mtrue = np.r_[np.log(sighalf), np.log(siglayer), np.log(sigblk), xc, yc, r, zh]
    dtrue = simulation.dpred(mtrue)
    perc = 0.0001
    floor = np.linalg.norm(dtrue) * 1e-8
    np.random.seed([1])
    uncert = np.random.randn(survey.nD) * perc + floor
    dobs = dtrue + uncert

    return dobs, uncert, simulation, xzlocs


def getPseudoLocs(xr, ntx, nmax, flag="PoleDipole"):
    """
    Compute the midpoint pseudolocation
    for each Transmitter-Receiver pair of a survey

    :param numpy.array xr: electrodes positions
    :param int ntx: number of transmitter
    :param int nmax: max number of receiver per source
    :param str flag: Survey Type 'PoleDipole', 'DipoleDipole', 'DipolePole'
    """
    xloc = []
    yloc = []
    for i in range(ntx):
        if i < ntx - nmax + 1:

            if flag == "DipoleDipole":
                txmid = xr[i] + dxr[i] * 0.5
                rxmid = xr[i + 1 : i + 1 + nmax] + dxr[i + 1 : i + 1 + nmax] * 0.5

            elif flag == "PoleDipole":
                txmid = xr[i]
                rxmid = xr[i + 1 : i + 1 + nmax] + dxr[i + 1 : i + 1 + nmax] * 0.5

            elif flag == "DipolePole":
                txmid = xr[i] + dxr[i] * 0.5
                rxmid = xr[i + 1 : i + 1 + nmax]

            elif flag == "PolePole":
                txmid = xr[i]
                rxmid = xr[i + 1 : i + 1 + nmax]

            mid = (txmid + rxmid) * 0.5
            xloc.append(mid)
            yloc.append(np.arange(nmax) + 1.0)
        else:
            if flag == "DipoleDipole":
                txmid = xr[i] + dxr[i] * 0.5
                rxmid = xr[i + 1 : ntx + 1] + dxr[i + 1 : ntx + 1] * 0.5

            elif flag == "PoleDipole":
                txmid = xr[i]
                rxmid = xr[i + 1 : ntx + 1] + dxr[i + 1 : ntx + 1] * 0.5

            elif flag == "DipolePole":
                txmid = xr[i] + dxr[i] * 0.5
                rxmid = xr[i + 1 : ntx + 1]

            elif flag == "PolePole":
                txmid = xr[i]
                rxmid = xr[i + 1 : ntx + 1]

            mid = (txmid + rxmid) * 0.5
            xloc.append(mid)
            yloc.append(np.arange(mid.size) + 1.0)
    xlocvec = np.hstack(xloc)
    ylocvec = np.hstack(yloc)
    return np.c_[xlocvec, ylocvec]


def PseudoSectionPlotfnc(i, j, survey, flag="PoleDipole"):
    """
    Plot the Pseudolocation associated with source i and receiver j

    :param int i: source index
    :param int j: receiver index
    :param SimPEG.survey survey: SimPEG survey object
    :param str flag: Survey Type 'PoleDipole', 'DipoleDipole', 'DipolePole'
    """
    matplotlib.rcParams["font.size"] = 14
    nmax = 8
    dx = 5
    xr = np.arange(-40, 41, dx)
    ntx = xr.size - 2
    # dxr = np.diff(xr)
    TxObj = survey.source_list
    TxLoc = TxObj[i].loc
    RxLoc = TxObj[i].receiver_list[0].locs
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(
        111, autoscale_on=False, xlim=(xr.min() - 5, xr.max() + 5), ylim=(nmax + 1, -2)
    )
    plt.plot(xr, np.zeros_like(xr), "ko", markersize=4)
    if flag == "PoleDipole":
        plt.plot(TxLoc[0][0], np.zeros(1), "rv", markersize=10)
        ax.annotate(
            "A",
            xy=(TxLoc[0][0], np.zeros(1)),
            xycoords="data",
            xytext=(-4.25, 7.5),
            textcoords="offset points",
        )
    else:
        plt.plot([TxLoc[0][0], TxLoc[1][0]], np.zeros(2), "rv", markersize=10)
        ax.annotate(
            "A",
            xy=(TxLoc[0][0], np.zeros(1)),
            xycoords="data",
            xytext=(-4.25, 7.5),
            textcoords="offset points",
        )
        ax.annotate(
            "B",
            xy=(TxLoc[1][0], np.zeros(1)),
            xycoords="data",
            xytext=(-4.25, 7.5),
            textcoords="offset points",
        )

    if i < ntx - nmax + 1:

        if flag in ["PoleDipole", "PolePole"]:
            txmid = TxLoc[0][0]
        else:
            txmid = (TxLoc[0][0] + TxLoc[1][0]) * 0.5

        MLoc = RxLoc[0][j]
        NLoc = RxLoc[1][j]

        if flag in ["DipolePole", "PolePole"]:
            plt.plot(MLoc[0], np.zeros(1), "bv", markersize=10)
            ax.annotate(
                "M",
                xy=(MLoc[0], np.zeros(1)),
                xycoords="data",
                xytext=(-4.25, 7.5),
                textcoords="offset points",
            )
            rxmid = MLoc[0]
        else:
            rxmid = (MLoc[0] + NLoc[0]) * 0.5
            plt.plot(MLoc[0], np.zeros(1), "bv", markersize=10)
            plt.plot(NLoc[0], np.zeros(1), "b^", markersize=10)
            ax.annotate(
                "M",
                xy=(MLoc[0], np.zeros(1)),
                xycoords="data",
                xytext=(-4.25, 7.5),
                textcoords="offset points",
            )
            ax.annotate(
                "N",
                xy=(NLoc[0], np.zeros(1)),
                xycoords="data",
                xytext=(-4.25, 7.5),
                textcoords="offset points",
            )
        mid = (txmid + rxmid) * 0.5
        midSep = np.sqrt(np.square(txmid - rxmid))
        plt.plot(txmid, np.zeros(1), "ro")
        plt.plot(rxmid, np.zeros(1), "bo")
        plt.plot(mid, midSep / 2.0, "go")
        plt.plot(np.r_[txmid, mid], np.r_[0, midSep / 2.0], "k:")
        plt.plot(np.r_[rxmid, mid], np.r_[0, midSep / 2.0], "k:")

    else:
        if flag in ["PoleDipole", "PolePole"]:
            txmid = TxLoc[0][0]
        else:
            txmid = (TxLoc[0][0] + TxLoc[1][0]) * 0.5

        MLoc = RxLoc[0][j]
        NLoc = RxLoc[1][j]

        if flag in ["DipolePole", "PolePole"]:
            plt.plot(MLoc[0], np.zeros(1), "bv", markersize=10)
            ax.annotate(
                "M",
                xy=(MLoc[0], np.zeros(1)),
                xycoords="data",
                xytext=(-4.25, 7.5),
                textcoords="offset points",
            )
            rxmid = MLoc[0]
        else:
            rxmid = (MLoc[0] + NLoc[0]) * 0.5
            plt.plot(MLoc[0], np.zeros(1), "bv", markersize=10)
            plt.plot(NLoc[0], np.zeros(1), "b^", markersize=10)
            ax.annotate(
                "M",
                xy=(MLoc[0], np.zeros(1)),
                xycoords="data",
                xytext=(-4.25, 7.5),
                textcoords="offset points",
            )
            ax.annotate(
                "N",
                xy=(NLoc[0], np.zeros(1)),
                xycoords="data",
                xytext=(-4.25, 7.5),
                textcoords="offset points",
            )

        mid = (txmid + rxmid) * 0.5
        plt.plot((txmid + rxmid) * 0.5, np.arange(mid.size) + 1.0, "bo")
        plt.plot(rxmid, np.zeros(rxmid.size), "go")
        plt.plot(np.r_[txmid, mid[-1]], np.r_[0, mid.size], "k:")
        for j in range(ntx - i):
            plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j + 1], "k:")
    plt.xlabel("X (m)")
    plt.ylabel("N-spacing")
    plt.xlim(xr.min() - 5, xr.max() + 5)
    plt.ylim(nmax * dx / 2 + dx, -2 * dx)
    plt.show()


def DipoleDipolefun(i):
    """
    Plotting function to display all receivers and pseudolocations
    of a dipole-dipole survey for each source i

    :param int i: source index
    """
    matplotlib.rcParams["font.size"] = 14
    plt.figure(figsize=(10, 3))
    nmax = 8
    xr = np.linspace(-40, 40, 20)
    ntx = xr.size - 2
    dxr = np.diff(xr)
    plt.plot(xr[:-1] + dxr * 0.5, np.zeros_like(xr[:-1]), "ko")
    plt.plot(xr[i] + dxr[i] * 0.5, np.zeros(1), "ro")
    # for i in range(ntx):
    if i < ntx - nmax + 1:
        txmid = xr[i] + dxr[i] * 0.5
        rxmid = xr[i + 1 : i + 1 + nmax] + dxr[i + 1 : i + 1 + nmax] * 0.5
        mid = (txmid + rxmid) * 0.5
        plt.plot(rxmid, np.zeros(rxmid.size), "go")
        plt.plot(mid, np.arange(nmax) + 1.0, "bo")
        plt.plot(np.r_[txmid, mid[-1]], np.r_[0, nmax], "k:")
        for j in range(nmax):
            plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j + 1], "k:")

    else:
        txmid = xr[i] + dxr[i] * 0.5
        rxmid = xr[i + 1 : ntx + 1] + dxr[i + 1 : ntx + 1] * 0.5
        mid = (txmid + rxmid) * 0.5
        plt.plot((txmid + rxmid) * 0.5, np.arange(mid.size) + 1.0, "bo")
        plt.plot(rxmid, np.zeros(rxmid.size), "go")
        plt.plot(np.r_[txmid, mid[-1]], np.r_[0, mid.size], "k:")
        for j in range(ntx - i):
            plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j + 1], "k:")
    plt.xlabel("X (m)")
    plt.ylabel("N-spacing")
    plt.xlim(xr.min(), xr.max())
    plt.ylim(nmax + 1, -1)
    plt.show()


def PseudoSectionWidget(survey, flag):
    """
    Wigdet to visualize the pseudolocations
    associated with a particular survey
    for each pair source-receiver

    :param SimPEG.survey survey: Survey object
    :param str flag: Survey Type 'PoleDipole', 'DipoleDipole', 'DipolePole'
    """
    dx = 5
    xr = np.arange(-40, 41, dx)
    if flag == "PoleDipole":
        ntx, nmax = xr.size - 2, 8
    elif flag == "DipolePole":
        ntx, nmax = xr.size - 1, 7
    elif flag == "DipoleDipole":
        ntx, nmax = xr.size - 3, 8
    elif flag == "PolePole":
        ntx, nmax = xr.size - 2, 8

    def PseudoSectionPlot(i, j):
        return PseudoSectionPlotfnc(i, j, survey, flag)

    return widgetify(
        PseudoSectionPlot,
        i=IntSlider(min=0, max=ntx - 1, step=1, value=0),
        j=IntSlider(min=0, max=nmax - 1, step=1, value=0),
    )


def MidpointPseudoSectionWidget():
    """
    Widget function to display receivers and pseudolocations
    of a dipole-dipole survey for each source i

    :param int i: source index
    """
    ntx = 18
    return widgetify(DipoleDipolefun, i=IntSlider(min=0, max=ntx - 1, step=1, value=0))


_cache = {}


def DC2Dfwdfun(
    mesh,
    simulation,
    mapping,
    xr,
    xzlocs,
    rhohalf,
    rhoblk,
    xc,
    yc,
    r,
    dobs,
    uncert,
    predmis,
    nmax=8,
    plotFlag=None,
):
    """
    Function to display the pseudosection obtained through a survey
    over a known geological model

    :param TensorMesh mesh: discretization of the model
    :param SimPEG.Simulation sim: simulation object
    :param SimPEG.SigmaMap mapping: sigmamap of the model
    :param numpy.array xr: electrodes positions
    :param numpy.array xzlocs: pseudolocations
    :param float rhohalf: Resistivity of the half-space
    :param float rhoblk: Resistivity of the cylinder
    :param float xc: horizontal center of the cylinder
    :param float zc: vertical center of the cylinder
    :param float r: radius of the cylinder
    :param numpy.array dobs: observed data
    :param numpy.array uncert: uncertainities of the data
    :param str predmis: Choose between 'mis' to display the data misfit
                        or 'pred' to display the predicted data
    :param int nmax: Maximum number of receivers for each source
    :param bool plotFlag: Plot only the predicted data
                          or also the observed and misfit
    """
    matplotlib.rcParams["font.size"] = 14
    sighalf, sigblk = 1.0 / rhohalf, 1.0 / rhoblk
    siglayer = 1e-3
    zh = -5
    mtrue = np.r_[np.log(sighalf), np.log(siglayer), np.log(sigblk), xc, yc, r, zh]
    dpred = simulation.dpred(mtrue)

    xi, yi = np.meshgrid(
        np.linspace(xr.min(), xr.max(), 120), np.linspace(1.0, nmax, 100)
    )
    survey = simulation.survey
    G = static_utils.geometric_factor(survey, survey_type=survey.survey_type)
    appres = np.abs(dpred * (1.0 / G))
    appresobs = np.abs(dobs * (1.0 / G))

    std = np.std(appres)
    pred = griddata(xzlocs, appres, (xi, yi), method="linear")

    if plotFlag is not None:
        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        dat1 = mesh.plotImage(
            np.log10(1.0 / (mapping * mtrue)),
            ax=ax1,
            clim=(1, 3),
            grid=True,
            gridOpts={"color": "k", "alpha": 0.5},
        )
        cb1ticks = [1.0, 2.0, 3.0]
        cb1 = plt.colorbar(dat1[0], ax=ax1, ticks=cb1ticks)
        cb1.ax.set_yticklabels(["{:.0f}".format(10.0 ** x) for x in cb1ticks])
        cb1.set_label("Resistivity (ohm-m)")
        ax1.set_ylim(-20, 1.0)
        ax1.set_xlim(-40, 40)
        ax1.set_xlabel("")
        ax1.set_ylabel("Depth (m)")
        ax1.set_aspect("equal")
        ax1.plot(xr, np.zeros_like(xr), "ko")
        if std < 1.0:
            dat2 = ax2.pcolormesh(xi, yi, pred)
        else:
            dat2 = ax2.contourf(xi, yi, pred, 10)
            ax2.contour(xi, yi, pred, 10, colors="k", alpha=0.5)
        ax2.plot(xzlocs[:, 0], xzlocs[:, 1], "k.", ms=3)
        cb2 = plt.colorbar(
            dat2,
            ax=ax2,
            ticks=np.linspace(appres.min(), appres.max(), 3),
            format="%.0f",
        )
        cb2.set_label("Apparent Resistivity \n (ohm-m)")
        ax2.text(-38, 7, "Predicted")

        ax2.set_ylim(nmax + 1, 0.0)
        ax2.set_ylabel("N-spacing")
        ax2.set_xlabel("Distance (m)")

    else:
        obs = griddata(xzlocs, appresobs, (xi, yi), method="linear")
        plt.figure(figsize=(12, 9))
        ax1 = plt.subplot(311)
        dat1 = mesh.plotImage(
            np.log10(1.0 / (mapping * mtrue)),
            ax=ax1,
            clim=(1, 3),
            grid=True,
            gridOpts={"color": "k", "alpha": 0.5},
        )
        cb1ticks = [1.0, 2.0, 3.0]
        cb1 = plt.colorbar(dat1[0], ax=ax1, ticks=cb1ticks)
        cb1.ax.set_yticklabels(["{:.0f}".format(10.0 ** x) for x in cb1ticks])
        cb1.set_label("Resistivity (ohm-m)")
        ax1.set_ylim(-20, 0.0)
        ax1.set_xlim(-40, 40)
        ax1.set_xlabel("")
        ax1.set_ylabel("Depth (m)")
        ax1.set_aspect("equal")

        ax2 = plt.subplot(312)
        dat2 = ax2.contourf(xi, yi, obs, 10)
        ax2.contour(xi, yi, obs, 10, colors="k", alpha=0.5)
        ax2.plot(xzlocs[:, 0], xzlocs[:, 1], "k.", ms=3)
        cb2 = plt.colorbar(dat2, ax=ax2)

        cb2.set_label("Apparent Resistivity \n (ohm-m)")
        ax2.set_ylim(nmax + 1, 0.0)
        ax2.set_ylabel("N-spacing")
        ax2.text(-38, 7, "Observed")

        ax3 = plt.subplot(313)
        if predmis == "pred":
            if std < 1.0:
                dat3 = ax3.pcolormesh(xi, yi, pred)
            else:
                dat3 = ax3.contourf(xi, yi, pred, 10)
                ax3.contour(xi, yi, pred, 10, colors="k", alpha=0.5)
            ax3.plot(xzlocs[:, 0], xzlocs[:, 1], "k.", ms=3)
            cb3 = plt.colorbar(
                dat3,
                ax=ax3,
                ticks=np.linspace(appres.min(), appres.max(), 5),
                format="%4.0f",
            )
            cb3.set_label("Apparent Resistivity \n (ohm-m)")
            ax3.text(-38, 7, "Predicted")
        elif predmis == "mis":
            mis = (appresobs - appres) / (appresobs) * 100
            Mis = griddata(xzlocs, mis, (xi, yi), method="linear")
            dat3 = ax3.contourf(xi, yi, Mis, 10)
            ax3.contour(xi, yi, Mis, 10, colors="k", alpha=0.5)
            ax3.plot(xzlocs[:, 0], xzlocs[:, 1], "k.", ms=3)
            cb3 = plt.colorbar(
                dat3, ax=ax3, ticks=np.linspace(mis.min(), mis.max(), 5), format="%4.2f"
            )
            cb3.set_label("Normalized misfit (%)")
            ax3.text(-38, 7, "Misifit")
        ax3.set_ylim(nmax + 1, 0.0)
        ax3.set_ylabel("N-spacing")
        ax3.set_xlabel("Distance (m)")

    plt.show()


def DC2DPseudoWidgetWrapper(rhohalf, rhosph, xc, zc, r, surveyType):
    if "surveyType" not in _cache or _cache["surveyType"] != surveyType:
        dobs, uncert, simulation, xzlocs = DC2Dsurvey(surveyType)
        _cache["surveyType"] = surveyType
        _cache["dobs"] = dobs
        _cache["uncert"] = uncert
        _cache["simulation"] = simulation
        _cache["xzlocs"] = xzlocs
    else:
        dobs = _cache["dobs"]
        uncert = _cache["uncert"]
        simulation = _cache["simulation"]
        xzlocs = _cache["xzlocs"]

    DC2Dfwdfun(
        mesh,
        simulation,
        mapping,
        xr,
        xzlocs,
        rhohalf,
        rhosph,
        xc,
        zc,
        r,
        dobs,
        uncert,
        "pred",
        plotFlag="PredOnly",
    )
    return None


def DC2DPseudoWidget():
    return interactive(
        DC2DPseudoWidgetWrapper,
        rhohalf=FloatText(
            min=10,
            max=1000,
            value=1000,
            continuous_update=False,
            description="$\\rho_1$",
        ),
        rhosph=FloatText(
            min=10,
            max=1000,
            value=1000,
            continuous_update=False,
            description="$\\rho_2$",
        ),
        xc=FloatText(min=-40, max=40, step=1, value=0, continuous_update=False),
        zc=FloatText(min=-20, max=0, step=1, value=-10, continuous_update=False),
        r=FloatText(min=0, max=15, step=0.5, value=5, continuous_update=False),
        surveyType=ToggleButtons(
            options=["PolePole", "PoleDipole", "DipolePole", "DipoleDipole"],
            value="DipoleDipole",
        ),
    )


def DC2DfwdWrapper(rhohalf, rhosph, xc, zc, r, predmis, surveyType):
    if "surveyType" not in _cache or _cache["surveyType"] != surveyType:
        dobs, uncert, simulation, xzlocs = DC2Dsurvey(surveyType)
        _cache["surveyType"] = surveyType
        _cache["dobs"] = dobs
        _cache["uncert"] = uncert
        _cache["simulation"] = simulation
        _cache["xzlocs"] = xzlocs
    else:
        dobs = _cache["dobs"]
        uncert = _cache["uncert"]
        simulation = _cache["simulation"]
        xzlocs = _cache["xzlocs"]
    DC2Dfwdfun(
        mesh,
        simulation,
        mapping,
        xr,
        xzlocs,
        rhohalf,
        rhosph,
        xc,
        zc,
        r,
        dobs,
        uncert,
        predmis,
    )
    return None


def DC2DfwdWidget():
    return widgetify(
        DC2DfwdWrapper,
        manual=False,
        rhohalf=FloatText(
            min=10,
            max=1000,
            value=1000,
            continuous_update=False,
            description="$\\rho_1$",
        ),
        rhosph=FloatText(
            min=10,
            max=1000,
            value=500,
            continuous_update=False,
            description="$\\rho_2$",
        ),
        xc=FloatSlider(min=-40, max=40, step=1, value=0, continuous_update=False),
        zc=FloatSlider(min=-20, max=0, step=1, value=-10, continuous_update=False),
        r=FloatSlider(min=0, max=15, step=0.5, value=5, continuous_update=False),
        predmis=ToggleButtons(options=["pred", "mis"]),
        surveyType=ToggleButtons(
            options=["PolePole", "PoleDipole", "DipolePole", "DipoleDipole"]
        ),
    )
