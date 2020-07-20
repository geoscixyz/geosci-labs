from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from scipy.constants import epsilon_0
from scipy.interpolate import griddata
import copy

from ipywidgets import (
    interact,
    interact_manual,
    IntSlider,
    FloatSlider,
    FloatText,
    ToggleButtons,
    fixed,
    Widget,
)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import LogFormatter
from matplotlib import colors, ticker, cm
from matplotlib.path import Path
import matplotlib.patches as patches

from discretize import TensorMesh

from SimPEG import maps, SolverLU, utils
from SimPEG.utils import ExtractCoreMesh
from SimPEG.electromagnetics.static import resistivity as DC
from SimPEG.electromagnetics.static import induced_polarization as IP
from pymatsolver import Pardiso

from ..base import widgetify

# Mesh, sigmaMap can be globals global
npad = 12
growrate = 2.0
cs = 20.0
hx = [(cs, npad, -growrate), (cs, 100), (cs, npad, growrate)]
hy = [(cs, npad, -growrate), (cs, 50)]
mesh = TensorMesh([hx, hy], "CN")
expmap = maps.ExpMap(mesh)
mapping = expmap
xmin = -1000.0
xmax = 1000.0
ymin = -1000.0
ymax = 100.0
dx = 60.0
xr = np.arange(xmin, xmax + 1.0, dx)
dxr = np.diff(xr)
xylim = np.c_[[xmin, ymin], [xmax, ymax]]
indCC, meshcore = ExtractCoreMesh(xylim, mesh)
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

_cache = {
    "A": None,
    "B": None,
    "mtrue": None,
    "mhalf": None,
    "mair": None,
    "mover": None,
    "whichprimary": None,
    "target_wide": None,
}

nmax = 8


def model_valley(
    lnsig_air=np.log(1e-8),
    ln_sigback=np.log(1e-4),
    ln_over=np.log(1e-2),
    ln_sigtarget=np.log(1e-3),
    overburden_thick=200.0,
    overburden_wide=1000.0,
    target_thick=200.0,
    target_wide=400.0,
    a=1000.0,
    b=500.0,
    xc=0.0,
    zc=250.0,
):

    mtrue = ln_sigback * np.ones(mesh.nC)
    mhalf = copy.deepcopy(mtrue)

    ellips = (
        ((mesh.gridCC[:, 0] - xc) ** 2.0) / a ** 2.0
        + ((mesh.gridCC[:, 1] - zc) ** 2.0) / b ** 2.0
    ) < 1.0
    mtrue[ellips] = lnsig_air
    mair = copy.deepcopy(mtrue)

    # overb = (mesh.gridCC[:, 1] >-overburden_thick) & (mesh.gridCC[:, 1]<=0)&(mesh.gridCC[:, 0] >-overburden_wide/2.)&(mesh.gridCC[:, 0] <overburden_wide/2.)
    # mtrue[overb] = ln_over*np.ones_like(mtrue[overb])
    if np.any(ellips):
        bottom_valley = mesh.gridCC[ellips, 1].min()
        overb = (
            (mesh.gridCC[:, 1] >= bottom_valley)
            & (mesh.gridCC[:, 1] < bottom_valley + overburden_thick)
            & ellips
        )
        mtrue[overb] = ln_over * np.ones_like(mtrue[overb])
        mair[overb] = ln_sigback
    else:
        bottom_valley = 0.0
    mover = copy.deepcopy(mtrue)

    target = (
        (mesh.gridCC[:, 1] > bottom_valley - target_thick)
        & (mesh.gridCC[:, 1] < bottom_valley)
        & (mesh.gridCC[:, 0] > -target_wide / 2.0)
        & (mesh.gridCC[:, 0] < target_wide / 2.0)
    )
    mtrue[target] = ln_sigtarget * np.ones_like(mtrue[target])

    mtrue = utils.mkvc(mtrue)

    return mtrue, mhalf, mair, mover


def findnearest(A):
    idx = np.abs(mesh.gridCC[:, 0, None] - A).argmin(axis=0)
    return mesh.gridCC[idx, 0]


def get_Surface(mtrue, A):
    active = mtrue > (np.log(1e-8))
    nearpoint = findnearest(A)
    columns = mesh.gridCC[:, 0, None] == nearpoint
    ind = np.logical_and(columns.T, active).T
    idm = []
    surface = []
    for i in range(ind.shape[1]):
        idm.append(
            np.where(
                np.all(
                    mesh.gridCC
                    == np.r_[nearpoint[i], np.max(mesh.gridCC[ind[:, i], 1])],
                    axis=1,
                )
            )
        )
        surface.append(mesh.gridCC[idm[-1], 1])
    return utils.mkvc(np.r_[idm]), utils.mkvc(np.r_[surface])


def model_fields(A, B, mtrue, mhalf, mair, mover, whichprimary="overburden"):
    re_run = (
        _cache["A"] != A
        or _cache["B"] != B
        or np.any(_cache["mtrue"] != mtrue)
        or np.any(_cache["mhalf"] != mhalf)
        or np.any(_cache["mair"] != mair)
        or np.any(_cache["mover"] != mover)
        or _cache["whichprimary"] != whichprimary
    )
    if re_run:

        idA, surfaceA = get_Surface(mtrue, A)
        idB, surfaceB = get_Surface(mtrue, B)
        if B == []:
            src = DC.sources.Pole([], np.r_[A, surfaceA])
        else:
            src = DC.sources.Dipole([], np.r_[A, surfaceA], np.r_[B, surfaceB])
        survey = DC.Survey([src])
        # Create three simulations so the fields object is accurate
        sim_primary = DC.Simulation2DCellCentered(
            mesh, survey=survey, sigmaMap=mapping, solver=Pardiso
        )
        sim_total = DC.Simulation2DCellCentered(
            mesh, survey=survey, sigmaMap=mapping, solver=Pardiso
        )
        sim_air = DC.Simulation2DCellCentered(
            mesh, survey=survey, sigmaMap=mapping, solver=Pardiso
        )

        mesh.setCellGradBC("neumann")

        if whichprimary == "air":
            primary_field = sim_primary.fields(mair)
        elif whichprimary == "half":
            primary_field = sim_primary.fields(mhalf)
        elif whichprimary == "overburden":
            primary_field = sim_primary.fields(mover)
        air_field = sim_total.fields(mtrue)
        total_field = sim_air.fields(mair)

        _cache["A"] = A
        _cache["B"] = B
        _cache["mtrue"] = mtrue
        _cache["mhalf"] = mhalf
        _cache["mair"] = mair
        _cache["mover"] = mover
        _cache["whichprimary"] = whichprimary

        _cache["src"] = src
        _cache["primary_field"] = primary_field
        _cache["air_field"] = air_field
        _cache["total_field"] = total_field
    else:
        src = _cache["src"]
        primary_field = _cache["primary_field"]
        air_field = _cache["air_field"]
        total_field = _cache["total_field"]

    return src, primary_field, air_field, total_field


def get_Surface_Potentials(mtrue, survey, src, field_obj):

    phi = field_obj[src, "phi"]
    # CCLoc = mesh.gridCC
    XLoc = np.unique(mesh.gridCC[:, 0])
    surfaceInd, zsurfaceLoc = get_Surface(mtrue, XLoc)
    phiSurface = phi[surfaceInd]
    phiScale = 0.0

    if survey == "Pole-Dipole" or survey == "Pole-Pole":
        refInd = utils.closestPoints(mesh, [xmax + 60.0, 0.0], gridLoc="CC")
        # refPoint =  CCLoc[refInd]
        # refSurfaceInd = np.where(xSurface == refPoint[0])
        # phiScale = np.median(phiSurface)
        phiScale = phi[refInd]
        phiSurface = phiSurface - phiScale

    return XLoc, phiSurface, phiScale


def getCylinderPoints(xc, zc, a, b):
    xLocOrig1 = np.arange(-a, a + a / 10.0, a / 10.0)
    xLocOrig2 = np.arange(a, -a - a / 10.0, -a / 10.0)
    # Top half of cylinder
    zLoc1 = b * np.sqrt(1.0 - (xLocOrig1 / a) ** 2) + zc
    # Bottom half of cylinder
    zLoc2 = -b * np.sqrt(1.0 - (xLocOrig2 / a) ** 2) + zc
    # Shift from x = 0 to xc
    xLoc1 = xLocOrig1 + xc * np.ones_like(xLocOrig1)
    xLoc2 = xLocOrig2 + xc * np.ones_like(xLocOrig2)

    cylinderPoints = np.vstack(
        [np.vstack([xLoc1, zLoc1]).T, np.vstack([xLoc2, zLoc2]).T]
    )
    return cylinderPoints


def get_OverburdenPoints(cylinderPoints, overburden_thick):
    bottom = cylinderPoints[:, 1].min()
    indb = np.where(cylinderPoints[:, 1] < 0.0)
    overburdenPoints = [
        np.maximum(cylinderPoints[i, 1], bottom + overburden_thick) for i in indb
    ]
    return np.vstack([cylinderPoints[indb, 0], overburdenPoints]).T


def getPlateCorners(target_thick, target_wide, cylinderPoints):

    bottom = cylinderPoints[:, 1].min()
    xc = 0.0
    zc = bottom - 0.5 * target_thick
    rotPlateCorners = np.array(
        [
            [-0.5 * target_wide, 0.5 * target_thick],
            [0.5 * target_wide, 0.5 * target_thick],
            [-0.5 * target_wide, -0.5 * target_thick],
            [0.5 * target_wide, -0.5 * target_thick],
        ]
    )
    plateCorners = rotPlateCorners + np.hstack(
        [np.repeat(xc, 4).reshape([4, 1]), np.repeat(zc, 4).reshape([4, 1])]
    )
    return plateCorners


# def get_TargetPoints(target_thick, target_wide, ellips_b, ellips_zc):
#     xLocOrig1 = np.arange(
#         -target_wide / 2.0, target_wide / 2.0 + target_wide / 10.0, target_wide / 10.0
#     )
#     xLocOrig2 = np.arange(
#         target_wide / 2.0, -target_wide / 2.0 - target_wide / 10.0, -target_wide / 10.0
#     )
#     zloc1 = np.ones_like(xLocOrig1) * (ellips_b + ellips_zc)
#     zloc1 = np.ones_like(xLocOrig1) * (ellips_b + ellips_zc - target_thick)

#     corner

#     targetpoint = np.vstack([np.vstack([xLoc1, zLoc1]).T, np.vstack([xLoc2, zLoc2]).T])


def getSensitivity(survey, A, B, M, N, model):
    src_type, rx_type = survey.split("-")
    if rx_type == "dipole":
        rx = DC.receivers.Dipole(np.r_[M, 0.0], np.r_[N, 0.0])
    else:
        rx = DC.receivers.Pole(np.r_[M, 0.0])

    if src_type == "dipole":
        src = DC.sources.Dipole([rx], np.r_[A, 0.0], np.r_[B, 0.0])
    else:
        src = DC.sources.Pole([rx], np.r_[A, 0.0])

    survey = DC.Survey([src])
    problem = DC.Simulation2DCellCentered(
        mesh, sigmaMap=mapping, solver=Pardiso, survey=survey
    )

    J = problem.getJ(model)[0]

    return J


def calculateRhoA(survey, VM, VN, A, B, M, N):

    # to stabilize division
    eps = 1e-9

    if survey == "Dipole-Dipole":
        G = 1.0 / (
            1.0 / (np.abs(A - M) + eps)
            - 1.0 / (np.abs(M - B) + eps)
            - 1.0 / (np.abs(N - A) + eps)
            + 1.0 / (np.abs(N - B) + eps)
        )
        rho_a = (VM - VN) * 2.0 * np.pi * G
    elif survey == "Pole-Dipole":
        G = 1.0 / (1.0 / (np.abs(A - M) + eps) - 1.0 / (np.abs(N - A) + eps))
        rho_a = (VM - VN) * 2.0 * np.pi * G
    elif survey == "Dipole-Pole":
        G = 1.0 / (1.0 / (np.abs(A - M) + eps) - 1.0 / (np.abs(M - B) + eps))
        rho_a = (VM) * 2.0 * np.pi * G
    elif survey == "Pole-Pole":
        G = 1.0 / (1.0 / (np.abs(A - M) + eps))
        rho_a = (VM) * 2.0 * np.pi * G

    return rho_a


def getPseudoLocs(xr, ntx, nmax, flag="PoleDipole"):
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

            mid = (txmid + rxmid) * 0.5
            xloc.append(mid)
            yloc.append(np.arange(mid.size) + 1.0)
    xlocvec = np.hstack(xloc)
    ylocvec = np.hstack(yloc)
    return np.c_[xlocvec, ylocvec]


def DC2Dsimulation(mtrue, flag="PoleDipole", nmax=8):

    if flag == "PoleDipole":
        ntx = xr.size - 2
    elif flag == "DipolePole":
        ntx = xr.size - 2
    elif flag == "DipoleDipole":
        ntx = xr.size - 3
    else:
        raise Exception("Not Implemented")
    xzlocs = getPseudoLocs(xr, ntx, nmax, flag)

    txList = []
    zloc = -cs / 2.0
    for i in range(ntx):
        if flag == "PoleDipole":
            A = np.r_[xr[i], zloc]
            B = np.r_[mesh.vectorCCx.min(), zloc]
            if i < ntx - nmax + 1:
                Mx = xr[i + 1 : i + 1 + nmax]
                _, Mz = get_Surface(mtrue, Mx)
                Nx = xr[i + 2 : i + 2 + nmax]
                _, Nz = get_Surface(mtrue, Nx)

                M = np.c_[Mx, Mz]
                N = np.c_[Nx, Nz]
            else:
                Mx = xr[i + 1 : ntx + 1]
                _, Mz = get_Surface(mtrue, Mx)
                Nx = xr[i + 2 : i + 2 + nmax]
                _, Nz = get_Surface(mtrue, Nx)

                M = np.c_[Mx, Mz]
                N = np.c_[Nx, Nz]

        elif flag == "DipolePole":
            A = np.r_[xr[i], zloc]
            B = np.r_[xr[i + 1], zloc]
            if i < ntx - nmax + 1:
                Mx = xr[i + 2 : i + 2 + nmax]
                _, Mz = get_Surface(mtrue, Mx)
                Nx = np.ones(nmax) * mesh.vectorCCx.max()
                _, Nz = get_Surface(mtrue, Nx)

                M = np.c_[Mx, Mz]
                N = np.c_[Nx, Nz]

            else:
                Mx = xr[i + 2 : ntx + 2]
                _, Mz = get_Surface(mtrue, Mx)
                Nx = np.ones(ntx - i) * mesh.vectorCCx.max()
                _, Nz = get_Surface(mtrue, Nx)
                M = np.c_[Mx, Mz]
                N = np.c_[Nx, Nz]

        elif flag == "DipoleDipole":
            A = np.r_[xr[i], zloc]
            B = np.r_[xr[i + 1], zloc]
            if i < ntx - nmax:
                Mx = xr[i + 2 : i + 2 + nmax]
                _, Mz = get_Surface(mtrue, Mx)
                Nx = xr[i + 3 : i + 3 + nmax]
                _, Nz = get_Surface(mtrue, Nx)
                M = np.c_[Mx, Mz]
                N = np.c_[Nx, Nz]

            else:
                Mx = xr[i + 2 : len(xr) - 1]
                _, Mz = get_Surface(mtrue, Mx)
                Nx = xr[i + 3 : len(xr)]
                _, Nz = get_Surface(mtrue, Nx)
                M = np.c_[Mx, Mz]
                N = np.c_[Nx, Nz]

        rx = DC.receivers.Dipole(M, N)
        src = DC.sources.Dipole([rx], A, B)
        txList.append(src)

    survey = DC.Survey(txList)
    simulation = DC.Simulation2DCellCentered(
        mesh, sigmaMap=mapping, survey=survey, solver=Pardiso
    )

    return simulation, xzlocs


def IP2Dsimulation(miptrue, sigmadc, flag="PoleDipole", nmax=8):

    if flag == "PoleDipole":
        ntx = xr.size - 2
    elif flag == "DipolePole":
        ntx = xr.size - 2
    elif flag == "DipoleDipole":
        ntx = xr.size - 3
    else:
        raise Exception("Not Implemented")
    xzlocs = getPseudoLocs(xr, ntx, nmax, flag)

    txList = []
    zloc = -cs / 2.0
    for i in range(ntx):
        if flag == "PoleDipole":
            A = np.r_[xr[i], zloc]
            B = np.r_[mesh.vectorCCx.min(), zloc]
            if i < ntx - nmax + 1:
                Mx = xr[i + 1 : i + 1 + nmax]
                _, Mz = get_Surface(miptrue, Mx)
                Nx = xr[i + 2 : i + 2 + nmax]
                _, Nz = get_Surface(miptrue, Nx)

                M = np.c_[Mx, Mz]
                N = np.c_[Nx, Nz]
            else:
                Mx = xr[i + 1 : ntx + 1]
                _, Mz = get_Surface(miptrue, Mx)
                Nx = xr[i + 2 : i + 2 + nmax]
                _, Nz = get_Surface(miptrue, Nx)

                M = np.c_[Mx, Mz]
                N = np.c_[Nx, Nz]

        elif flag == "DipolePole":
            A = np.r_[xr[i], zloc]
            B = np.r_[xr[i + 1], zloc]
            if i < ntx - nmax + 1:
                Mx = xr[i + 2 : i + 2 + nmax]
                _, Mz = get_Surface(miptrue, Mx)
                Nx = np.ones(nmax) * mesh.vectorCCx.max()
                _, Nz = get_Surface(miptrue, Nx)

                M = np.c_[Mx, Mz]
                N = np.c_[Nx, Nz]

            else:
                Mx = xr[i + 2 : ntx + 2]
                _, Mz = get_Surface(miptrue, Mx)
                Nx = np.ones(ntx - i) * mesh.vectorCCx.max()
                _, Nz = get_Surface(miptrue, Nx)
                M = np.c_[Mx, Mz]
                N = np.c_[Nx, Nz]

        elif flag == "DipoleDipole":
            A = np.r_[xr[i], zloc]
            B = np.r_[xr[i + 1], zloc]
            if i < ntx - nmax:
                Mx = xr[i + 2 : i + 2 + nmax]
                _, Mz = get_Surface(miptrue, Mx)
                Nx = xr[i + 3 : i + 3 + nmax]
                _, Nz = get_Surface(miptrue, Nx)
                M = np.c_[Mx, Mz]
                N = np.c_[Nx, Nz]

            else:
                Mx = xr[i + 2 : len(xr) - 1]
                _, Mz = get_Surface(miptrue, Mx)
                Nx = xr[i + 3 : len(xr)]
                _, Nz = get_Surface(miptrue, Nx)
                M = np.c_[Mx, Mz]
                N = np.c_[Nx, Nz]

        rx = DC.receivers.Dipole(M, N)
        src = DC.sources.Dipole([rx], A, B)
        txList.append(src)

    survey = IP.Survey(txList)
    simulation = IP.Simulation2DCellCentred(
        mesh,
        sigma=sigmadc,
        etaMap=maps.IdentityMap(mesh),
        survey=survey,
        solver=Pardiso,
    )

    return simulation, xzlocs


def PseudoSectionPlotfnc(i, j, survey, flag="PoleDipole"):
    matplotlib.rcParams["font.size"] = 14
    ntx = xr.size - 2
    TxObj = survey.srcList
    TxLoc = TxObj[i].loc
    RxLoc = TxObj[i].rxList[0].locs
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(
        111, autoscale_on=False, xlim=(xr.min() - 5, xr.max() + 5), ylim=(nmax + 1, -2)
    )
    plt.plot(xr, np.zeros_like(xr), "ko", markersize=4)
    if flag == "PoleDipole":
        plt.plot(TxLoc[0][0], np.zeros(1), "rv", markersize=10)
        # print([TxLoc[0][0],0])
        ax.annotate(
            "A",
            xy=(TxLoc[0][0], np.zeros(1)),
            xycoords="data",
            xytext=(-4.25, 7.5),
            textcoords="offset points",
        )
    else:
        plt.plot([TxLoc[0][0], TxLoc[1][0]], np.zeros(2), "rv", markersize=10)
        # print([[TxLoc[0][0],0],[TxLoc[1][0],0]])
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
    # for i in range(ntx):
    if i < ntx - nmax + 1:
        if flag == "PoleDipole":
            txmid = TxLoc[0][0]
        else:
            txmid = (TxLoc[0][0] + TxLoc[1][0]) * 0.5

        MLoc = RxLoc[0][j]
        NLoc = RxLoc[1][j]

        if flag == "DipolePole":
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
        if flag == "PoleDipole":
            txmid = TxLoc[0][0]
        else:
            txmid = (TxLoc[0][0] + TxLoc[1][0]) * 0.5

        MLoc = RxLoc[0][j]
        NLoc = RxLoc[1][j]
        if flag == "DipolePole":
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
    return


def DipoleDipolefun(i):
    matplotlib.rcParams["font.size"] = 14
    plt.figure(figsize=(10, 3))
    ntx = xr.size - 2
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
    return


def PseudoSectionWidget(simulation, flag):
    if flag == "PoleDipole":
        ntx, nmax = xr.size - 2, 8
    elif flag == "DipolePole":
        ntx, nmax = xr.size - 1, 7
    elif flag == "DipoleDipole":
        ntx, nmax = xr.size - 3, 8

    def PseudoSectionPlot(i, j, flag):
        return PseudoSectionPlotfnc(i, j, simulation.survey, flag)

    return widgetify(
        PseudoSectionPlot,
        i=IntSlider(min=0, max=ntx - 1, step=1, value=0),
        j=IntSlider(min=0, max=nmax - 1, step=1, value=0),
        flag=ToggleButtons(
            options=["DipoleDipole", "PoleDipole", "DipolePole"],
            description="Array Type",
        ),
    )


def MidpointPseudoSectionWidget():
    ntx = xr.size - 2
    return widgetify(DipoleDipolefun, i=IntSlider(min=0, max=ntx - 1, step=1, value=0))


def DCIP2Dfwdfun(
    mesh,
    mapping,
    rhohalf,
    rholayer,
    rhoTarget,
    chghalf,
    chglayer,
    chgTarget,
    overburden_thick,
    overburden_wide,
    target_thick,
    target_wide,
    ellips_a,
    ellips_b,
    xc,
    zc,
    predmis,
    surveyType,
    nmax=8,
    which="DC",
    Scale="Linear",
):

    matplotlib.rcParams["font.size"] = 14

    ln_sigTarget = np.log(1.0 / rhoTarget)
    ln_sigLayer = np.log(1.0 / rholayer)
    ln_sigHalf = np.log(1.0 / rhohalf)

    mtrue, mhalf, mair, mover = model_valley(
        lnsig_air=np.log(1e-8),
        ln_sigback=ln_sigHalf,
        ln_over=ln_sigLayer,
        ln_sigtarget=ln_sigTarget,
        overburden_thick=overburden_thick,
        target_thick=target_thick,
        target_wide=target_wide,
        a=ellips_a,
        b=ellips_b,
        xc=xc,
        zc=zc,
    )
    mdctrue = mtrue

    if which == "IP":
        mtrue, mhalf, mair, mover = model_valley(
            lnsig_air=0.0,
            ln_sigback=chghalf,
            ln_over=chglayer,
            ln_sigtarget=chgTarget,
            overburden_thick=overburden_thick,
            target_thick=target_thick,
            target_wide=target_wide,
            a=ellips_a,
            b=ellips_b,
            xc=xc,
            zc=zc,
        )

        sigmadc = 1.0 / (mapping * mdctrue)
        simulation, xzlocs = IP2Dsimulation(mtrue, sigmadc, surveyType, nmax=nmax)

    else:
        simulation, xzlocs = DC2Dsimulation(mtrue, surveyType, nmax=nmax)

    dmover = simulation.dpred(mover)
    dpred = simulation.dpred(mtrue)
    xi, yi = np.meshgrid(
        np.linspace(xr.min(), xr.max(), 120), np.linspace(1.0, nmax, 100)
    )

    # Cheat to compute a geometric factor
    # define as G = dV_halfspace / rho_halfspace
    if which == "IP":
        mtest = 10.0 * np.ones_like(mtrue)
        mtest[mdctrue == np.log(1e-8)] = 0.0
        dhalf = simulation.dpred(mtest)
        appresover = 10.0 * (dmover / dhalf)
        apprestrue = 10.0 * (dpred / dhalf)
    else:
        dmair = simulation.dpred(mair)
        appresover = dmover / dmair / np.exp(ln_sigHalf)
        apprestrue = dpred / dmair / np.exp(ln_sigHalf)

    dtrue = griddata(xzlocs, apprestrue, (xi, yi), method="linear")
    dtrue = np.ma.masked_where(np.isnan(dtrue), dtrue)

    dover = griddata(xzlocs, appresover, (xi, yi), method="linear")
    dover = np.ma.masked_where(np.isnan(dover), dover)

    if which == "IP":
        label = "Chargeability"
    else:
        label = "Resistivity (Ohm-m)"

    plt.figure(figsize=(12, 9))
    ax1 = plt.subplot(311)

    if which == "IP":
        u = np.ma.masked_where(mdctrue <= np.log(1e-8), mtrue)
    else:
        u = np.ma.masked_where(mtrue <= np.log(1e-8), np.log10(1.0 / (mapping * mtrue)))
    dat1 = mesh.plotImage(
        u,
        ax=ax1,
        clim=(u.min(), u.max()),
        grid=True,
        gridOpts={"color": "k", "alpha": 0.5},
    )

    if which == "IP":
        cb1 = plt.colorbar(dat1[0], ax=ax1)
    else:
        cb1ticks = np.linspace(u.min(), u.max(), 3)
        cb1 = plt.colorbar(dat1[0], ax=ax1, ticks=cb1ticks)
        cb1.ax.set_yticklabels(["{:.0f}".format(10 ** x) for x in cb1ticks])
    cb1.set_label(label)

    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    ax1.set_xlabel("")
    ax1.set_ylabel("Depth (m)")

    ax2 = plt.subplot(312)
    if Scale == "Log":
        lev_exp = np.arange(
            np.floor(np.log10(np.abs(dtrue.min()))),
            np.ceil(np.log10(dtrue.max())) + 0.1,
            0.1,
        )

        lev = np.power(10, lev_exp)
        dat2 = ax2.contourf(xi, yi, dtrue, lev, locator=ticker.LogLocator())
        ax2.contour(
            xi, yi, dtrue, lev, locator=ticker.LogLocator(), colors="k", alpha=0.5
        )
        ax2.plot(xzlocs[:, 0], xzlocs[:, 1], "k.", ms=3)

        cb2 = plt.colorbar(
            dat2,
            ax=ax2,
            ticks=np.linspace(appresover.min(), appresover.max(), 5),
            format="%4.0f",
        )

    else:
        dat2 = ax2.contourf(xi, yi, dtrue, 10)
        ax2.contour(xi, yi, dtrue, 10, colors="k", alpha=0.5)
        ax2.plot(xzlocs[:, 0], xzlocs[:, 1], "k.", ms=3)
        cb2 = plt.colorbar(dat2, ax=ax2)

    cb2.set_label("Apparent\n" + label)
    ax2.set_ylim(nmax + 1, 0.0)
    ax2.set_ylabel("N-spacing")
    ax2.text(250, nmax - 1, "Observed")

    ax3 = plt.subplot(313)
    if predmis == "Data Without Target":
        if Scale == "Log":
            dat3 = ax3.contourf(xi, yi, dover, lev, locator=ticker.LogLocator())
            ax3.contour(
                xi, yi, dover, lev, locator=ticker.LogLocator(), colors="k", alpha=0.5
            )
            ax3.plot(xzlocs[:, 0], xzlocs[:, 1], "k.", ms=3)
            cb3 = plt.colorbar(
                dat3,
                ax=ax3,
                ticks=np.linspace(appresover.min(), appresover.max(), 5),
                format="%4.0f",
            )
        else:
            dat3 = ax3.contourf(xi, yi, dover, 10, vmin=dtrue.min(), vmax=dtrue.max())
            ax3.contour(
                xi,
                yi,
                dover,
                10,
                vmin=dtrue.min(),
                vmax=dtrue.max(),
                colors="k",
                alpha=0.5,
            )
            ax3.plot(xzlocs[:, 0], xzlocs[:, 1], "k.", ms=3)
            cb3 = plt.colorbar(
                dat3, ax=ax3, format="%4.0f", boundaries=(dtrue.min(), dtrue.max())
            )
        cb3.set_label("Apparent\n" + label)
        ax3.text(250, nmax - 1, "Predicted\nwithout Target")

    else:
        if predmis == "Difference":
            mis = apprestrue - appresover
            Mis = griddata(xzlocs, mis, (xi, yi), method="linear")
            if which == "IP":
                diflabel = "Difference (chg unit)"
            else:
                diflabel = "Difference (Ohm-m)"

        else:
            mis = (apprestrue - appresover) / apprestrue
            Mis = griddata(xzlocs, mis, (xi, yi), method="linear")
            diflabel = "Normalized Difference (%)"

        dat3 = ax3.contourf(xi, yi, Mis, 10)
        ax3.contour(xi, yi, Mis, 10, colors="k", alpha=0.5)
        ax3.plot(xzlocs[:, 0], xzlocs[:, 1], "k.", ms=3)
        cb3 = plt.colorbar(dat3, ax=ax3, format="%4.2f")
        cb3.set_label(diflabel)
        ax3.text(-38, 7, diflabel)
    ax3.set_ylim(nmax + 1, 0.0)
    ax3.set_ylabel("N-spacing")
    ax3.set_xlabel("Distance (m)")

    plt.show()
    return


def DC2DfwdWrapper(
    rhohalf,
    rholayer,
    rhoTarget,
    chghalf,
    chglayer,
    chgTarget,
    overburden_thick,
    overburden_wide,
    target_thick,
    target_wide,
    ellips_a,
    ellips_b,
    xc,
    zc,
    predmis,
    surveyType,
    nmax,
    which,
    Scale,
):
    DCIP2Dfwdfun(
        mesh,
        mapping,
        rhohalf,
        rholayer,
        rhoTarget,
        chghalf,
        chglayer,
        chgTarget,
        overburden_thick,
        overburden_wide,
        target_thick,
        target_wide,
        ellips_a,
        ellips_b,
        xc,
        zc,
        predmis,
        surveyType,
        nmax,
        which,
        Scale,
    )
    return None


def DCIP2DfwdWidget():
    return widgetify(
        DC2DfwdWrapper,
        xc=FloatSlider(
            min=-1005.0, max=1000.0, step=10.0, value=0.0, continuous_update=False
        ),
        zc=FloatSlider(
            min=-1000.0, max=1000.0, step=10.0, value=250.0, continuous_update=False
        ),
        ellips_a=FloatSlider(
            min=10.0, max=10000.0, step=100.0, value=1000.0, continuous_update=False
        ),
        ellips_b=FloatSlider(
            min=10.0, max=10000.0, step=100.0, value=500.0, continuous_update=False
        ),
        rhohalf=FloatText(
            min=1e-8,
            max=1e8,
            value=1000.0,
            description="$\\rho_1$",
            continuous_update=False,
        ),
        chghalf=FloatText(
            min=0.0,
            max=100,
            value=0.0,
            description="$\\eta_1$",
            continuous_update=False,
        ),
        rholayer=FloatText(
            min=1e-8,
            max=1e8,
            value=100.0,
            description="$\\rho_2$",
            continuous_update=False,
        ),
        chglayer=FloatText(
            min=0.0,
            max=100,
            value=20.0,
            description="$\\eta_2$",
            continuous_update=False,
        ),
        rhoTarget=FloatText(
            min=1e-8,
            max=1e8,
            value=500.0,
            description="$\\rho_3$",
            continuous_update=False,
        ),
        chgTarget=FloatText(
            min=0.0,
            max=100,
            value=10.0,
            description="$\\eta_3$",
            continuous_update=False,
        ),
        overburden_thick=FloatSlider(
            min=0.0, max=1000.0, step=10.0, value=250.0, continuous_update=False
        ),
        overburden_wide=fixed(2000.0),
        target_thick=FloatSlider(
            min=0.0, max=1000.0, step=10.0, value=200.0, continuous_update=False
        ),
        target_wide=FloatSlider(
            min=0.0, max=1000.0, step=10.0, value=200.0, continuous_update=False
        ),
        predmis=ToggleButtons(
            options=["Data Without Target", "Difference", "Normalized Difference"]
        ),
        surveyType=ToggleButtons(
            options=["DipoleDipole", "PoleDipole", "DipolePole"],
            desciption="Array Type",
        ),
        which=ToggleButtons(options=["DC", "IP"], description="Survey"),
        nmax=IntSlider(min=1, max=16, value=8, description="Rx per Tx"),
        Scale=ToggleButtons(options=["Linear", "Log"]),
    )
