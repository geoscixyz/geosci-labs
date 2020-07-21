from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
from scipy.constants import epsilon_0
import copy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import LogFormatter
from matplotlib.path import Path
import matplotlib.patches as patches

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

from discretize import TensorMesh
from SimPEG import maps, utils
from SimPEG.utils import ExtractCoreMesh
from SimPEG.electromagnetics.static import resistivity as DC

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

    bottom_valley = mesh.gridCC[ellips, 1].min()
    overb = (
        (mesh.gridCC[:, 1] >= bottom_valley)
        & (mesh.gridCC[:, 1] < bottom_valley + overburden_thick)
        & ellips
    )
    mtrue[overb] = ln_over * np.ones_like(mtrue[overb])
    mair[overb] = ln_sigback
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


def model_fields(A, B, mtrue, mhalf, mair, mover, whichprimary="air"):

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
        survey = DC.survey.Survey([src])
        problem = DC.Simulation2DCellCentered(
            mesh, survey=survey, sigmaMap=mapping, solver=Pardiso
        )
        problem_prim = DC.Simulation2DCellCentered(
            mesh, survey=survey, sigmaMap=mapping, solver=Pardiso
        )
        problem_air = DC.Simulation2DCellCentered(
            mesh, survey=survey, sigmaMap=mapping, solver=Pardiso
        )

        if whichprimary == "air":
            primary_field = problem_prim.fields(mair)
        elif whichprimary == "half":
            primary_field = problem_prim.fields(mhalf)
        elif whichprimary == "overburden":
            primary_field = problem_prim.fields(mover)
        total_field = problem.fields(mtrue)
        air_field = problem_air.fields(mair)

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


# In[30]:


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
#     # xLocOrig2 = np.arange(
#     #     target_wide / 2.0, -target_wide / 2.0 - target_wide / 10.0, -target_wide / 10.0
#     # )
#     # zloc1 = np.ones_like(xLocOrig1) * (ellips_b + ellips_zc)
#     zloc1 = np.ones_like(xLocOrig1) * (ellips_b + ellips_zc - target_thick)

#     # corner

#     targetpoint = np.vstack([np.vstack([xLoc1, zLoc1]).T, np.vstack([xLoc2, zLoc2]).T])


def getSensitivity(survey, A, B, M, N, model):
    src_type, rx_type = survey.split("-")
    if rx_type == "Pole":
        rx = DC.receivers.Pole(np.r_[M, 0.0])
    else:
        rx = DC.receivers.Dipole(np.r_[M, 0.0], np.r_[N, 0.0])
    if src_type == "Pole":
        src = DC.sources.Pole([rx], np.r_[A, 0.0])
    else:
        src = DC.sources.Dipole([rx], np.r_[A, 0.0], np.r_[B, 0.0])

    Src = DC.Survey([src])
    sim = DC.Simulation2DCellCentered(
        mesh, survey=Src, sigmaMap=mapping, solver=Pardiso
    )
    J = sim.getJ(model)[0]

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


def PLOT(
    survey,
    A,
    B,
    M,
    N,
    rhohalf,
    rholayer,
    rhoTarget,
    overburden_thick,
    overburden_wide,
    target_thick,
    target_wide,
    whichprimary,
    ellips_a,
    ellips_b,
    xc,
    zc,
    Field,
    Type,
    Scale,
):

    labelsize = 12.0
    ticksize = 10.0

    if survey == "Pole-Dipole" or survey == "Pole-Pole":
        B = []

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

    src, primary_field, air_field, total_field = model_fields(
        A, B, mtrue, mhalf, mair, mover, whichprimary=whichprimary
    )

    fig, ax = plt.subplots(2, 1, figsize=(9 * 1.5, 9 * 1.5), sharex=True)
    fig.subplots_adjust(right=0.8)

    xSurface, phiTotalSurface, phiScaleTotal = get_Surface_Potentials(
        mtrue, survey, src, total_field
    )
    xSurface, phiPrimSurface, phiScalePrim = get_Surface_Potentials(
        mtrue, survey, src, primary_field
    )
    xSurface, phiAirSurface, phiScaleAir = get_Surface_Potentials(
        mtrue, survey, src, air_field
    )
    ylim = np.r_[-1.0, 1.0] * np.max(np.abs(phiTotalSurface))
    xlim = np.array([-1000.0, 1000.0])

    if survey == "Dipole-Pole" or survey == "Pole-Pole":
        MInd = np.where(xSurface == findnearest(M))
        N = []

        VM = phiTotalSurface[MInd[0]]
        VN = 0.0

        # VMprim = phiPrimSurface[MInd[0]]
        # VNprim = 0.0

        VMair = phiAirSurface[MInd[0]]
        VNair = 0.0

    else:
        MInd = np.where(xSurface == findnearest(M))
        NInd = np.where(xSurface == findnearest(N))

        VM = phiTotalSurface[MInd[0]]
        VN = phiTotalSurface[NInd[0]]

        # VMprim = phiPrimSurface[MInd[0]]
        # VNprim = phiPrimSurface[NInd[0]]

        VMair = phiAirSurface[MInd[0]]
        VNair = phiAirSurface[NInd[0]]

    # 2D geometric factor
    G2D = rhohalf / (calculateRhoA(survey, VMair, VNair, A, B, M, N))
    # print G2D
    # Subplot 1: Full set of surface potentials
    ax[0].plot(xSurface, phiPrimSurface, linestyle="dashed", linewidth=2.0, color="k")
    ax[0].plot(xSurface, phiTotalSurface, color=[0.1, 0.5, 0.1], linewidth=1.0)
    ax[0].grid(
        which="both", linestyle="-", linewidth=0.5, color=[0.2, 0.2, 0.2], alpha=0.5
    )

    if survey == "Pole-Dipole" or survey == "Pole-Pole":
        ax[0].plot(A, 0, "+", markersize=12, markeredgewidth=3, color=[1.0, 0.0, 0.0])
    else:
        ax[0].plot(A, 0, "+", markersize=12, markeredgewidth=3, color=[1.0, 0.0, 0.0])
        ax[0].plot(B, 0, "_", markersize=12, markeredgewidth=3, color=[0.0, 0.0, 1.0])
    ax[0].set_ylabel("Potential, (V)", fontsize=labelsize)
    ax[0].set_xlabel("x (m)", fontsize=labelsize)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)

    if survey == "Dipole-Pole" or survey == "Pole-Pole":
        ax[0].plot(M, VM, "o", color="k")

        xytextM = (M + 0.5, np.max([np.min([VM, ylim.max()]), ylim.min()]) + 0.5)
        ax[0].annotate("%2.1e" % (VM), xy=xytextM, xytext=xytextM, fontsize=labelsize)

    else:
        ax[0].plot(M, VM, "o", color="k")
        ax[0].plot(N, VN, "o", color="k")

        xytextM = (M + 0.5, np.max([np.min([VM, ylim.max()]), ylim.min()]) + 0.5)
        xytextN = (N + 0.5, np.max([np.min([VN, ylim.max()]), ylim.min()]) + 0.5)
        ax[0].annotate("%2.1e" % (VM), xy=xytextM, xytext=xytextM, fontsize=labelsize)
        ax[0].annotate("%2.1e" % (VN), xy=xytextN, xytext=xytextN, fontsize=labelsize)

    ax[0].tick_params(axis="both", which="major", labelsize=ticksize)

    props = dict(boxstyle="round", facecolor="grey", alpha=0.4)
    ax[0].text(
        xlim.max() + 1,
        ylim.max() - 0.1 * ylim.max(),
        "$\\rho_a$ = %2.2f" % (G2D * calculateRhoA(survey, VM, VN, A, B, M, N)),
        verticalalignment="bottom",
        bbox=props,
        fontsize=14,
    )

    ax[0].legend(["Reference Potential", "Model Potential"], loc=3, fontsize=labelsize)
    if Scale == "Log":
        ax[0].set_yscale("symlog", linthreshy=1e-5)

    if Field == "Model":

        label = "Resisitivity (ohm-m)"
        xtype = "CC"
        view = "real"
        streamOpts = None
        ind = indCC

        formatter = "%.1e"
        pcolorOpts = {"cmap": "jet_r"}
        if Scale == "Log":
            pcolorOpts = {"norm": matplotlib.colors.LogNorm(), "cmap": "jet_r"}

        if whichprimary == "air":
            mprimary = mair
        elif whichprimary == "overburden":
            mprimary = mover
        elif whichprimary == "half":
            mprimary = mhalf

        if Type == "Total":
            u = 1.0 / (mapping * mtrue)
        elif Type == "Primary":
            u = 1.0 / (mapping * mprimary)
        elif Type == "Secondary":
            u = 1.0 / (mapping * mtrue) - 1.0 / (mapping * mprimary)
            if Scale == "Log":
                linthresh = 10.0
                pcolorOpts = {
                    "norm": matplotlib.colors.SymLogNorm(
                        linthresh=linthresh, linscale=0.2
                    ),
                    "cmap": "jet_r",
                }

        # prepare for masking arrays - 'conventional' arrays won't do it
        u = np.ma.array(u)
        # mask values below a certain threshold
        u = np.ma.masked_where(mtrue <= np.log(1e-8), u)

    elif Field == "Potential":

        label = "Potential (V)"
        xtype = "CC"
        view = "real"
        streamOpts = None
        ind = indCC

        formatter = "%.1e"
        pcolorOpts = {"cmap": "viridis"}
        if Scale == "Log":
            linthresh = 10.0
            pcolorOpts = {
                "norm": matplotlib.colors.SymLogNorm(linthresh=linthresh, linscale=0.2),
                "cmap": "viridis",
            }

        if Type == "Total":
            # formatter = LogFormatter(10, labelOnlyBase =False)
            # pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh =10, linscale =0.1)}

            u = total_field[src, "phi"] - phiScaleTotal

        elif Type == "Primary":
            # formatter = LogFormatter(10, labelOnlyBase =False)
            # pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh =10, linscale =0.1)}

            u = primary_field[src, "phi"] - phiScalePrim

        elif Type == "Secondary":
            # formatter = None
            # pcolorOpts = {"cmap":"viridis"}

            uTotal = total_field[src, "phi"] - phiScaleTotal
            uPrim = primary_field[src, "phi"] - phiScalePrim
            u = uTotal - uPrim

    elif Field == "E":

        label = "Electric Field (V/m)"
        xtype = "F"
        view = "vec"
        streamOpts = {"color": "w"}
        ind = indF

        # formatter = LogFormatter(10, labelOnlyBase =False)
        pcolorOpts = {"cmap": "viridis"}
        if Scale == "Log":
            pcolorOpts = {"norm": matplotlib.colors.LogNorm(), "cmap": "viridis"}
        formatter = "%.1e"

        if Type == "Total":
            u = total_field[src, "e"]

        elif Type == "Primary":
            u = primary_field[src, "e"]

        elif Type == "Secondary":
            uTotal = total_field[src, "e"]
            uPrim = primary_field[src, "e"]
            u = uTotal - uPrim

    elif Field == "J":

        label = "Current density ($A/m^2$)"
        xtype = "F"
        view = "vec"
        streamOpts = {"color": "w"}
        ind = indF

        # formatter = LogFormatter(10, labelOnlyBase =False)
        pcolorOpts = {"cmap": "viridis"}
        if Scale == "Log":
            pcolorOpts = {"norm": matplotlib.colors.LogNorm(), "cmap": "viridis"}
        formatter = "%.1e"

        if Type == "Total":
            u = total_field[src, "j"]

        elif Type == "Primary":
            u = primary_field[src, "j"]

        elif Type == "Secondary":
            uTotal = total_field[src, "j"]
            uPrim = primary_field[src, "j"]
            u = uTotal - uPrim

    elif Field == "Charge":

        label = "Charge Density ($C/m^2$)"
        xtype = "CC"
        view = "real"
        streamOpts = None
        ind = indCC

        # formatter = LogFormatter(10, labelOnlyBase =False)
        pcolorOpts = {"cmap": "RdBu_r"}
        if Scale == "Log":
            linthresh = 1e-12
            pcolorOpts = {
                "norm": matplotlib.colors.SymLogNorm(linthresh=linthresh, linscale=0.2),
                "cmap": "RdBu_r",
            }
        formatter = "%.1e"

        if Type == "Total":
            u = total_field[src, "charge"]

        elif Type == "Primary":
            u = primary_field[src, "charge"]

        elif Type == "Secondary":
            uTotal = total_field[src, "charge"]
            uPrim = primary_field[src, "charge"]
            u = uTotal - uPrim

    elif Field == "Sensitivity":

        label = "Sensitivity"
        xtype = "CC"
        view = "real"
        streamOpts = None
        ind = indCC

        # formatter = None
        # pcolorOpts = {"cmap":"viridis"}
        # formatter = LogFormatter(10, labelOnlyBase =False)
        pcolorOpts = {"cmap": "viridis"}
        if Scale == "Log":
            linthresh = 1e-4
            pcolorOpts = {
                "norm": matplotlib.colors.SymLogNorm(linthresh=linthresh, linscale=0.2),
                "cmap": "viridis",
            }
        # formatter = formatter = "$10^{%.1f}$"
        formatter = "%.1e"

        if Type == "Total":
            u = getSensitivity(survey, A, B, M, N, mtrue)

        elif Type == "Primary":
            u = getSensitivity(survey, A, B, M, N, mhalf)

        elif Type == "Secondary":
            uTotal = getSensitivity(survey, A, B, M, N, mtrue)
            uPrim = getSensitivity(survey, A, B, M, N, mhalf)
            u = uTotal - uPrim
        # u = np.log10(abs(u))

    if Scale == "Log":
        eps = 1e-16
    else:
        eps = 0.0
    # print ind.shape
    # print u.shape
    # print xtype
    dat = meshcore.plotImage(
        u[ind] + eps,
        v_type=xtype,
        ax=ax[1],
        grid=False,
        view=view,
        stream_opts=streamOpts,
        pcolor_opts=pcolorOpts,
    )  # gridOpts ={'color':'k', 'alpha':0.5}

    # Get cylinder outline
    cylinderPoints = getCylinderPoints(xc, zc, ellips_a, ellips_b)

    if rhoTarget != rhohalf:
        # Get plate corners
        plateCorners = getPlateCorners(target_thick, target_wide, cylinderPoints)

        # plot top of plate outline
        ax[1].plot(
            plateCorners[[0, 1], 0],
            plateCorners[[0, 1], 1],
            linestyle="dashed",
            color="k",
        )
        # plot east side of plate outline
        ax[1].plot(
            plateCorners[[1, 3], 0],
            plateCorners[[1, 3], 1],
            linestyle="dashed",
            color="k",
        )
        # plot bottom of plate outline
        ax[1].plot(
            plateCorners[[2, 3], 0],
            plateCorners[[2, 3], 1],
            linestyle="dashed",
            color="k",
        )
        # plot west side of plate outline
        ax[1].plot(
            plateCorners[[0, 2], 0],
            plateCorners[[0, 2], 1],
            linestyle="dashed",
            color="k",
        )

    if rholayer != rhohalf:
        OverburdenPoints = get_OverburdenPoints(cylinderPoints, overburden_thick)
        if np.all(OverburdenPoints[:, 1] <= 0.0):
            ax[1].plot(
                OverburdenPoints[:, 0],
                OverburdenPoints[:, 1],
                linestyle="dashed",
                color="k",
            )
            ax[1].plot(
                OverburdenPoints[:, 0],
                OverburdenPoints[:, 1],
                linestyle="dashed",
                color="k",
            )
        idcyl = cylinderPoints[:, 1] <= 0.0
        ax[1].plot(
            cylinderPoints[idcyl, 0],
            cylinderPoints[idcyl, 1],
            linestyle="dashed",
            color="k",
        )

    # if (Field == 'Charge') and (Type != 'Primary') and (Type != 'Total'):
    #    qTotal = total_field['q']
    #    qPrim = primary_field['q']
    #    qSecondary = qTotal - qPrim
    #    qPosSum, qNegSum, qPosAvgLoc, qNegAvgLoc = sumCylinderCharges(xc, zc, r, qSecondary)
    #    ax[1].plot(qPosAvgLoc[0], qPosAvgLoc[1], marker = '.', color ='black', markersize = labelsize)
    #    ax[1].plot(qNegAvgLoc[0], qNegAvgLoc[1], marker = '.',  color ='black', markersize = labelsize)
    #    if(qPosAvgLoc[0] > qNegAvgLoc[0]):
    #        xytext_qPos = (qPosAvgLoc[0] + 1., qPosAvgLoc[1] - 0.5)
    #        xytext_qNeg = (qNegAvgLoc[0] - 15., qNegAvgLoc[1] - 0.5)
    #    else:
    #        xytext_qPos = (qPosAvgLoc[0] - 15., qPosAvgLoc[1] - 0.5)
    #        xytext_qNeg = (qNegAvgLoc[0] + 1., qNegAvgLoc[1] - 0.5)
    #    ax[1].annotate('+Q = %2.1e'%(qPosSum), xy =xytext_qPos, xytext =xytext_qPos , fontsize = labelsize)
    #    ax[1].annotate('-Q = %2.1e'%(qNegSum), xy =xytext_qNeg, xytext =xytext_qNeg , fontsize = labelsize)

    ax[1].set_xlabel("x (m)", fontsize=labelsize)
    ax[1].set_ylabel("z (m)", fontsize=labelsize)

    _, surfaceA = get_Surface(mtrue, A)
    _, surfaceB = get_Surface(mtrue, B)
    _, surfaceM = get_Surface(mtrue, M)
    _, surfaceN = get_Surface(mtrue, N)

    if survey == "Dipole-Dipole":

        ax[1].plot(A, surfaceA + 1.0, marker="v", color="red", markersize=labelsize)
        ax[1].plot(B, surfaceB + 1.0, marker="v", color="blue", markersize=labelsize)
        ax[1].plot(M, surfaceM + 1.0, marker="^", color="yellow", markersize=labelsize)
        ax[1].plot(N, surfaceN + 1.0, marker="^", color="green", markersize=labelsize)

        xytextA1 = (A - 0.5, surfaceA + 2.0)
        xytextB1 = (B - 0.5, surfaceB + 2.0)
        xytextM1 = (M - 0.5, surfaceM + 2.0)
        xytextN1 = (N - 0.5, surfaceN + 2.0)
        ax[1].annotate("A", xy=xytextA1, xytext=xytextA1, fontsize=labelsize)
        ax[1].annotate("B", xy=xytextB1, xytext=xytextB1, fontsize=labelsize)
        ax[1].annotate("M", xy=xytextM1, xytext=xytextM1, fontsize=labelsize)
        ax[1].annotate("N", xy=xytextN1, xytext=xytextN1, fontsize=labelsize)
    elif survey == "Pole-Dipole":
        ax[1].plot(A, surfaceA + 1.0, marker="v", color="red", markersize=labelsize)
        ax[1].plot(M, surfaceM + 1.0, marker="^", color="yellow", markersize=labelsize)
        ax[1].plot(N, surfaceN + 1.0, marker="^", color="green", markersize=labelsize)

        xytextA1 = (A - 0.5, surfaceA + 2.0)
        xytextM1 = (M - 0.5, surfaceM + 2.0)
        xytextN1 = (N - 0.5, surfaceN + 2.0)
        ax[1].annotate("A", xy=xytextA1, xytext=xytextA1, fontsize=labelsize)
        ax[1].annotate("M", xy=xytextM1, xytext=xytextM1, fontsize=labelsize)
        ax[1].annotate("N", xy=xytextN1, xytext=xytextN1, fontsize=labelsize)
    elif survey == "Dipole-Pole":
        ax[1].plot(A, surfaceA + 1.0, marker="v", color="red", markersize=labelsize)
        ax[1].plot(B, surfaceB + 1.0, marker="v", color="blue", markersize=labelsize)
        ax[1].plot(M, surfaceM + 1.0, marker="^", color="yellow", markersize=labelsize)

        xytextA1 = (A - 0.5, surfaceA + 2.0)
        xytextB1 = (B - 0.5, surfaceB + 2.0)
        xytextM1 = (M - 0.5, surfaceM + 2.0)
        ax[1].annotate("A", xy=xytextA1, xytext=xytextA1, fontsize=labelsize)
        ax[1].annotate("B", xy=xytextB1, xytext=xytextB1, fontsize=labelsize)
        ax[1].annotate("M", xy=xytextM1, xytext=xytextM1, fontsize=labelsize)
    elif survey == "Pole-Pole":
        ax[1].plot(A, surfaceA + 1.0, marker="v", color="red", markersize=labelsize)
        ax[1].plot(M, surfaceM + 1.0, marker="^", color="yellow", markersize=labelsize)

        xytextA1 = (A - 0.5, surfaceA + 2.0)
        xytextM1 = (M - 0.5, surfaceM + 2.0)
        ax[1].annotate("A", xy=xytextA1, xytext=xytextA1, fontsize=labelsize)
        ax[1].annotate("M", xy=xytextM1, xytext=xytextM1, fontsize=labelsize)

    ax[1].tick_params(axis="both", which="major", labelsize=ticksize)
    cbar_ax = fig.add_axes([0.8, 0.05, 0.08, 0.5])
    cbar_ax.axis("off")

    vmin, vmax = dat[0].get_clim()

    # if Field == 'Model':
    #    vmax =(np.r_[rhohalf, rholayer, rhoTarget]).max()

    if Scale == "Log":

        if (Field == "E") or (Field == "J"):
            cb = plt.colorbar(
                dat[0],
                ax=cbar_ax,
                format=formatter,
                ticks=np.logspace(np.log10(vmin), np.log10(vmax), 5),
            )

        elif Field == "Model":
            if Type == "Secondary":
                cb = plt.colorbar(
                    dat[0],
                    ax=cbar_ax,
                    format=formatter,
                    ticks=np.r_[np.minimum(0.0, vmin), np.maximum(0.0, vmax)],
                )
            else:
                cb = plt.colorbar(
                    dat[0],
                    ax=cbar_ax,
                    format=formatter,
                    ticks=np.logspace(np.log10(vmin), np.log10(vmax), 5),
                )

        else:
            cb = plt.colorbar(
                dat[0],
                ax=cbar_ax,
                format=formatter,
                ticks=np.r_[
                    -1.0
                    * np.logspace(np.log10(-vmin - eps), np.log10(linthresh), 3)[:-1],
                    0.0,
                    np.logspace(np.log10(linthresh), np.log10(vmax), 3)[1:],
                ],
            )
    else:
        if (Field == "Model") and (Type == "Secondary"):
            cb = plt.colorbar(
                dat[0],
                ax=cbar_ax,
                format=formatter,
                ticks=np.r_[np.minimum(0.0, vmin), np.maximum(0.0, vmax)],
            )
        else:
            cb = plt.colorbar(
                dat[0], ax=cbar_ax, format=formatter, ticks=np.linspace(vmin, vmax, 5)
            )
    # t_logloc = matplotlib.ticker.LogLocator(base =10.0, subs =[1.0, 2.], numdecs =4, numticks =8)
    # tick_locator = matplotlib.ticker.SymmetricalLogLocator(t_logloc)
    # cb.locator = tick_locator
    # cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    # cb.update_ticks()
    cb.ax.tick_params(labelsize=ticksize)
    cb.set_label(label, fontsize=labelsize)
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    # ax[1].set_aspect('equal')

    plt.show()


def valley_app():
    app = widgetify(
        PLOT,
        survey=ToggleButtons(
            options=["Dipole-Dipole", "Dipole-Pole", "Pole-Dipole", "Pole-Pole"],
            value="Dipole-Dipole",
        ),
        xc=FloatSlider(
            min=-1005.0, max=1000.0, step=10.0, value=0.0
        ),  # , continuous_update=False),
        zc=FloatSlider(
            min=-1000.0, max=1000.0, step=10.0, value=250.0
        ),  # , continuous_update=False),
        ellips_a=FloatSlider(
            min=10.0, max=10000.0, step=100.0, value=1000.0
        ),  # , continuous_update=False),
        ellips_b=FloatSlider(
            min=10.0, max=10000.0, step=100.0, value=500.0
        ),  # , continuous_update=False),
        rhohalf=FloatText(
            min=1e-8, max=1e8, value=1000.0, description="$\\rho_1$"
        ),  # , continuous_update=False, description='$\\rho_1$'),
        rholayer=FloatText(
            min=1e-8, max=1e8, value=100.0, description="$\\rho_2$"
        ),  # , continuous_update=False, description='$\\rho_2$'),
        rhoTarget=FloatText(
            min=1e-8, max=1e8, value=500.0, description="$\\rho_3$"
        ),  # , continuous_update=False, description='$\\rho_3$'),
        overburden_thick=FloatSlider(
            min=0.0, max=1000.0, step=10.0, value=200.0
        ),  # , continuous_update=False),
        overburden_wide=fixed(2000.0),  # , continuous_update=False),
        target_thick=FloatSlider(
            min=0.0, max=1000.0, step=10.0, value=200.0
        ),  # , continuous_update=False),
        target_wide=FloatSlider(
            min=0.0, max=1000.0, step=10.0, value=200.0
        ),  # , continuous_update=False),
        A=FloatSlider(
            min=-1010.0, max=1010.0, step=20.0, value=-510.0
        ),  # , continuous_update=False),
        B=FloatSlider(
            min=-1010.0, max=1010.0, step=20.0, value=510.0
        ),  # , continuous_update=False),
        M=FloatSlider(
            min=-1010.0, max=1010.0, step=20.0, value=-210.0
        ),  # , continuous_update=False),
        N=FloatSlider(
            min=-1010.0, max=1010.0, step=20.0, value=210.0
        ),  # , continuous_update=False),
        Field=ToggleButtons(
            options=["Model", "Potential", "E", "J", "Charge", "Sensitivity"], value="J"
        ),
        whichprimary=ToggleButtons(options=["air", "overburden"], value="overburden"),
        Type=ToggleButtons(options=["Total", "Primary", "Secondary"], value="Total"),
        Scale=ToggleButtons(options=["Linear", "Log"], value="Log"),
    )
    return app


if __name__ == "__main__":
    app = valley_app()
