from discretize import TensorMesh
from SimPEG import maps, SolverLU, utils
from SimPEG.utils import extract_core_mesh
import numpy as np
from SimPEG.electromagnetics.static import resistivity as DC
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import LogFormatter
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.constants import epsilon_0
import copy
from pymatsolver import Pardiso

from ipywidgets import interact, IntSlider, FloatSlider, FloatText, ToggleButtons

from ..base import widgetify

# Mesh, sigmaMap can be globals global
npad = 15
growrate = 2.0
cs = 0.5
hx = [(cs, npad, -growrate), (cs, 200), (cs, npad, growrate)]
hy = [(cs, npad, -growrate), (cs, 100)]
mesh = TensorMesh([hx, hy], "CN")
idmap = maps.IdentityMap(mesh)
sigmaMap = idmap
dx = 5
xr = np.arange(-40, 41, dx)
dxr = np.diff(xr)
xmin = -40.0
xmax = 40.0
ymin = -40.0
ymax = 8.0
xylim = np.c_[[xmin, ymin], [xmax, ymax]]
indCC, meshcore = extract_core_mesh(xylim, mesh)
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
    "zcLayer": None,
    "dzLayer": None,
    "xc": None,
    "zc": None,
    "r": None,
    "sigLayer": None,
    "sigTarget": None,
    "sigHalf": None,
}


def model_fields(A, B, zcLayer, dzLayer, xc, zc, r, sigLayer, sigTarget, sigHalf):
    re_run = (
        _cache["A"] != A
        or _cache["B"] != B
        or _cache["zcLayer"] != zcLayer
        or _cache["dzLayer"] != dzLayer
        or _cache["xc"] != xc
        or _cache["zc"] != zc
        or _cache["r"] != r
        or _cache["sigLayer"] != sigLayer
        or _cache["sigTarget"] != sigTarget
        or _cache["sigHalf"] != sigHalf
    )
    if re_run:
        # Create halfspace model
        mhalf = sigHalf * np.ones([mesh.nC])
        # Add layer to model
        mLayer = addLayer2Mod(zcLayer, dzLayer, mhalf, sigLayer)
        # Add plate or cylinder
        mtrue = addCylinder2Mod(xc, zc, r, mLayer, sigTarget)

        Mx = np.empty(shape=(0, 2))
        Nx = np.empty(shape=(0, 2))
        rx = DC.receivers.Dipole(Mx, Nx)
        if B == []:
            src = DC.sources.Pole([rx], np.r_[A, 0.0])
        else:
            src = DC.sources.Dipole([rx], np.r_[A, 0.0], np.r_[B, 0.0])

        survey = DC.Survey([src])

        problem = DC.Simulation3DCellCentered(
            mesh, sigmaMap=sigmaMap, solver=Pardiso, survey=survey
        )
        problem_prim = DC.Simulation3DCellCentered(
            mesh, sigmaMap=sigmaMap, solver=Pardiso, survey=survey
        )

        primary_field = problem_prim.fields(mhalf)

        total_field = problem.fields(mtrue)

        _cache["A"] = A
        _cache["B"] = B
        _cache["zcLayer"] = zcLayer
        _cache["dzLayer"] = dzLayer
        _cache["xc"] = xc
        _cache["zc"] = zc
        _cache["r"] = r
        _cache["sigLayer"] = sigLayer
        _cache["sigTarget"] = sigTarget
        _cache["sigHalf"] = sigHalf

        _cache["mtrue"] = mtrue
        _cache["mhalf"] = mhalf
        _cache["src"] = src
        _cache["total_field"] = total_field
        _cache["primary_field"] = primary_field
    else:
        mtrue = _cache["mtrue"]
        mhalf = _cache["mhalf"]
        src = _cache["src"]
        total_field = _cache["total_field"]
        primary_field = _cache["primary_field"]

    return mtrue, mhalf, src, primary_field, total_field


def addLayer2Mod(zcLayer, dzLayer, modd, sigLayer):

    CCLocs = mesh.gridCC
    mod = copy.copy(modd)

    zmax = zcLayer + dzLayer / 2.0
    zmin = zcLayer - dzLayer / 2.0

    belowInd = np.where(CCLocs[:, 1] <= zmax)[0]
    aboveInd = np.where(CCLocs[:, 1] >= zmin)[0]
    layerInds = list(set(belowInd).intersection(aboveInd))

    # # Check selected cell centers by plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.scatter(CCLocs[layerInds,0],CCLocs[layerInds,1])
    # ax.set_xlim(-40,40)
    # ax.set_ylim(-35,0)
    # plt.axes().set_aspect('equal')
    # plt.show()

    mod[layerInds] = sigLayer
    return mod


def getCylinderPoints(xc, zc, r):
    angle = np.linspace(-np.pi, np.pi, 250)
    xs = np.cos(angle) * r + xc
    zs = np.sin(angle) * r + zc
    return np.c_[xs, zs]


def addCylinder2Mod(xc, zc, r, modd, sigCylinder):

    # Get points for cylinder outline
    cylinderPoints = getCylinderPoints(xc, zc, r)
    mod = copy.copy(modd)

    verts = []
    codes = []
    for ii in range(0, cylinderPoints.shape[0]):
        verts.append(cylinderPoints[ii, :])

        if ii == 0:
            codes.append(Path.MOVETO)
        elif ii == cylinderPoints.shape[0] - 1:
            codes.append(Path.CLOSEPOLY)
        else:
            codes.append(Path.LINETO)

    path = Path(verts, codes)
    CCLocs = mesh.gridCC
    insideInd = np.where(path.contains_points(CCLocs))

    # #Check selected cell centers by plotting
    # # print insideInd
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # patch = patches.PathPatch(path, facecolor='none', lw=2)
    # ax.add_patch(patch)
    # plt.scatter(CCLocs[insideInd,0],CCLocs[insideInd,1])
    # ax.set_xlim(-40,40)
    # ax.set_ylim(-35,0)
    # plt.axes().set_aspect('equal')
    # plt.show()

    mod[insideInd] = sigCylinder
    return mod


# def getPlateCorners(xc, zc, dx, dz, rotAng):

#     # Form rotation matix
#     rotMat = np.array([[np.cos(rotAng*(np.pi/180.)), -np.sin(rotAng*(np.pi/180.))],[np.sin(rotAng*(np.pi/180.)), np.cos(rotAng*(np.pi/180.))]])
#     originCorners = np.array([[-0.5*dx, 0.5*dz], [0.5*dx, 0.5*dz], [-0.5*dx, -0.5*dz], [0.5*dx, -0.5*dz]])

#     rotPlateCorners = np.dot(originCorners,rotMat)
#     plateCorners = rotPlateCorners + np.hstack([np.repeat(xc,4).reshape([4,1]),np.repeat(zc,4).reshape([4,1])])
#     return plateCorners


# def addPlate2Mod(xc, zc, dx, dz, rotAng, mod, sigPlate):
#     # use matplotlib paths to find CC inside of polygon
#     plateCorners = getPlateCorners(xc,zc,dx,dz,rotAng)

#     verts = [
#         (plateCorners[0,:]), # left, top
#         (plateCorners[1,:]), # right, top
#         (plateCorners[3,:]), # right, bottom
#         (plateCorners[2,:]), # left, bottom
#         (plateCorners[0,:]), # left, top (closes polygon)
#         ]

#     codes = [Path.MOVETO,
#              Path.LINETO,
#              Path.LINETO,
#              Path.LINETO,
#              Path.CLOSEPOLY,
#              ]

#     path = Path(verts, codes)
#     CCLocs = mesh.gridCC
#     insideInd = np.where(path.contains_points(CCLocs))

#     #Check selected cell centers by plotting
#     # print insideInd
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     # patch = patches.PathPatch(path, facecolor='none', lw=2)
#     # ax.add_patch(patch)
#     # plt.scatter(CCLocs[insideInd,0],CCLocs[insideInd,1])
#     # ax.set_xlim(-10,10)
#     # ax.set_ylim(-20,0)
#     # plt.axes().set_aspect('equal')
#     # plt.show()

#     mod[insideInd] = sigPlate
#     return mod


def get_Surface_Potentials(survey, src, field_obj):

    phi = field_obj[src, "phi"]
    CCLoc = mesh.gridCC
    zsurfaceLoc = np.max(CCLoc[:, 1])
    surfaceInd = np.where(CCLoc[:, 1] == zsurfaceLoc)
    phiSurface = phi[surfaceInd]
    xSurface = CCLoc[surfaceInd, 0].T
    phiScale = 0.0

    if survey == "Pole-Dipole" or survey == "Pole-Pole":
        refInd = utils.closestPoints(mesh, [xmax + 60.0, 0.0], gridLoc="CC")
        # refPoint =  CCLoc[refInd]
        # refSurfaceInd = np.where(xSurface == refPoint[0])
        # phiScale = np.median(phiSurface)
        phiScale = phi[refInd]
        phiSurface = phiSurface - phiScale

    return xSurface, phiSurface, phiScale


def sumCylinderCharges(xc, zc, r, qSecondary):
    chargeRegionVerts = getCylinderPoints(xc, zc, r + 0.5)

    codes = chargeRegionVerts.shape[0] * [Path.LINETO]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY

    chargeRegionPath = Path(chargeRegionVerts, codes)
    CCLocs = mesh.gridCC
    chargeRegionInsideInd = np.where(chargeRegionPath.contains_points(CCLocs))

    plateChargeLocs = CCLocs[chargeRegionInsideInd]
    plateCharge = qSecondary[chargeRegionInsideInd]
    posInd = np.where(plateCharge >= 0)
    negInd = np.where(plateCharge < 0)
    qPos = utils.mkvc(plateCharge[posInd])
    qNeg = utils.mkvc(plateCharge[negInd])

    qPosLoc = plateChargeLocs[posInd, :][0]
    qNegLoc = plateChargeLocs[negInd, :][0]

    # qPosData = np.vstack([qPosLoc[:, 0], qPosLoc[:, 1], qPos]).T
    # qNegData = np.vstack([qNegLoc[:, 0], qNegLoc[:, 1], qNeg]).T

    if qNeg.shape == (0,) or qPos.shape == (0,):
        qNegAvgLoc = np.r_[-10, -10]
        qPosAvgLoc = np.r_[+10, -10]
    else:
        qNegAvgLoc = np.average(qNegLoc, axis=0, weights=qNeg)
        qPosAvgLoc = np.average(qPosLoc, axis=0, weights=qPos)

    qPosSum = np.sum(qPos)
    qNegSum = np.sum(qNeg)

    # # Check things by plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # platePatch = patches.PathPatch(platePath, facecolor='none', lw=2)
    # ax.add_patch(platePatch)
    # chargeRegionPatch = patches.PathPatch(chargeRegionPath, facecolor='none', lw=2)
    # ax.add_patch(chargeRegionPatch)
    # plt.scatter(qNegAvgLoc[0],qNegAvgLoc[1],color='b')
    # plt.scatter(qPosAvgLoc[0],qPosAvgLoc[1],color='r')
    # ax.set_xlim(-15,5)
    # ax.set_ylim(-25,-5)
    # plt.axes().set_aspect('equal')
    # plt.show()

    return qPosSum, qNegSum, qPosAvgLoc, qNegAvgLoc


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

    survey = DC.survey.Survey([src])
    problem = DC.Simulation3DCellCentered(
        mesh, survey=survey, sigmaMap=sigmaMap, solver=Pardiso
    )
    J = problem.getJ(model)

    return J[0]


def calculateRhoA(survey, VM, VN, A, B, M, N):

    eps = 1e-9  # to stabilize division

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


# Inline functions for computing apparent resistivity
# eps = 1e-9 #to stabilize division
# G = lambda A, B, M, N: 1. / ( 1./(np.abs(A-M)+eps) - 1./(np.abs(M-B)+eps) - 1./(np.abs(N-A)+eps) + 1./(np.abs(N-B)+eps) )
# rho_a = lambda VM,VN, A,B,M,N: (VM-VN)*2.*np.pi*G(A,B,M,N)


def plot_Surface_Potentials(
    survey,
    A,
    B,
    M,
    N,
    zcLayer,
    dzLayer,
    xc,
    zc,
    r,
    rhoHalf,
    rhoLayer,
    rhoTarget,
    Field,
    Type,
    Scale,
):
    labelsize = 16.0
    ticksize = 16.0

    sigTarget = 1.0 / rhoTarget
    # rhoLayer = np.exp(logRhoLayer)
    sigLayer = 1.0 / rhoLayer
    sigHalf = 1.0 / rhoHalf

    if survey == "Pole-Dipole" or survey == "Pole-Pole":
        B = []

    mtrue, mhalf, src, primary_field, total_field = model_fields(
        A, B, zcLayer, dzLayer, xc, zc, r, sigLayer, sigTarget, sigHalf
    )

    fig, ax = plt.subplots(2, 1, figsize=(9 * 1.5, 9 * 1.8), sharex=True)
    fig.subplots_adjust(right=0.8, wspace=0.05, hspace=0.05)

    xSurface, phiTotalSurface, phiScaleTotal = get_Surface_Potentials(
        survey, src, total_field
    )
    xSurface, phiPrimSurface, phiScalePrim = get_Surface_Potentials(
        survey, src, primary_field
    )
    ylim = np.r_[-1.0, 1.0] * np.max(np.abs(phiTotalSurface))
    xlim = np.array([-40, 40])

    if survey == "Dipole-Pole" or survey == "Pole-Pole":
        MInd = np.where(xSurface == M)
        N = []

        VM = phiTotalSurface[MInd[0]]
        VN = 0.0

        VMprim = phiPrimSurface[MInd[0]]
        VNprim = 0.0

    else:
        MInd = np.where(xSurface == M)
        NInd = np.where(xSurface == N)

        VM = phiTotalSurface[MInd[0]]
        VN = phiTotalSurface[NInd[0]]

        VMprim = phiPrimSurface[MInd[0]]
        VNprim = phiPrimSurface[NInd[0]]

    # 2D geometric factor
    G2D = rhoHalf / (calculateRhoA(survey, VMprim, VNprim, A, B, M, N))

    ax[0].plot(xSurface, phiTotalSurface, color=[0.1, 0.5, 0.1], linewidth=2)
    ax[0].plot(xSurface, phiPrimSurface, linestyle="dashed", linewidth=0.5, color="k")
    ax[0].grid(
        which="both", linestyle="-", linewidth=0.5, color=[0.2, 0.2, 0.2], alpha=0.5
    )

    if survey == "Pole-Dipole" or survey == "Pole-Pole":
        ax[0].plot(A, 0, "+", markersize=12, markeredgewidth=3, color=[1.0, 0.0, 0])
    else:
        ax[0].plot(A, 0, "+", markersize=12, markeredgewidth=3, color=[1.0, 0.0, 0])
        ax[0].plot(B, 0, "_", markersize=12, markeredgewidth=3, color=[0.0, 0.0, 1.0])
    ax[0].set_ylabel("Potential, (V)", fontsize=14)
    ax[0].set_xlabel("x (m)", fontsize=14)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)

    if survey == "Dipole-Pole" or survey == "Pole-Pole":
        ax[0].plot(M, VM, "o", color="k")

        xytextM = (M + 0.5, max(min(VM, ylim.max()), ylim.min()) + 0.5)
        ax[0].annotate("%2.1e" % (VM), xy=xytextM, xytext=xytextM, fontsize=labelsize)

    else:
        ax[0].plot(M, VM, "o", color="k")
        ax[0].plot(N, VN, "o", color="k")

        xytextM = (M + 0.5, max(min(VM, ylim.max()), ylim.min()) + 1.0)
        xytextN = (N + 0.5, max(min(VN, ylim.max()), ylim.min()) + 1.0)
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
        fontsize=labelsize,
    )

    ax[0].legend(
        ["Model Potential", "Layered Earth Potential"], loc=3, fontsize=labelsize
    )

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

        if Type == "Total":
            u = 1.0 / (sigmaMap * mtrue)
        elif Type == "Primary":
            u = 1.0 / (sigmaMap * mhalf)
        elif Type == "Secondary":
            u = 1.0 / (sigmaMap * mtrue) - 1.0 / (sigmaMap * mhalf)
            if Scale == "Log":
                linthresh = 10.0
                pcolorOpts = {
                    "norm": matplotlib.colors.SymLogNorm(
                        linthresh=linthresh, linscale=0.2
                    ),
                    "cmap": "jet_r",
                }

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
            # formatter = LogFormatter(10, labelOnlyBase=False)
            # pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=10, linscale=0.1)}

            u = total_field[src, "phi"] - phiScaleTotal

        elif Type == "Primary":
            # formatter = LogFormatter(10, labelOnlyBase=False)
            # pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=10, linscale=0.1)}

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

        # formatter = LogFormatter(10, labelOnlyBase=False)
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

        # formatter = LogFormatter(10, labelOnlyBase=False)
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

        # formatter = LogFormatter(10, labelOnlyBase=False)
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
        # formatter = LogFormatter(10, labelOnlyBase=False)
        pcolorOpts = {"cmap": "viridis"}
        if Scale == "Log":
            linthresh = 1.0
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

    if Scale == "Log":
        eps = 1e-16
    else:
        eps = 0.0
    dat = meshcore.plot_image(
        u[ind] + eps,
        v_type=xtype,
        ax=ax[1],
        grid=False,
        view=view,
        stream_opts=streamOpts,
        pcolor_opts=pcolorOpts,
    )  # gridOpts={'color':'k', 'alpha':0.5}

    # Get cylinder outline
    cylinderPoints = getCylinderPoints(xc, zc, r)

    if rhoTarget != rhoHalf:
        ax[1].plot(
            cylinderPoints[:, 0], cylinderPoints[:, 1], linestyle="dashed", color="k"
        )

    if rhoLayer != rhoHalf:
        layerX = np.arange(xmin, xmax + 1)
        layerTopY = (zcLayer + dzLayer / 2.0) * np.ones_like(layerX)
        layerBottomY = (zcLayer - dzLayer / 2.0) * np.ones_like(layerX)
        ax[1].plot(layerX, layerTopY, linestyle="dashed", color="k")
        ax[1].plot(layerX, layerBottomY, linestyle="dashed", color="k")

    if (Field == "Charge") and (Type != "Primary") and (Type != "Total"):
        qTotal = total_field[src, "charge"]
        qPrim = primary_field[src, "charge"]
        qSecondary = qTotal - qPrim
        qPosSum, qNegSum, qPosAvgLoc, qNegAvgLoc = sumCylinderCharges(
            xc, zc, r, qSecondary
        )
        ax[1].plot(
            qPosAvgLoc[0],
            qPosAvgLoc[1],
            marker=".",
            color="black",
            markersize=labelsize,
        )
        ax[1].plot(
            qNegAvgLoc[0],
            qNegAvgLoc[1],
            marker=".",
            color="black",
            markersize=labelsize,
        )
        if qPosAvgLoc[0] > qNegAvgLoc[0]:
            xytext_qPos = (qPosAvgLoc[0] + 1.0, qPosAvgLoc[1] - 0.5)
            xytext_qNeg = (qNegAvgLoc[0] - 15.0, qNegAvgLoc[1] - 0.5)
        else:
            xytext_qPos = (qPosAvgLoc[0] - 15.0, qPosAvgLoc[1] - 0.5)
            xytext_qNeg = (qNegAvgLoc[0] + 1.0, qNegAvgLoc[1] - 0.5)
        ax[1].annotate(
            "+Q = %2.1e" % (qPosSum),
            xy=xytext_qPos,
            xytext=xytext_qPos,
            fontsize=labelsize,
        )
        ax[1].annotate(
            "-Q = %2.1e" % (qNegSum),
            xy=xytext_qNeg,
            xytext=xytext_qNeg,
            fontsize=labelsize,
        )

    ax[1].set_xlabel("x (m)", fontsize=labelsize)
    ax[1].set_ylabel("z (m)", fontsize=labelsize)

    if survey == "Dipole-Dipole":
        ax[1].plot(A, 1.0, marker="v", color="red", markersize=labelsize)
        ax[1].plot(B, 1.0, marker="v", color="blue", markersize=labelsize)
        ax[1].plot(M, 1.0, marker="^", color="yellow", markersize=labelsize)
        ax[1].plot(N, 1.0, marker="^", color="green", markersize=labelsize)

        xytextA1 = (A - 0.5, 2.5)
        xytextB1 = (B - 0.5, 2.5)
        xytextM1 = (M - 0.5, 2.5)
        xytextN1 = (N - 0.5, 2.5)
        ax[1].annotate("A", xy=xytextA1, xytext=xytextA1, fontsize=labelsize)
        ax[1].annotate("B", xy=xytextB1, xytext=xytextB1, fontsize=labelsize)
        ax[1].annotate("M", xy=xytextM1, xytext=xytextM1, fontsize=labelsize)
        ax[1].annotate("N", xy=xytextN1, xytext=xytextN1, fontsize=labelsize)
    elif survey == "Pole-Dipole":
        ax[1].plot(A, 1.0, marker="v", color="red", markersize=labelsize)
        ax[1].plot(M, 1.0, marker="^", color="yellow", markersize=labelsize)
        ax[1].plot(N, 1.0, marker="^", color="green", markersize=labelsize)

        xytextA1 = (A - 0.5, 2.5)
        xytextM1 = (M - 0.5, 2.5)
        xytextN1 = (N - 0.5, 2.5)
        ax[1].annotate("A", xy=xytextA1, xytext=xytextA1, fontsize=labelsize)
        ax[1].annotate("M", xy=xytextM1, xytext=xytextM1, fontsize=labelsize)
        ax[1].annotate("N", xy=xytextN1, xytext=xytextN1, fontsize=labelsize)
    elif survey == "Dipole-Pole":
        ax[1].plot(A, 1.0, marker="v", color="red", markersize=labelsize)
        ax[1].plot(B, 1.0, marker="v", color="blue", markersize=labelsize)
        ax[1].plot(M, 1.0, marker="^", color="yellow", markersize=labelsize)

        xytextA1 = (A - 0.5, 2.5)
        xytextB1 = (B - 0.5, 2.5)
        xytextM1 = (M - 0.5, 2.5)
        ax[1].annotate("A", xy=xytextA1, xytext=xytextA1, fontsize=labelsize)
        ax[1].annotate("B", xy=xytextB1, xytext=xytextB1, fontsize=labelsize)
        ax[1].annotate("M", xy=xytextM1, xytext=xytextM1, fontsize=labelsize)
    elif survey == "Pole-Pole":
        ax[1].plot(A, 1.0, marker="v", color="red", markersize=labelsize)
        ax[1].plot(M, 1.0, marker="^", color="yellow", markersize=labelsize)

        xytextA1 = (A - 0.5, 2.5)
        xytextM1 = (M - 0.5, 2.5)
        ax[1].annotate("A", xy=xytextA1, xytext=xytextA1, fontsize=labelsize)
        ax[1].annotate("M", xy=xytextM1, xytext=xytextM1, fontsize=labelsize)

    ax[1].tick_params(axis="both", which="major", labelsize=ticksize)
    cbar_ax = fig.add_axes([0.8, 0.05, 0.08, 0.5])
    cbar_ax.axis("off")
    vmin, vmax = dat[0].get_clim()
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

    cb.ax.tick_params(labelsize=ticksize)
    cb.set_label(label, fontsize=labelsize)
    ax[1].set_xlim([-40.0, 40.0])
    ax[1].set_ylim([-40.0, 8.0])
    ax[1].set_aspect("equal")

    plt.show()


def ResLayer_app():
    app = widgetify(
        plot_Surface_Potentials,
        survey=ToggleButtons(
            options=["Dipole-Dipole", "Dipole-Pole", "Pole-Dipole", "Pole-Pole"],
            value="Dipole-Dipole",
        ),
        zcLayer=FloatSlider(
            min=-10.0,
            max=0.0,
            step=1.0,
            value=-10.0,
            continuous_update=False,
            description="$zc_{layer}$",
        ),
        dzLayer=FloatSlider(
            min=0.5,
            max=5.0,
            step=0.5,
            value=1.0,
            continuous_update=False,
            description="$dz_{layer}$",
        ),
        rhoLayer=FloatText(
            min=1e-8,
            max=1e8,
            value=5000.0,
            continuous_update=False,
            description="$\\rho_{2}$",
        ),
        xc=FloatSlider(
            min=-30.0, max=30.0, step=1.0, value=0.0, continuous_update=False
        ),
        zc=FloatSlider(
            min=-30.0, max=-15.0, step=0.5, value=-25.0, continuous_update=False
        ),
        r=FloatSlider(min=1.0, max=10.0, step=0.5, value=5.0, continuous_update=False),
        rhoHalf=FloatText(
            min=1e-8,
            max=1e8,
            value=500.0,
            continuous_update=False,
            description="$\\rho_{1}$",
        ),
        rhoTarget=FloatText(
            min=1e-8,
            max=1e8,
            value=500.0,
            continuous_update=False,
            description="$\\rho_{3}$",
        ),
        A=FloatSlider(
            min=-30.25, max=30.25, step=0.5, value=-30.25, continuous_update=False
        ),
        B=FloatSlider(
            min=-30.25, max=30.25, step=0.5, value=30.25, continuous_update=False
        ),
        M=FloatSlider(
            min=-30.25, max=30.25, step=0.5, value=-10.25, continuous_update=False
        ),
        N=FloatSlider(
            min=-30.25, max=30.25, step=0.5, value=10.25, continuous_update=False
        ),
        Field=ToggleButtons(
            options=["Model", "Potential", "E", "J", "Charge", "Sensitivity"],
            value="Model",
        ),
        Type=ToggleButtons(options=["Total", "Primary", "Secondary"], value="Total"),
        Scale=ToggleButtons(options=["Linear", "Log"], value="Linear"),
    )
    return app
