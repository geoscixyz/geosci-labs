from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from SimPEG import Mesh, Maps, SolverLU, Utils
from SimPEG.Utils import ExtractCoreMesh
import numpy as np
from SimPEG.EM.Static import DC
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import LogFormatter
from matplotlib.path import Path
import matplotlib.patches as patches
from scipy.constants import epsilon_0
from scipy.ndimage.measurements import center_of_mass

from ipywidgets import IntSlider, FloatSlider, FloatText, ToggleButtons

from .Base import widgetify

# Mesh, mapping can be globals global
npad = 15
growrate = 2.
cs = 0.5
hx = [(cs,npad, -growrate),(cs,200),(cs,npad, growrate)]
hy = [(cs,npad, -growrate),(cs,100)]
mesh = Mesh.TensorMesh([hx, hy], "CN")
expmap = Maps.ExpMap(mesh)
mapping = expmap
dx = 5
xr = np.arange(-40,41,dx)
dxr = np.diff(xr)
xmin = -40.
xmax = 40.
ymin = -40.
ymax = 5.
xylim = np.c_[[xmin,ymin],[xmax,ymax]]
indCC, meshcore = ExtractCoreMesh(xylim,mesh)
indx = (mesh.gridFx[:,0]>=xmin) & (mesh.gridFx[:,0]<=xmax) \
    & (mesh.gridFx[:,1]>=ymin) & (mesh.gridFx[:,1]<=ymax)
indy = (mesh.gridFy[:,0]>=xmin) & (mesh.gridFy[:,0]<=xmax) \
    & (mesh.gridFy[:,1]>=ymin) & (mesh.gridFy[:,1]<=ymax)
indF = np.concatenate((indx,indy))


def plate_fields(A, B, dx, dz, xc, zc, rotAng, sigplate, sighalf):
    # Create halfspace model
    mhalf = np.log(sighalf*np.ones([mesh.nC,]))

    #Create true model with plate
    mtrue = createPlateMod(xc,zc,dx,dz,rotAng,sigplate,sighalf)

    Mx = np.empty(shape=(0, 2))
    Nx = np.empty(shape=(0, 2))
    rx = DC.Rx.Dipole(Mx,Nx)
    if(B == []):
        src = DC.Src.Pole([rx], np.r_[A,0.])
    else:
        src = DC.Src.Dipole([rx], np.r_[A,0.], np.r_[B,0.])


    survey = DC.Survey([src])
    survey_prim = DC.Survey([src])

    problem = DC.Problem3D_CC(mesh, sigmaMap = mapping)
    problem_prim = DC.Problem3D_CC(mesh, sigmaMap = mapping)
    problem.Solver = SolverLU
    problem_prim.Solver = SolverLU
    problem.pair(survey)
    problem_prim.pair(survey_prim)

    primary_field = problem_prim.fields(mhalf)

    total_field = problem.fields(mtrue)

    return mtrue,mhalf, src, primary_field, total_field



def getPlateCorners(xc, zc, dx, dz, rotAng):

    # Form rotation matix
    rotMat = np.array([[np.cos(rotAng*(np.pi/180.)), -np.sin(rotAng*(np.pi/180.))],[np.sin(rotAng*(np.pi/180.)), np.cos(rotAng*(np.pi/180.))]])
    originCorners = np.array([[-0.5*dx, 0.5*dz], [0.5*dx, 0.5*dz], [-0.5*dx, -0.5*dz], [0.5*dx, -0.5*dz]])

    rotPlateCorners = np.dot(originCorners,rotMat)
    plateCorners = rotPlateCorners + np.hstack([np.repeat(xc,4).reshape([4,1]),np.repeat(zc,4).reshape([4,1])])
    return plateCorners


def createPlateMod(xc, zc, dx, dz, rotAng, sigplate, sighalf):
    # use matplotlib paths to find CC inside of polygon
    plateCorners = getPlateCorners(xc,zc,dx,dz,rotAng)

    verts = [
        (plateCorners[0,:]), # left, top
        (plateCorners[1,:]), # right, top
        (plateCorners[3,:]), # right, bottom
        (plateCorners[2,:]), # left, bottom
        (plateCorners[0,:]), # left, top (closes polygon)
        ]

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]

    path = Path(verts, codes)
    CCLocs = mesh.gridCC
    insideInd = np.where(path.contains_points(CCLocs))

    #Check selected cell centers by plotting
    # print insideInd
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # patch = patches.PathPatch(path, facecolor='none', lw=2)
    # ax.add_patch(patch)
    # plt.scatter(CCLocs[insideInd,0],CCLocs[insideInd,1])
    # ax.set_xlim(-10,10)
    # ax.set_ylim(-20,0)
    # plt.axes().set_aspect('equal')
    # plt.show()

    mtrue = sighalf*np.ones([mesh.nC,])
    mtrue[insideInd] = sigplate
    mtrue = np.log(mtrue)
    return mtrue


def get_Surface_Potentials(survey, src, field_obj):

    phi = field_obj[src, 'phi']
    CCLoc = mesh.gridCC
    zsurfaceLoc = np.max(CCLoc[:,1])
    surfaceInd = np.where(CCLoc[:,1] == zsurfaceLoc)
    xSurface = CCLoc[surfaceInd,0].T
    phiSurface = phi[surfaceInd]
    phiScale = 0.

    if(survey == "Pole-Dipole" or survey == "Pole-Pole"):
        refInd = Utils.closestPoints(mesh, [xmax+60.,0.], gridLoc='CC')
        # refPoint =  CCLoc[refInd]
        # refSurfaceInd = np.where(xSurface == refPoint[0])
        # phiScale = np.median(phiSurface)
        phiScale = phi[refInd]
        phiSurface = phiSurface - phiScale

    return xSurface,phiSurface,phiScale


def sumPlateCharges(xc, zc, dx, dz, rotAng, qSecondary):
    # plateCorners = getPlateCorners(xc,zc,dx,dz,rotAng)
    chargeRegionCorners = getPlateCorners(xc,zc,dx+1.,dz+1.,rotAng)

    # plateVerts = [
    #     (plateCorners[0,:]), # left, top
    #     (plateCorners[1,:]), # right, top
    #     (plateCorners[3,:]), # right, bottom
    #     (plateCorners[2,:]), # left, bottom
    #     (plateCorners[0,:]), # left, top (closes polygon)
    #     ]

    chargeRegionVerts = [
        (chargeRegionCorners[0,:]), # left, top
        (chargeRegionCorners[1,:]), # right, top
        (chargeRegionCorners[3,:]), # right, bottom
        (chargeRegionCorners[2,:]), # left, bottom
        (chargeRegionCorners[0,:]), # left, top (closes polygon)
        ]

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]

    # platePath = Path(plateVerts, codes)
    chargeRegionPath = Path(chargeRegionVerts, codes)
    CCLocs = mesh.gridCC
    # plateInsideInd = np.where(platePath.contains_points(CCLocs))
    chargeRegionInsideInd = np.where(chargeRegionPath.contains_points(CCLocs))

    plateChargeLocs = CCLocs[chargeRegionInsideInd]
    plateCharge = qSecondary[chargeRegionInsideInd]
    posInd = np.where(plateCharge >= 0)
    negInd = np.where(plateCharge < 0)
    qPos = Utils.mkvc(plateCharge[posInd])
    qNeg = Utils.mkvc(plateCharge[negInd])

    qPosLoc = plateChargeLocs[posInd,:][0]
    qNegLoc = plateChargeLocs[negInd,:][0]

    qPosData = np.vstack([qPosLoc[:,0], qPosLoc[:,1], qPos]).T
    qNegData = np.vstack([qNegLoc[:,0], qNegLoc[:,1], qNeg]).T

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


def getSensitivity(survey,A,B,M,N,model):

    if(survey == "Dipole-Dipole"):
        rx = DC.Rx.Dipole(np.r_[M,0.], np.r_[N,0.])
        src = DC.Src.Dipole([rx], np.r_[A,0.], np.r_[B,0.])
    elif(survey == "Pole-Dipole"):
        rx = DC.Rx.Dipole(np.r_[M,0.], np.r_[N,0.])
        src = DC.Src.Pole([rx], np.r_[A,0.])
    elif(survey == "Dipole-Pole"):
        rx = DC.Rx.Pole(np.r_[M,0.])
        src = DC.Src.Dipole([rx], np.r_[A,0.], np.r_[B,0.])
    elif(survey == "Pole-Pole"):
        rx = DC.Rx.Pole(np.r_[M,0.])
        src = DC.Src.Pole([rx], np.r_[A,0.])

    # Model mappings
    expmap = Maps.ExpMap(mesh)
    mapping = expmap

    survey = DC.Survey([src])
    problem = DC.Problem3D_CC(mesh, sigmaMap = mapping)
    problem.Solver = SolverLU
    problem.pair(survey)
    fieldObj = problem.fields(model)

    J = problem.Jtvec(model, np.array([1.]), f=fieldObj)

    return J


def calculateRhoA(survey,VM,VN,A,B,M,N):

    eps = 1e-9 #to stabilize division

    if(survey == "Dipole-Dipole"):
        G =  1. / ( 1./(np.abs(A-M)+eps) - 1./(np.abs(M-B)+eps) - 1./(np.abs(N-A)+eps) + 1./(np.abs(N-B)+eps) )
        rho_a = (VM-VN)*2.*np.pi*G
    elif(survey == "Pole-Dipole"):
        G =  1. / ( 1./(np.abs(A-M)+eps) - 1./(np.abs(N-A)+eps) )
        rho_a = (VM-VN)*2.*np.pi*G
    elif(survey == "Dipole-Pole"):
        G =  1. / ( 1./(np.abs(A-M)+eps) - 1./(np.abs(M-B)+eps) )
        rho_a = (VM)*2.*np.pi*G
    elif(survey == "Pole-Pole"):
        G =  1. / ( 1./(np.abs(A-M)+eps) )
        rho_a = (VM)*2.*np.pi*G

    return rho_a


def plot_Surface_Potentials(survey,A,B,M,N,dx,dz,xc,zc,rotAng,rhohalf,rhoplate,Field,Type,Scale):

    labelsize = 12.
    ticksize = 10.

    sigplate = 1./rhoplate
    sighalf = 1./rhohalf

    if(survey == "Pole-Dipole" or survey == "Pole-Pole"):
        B = []

    mtrue, mhalf,src, primary_field, total_field = plate_fields(A,B,dx,dz,xc,zc,rotAng,sigplate,sighalf)

    fig, ax = plt.subplots(2,1,figsize=(9*1.5,9*1.5),sharex=True)
    fig.subplots_adjust(right=0.8)

    xSurface, phiTotalSurface, phiScaleTotal = get_Surface_Potentials(survey, src, total_field)
    xSurface, phiPrimSurface, phiScalePrim = get_Surface_Potentials(survey, src, primary_field)
    ylim = np.r_[-1., 1.]*np.max(np.abs(phiTotalSurface))
    xlim = np.array([-40,40])

    if(survey == "Dipole-Pole" or survey == "Pole-Pole"):
        MInd = np.where(xSurface == M)
        N = []

        VM = phiTotalSurface[MInd[0]]
        VN = 0.

        VMprim = phiPrimSurface[MInd[0]]
        VNprim = 0.

    else:
        MInd = np.where(xSurface == M)
        NInd = np.where(xSurface == N)

        VM = phiTotalSurface[MInd[0]]
        VN = phiTotalSurface[NInd[0]]

        VMprim = phiPrimSurface[MInd[0]]
        VNprim = phiPrimSurface[NInd[0]]

    #2D geometric factor
    G2D = rhohalf/(calculateRhoA(survey,VMprim,VNprim,A,B,M,N))

    # Subplot 1: Full set of surface potentials
    ax[0].plot(xSurface,phiTotalSurface,color=[0.1,0.5,0.1],linewidth=2)
    ax[0].plot(xSurface,phiPrimSurface ,linestyle='dashed',linewidth=0.5,color='k')
    ax[0].grid(which='both',linestyle='-',linewidth=0.5,color=[0.2,0.2,0.2],alpha=0.5)

    if(survey == "Pole-Dipole" or survey == "Pole-Pole"):
        ax[0].plot(A,0,'+',markersize = 12, markeredgewidth = 3, color=[1.,0.,0])
    else:
        ax[0].plot(A,0,'+',markersize = 12, markeredgewidth = 3, color=[1.,0.,0])
        ax[0].plot(B,0,'_',markersize = 12, markeredgewidth = 3, color=[0.,0.,1.])
    ax[0].set_ylabel('Potential, (V)',fontsize = labelsize)
    ax[0].set_xlabel('x (m)',fontsize = labelsize)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)

    if(survey == "Dipole-Pole" or survey == "Pole-Pole"):
        ax[0].plot(M,VM,'o',color='k')

        xytextM = (M+0.5,np.max([np.min([VM,ylim.max()]),ylim.min()])+0.5)
        ax[0].annotate('%2.1e'%(VM), xy=xytextM, xytext=xytextM,fontsize = labelsize)

    else:
        ax[0].plot(M,VM,'o',color='k')
        ax[0].plot(N,VN,'o',color='k')

        xytextM = (M+0.5,np.max([np.min([VM,ylim.max()]),ylim.min()])+0.5)
        xytextN = (N+0.5,np.max([np.min([VN,ylim.max()]),ylim.min()])+0.5)
        ax[0].annotate('%2.1e'%(VM), xy=xytextM, xytext=xytextM,fontsize = labelsize)
        ax[0].annotate('%2.1e'%(VN), xy=xytextN, xytext=xytextN,fontsize = labelsize)

    ax[0].tick_params(axis='both', which='major', labelsize=ticksize)

    props = dict(boxstyle='round', facecolor='grey', alpha=0.4)
    ax[0].text(xlim.max()+1,ylim.max()-0.1*ylim.max(),'$\\rho_a$ = %2.2f'%(G2D*calculateRhoA(survey,VM,VN,A,B,M,N)),
                verticalalignment='bottom', bbox=props, fontsize = labelsize)

    ax[0].legend(['Model Potential','Half-Space Potential'], loc=3, fontsize = labelsize)

    # # Subplot 2: Surface potentials with gaps around current electrodes

    # # Select points more than 5m from Tx electrodes of plotting
    # xSurface_AInd = np.where(np.abs(xSurface - A) >= 5.)[0]
    # xSurface_BInd = np.where(np.abs(xSurface - B) >= 5.)[0]
    # xSurfaceTxGapInd = list(set(xSurface_AInd).intersection(xSurface_BInd))
    # xSurface_TxGap = xSurface[xSurfaceTxGapInd]
    # phiTotalSurface_TxGap = phiTotalSurface[xSurfaceTxGapInd]
    # phiPrimSurface_TxGap = phiPrimSurface[xSurfaceTxGapInd]
    # ylim = np.r_[-1., 1.]*(np.max(np.abs(phiTotalSurface_TxGap)) - 0.05*np.max(np.abs(phiTotalSurface_TxGap)))

    # ax[1].plot(xSurface_TxGap,phiTotalSurface_TxGap ,color=[0.1,0.5,0.1],linewidth=2)
    # ax[1].plot(xSurface_TxGap,phiPrimSurface_TxGap ,linestyle='dashed',linewidth=0.5,color='k')
    # ax[1].grid(which='both',linestyle='-',linewidth=0.5,color=[0.2,0.2,0.2],alpha=0.5)
    # ax[1].plot(A,0,'+',markersize = 12, markeredgewidth = 3, color=[1.,0.,0])
    # ax[1].plot(B,0,'_',markersize = 12, markeredgewidth = 3, color=[0.,0.,1.])
    # ax[1].set_ylabel('Potential, (V)',fontsize = labelsize)
    # ax[1].set_xlabel('x (m)',fontsize = labelsize)
    # ax[1].set_xlim(xlim)
    # ax[1].set_ylim(ylim)

    # ax[1].plot(M,VM,'o',color='k')
    # ax[1].plot(N,VN,'o',color='k')

    # ax[1].annotate('%2.1e'%(VM), xy=xytextM, xytext=xytextM,fontsize = labelsize)
    # ax[1].annotate('%2.1e'%(VN), xy=xytextN, xytext=xytextN,fontsize = labelsize)

    # ax[1].tick_params(axis='both', which='major', labelsize=ticksize)

    # props = dict(boxstyle='round', facecolor='grey', alpha=0.4)
    # ax[1].text(xlim.max()+1,ylim.max()-0.1*ylim.max(),'$\\rho_a$ = %2.2f'%(G2D*rho_a(VM,VN,A,B,M,N)),
    #             verticalalignment='bottom', bbox=props, fontsize = labelsize)

    # ax[1].legend(['Model Potential','Half-Space Potential'], loc=3, fontsize = labelsize)


    if Field == 'Model':

        label = 'Resisitivity (ohm-m)'
        xtype = 'CC'
        view = 'real'
        streamOpts = None
        ind = indCC

        formatter = "%.1e"
        pcolorOpts = {"cmap":"jet_r"}
        if Scale == 'Log':
            pcolorOpts = {'norm':matplotlib.colors.LogNorm(),"cmap":"jet_r"}

        if Type == 'Total':
            u = 1./(mapping*mtrue)
        elif Type == 'Primary':
            u = 1./(mapping*mhalf)
        elif Type == 'Secondary':
            u = 1./(mapping*mtrue) - 1./(mapping*mhalf)
            if Scale == 'Log':
                linthresh = 10.
                pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=linthresh, linscale=0.2),"cmap":"jet_r"}

    elif Field == 'Potential':

        label = 'Potential (V)'
        xtype = 'CC'
        view = 'real'
        streamOpts = None
        ind = indCC

        formatter = "%.1e"
        pcolorOpts = {"cmap":"viridis"}
        if Scale == 'Log':
            linthresh = 10.
            pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=linthresh, linscale=0.2),"cmap":"viridis"}

        if Type == 'Total':
            # formatter = LogFormatter(10, labelOnlyBase=False)
            # pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=10, linscale=0.1)}

            u = total_field[src,'phi'] - phiScaleTotal

        elif Type == 'Primary':
            # formatter = LogFormatter(10, labelOnlyBase=False)
            # pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=10, linscale=0.1)}

            u = primary_field[src,'phi'] - phiScalePrim

        elif Type == 'Secondary':
            # formatter = None
            # pcolorOpts = {"cmap":"viridis"}

            uTotal = total_field[src,'phi'] - phiScaleTotal
            uPrim = primary_field[src,'phi'] - phiScalePrim
            u = uTotal - uPrim

    elif Field == 'E':

        label = 'Electric Field (V/m)'
        xtype = 'F'
        view = 'vec'
        streamOpts = {'color':'w'}
        ind = indF

        #formatter = LogFormatter(10, labelOnlyBase=False)
        pcolorOpts = {"cmap":"viridis"}
        if Scale == 'Log':
            pcolorOpts = {'norm':matplotlib.colors.LogNorm(),"cmap":"viridis"}
        formatter = "%.1e"

        if Type == 'Total':
            u = total_field[src,'e']

        elif Type == 'Primary':
            u = primary_field[src,'e']

        elif Type == 'Secondary':
            uTotal = total_field[src,'e']
            uPrim = primary_field[src,'e']
            u = uTotal - uPrim

    elif Field == 'J':

        label = 'Current density ($A/m^2$)'
        xtype = 'F'
        view = 'vec'
        streamOpts = {'color':'w'}
        ind = indF

        #formatter = LogFormatter(10, labelOnlyBase=False)
        pcolorOpts = {"cmap":"viridis"}
        if Scale == 'Log':
            pcolorOpts = {'norm':matplotlib.colors.LogNorm(),"cmap":"viridis"}
        formatter = "%.1e"

        if Type == 'Total':
            u = total_field[src,'j']

        elif Type == 'Primary':
            u = primary_field[src,'j']

        elif Type == 'Secondary':
            uTotal = total_field[src,'j']
            uPrim = primary_field[src,'j']
            u = uTotal - uPrim

    elif Field == 'Charge':

        label = 'Charge Density ($C/m^2$)'
        xtype = 'CC'
        view = 'real'
        streamOpts = None
        ind = indCC

       # formatter = LogFormatter(10, labelOnlyBase=False)
        pcolorOpts = {"cmap":"RdBu_r"}
        if Scale == 'Log':
            linthresh = 1e-12
            pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=linthresh, linscale=0.2),"cmap":"RdBu_r"}
        formatter = "%.1e"

        if Type == 'Total':
            u = total_field[src,'charge']

        elif Type == 'Primary':
            u = primary_field[src,'charge']

        elif Type == 'Secondary':
            uTotal = total_field[src,'charge']
            uPrim = primary_field[src,'charge']
            u = uTotal - uPrim

    elif Field == 'Sensitivity':

        label = 'Sensitivity'
        xtype = 'CC'
        view = 'real'
        streamOpts = None
        ind = indCC

        # formatter = None
        # pcolorOpts = {"cmap":"viridis"}
        # formatter = LogFormatter(10, labelOnlyBase=False)
        pcolorOpts = {"cmap":"viridis"}
        if Scale == 'Log':
            linthresh =1.
            pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=linthresh, linscale=0.2),"cmap":"viridis"}
        # formatter = formatter = "$10^{%.1f}$"
        formatter = "%.1e"

        if Type == 'Total':
            u = getSensitivity(survey,A,B,M,N,mtrue)

        elif Type == 'Primary':
            u = getSensitivity(survey,A,B,M,N,mhalf)

        elif Type == 'Secondary':
            uTotal = getSensitivity(survey,A,B,M,N,mtrue)
            uPrim = getSensitivity(survey,A,B,M,N,mhalf)
            u = uTotal - uPrim
        # u = np.log10(abs(u))

    if Scale == 'Log':
        eps = 1e-16
    else:
        eps = 0.
    dat = meshcore.plotImage(u[ind]+eps, vType = xtype, ax=ax[1], grid=False,view=view, streamOpts=streamOpts, pcolorOpts = pcolorOpts) #gridOpts={'color':'k', 'alpha':0.5}

    # Get plate corners
    plateCorners = getPlateCorners(xc,zc,dx,dz,rotAng)

    if(rhoplate != rhohalf):
        # plot top of plate outline
        ax[1].plot(plateCorners[[0,1],0],plateCorners[[0,1],1],linestyle = 'dashed',color='k')
        # plot east side of plate outline
        ax[1].plot(plateCorners[[1,3],0],plateCorners[[1,3],1],linestyle = 'dashed',color='k')
        # plot bottom of plate outline
        ax[1].plot(plateCorners[[2,3],0],plateCorners[[2,3],1],linestyle = 'dashed',color='k')
        # plot west side of plate outline
        ax[1].plot(plateCorners[[0,2],0],plateCorners[[0,2],1],linestyle = 'dashed',color='k')

    if (Field == 'Charge') and (Type != 'Primary') and (Type != 'Total'):
        qTotal = total_field[src,'charge']
        qPrim = primary_field[src,'charge']
        qSecondary = qTotal - qPrim
        qPosSum, qNegSum, qPosAvgLoc, qNegAvgLoc = sumPlateCharges(xc,zc,dx,dz,rotAng,qSecondary)
        ax[1].plot(qPosAvgLoc[0],qPosAvgLoc[1], marker = '.', color='black', markersize= labelsize)
        ax[1].plot(qNegAvgLoc[0],qNegAvgLoc[1], marker = '.',  color='black', markersize= labelsize)
        if(qPosAvgLoc[0] > qNegAvgLoc[0]):
            xytext_qPos = (qPosAvgLoc[0] + 1., qPosAvgLoc[1] - 0.5)
            xytext_qNeg = (qNegAvgLoc[0] - 15., qNegAvgLoc[1] - 0.5)
        else:
            xytext_qPos = (qPosAvgLoc[0] - 15., qPosAvgLoc[1] - 0.5)
            xytext_qNeg = (qNegAvgLoc[0] + 1., qNegAvgLoc[1] - 0.5)
        ax[1].annotate('+Q = %2.1e'%(qPosSum), xy=xytext_qPos, xytext=xytext_qPos ,fontsize = labelsize)
        ax[1].annotate('-Q = %2.1e'%(qNegSum), xy=xytext_qNeg, xytext=xytext_qNeg ,fontsize = labelsize)

    ax[1].set_xlabel('x (m)', fontsize= labelsize)
    ax[1].set_ylabel('z (m)', fontsize= labelsize)

    if(survey == "Dipole-Dipole"):
        ax[1].plot(A,1.,marker = 'v',color='red',markersize= labelsize)
        ax[1].plot(B,1.,marker = 'v',color='blue',markersize= labelsize)
        ax[1].plot(M,1.,marker = '^',color='yellow',markersize= labelsize)
        ax[1].plot(N,1.,marker = '^',color='green',markersize= labelsize)

        xytextA1 = (A-0.5,2.)
        xytextB1 = (B-0.5,2.)
        xytextM1 = (M-0.5,2.)
        xytextN1 = (N-0.5,2.)
        ax[1].annotate('A', xy=xytextA1, xytext=xytextA1,fontsize = labelsize)
        ax[1].annotate('B', xy=xytextB1, xytext=xytextB1,fontsize = labelsize)
        ax[1].annotate('M', xy=xytextM1, xytext=xytextM1,fontsize = labelsize)
        ax[1].annotate('N', xy=xytextN1, xytext=xytextN1,fontsize = labelsize)
    elif(survey == "Pole-Dipole"):
        ax[1].plot(A,1.,marker = 'v',color='red',markersize= labelsize)
        ax[1].plot(M,1.,marker = '^',color='yellow',markersize= labelsize)
        ax[1].plot(N,1.,marker = '^',color='green',markersize= labelsize)

        xytextA1 = (A-0.5,2.)
        xytextM1 = (M-0.5,2.)
        xytextN1 = (N-0.5,2.)
        ax[1].annotate('A', xy=xytextA1, xytext=xytextA1,fontsize = labelsize)
        ax[1].annotate('M', xy=xytextM1, xytext=xytextM1,fontsize = labelsize)
        ax[1].annotate('N', xy=xytextN1, xytext=xytextN1,fontsize = labelsize)
    elif(survey == "Dipole-Pole"):
        ax[1].plot(A,1.,marker = 'v',color='red',markersize= labelsize)
        ax[1].plot(B,1.,marker = 'v',color='blue',markersize= labelsize)
        ax[1].plot(M,1.,marker = '^',color='yellow',markersize= labelsize)

        xytextA1 = (A-0.5,2.)
        xytextB1 = (B-0.5,2.)
        xytextM1 = (M-0.5,2.)
        ax[1].annotate('A', xy=xytextA1, xytext=xytextA1,fontsize = labelsize)
        ax[1].annotate('B', xy=xytextB1, xytext=xytextB1,fontsize = labelsize)
        ax[1].annotate('M', xy=xytextM1, xytext=xytextM1,fontsize = labelsize)
    elif(survey == "Pole-Pole"):
        ax[1].plot(A,1.,marker = 'v',color='red',markersize= labelsize)
        ax[1].plot(M,1.,marker = '^',color='yellow',markersize= labelsize)

        xytextA1 = (A-0.5,2.)
        xytextM1 = (M-0.5,2.)
        ax[1].annotate('A', xy=xytextA1, xytext=xytextA1,fontsize = labelsize)
        ax[1].annotate('M', xy=xytextM1, xytext=xytextM1,fontsize = labelsize)

    ax[1].tick_params(axis='both', which='major', labelsize=ticksize)
    cbar_ax = fig.add_axes([0.8, 0.05, 0.08, 0.5])
    cbar_ax.axis('off')
    vmin, vmax = dat[0].get_clim()
    if Scale == 'Log':

        if (Field=='E') or (Field == 'J'):
            cb = plt.colorbar(dat[0], ax=cbar_ax,format = formatter, ticks = np.logspace(np.log10(vmin), np.log10(vmax), 5))

        elif (Field == 'Model'):

            if (Type == 'Secondary'):
                cb = plt.colorbar(dat[0], ax=cbar_ax,format = formatter, ticks = np.r_[np.minimum(0.,vmin),np.maximum(0.,vmax)])
            else:
                cb = plt.colorbar(dat[0], ax=cbar_ax,format = formatter, ticks = np.logspace(np.log10(vmin), np.log10(vmax), 5))

        else:
            cb = plt.colorbar(dat[0], ax=cbar_ax,format = formatter, ticks = np.r_[-1.*np.logspace(np.log10(-vmin-eps), np.log10(linthresh), 3)[:-1],0.,np.logspace(np.log10(linthresh), np.log10(vmax), 3)[1:]])
    else:
        if (Field == 'Model') and (Type == 'Secondary'):
            cb = plt.colorbar(dat[0], ax=cbar_ax,format = formatter, ticks = np.r_[np.minimum(0.,vmin),np.maximum(0.,vmax)])
        else:
            cb = plt.colorbar(dat[0], ax=cbar_ax,format = formatter, ticks = np.linspace(vmin, vmax, 5))

    cb.ax.tick_params(labelsize=ticksize)
    cb.set_label(label, fontsize=labelsize)
    ax[1].set_xlim([-40.,40.])
    ax[1].set_ylim([-40.,5.])
    ax[1].set_aspect('equal')
    plt.show()
    # return fig, ax


def plate_app():
    app = widgetify(plot_Surface_Potentials,
        survey = ToggleButtons(options =['Dipole-Dipole','Dipole-Pole','Pole-Dipole','Pole-Pole'],value='Dipole-Dipole'),
        dx = FloatSlider(min=1.,max=1000.,step=1.,value=10., continuous_update=False),
        dz = FloatSlider(min=1.,max=200.,step=1.,value=10., continuous_update=False),
        xc = FloatSlider(min=-30.,max=30.,step=1.,value=0., continuous_update=False),
        zc = FloatSlider(min=-30.,max=0.,step=1.,value=-10., continuous_update=False),
        rotAng = FloatSlider(min=-90.,max=90.,step=1.,value=0., continuous_update=False,description='$\\theta$'),
        rhoplate = FloatText(min=1e-8,max=1e8, value = 500., continuous_update=False,description='$\\rho_2$'),
        rhohalf  = FloatText(min=1e-8,max=1e8, value = 500., continuous_update=False,description='$\\rho_1$'),
        A = FloatSlider(min=-30.25,max=30.25,step=0.5,value=-30.25, continuous_update=False),
        B = FloatSlider(min=-30.25,max=30.25,step=0.5,value=30.25, continuous_update=False),
        M = FloatSlider(min=-30.25,max=30.25,step=0.5,value=-10.25, continuous_update=False),
        N = FloatSlider(min=-30.25,max=30.25,step=0.5,value=10.25, continuous_update=False),
        Field = ToggleButtons(options =['Model','Potential','E','J','Charge','Sensitivity'],value='Model'),
        Type = ToggleButtons(options =['Total','Primary','Secondary'],value='Total'),
        Scale = ToggleButtons(options =['Linear','Log'],value='Linear')
    )
    return app
