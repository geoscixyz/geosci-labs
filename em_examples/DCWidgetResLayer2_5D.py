#import sys
#sys.path.append("./simpeg")
#sys.path.append("./simpegdc/")

#import warnings
#warnings.filterwarnings('ignore')

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
import copy


import warnings
warnings.filterwarnings('ignore') # ignore warnings: only use this once you are sure things are working

try:
    from IPython.html.widgets import  interact, IntSlider, FloatSlider, FloatText, ToggleButtons
    pass
except Exception, e:    
    from ipywidgets import interact, IntSlider, FloatSlider, FloatText, ToggleButtons

# Mesh, sigmaMap can be globals global
npad = 15
growrate = 2.
cs = 0.5
hx = [(cs,npad, -growrate),(cs,200),(cs,npad, growrate)]
hy = [(cs,npad, -growrate),(cs,100)]
mesh = Mesh.TensorMesh([hx, hy], "CN")
idmap = Maps.IdentityMap(mesh)
# actmap = Maps.InjectActiveCells(mesh, ~airInd, np.log(1e-8))
sigmaMap = idmap
# sigmaMap = Maps.IdentityMap(mesh)
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


def model_fields(A,B,zcLayer,dzLayer,xc,zc,r,sigLayer,sigTarget,sigHalf):
    # Create halfspace model
    mHalf = sigHalf*np.ones([mesh.nC,])
    # Add layer to model
    mLayer = addLayer2Mod(zcLayer,dzLayer,mHalf,sigLayer)
    # Add plate or cylinder
    # fullMod = addPlate2Mod(xc,zc,dx,dz,rotAng,LayerMod,sigTarget)
    mFull = addCylinder2Mod(xc,zc,r,mLayer,sigTarget)
    
    Mx =  mesh.gridCC
    # Nx = np.empty(shape=(mesh.nC, 2))
    rx = DC.Rx.Pole_ky(Mx)
    # rx = DC.Rx.Dipole(Mx,Nx)
    src = DC.Src.Dipole([rx], np.r_[A,0.], np.r_[B,0.])
    # src = DC.Src.Dipole_ky([rx], np.r_[A,0.], np.r_[B,0.])
    survey = DC.Survey_ky([src])
    # survey = DC.Survey([src])
    # survey_prim = DC.Survey([src])
    survey_prim = DC.Survey_ky([src])
    #problem = DC.Problem3D_CC(mesh, sigmaMap = sigmaMap)
    problem = DC.Problem2D_CC(mesh, sigmaMap = sigmaMap)
    # problem_prim = DC.Problem3D_CC(mesh, sigmaMap = sigmaMap)
    problem_prim = DC.Problem2D_CC(mesh, sigmaMap = sigmaMap)
    problem.Solver = SolverLU
    problem_prim.Solver = SolverLU
    problem.pair(survey)
    problem_prim.pair(survey_prim)


    mesh.setCellGradBC("neumann")
    cellGrad = mesh.cellGrad
    faceDiv = mesh.faceDiv

    phi_primary = survey_prim.dpred(mHalf)
    e_primary = -cellGrad*phi_primary
    j_primary = problem_prim.MfRhoI*problem_prim.Grad*phi_primary
    q_primary = epsilon_0*problem_prim.Vol*(faceDiv*e_primary)
    primary_field = {'phi': phi_primary, 'e': e_primary, 'j': j_primary, 'q': q_primary}

    phi_total = survey.dpred(mFull)
    e_total = -cellGrad*phi_total
    j_total = problem.MfRhoI*problem.Grad*phi_total
    q_total = epsilon_0*problem.Vol*(faceDiv*e_total)
    total_field = {'phi': phi_total, 'e': e_total, 'j': j_total, 'q': q_total}

    return mFull,mHalf, src, primary_field, total_field


def addLayer2Mod(zcLayer,dzLayer,modd,sigLayer):

    CCLocs = mesh.gridCC
    mod = copy.copy(modd)

    zmax = zcLayer + dzLayer/2.
    zmin = zcLayer - dzLayer/2.

    belowInd = np.where(CCLocs[:,1] <= zmax)[0]
    aboveInd = np.where(CCLocs[:,1] >= zmin)[0]
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

def getCylinderPoints(xc,zc,r):
    xLocOrig1 = np.arange(-r,r+r/10.,r/10.)
    xLocOrig2 = np.arange(r,-r-r/10.,-r/10.)
    # Top half of cylinder
    zLoc1 = np.sqrt(-xLocOrig1**2.+r**2.)+zc
    # Bottom half of cylinder
    zLoc2 = -np.sqrt(-xLocOrig2**2.+r**2.)+zc
    # Shift from x = 0 to xc
    xLoc1 = xLocOrig1 + xc*np.ones_like(xLocOrig1)
    xLoc2 = xLocOrig2 + xc*np.ones_like(xLocOrig2)

    cylinderPoints = np.vstack([np.vstack([xLoc1,zLoc1]).T,np.vstack([xLoc2,zLoc2]).T])
    return cylinderPoints

def addCylinder2Mod(xc,zc,r,modd,sigCylinder):

    # Get points for cylinder outline
    cylinderPoints = getCylinderPoints(xc,zc,r)
    mod = copy.copy(modd)

    verts = []
    codes = []
    for ii in range(0,cylinderPoints.shape[0]):
        verts.append(cylinderPoints[ii,:])
        
        if(ii == 0):
            codes.append(Path.MOVETO)
        elif(ii == cylinderPoints.shape[0]-1):
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


def getPlateCorners(xc,zc,dx,dz,rotAng):

    # Form rotation matix
    rotMat = np.array([[np.cos(rotAng*(np.pi/180.)), -np.sin(rotAng*(np.pi/180.))],[np.sin(rotAng*(np.pi/180.)), np.cos(rotAng*(np.pi/180.))]])
    originCorners = np.array([[-0.5*dx, 0.5*dz], [0.5*dx, 0.5*dz], [-0.5*dx, -0.5*dz], [0.5*dx, -0.5*dz]])
    
    rotPlateCorners = np.dot(originCorners,rotMat)
    plateCorners = rotPlateCorners + np.hstack([np.repeat(xc,4).reshape([4,1]),np.repeat(zc,4).reshape([4,1])])
    return plateCorners


def addPlate2Mod(xc,zc,dx,dz,rotAng,modd,sigPlate):
    # use matplotlib paths to find CC inside of polygon
    plateCorners = getPlateCorners(xc,zc,dx,dz,rotAng)
    mod = copy.copy(modd)

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

    mod[insideInd] = sigPlate
    return mod   


def get_Surface_Potentials(survey, src,field_obj):

    phi = field_obj['phi']
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


# Inline functions for computing apparent resistivity
#eps = 1e-9 #to stabilize division
#G = lambda A, B, M, N: 1. / ( 1./(np.abs(A-M)+eps) - 1./(np.abs(M-B)+eps) - 1./(np.abs(N-A)+eps) + 1./(np.abs(N-B)+eps) )
#rho_a = lambda VM,VN, A,B,M,N: (VM-VN)*2.*np.pi*G(A,B,M,N)

def sumCylinderCharges(xc, zc, r, qSecondary):
    chargeRegionVerts = getCylinderPoints(xc, zc, r+0.5)

    codes = chargeRegionVerts.shape[0]*[Path.LINETO]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY

    chargeRegionPath = Path(chargeRegionVerts, codes)
    CCLocs = mesh.gridCC
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

    survey = DC.Survey([src])
    problem = DC.Problem3D_CC(mesh, sigmaMap = sigmaMap)
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

def plot_Surface_Potentials(survey,A,B,M,N,zcLayer,dzLayer,xc,zc,r,rhohalf,rholayer,rhocyl,Field,Type):

    labelsize = 12.
    ticksize = 10.

    sigTarget = 1./rhocyl
    sigLayer = 1./rholayer
    sigHalf = 1./rhohalf

    if(survey == "Pole-Dipole" or survey == "Pole-Pole"):
        B = []

    mtrue, mhalf, src, total_field, primary_field = model_fields(A,B,zcLayer,dzLayer,xc,zc,r,sigLayer,sigTarget,sigHalf)

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
                verticalalignment='bottom', bbox=props, fontsize = 14)

    ax[0].legend(['Model Potential','Half-Space Potential'], loc=3, fontsize = labelsize)

    if Field == 'Model':

        label = 'Resisitivity (ohm-m)'
        xtype = 'CC'
        view = 'real'
        streamOpts = None
        ind = indCC

        formatter = "1e%.1f"
        pcolorOpts = {"cmap":"jet_r"}


        if Type == 'Total':
            u = 1./(mtrue)
            u = np.log10(u)
        elif Type == 'Primary':
            u = 1./(mhalf)
            u = np.log10(u)
        elif Type == 'Secondary':
            u = 1./(mtrue) - 1./(mhalf)
            formatter = "%.1e"
            pcolorOpts = {"cmap":"jet_r"}


    elif Field == 'Potential':

        label = 'Potential (V)'
        xtype = 'CC'
        view = 'real'
        streamOpts = None
        ind = indCC

        formatter = "%.1e"
        pcolorOpts = {"cmap":"viridis"}

        if Type == 'Total':
            # formatter = LogFormatter(10, labelOnlyBase=False)
            # pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=10, linscale=0.1)}

            u = total_field['phi'] - phiScaleTotal

        elif Type == 'Primary':
            # formatter = LogFormatter(10, labelOnlyBase=False)
            # pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=10, linscale=0.1)}

            u = primary_field['phi'] - phiScalePrim

        elif Type == 'Secondary':
            # formatter = None
            # pcolorOpts = {"cmap":"viridis"}

            uTotal = total_field['phi'] - phiScaleTotal
            uPrim = primary_field['phi'] - phiScalePrim
            u = uTotal - uPrim

    elif Field == 'E':

        label = 'Electric Field (V/m)'
        xtype = 'F'
        view = 'vec'
        streamOpts = {'color':'w'}
        ind = indF

        # formatter = LogFormatter(10, labelOnlyBase=False)
        # pcolorOpts = {'norm':matplotlib.colors.LogNorm()}
        formatter = "%.1e"
        pcolorOpts = {"cmap":"viridis"}

        if Type == 'Total':
            u = total_field['e']

        elif Type == 'Primary':
            u = primary_field['e']

        elif Type == 'Secondary':
            uTotal = total_field['e']
            uPrim = primary_field['e']
            u = uTotal - uPrim

    elif Field == 'J':

        label = 'Current density ($A/m^2$)'
        xtype = 'F'
        view = 'vec'
        streamOpts = {'color':'w'}
        ind = indF

        # formatter = LogFormatter(10, labelOnlyBase=False)
        # pcolorOpts = {'norm':matplotlib.colors.LogNorm()}
        formatter = "%.1e"
        pcolorOpts = {"cmap":"viridis"}

        if Type == 'Total':
            u = total_field['j']

        elif Type == 'Primary':
            u = primary_field['j']

        elif Type == 'Secondary':
            uTotal = total_field['j']
            uPrim = primary_field['j']
            u = uTotal - uPrim

    elif Field == 'Charge':

        label = 'Charge Density ($C/m^2$)'
        xtype = 'CC'
        view = 'real'
        streamOpts = None
        ind = indCC

        # formatter = LogFormatter(10, labelOnlyBase=False)
        # pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=1e-11, linscale=0.01)}
        formatter = "%.1e"
        pcolorOpts = {"cmap":"RdBu_r"}

        if Type == 'Total':
            u = total_field['q']

        elif Type == 'Primary':
            u = primary_field['q']

        elif Type == 'Secondary':
            uTotal = total_field['q']
            uPrim = primary_field['q']
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
        # pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=1e-2, linscale=0.01)}
        # formatter = formatter = "$10^{%.1f}$"
        formatter = "%.1e"
        pcolorOpts = {"cmap":"viridis"}

        if Type == 'Total':
            u = getSensitivity(survey,A,B,M,N,mtrue)

        elif Type == 'Primary':
            u = getSensitivity(survey,A,B,M,N,mhalf)

        elif Type == 'Secondary':
            uTotal = getSensitivity(survey,A,B,M,N,mtrue)
            uPrim = getSensitivity(survey,A,B,M,N,mhalf)
            u = uTotal - uPrim
        # u = np.log10(abs(u))

    dat = meshcore.plotImage(u[ind], vType = xtype, ax=ax[1], grid=False,view=view, streamOpts=streamOpts, pcolorOpts = pcolorOpts) #gridOpts={'color':'k', 'alpha':0.5}
    
    # Get cylinder outline
    cylinderPoints = getCylinderPoints(xc,zc,r)

    if(rhocyl != rhohalf):
        ax[1].plot(cylinderPoints[:,0],cylinderPoints[:,1], linestyle = 'dashed', color='k')

    # if (Field == 'Charge') and (Type != 'Primary') and (Type != 'Total'):
    #     qTotal = total_field['q']
    #     qPrim = primary_field['q']
    #     qSecondary = qTotal - qPrim
    #     qPosSum, qNegSum, qPosAvgLoc, qNegAvgLoc = sumCylinderCharges(xc,zc,r,qSecondary)
    #     ax[1].plot(qPosAvgLoc[0],qPosAvgLoc[1], marker = '.', color='black', markersize= labelsize)
    #     ax[1].plot(qNegAvgLoc[0],qNegAvgLoc[1], marker = '.',  color='black', markersize= labelsize)
    #     if(qPosAvgLoc[0] > qNegAvgLoc[0]):
    #         xytext_qPos = (qPosAvgLoc[0] + 1., qPosAvgLoc[1] - 0.5)
    #         xytext_qNeg = (qNegAvgLoc[0] - 15., qNegAvgLoc[1] - 0.5)
    #     else:
    #         xytext_qPos = (qPosAvgLoc[0] - 15., qPosAvgLoc[1] - 0.5)
    #         xytext_qNeg = (qNegAvgLoc[0] + 1., qNegAvgLoc[1] - 0.5)
    #     ax[1].annotate('+Q = %2.1e'%(qPosSum), xy=xytext_qPos, xytext=xytext_qPos ,fontsize = labelsize)
    #     ax[1].annotate('-Q = %2.1e'%(qNegSum), xy=xytext_qNeg, xytext=xytext_qNeg ,fontsize = labelsize)

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
    cb = plt.colorbar(dat[0], ax=cbar_ax,format = formatter, ticks = np.linspace(vmin, vmax, 5))
    #t_logloc = matplotlib.ticker.LogLocator(base=10.0, subs=[1.0,2.], numdecs=4, numticks=8)
    #tick_locator = matplotlib.ticker.SymmetricalLogLocator(t_logloc)
    #cb.locator = tick_locator
    #cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    #cb.update_ticks()
    cb.ax.tick_params(labelsize=ticksize)
    cb.set_label(label, fontsize=labelsize)
    ax[1].set_xlim([xmin,xmax])
    ax[1].set_ylim([ymin,ymax])
    ax[1].set_aspect('equal')


def ResLayer_app():
    app = interact(plot_Surface_Potentials,
                survey = ToggleButtons(options =['Dipole-Dipole','Dipole-Pole','Pole-Dipole','Pole-Pole'],value='Dipole-Dipole'),
                zcLayer = FloatSlider(min=-10.,max=0.,step=1.,value=-10., continuous_update=False),
                dzLayer = FloatSlider(min=0.5,max=5.,step=0.5,value=1., continuous_update=False),
                rholayer = FloatText(min=1e-8,max=1e8, value = 5000., continuous_update=False,description='$\\rho_{Layer}$'),
                xc = FloatSlider(min=-30.,max=30.,step=1.,value=0., continuous_update=False),
                zc = FloatSlider(min=-30.,max=-15.,step=0.5,value=-25., continuous_update=False),
                r = FloatSlider(min=1.,max=10.,step=0.5,value=5., continuous_update=False),
                rhocyl = FloatText(min=1e-8,max=1e8, value = 500., continuous_update=False,description='$\\rho_{Cyl}$'),
                rhohalf = FloatText(min=1e-8,max=1e8, value = 500., continuous_update=False,description='$\\rho_{Half}$'),
                A = FloatSlider(min=-30.25,max=30.25,step=0.5,value=-30.25, continuous_update=False),
                B = FloatSlider(min=-30.25,max=30.25,step=0.5,value=30.25, continuous_update=False),
                M = FloatSlider(min=-30.25,max=30.25,step=0.5,value=-10.25, continuous_update=False),
                N = FloatSlider(min=-30.25,max=30.25,step=0.5,value=10.25, continuous_update=False),
                Field = ToggleButtons(options =['Model','Potential','E','J','Charge','Sensitivity'],value='Model'),
                Type = ToggleButtons(options =['Total','Primary','Secondary'],value='Total')
                )
    return app

