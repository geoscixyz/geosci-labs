#import sys
#sys.path.append("./simpeg")
#sys.path.append("./simpegdc/")

#import warnings
#warnings.filterwarnings('ignore')

from SimPEG import Mesh, Maps, SolverLU, Utils
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


import warnings
warnings.filterwarnings('ignore') # ignore warnings: only use this once you are sure things are working

try:
    from IPython.html.widgets import  interact, IntSlider, FloatSlider, FloatText, ToggleButtons
    pass
except Exception, e:
    from ipywidgets import interact, IntSlider, FloatSlider, FloatText, ToggleButtons


def ExtractCoreMesh(xyzlim, mesh, meshType='tensor'):
    """
    Extracts Core Mesh from Global mesh

    :param numpy.ndarray xyzlim: 2D array [ndim x 2]
    :param BaseMesh mesh: The mesh

    This function ouputs::

        - actind: corresponding boolean index from global to core
        - meshcore: core SimPEG mesh

    Warning: 1D and 2D has not been tested
    """
    if mesh.dim == 1:
        xyzlim = xyzlim.flatten()
        xmin, xmax = xyzlim[0], xyzlim[1]

        xind = np.logical_and(mesh.vectorCCx>xmin, mesh.vectorCCx<xmax)

        xc = mesh.vectorCCx[xind]

        hx = mesh.hx[xind]

        x0 = [xc[0]-hx[0]*0.5, zc[0]-hy[0]*0.5]

        meshCore = Mesh.TensorMesh([hx, hy], x0=x0)

        actind = (mesh.gridCC[:,0]>xmin) & (mesh.gridCC[:,0]<xmax)

    elif mesh.dim == 2:
        xmin, xmax = xyzlim[0,0], xyzlim[0,1]
        ymin, ymax = xyzlim[1,0], xyzlim[1,1]

        xind = np.logical_and(mesh.vectorCCx>xmin, mesh.vectorCCx<xmax)
        yind = np.logical_and(mesh.vectorCCy>ymin, mesh.vectorCCy<ymax)

        xc = mesh.vectorCCx[xind]
        zc = mesh.vectorCCy[yind]

        hx = mesh.hx[xind]
        hy = mesh.hy[yind]

        x0 = [xc[0]-hx[0]*0.5, zc[0]-hy[0]*0.5]

        meshCore = Mesh.TensorMesh([hx, hy], x0=x0)

        actind = (mesh.gridCC[:,0]>xmin) & (mesh.gridCC[:,0]<xmax) \
               & (mesh.gridCC[:,1]>ymin) & (mesh.gridCC[:,1]<ymax) \

    elif mesh.dim == 3:
        xmin, xmax = xyzlim[0,0], xyzlim[0,1]
        ymin, ymax = xyzlim[1,0], xyzlim[1,1]
        zmin, zmax = xyzlim[2,0], xyzlim[2,1]

        xind = np.logical_and(mesh.vectorCCx>xmin, mesh.vectorCCx<xmax)
        yind = np.logical_and(mesh.vectorCCy>ymin, mesh.vectorCCy<ymax)
        zind = np.logical_and(mesh.vectorCCz>zmin, mesh.vectorCCz<zmax)

        xc = mesh.vectorCCx[xind]
        zc = mesh.vectorCCy[yind]
        zc = mesh.vectorCCz[zind]

        hx = mesh.hx[xind]
        hy = mesh.hy[yind]
        hz = mesh.hz[zind]

        x0 = [xc[0]-hx[0]*0.5, zc[0]-hy[0]*0.5, zc[0]-hz[0]*0.5]

        meshCore = Mesh.TensorMesh([hx, hy, hz], x0=x0)

        actind = (mesh.gridCC[:,0]>xmin) & (mesh.gridCC[:,0]<xmax) \
               & (mesh.gridCC[:,1]>ymin) & (mesh.gridCC[:,1]<ymax) \
               & (mesh.gridCC[:,2]>zmin) & (mesh.gridCC[:,2]<zmax)

    else:
        raise(Exception("Not implemented!"))


    return actind, meshCore



# Mesh, mapping can be globals global
npad = 12
growrate = 3.
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


def plate_fields(A,B,dx,dz,xc,zc,rotAng,sigplate,sighalf):
    # Create halfspace model
    mhalf = np.log(sighalf*np.ones([mesh.nC,]))

    #Create true model with plate
    mtrue = createPlateMod(xc,zc,dx,dz,rotAng,sigplate,sighalf)

    Mx = np.empty(shape=(0, 2))
    Nx = np.empty(shape=(0, 2))
    rx = DC.Rx.Dipole(Mx,Nx)
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



# def plate_wrapper(A,B,dx,dz,xc,zc,rotAng,rhoplate,rhohalf,Field,Type):
#     sigplate = 1./rhoplate
#     sighalf = 1./rhohalf

#     mtrue,mhalf, src, primary_field, total_field = plate_fields(A,B,dx,dz,xc,zc,rotAng,sigplate,sighalf)

#     fig = plt.figure(figsize=(15, 5))
#     ax = fig.add_subplot(111,autoscale_on=False)
#     ax.plot(A,1.,'+',markersize = 12, markeredgewidth = 3, color=[1.,0.,0])
#     ax.plot(B,1.,'_',markersize = 12, markeredgewidth = 3, color=[0.,0.,1.])
#     # Get plate corners
#     plateCorners = getPlateCorners(xc,zc,dx,dz,rotAng)
#     # plot top of plate outline
#     ax.plot(plateCorners[[0,1],0],plateCorners[[0,1],1],linestyle = 'dashed',color='k')
#     # plot east side of plate outline
#     ax.plot(plateCorners[[1,3],0],plateCorners[[1,3],1],linestyle = 'dashed',color='k')
#     # plot bottom of plate outline
#     ax.plot(plateCorners[[2,3],0],plateCorners[[2,3],1],linestyle = 'dashed',color='k')
#     # plot west side of plate outline
#     ax.plot(plateCorners[[0,2],0],plateCorners[[0,2],1],linestyle = 'dashed',color='k')

#     if Field == 'Model':

#         label = 'Resisitivity (ohm-m)'
#         xtype = 'CC'
#         view = 'real'
#         streamOpts = None
#         ind = indCC

#         formatter = None
#         pcolorOpts = None


#         if Type == 'Total':
#             u = 1./(mapping*mtrue)
#         elif Type == 'Primary':
#             u = 1./(mapping*mhalf)
#         elif Type == 'Secondary':
#             u = 1./(mapping*mtrue) - 1./(mapping*mhalf)

#     elif Field == 'Potential':

#         label = 'Potential (V)'
#         xtype = 'CC'
#         view = 'real'
#         streamOpts = None
#         ind = indCC

#         if Type == 'Total':
#             formatter = LogFormatter(10, labelOnlyBase=False)
#             pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=10, linscale=0.1)}

#             u = total_field['phi']

#         elif Type == 'Primary':
#             formatter = LogFormatter(10, labelOnlyBase=False)
#             pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=10, linscale=0.1)}

#             u = primary_field['phi']

#         elif Type == 'Secondary':
#             formatter = None
#             pcolorOpts = None

#             uTotal = total_field['phi']
#             uPrim = primary_field['phi']
#             u = uTotal - uPrim

#     elif Field == 'E':

#         label = 'Electric Field (V/m)'
#         xtype = 'F'
#         view = 'vec'
#         streamOpts = {'color':'w'}
#         ind = indF

#         formatter = LogFormatter(10, labelOnlyBase=False)
#         pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=1e-4, linscale=0.01)}

#         if Type == 'Total':
#             u = total_field['e']

#         elif Type == 'Primary':
#             u = primary_field['e']

#         elif Type == 'Secondary':
#             uTotal = total_field['e']
#             uPrim = primary_field['e']
#             u = uTotal - uPrim

#     elif Field == 'J':

#         label = 'Current density ($A/m^2$)'
#         xtype = 'F'
#         view = 'vec'
#         streamOpts = {'color':'w'}
#         ind = indF

#         formatter = LogFormatter(10, labelOnlyBase=False)
#         pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=1e-4, linscale=0.01)}


#         if Type == 'Total':
#             u = total_field['j']

#         elif Type == 'Primary':
#             u = primary_field['j']

#         elif Type == 'Secondary':
#             uTotal = total_field['j']
#             uPrim = primary_field['j']
#             u = uTotal - uPrim

#     elif Field == 'Charge':

#         label = 'Charge Density ($C/m^2$)'
#         xtype = 'CC'
#         view = 'real'
#         streamOpts = None
#         ind = indCC

#         formatter = LogFormatter(10, labelOnlyBase=False)
#         pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=1e-13, linscale=0.01)}

#         if Type == 'Total':
#             u = total_field['q']

#         elif Type == 'Primary':
#             u = primary_field['q']

#         elif Type == 'Secondary':
#             uTotal = total_field['q']
#             uPrim = primary_field['q']
#             u = uTotal - uPrim


#     dat = meshcore.plotImage(u[ind], vType = xtype, ax=ax, grid=False,view=view, streamOpts=streamOpts, pcolorOpts = pcolorOpts) #gridOpts={'color':'k', 'alpha':0.5}

#     cb = plt.colorbar(dat[0], ax=ax,format = formatter)
#     cb.set_label(label)
#     ax.set_xlim([-40.,40.])
#     ax.set_ylim([-40.,5.])
#     ax.set_aspect('equal')
#     plt.show()
#     return


# def plate_app():
#     app = interact(plate_wrapper,
#             rhohalf = FloatSlider(min=10.,max=1000.,step=10., value = 500., continuous_update=False),
#             rhoplate = FloatSlider(min=10.,max=1000.,step=10., value = 500., continuous_update=False),
#             dx = FloatSlider(min=1.,max=20.,step=1.,value=10., continuous_update=False),
#             dz = FloatSlider(min=1.,max=20.,step=1.,value=10., continuous_update=False),
#             xc = FloatSlider(min=-30.,max=30.,step=1.,value=0., continuous_update=False),
#             zc = FloatSlider(min=-30.,max=0.,step=1.,value=-10., continuous_update=False),
#             rotAng = FloatSlider(min=-90.,max=90.,step=1.,value=0., continuous_update=False),
#             A = FloatSlider(min=-40.,max=40.,step=1.,value=-30., continuous_update=False),
#             B = FloatSlider(min=-40.,max=40.,step=1.,value=30., continuous_update=False),
#             Field = ToggleButtons(options =['Model','Potential','E','J','Charge'],value='Model'),
#             Type = ToggleButtons(options =['Total','Primary','Secondary'],value='Total')
#             #Scale = ToggleButtons(options = ['Scalar','Log'],value='Scalar')
#             )
#     return app

def getPlateCorners(xc,zc,dx,dz,rotAng):

    # Form rotation matix
    rotMat = np.array([[np.cos(rotAng*(np.pi/180.)), -np.sin(rotAng*(np.pi/180.))],[np.sin(rotAng*(np.pi/180.)), np.cos(rotAng*(np.pi/180.))]])
    originCorners = np.array([[-0.5*dx, 0.5*dz], [0.5*dx, 0.5*dz], [-0.5*dx, -0.5*dz], [0.5*dx, -0.5*dz]])

    rotPlateCorners = np.dot(originCorners,rotMat)
    plateCorners = rotPlateCorners + np.hstack([np.repeat(xc,4).reshape([4,1]),np.repeat(zc,4).reshape([4,1])])
    return plateCorners


def createPlateMod(xc,zc,dx,dz,rotAng,sigplate,sighalf):
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


def get_Surface_Potentials(src,field_obj):

    phi = field_obj[src, 'phi']
    CCLoc = mesh.gridCC
    zsurfaceLoc = np.max(CCLoc[:,1])
    surfaceInd = np.where(CCLoc[:,1] == zsurfaceLoc)
    phiSurface = phi[surfaceInd]
    xSurface = CCLoc[surfaceInd,0].T
    return xSurface,phiSurface

def sumPlateCharges(xc,zc,dx,dz,rotAng,qSecondary):
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
    print qPosData.shape
    qNegData = np.vstack([qNegLoc[:,0], qNegLoc[:,1], qNeg]).T

    qNegAvgLoc = np.average(qNegLoc,axis=0, weights=qNeg)
    qPosAvgLoc = np.average(qPosLoc,axis=0, weights=qPos)

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


# Inline functions for computing apparent resistivity
eps = 1e-9 #to stabilize division
G = lambda A, B, M, N: 1. / ( 1./(np.abs(A-M)+eps) - 1./(np.abs(M-B)+eps) - 1./(np.abs(N-A)+eps) + 1./(np.abs(N-B)+eps) )
rho_a = lambda VM,VN, A,B,M,N: (VM-VN)*2.*np.pi*G(A,B,M,N)


def plot_Surface_Potentials(dx,dz,xc,zc,rotAng,rhoplate,rhohalf,A,B,M,N,Field,Type):

    labelsize = 18.
    ticksize = 16.

    sigplate = 1./rhoplate
    sighalf = 1./rhohalf

    mtrue, mhalf,src, primary_field, total_field = plate_fields(A,B,dx,dz,xc,zc,rotAng,sigplate,sighalf)

    fig, ax = plt.subplots(2,1,figsize=(15,16),sharex=True)
    fig.subplots_adjust(right=0.8)

    xSurface, phiTotalSurface = get_Surface_Potentials(src, total_field)
    xSurface, phiPrimSurface = get_Surface_Potentials(src, primary_field)
    ylim = np.r_[-1., 1.]*np.max(np.abs(phiTotalSurface))
    xlim = np.array([-40,40])

    MInd = np.where(xSurface == M)
    NInd = np.where(xSurface == N)

    VM = phiTotalSurface[MInd[0]]
    VN = phiTotalSurface[NInd[0]]

    VMprim = phiPrimSurface[MInd[0]]
    VNprim = phiPrimSurface[NInd[0]]

    #2D geometric factor
    G2D = rhohalf/(rho_a(VMprim,VNprim,A,B,M,N))

    # Subplot 1: Full set of surface potentials
    ax[0].plot(xSurface,phiTotalSurface,color=[0.1,0.5,0.1],linewidth=2)
    ax[0].plot(xSurface,phiPrimSurface ,linestyle='dashed',linewidth=0.5,color='k')
    ax[0].grid(which='both',linestyle='-',linewidth=0.5,color=[0.2,0.2,0.2],alpha=0.5)
    ax[0].plot(A,0,'+',markersize = 12, markeredgewidth = 3, color=[1.,0.,0])
    ax[0].plot(B,0,'_',markersize = 12, markeredgewidth = 3, color=[0.,0.,1.])
    # ax[0].set_yscale('log')
    ax[0].set_ylabel('Potential, (V)',fontsize = labelsize)
    ax[0].set_xlabel('x (m)',fontsize = labelsize)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)

    ax[0].plot(M,VM,'o',color='k')
    ax[0].plot(N,VN,'o',color='k')

    xytextM = (M+0.5,np.max([np.min([VM,ylim.max()]),ylim.min()])+0.5)
    xytextN = (N+0.5,np.max([np.min([VN,ylim.max()]),ylim.min()])+0.5)

    ax[0].annotate('%2.1e'%(VM), xy=xytextM, xytext=xytextM,fontsize = labelsize)
    ax[0].annotate('%2.1e'%(VN), xy=xytextN, xytext=xytextN,fontsize = labelsize)

    ax[0].tick_params(axis='both', which='major', labelsize=ticksize)

    props = dict(boxstyle='round', facecolor='grey', alpha=0.4)
    ax[0].text(xlim.max()+1,ylim.max()-0.1*ylim.max(),'$\\rho_a$ = %2.2f'%(G2D*rho_a(VM,VN,A,B,M,N)),
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

        formatter = None
        pcolorOpts = None


        if Type == 'Total':
            u = 1./(mapping*mtrue)
        elif Type == 'Primary':
            u = 1./(mapping*mhalf)
        elif Type == 'Secondary':
            u = 1./(mapping*mtrue) - 1./(mapping*mhalf)

    elif Field == 'Potential':

        label = 'Potential (V)'
        xtype = 'CC'
        view = 'real'
        streamOpts = None
        ind = indCC

        formatter = None
        pcolorOpts = None

        if Type == 'Total':
            # formatter = LogFormatter(10, labelOnlyBase=False)
            # pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=10, linscale=0.1)}

            u = total_field[src,'phi']

        elif Type == 'Primary':
            # formatter = LogFormatter(10, labelOnlyBase=False)
            # pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=10, linscale=0.1)}

            u = primary_field[src,'phi']

        elif Type == 'Secondary':
            # formatter = None
            # pcolorOpts = None

            uTotal = total_field[src,'phi']
            uPrim = primary_field[src,'phi']
            u = uTotal - uPrim

    elif Field == 'E':

        label = 'Electric Field (V/m)'
        xtype = 'F'
        view = 'vec'
        streamOpts = {'color':'w'}
        ind = indF

        formatter = LogFormatter(10, labelOnlyBase=False)
        pcolorOpts = {'norm':matplotlib.colors.LogNorm()}

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

        formatter = LogFormatter(10, labelOnlyBase=False)
        pcolorOpts = {'norm':matplotlib.colors.LogNorm()}


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

        formatter = LogFormatter(10, labelOnlyBase=False)
        pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=1e-11, linscale=0.01)}

        if Type == 'Total':
            u = total_field[src,'charge']

        elif Type == 'Primary':
            u = primary_field[src,'charge']

        elif Type == 'Secondary':
            uTotal = total_field[src,'charge']
            uPrim = primary_field[src,'charge']
            u = uTotal - uPrim

    dat = meshcore.plotImage(u[ind], vType = xtype, ax=ax[1], grid=False,view=view, streamOpts=streamOpts, pcolorOpts = pcolorOpts) #gridOpts={'color':'k', 'alpha':0.5}
    # Get plate corners
    plateCorners = getPlateCorners(xc,zc,dx,dz,rotAng)
    # plot top of plate outline
    ax[1].plot(plateCorners[[0,1],0],plateCorners[[0,1],1],linestyle = 'dashed',color='k')
    # plot east side of plate outline
    ax[1].plot(plateCorners[[1,3],0],plateCorners[[1,3],1],linestyle = 'dashed',color='k')
    # plot bottom of plate outline
    ax[1].plot(plateCorners[[2,3],0],plateCorners[[2,3],1],linestyle = 'dashed',color='k')
    # plot west side of plate outline
    ax[1].plot(plateCorners[[0,2],0],plateCorners[[0,2],1],linestyle = 'dashed',color='k')

    if (Field == 'Charge') and (Type != 'Primary'):
        qTotal = total_field[src,'charge']
        qPrim = primary_field[src,'charge']
        qSecondary = qTotal - qPrim
        qPosSum, qNegSum, qPosAvgLoc, qNegAvgLoc = sumPlateCharges(xc,zc,dx,dz,rotAng,qSecondary)
        print(qPosAvgLoc[0])
        print(qPosAvgLoc[1])
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
    ax[1].tick_params(axis='both', which='major', labelsize=ticksize)
    cbar_ax = fig.add_axes([0.8, 0.05, 0.08, 0.5])
    cbar_ax.axis('off')
    cb = plt.colorbar(dat[0], ax=cbar_ax,format = formatter)
    cb.ax.tick_params(labelsize=ticksize)
    cb.set_label(label, fontsize=labelsize)
    ax[1].set_xlim([-40.,40.])
    ax[1].set_ylim([-40.,5.])
    ax[1].set_aspect('equal')

    plt.show()
    return fig, ax


def plate_app():
    app = interact(plot_Surface_Potentials,
                dx = FloatSlider(min=1.,max=20.,step=1.,value=10., continuous_update=False),
                dz = FloatSlider(min=1.,max=20.,step=1.,value=10., continuous_update=False),
                xc = FloatSlider(min=-30.,max=30.,step=1.,value=0., continuous_update=False),
                zc = FloatSlider(min=-30.,max=0.,step=1.,value=-10., continuous_update=False),
                rotAng = FloatSlider(min=-90.,max=90.,step=1.,value=0., continuous_update=False),
                rhoplate = FloatSlider(min=10.,max=1000.,step=10., value = 500., continuous_update=False),
                rhohalf = FloatSlider(min=10.,max=1000.,step=10., value = 500., continuous_update=False),
                A = FloatSlider(min=-30.25,max=30.25,step=0.5,value=-30.25, continuous_update=False),
                B = FloatSlider(min=-30.25,max=30.25,step=0.5,value=30.25, continuous_update=False),
                M = FloatSlider(min=-30.25,max=30.25,step=0.5,value=-10.25, continuous_update=False),
                N = FloatSlider(min=-30.25,max=30.25,step=0.5,value=10.25, continuous_update=False),
                Field = ToggleButtons(options =['Model','Potential','E','J','Charge'],value='Model'),
                Type = ToggleButtons(options =['Total','Primary','Secondary'],value='Total')
                )
    return app
