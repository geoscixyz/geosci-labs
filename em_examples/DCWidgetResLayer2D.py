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


def model_fields(A,B,zcLayer,dzLayer,xc,zc,r,sigLayer,sigTarget,sigHalf):
    # Create halfspace model
    halfspaceMod = sigHalf*np.ones([mesh.nC,])
    mHalf = np.log(halfspaceMod)
    # Add layer to model
    LayerMod = addLayer2Mod(zcLayer,dzLayer,halfspaceMod,sigLayer)
    mLayer = np.log(LayerMod)

    # Add plate or cylinder
    # fullMod = addPlate2Mod(xc,zc,dx,dz,rotAng,LayerMod,sigTarget)
    fullMod = addCylinder2Mod(xc,zc,r,LayerMod,sigTarget)
    mFull = np.log(fullMod)

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

    primary_field = problem_prim.fields(mHalf)

    total_field = problem.fields(mFull)

    return mFull,mHalf, src, primary_field, total_field


def addLayer2Mod(zcLayer,dzLayer,mod,sigLayer):

    CCLocs = mesh.gridCC

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


def addCylinder2Mod(xc,zc,r,mod,sigCylinder):

    # Get points for cylinder outline
    cylinderPoints = getCylinderPoints(xc,zc,r)

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


def addPlate2Mod(xc,zc,dx,dz,rotAng,mod,sigPlate):
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

    mod[insideInd] = sigPlate
    return mod


def get_Surface_Potentials(src,field_obj):

    phi = field_obj[src, 'phi']
    CCLoc = mesh.gridCC
    zsurfaceLoc = np.max(CCLoc[:,1])
    surfaceInd = np.where(CCLoc[:,1] == zsurfaceLoc)
    phiSurface = phi[surfaceInd]
    xSurface = CCLoc[surfaceInd,0].T
    return xSurface,phiSurface


# Inline functions for computing apparent resistivity
eps = 1e-9 #to stabilize division
G = lambda A, B, M, N: 1. / ( 1./(np.abs(A-M)+eps) - 1./(np.abs(M-B)+eps) - 1./(np.abs(N-A)+eps) + 1./(np.abs(N-B)+eps) )
rho_a = lambda VM,VN, A,B,M,N: (VM-VN)*2.*np.pi*G(A,B,M,N)


def plot_Surface_Potentials(zcLayer,dzLayer,logRhoLayer,xc,zc,r,rhoTarget,rhoHalf,A,B,M,N,Field,Type):

    labelsize = 18.
    ticksize = 16.

    sigTarget = 1./rhoTarget
    rhoLayer = np.exp(logRhoLayer)
    sigLayer = 1./rhoLayer
    sigHalf = 1./rhoHalf

    mFull, mHalf,src, primary_field, total_field = model_fields(A,B,zcLayer,dzLayer,xc,zc,r,sigLayer,sigTarget,sigHalf)

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
    G2D = rhoHalf/(rho_a(VMprim,VNprim,A,B,M,N))

    ax[0].plot(xSurface,phiTotalSurface,color=[0.1,0.5,0.1],linewidth=2)
    ax[0].plot(xSurface,phiPrimSurface ,linestyle='dashed',linewidth=0.5,color='k')
    ax[0].grid(which='both',linestyle='-',linewidth=0.5,color=[0.2,0.2,0.2],alpha=0.5)
    ax[0].plot(A,0,'+',markersize = 12, markeredgewidth = 3, color=[1.,0.,0])
    ax[0].plot(B,0,'_',markersize = 12, markeredgewidth = 3, color=[0.,0.,1.])
    ax[0].set_ylabel('Potential, (V)',fontsize = 14)
    ax[0].set_xlabel('x (m)',fontsize = 14)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)

    ax[0].plot(M,VM,'o',color='k')
    ax[0].plot(N,VN,'o',color='k')

    props = dict(boxstyle='round', facecolor='grey', alpha=0.3)

    txtsp = 1

    xytextM = (M+0.5,np.max([np.min([VM,ylim.max()]),ylim.min()])+0.5)
    xytextN = (N+0.5,np.max([np.min([VN,ylim.max()]),ylim.min()])+0.5)


    props = dict(boxstyle='round', facecolor='grey', alpha=0.4)

    ax[0].annotate('%2.1e'%(VM), xy=xytextM, xytext=xytextM,fontsize = 14)
    ax[0].annotate('%2.1e'%(VN), xy=xytextN, xytext=xytextN,fontsize = 14)

    # ax[0].plot(np.r_[M,N],np.ones(2)*VN,color='k')
    # ax[0].plot(np.r_[M,M],np.r_[VM, VN],color='k')
    # ax[0].annotate('%2.1e'%(VM-VN) , xy=(M,(VM+VN)/2), xytext=(M-9,(VM+VN)/2.),fontsize = 14)

    props = dict(boxstyle='round', facecolor='grey', alpha=0.4)
    ax[0].text(xlim.max()+1,ylim.max()-0.1*ylim.max(),'$\\rho_a$ = %2.2f'%(G2D*rho_a(VM,VN,A,B,M,N)),
                verticalalignment='bottom', bbox=props, fontsize = labelsize)

    if Field == 'Model':

        label = 'Resisitivity (ohm-m)'
        xtype = 'CC'
        view = 'real'
        streamOpts = None
        ind = indCC

        formatter = LogFormatter(10, labelOnlyBase=False)
        pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=10, linscale=0.1)}


        if Type == 'Total':
            u = 1./(mapping*mFull)
        elif Type == 'Primary':
            u = 1./(mapping*mHalf)
        elif Type == 'Secondary':
            u = 1./(mapping*mFull) - 1./(mapping*mHalf)

    elif Field == 'Potential':

        label = 'Potential (V)'
        xtype = 'CC'
        view = 'real'
        streamOpts = None
        ind = indCC

        if Type == 'Total':
            formatter = LogFormatter(10, labelOnlyBase=False)
            pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=10, linscale=0.1)}

            u = total_field[src,'phi']

        elif Type == 'Primary':
            formatter = LogFormatter(10, labelOnlyBase=False)
            pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=10, linscale=0.1)}

            u = primary_field[src,'phi']

        elif Type == 'Secondary':
            formatter = None
            pcolorOpts = None

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
        pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=1e-4, linscale=0.01)}

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
        pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=1e-4, linscale=0.01)}


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
        pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=1e-13, linscale=0.01)}

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
    cylinderPoints = getCylinderPoints(xc,zc,r)

    ax[1].plot(cylinderPoints[:,0],cylinderPoints[:,1], linestyle = 'dashed', color='k')
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
    cb.set_label(label)
    ax[1].set_xlim([-40.,40.])
    ax[1].set_ylim([-40.,5.])
    ax[1].set_aspect('equal')

    plt.show()
    return fig, ax


def ResLayer_app():
    app = interact(plot_Surface_Potentials,
                zcLayer = FloatSlider(min=-10.,max=0.,step=1.,value=-10., continuous_update=False, description="$dz_{layer}$"),
                dzLayer = FloatSlider(min=0.5,max=5.,step=0.5,value=1., continuous_update=False, description="$dx_{layer}$"),
                logRhoLayer = FloatSlider(min=-20.,max=20.,step=0.5, value = 10., continuous_update=False, description="$log\\left(\\rho_{layer}\\right)$"),
                xc = FloatSlider(min=-30.,max=30.,step=1.,value=0., continuous_update=False),
                zc = FloatSlider(min=-30.,max=-15.,step=0.5,value=-25., continuous_update=False),
                r = FloatSlider(min=1.,max=10.,step=0.5,value=5., continuous_update=False),
                rhoTarget = FloatSlider(min=10.,max=1000.,step=10., value = 100., continuous_update=False, description="$\\rho_{target}$"),
                rhoHalf = FloatSlider(min=10.,max=1000.,step=10., value = 500., continuous_update=False, description="$\\rho_{half}$"),
                A = FloatSlider(min=-30.25,max=30.25,step=0.5,value=-30.25, continuous_update=False),
                B = FloatSlider(min=-30.25,max=30.25,step=0.5,value=30.25, continuous_update=False),
                M = FloatSlider(min=-30.25,max=30.25,step=0.5,value=-10.25, continuous_update=False),
                N = FloatSlider(min=-30.25,max=30.25,step=0.5,value=10.25, continuous_update=False),
                Field = ToggleButtons(options =['Model','Potential','E','J','Charge'],value='Model'),
                Type = ToggleButtons(options =['Total','Primary','Secondary'],value='Total')
                )
    return app

# if __name__ == '__main__':
#     rhohalf = 500.
#     rhoplate = 500.
#     dx = 10.
#     dz = 10.
#     xc = 0.
#     zc = -10.
#     rotAng = 0.
#     A,B = -30.25, 30.25
#     M,N = -10.25, 10.25
#     Field =  'Model'
#     Type = 'Total'
#     plot_Surface_Potentials(dx,dz,xc,zc,rotAng,rhoplate,rhohalf,A,B,M,N,Field,Type)


# def DC2Dsurvey(flag="PoleDipole"):

#     if flag =="PoleDipole":
#         ntx, nmax = xr.size-2, 8
#     elif flag =="DipolePole":
#         ntx, nmax = xr.size-2, 8
#     elif flag =="DipoleDipole":
#         ntx, nmax = xr.size-3, 8
#     else:
#         raise Exception('Not Implemented')
#     xzlocs = getPseudoLocs(xr, ntx, nmax, flag)

#     txList = []
#     zloc = -2.5
#     for i in range(ntx):
#         if flag == "PoleDipole":
#             A = np.r_[xr[i], zloc]
#             B = np.r_[mesh.vectorCCx.min(), zloc]
#             if i < ntx-nmax+1:
#                 M = np.c_[xr[i+1:i+1+nmax], np.ones(nmax)*zloc]
#                 N = np.c_[xr[i+2:i+2+nmax], np.ones(nmax)*zloc]
#             else:
#                 M = np.c_[xr[i+1:ntx+1], np.ones(ntx-i)*zloc]
#                 N = np.c_[xr[i+2:i+2+nmax], np.ones(ntx-i)*zloc]
#         elif flag =="DipolePole":
#             A = np.r_[xr[i], zloc]
#             B = np.r_[xr[i+1], zloc]
#             if i < ntx-nmax+1:
#                 M = np.c_[xr[i+2:i+2+nmax], np.ones(nmax)*zloc]
#                 N = np.c_[np.ones(nmax)*mesh.vectorCCx.max(), np.ones(nmax)*zloc]
#             else:
#                 M = np.c_[xr[i+2:ntx+2], np.ones(ntx-i)*zloc]
#                 N = np.c_[np.ones(ntx-i)*mesh.vectorCCx.max(), np.ones(ntx-i)*zloc]
#         elif flag =="DipoleDipole":
#             A = np.r_[xr[i], zloc]
#             B = np.r_[xr[i+1], zloc]
#             if i < ntx-nmax:
#                 M = np.c_[xr[i+2:i+2+nmax], np.ones(len(xr[i+2:i+2+nmax]))*zloc]
#                 N = np.c_[xr[i+3:i+3+nmax], np.ones(len(xr[i+3:i+3+nmax]))*zloc]
#             else:
#                 M = np.c_[xr[i+2:len(xr)-1], np.ones(len(xr[i+2:len(xr)-1]))*zloc]
#                 N = np.c_[xr[i+3:len(xr)], np.ones(len(xr[i+3:len(xr)]))*zloc]

#         rx = DC.Rx.Dipole(M, N)
#         src = DC.Src.Dipole([rx], A, B)
#         txList.append(src)

#     survey = DC.Survey(txList)
#     problem = DC.Problem3D_CC(mesh, sigmaMap = mapping)
#     problem.pair(survey)

#     sigblk, sighalf = 2e-2, 2e-3
#     xc, zc, r = -15, -8, 4
#     mtrue = np.r_[np.log(sigblk), np.log(sighalf), xc, zc, r]
#     dtrue = survey.dpred(mtrue)
#     perc = 0.1
#     floor = np.linalg.norm(dtrue)*1e-3
#     np.random.seed([1])
#     uncert = np.random.randn(survey.nD)*perc + floor
#     dobs = dtrue + uncert

#     return dobs, uncert, survey, xzlocs

# def getPseudoLocs(xr, ntx, nmax, flag = "PoleDipole"):
#     xloc = []
#     yloc = []
#     for i in range(ntx):
#         if i < ntx-nmax+1:

#             if flag is 'DipoleDipole':
#                 txmid = xr[i]+dxr[i]*0.5
#                 rxmid = xr[i+1:i+1+nmax]+dxr[i+1:i+1+nmax]*0.5

#             elif flag is 'PoleDipole':
#                 txmid = xr[i]
#                 rxmid = xr[i+1:i+1+nmax]+dxr[i+1:i+1+nmax]*0.5

#             elif flag is 'DipolePole':
#                 txmid = xr[i]+dxr[i]*0.5
#                 rxmid = xr[i+1:i+1+nmax]

#             mid = (txmid+rxmid)*0.5
#             xloc.append(mid)
#             yloc.append(np.arange(nmax)+1.)
#         else:
#             if flag is 'DipoleDipole':
#                 txmid = xr[i]+dxr[i]*0.5
#                 rxmid = xr[i+1:ntx+1]+dxr[i+1:ntx+1]*0.5

#             elif flag is 'PoleDipole':
#                 txmid = xr[i]
#                 rxmid = xr[i+1:ntx+1]+dxr[i+1:ntx+1]*0.5

#             elif flag is 'DipolePole':
#                 txmid = xr[i]+dxr[i]*0.5
#                 rxmid = xr[i+1:ntx+1]

#             mid = (txmid+rxmid)*0.5
#             xloc.append(mid)
#             yloc.append(np.arange(mid.size)+1.)
#     xlocvec = np.hstack(xloc)
#     ylocvec = np.hstack(yloc)
#     return np.c_[xlocvec, ylocvec]

# def PseudoSectionPlotfnc(i,j,survey,flag="PoleDipole"):
#     matplotlib.rcParams['font.size'] = 14
#     nmax = 8
#     dx = 5
#     xr = np.arange(-40,41,dx)
#     ntx = xr.size-2
#     dxr = np.diff(xr)
#     TxObj = survey.srcList
#     TxLoc = TxObj[i].loc
#     RxLoc = TxObj[i].rxList[0].locs
#     fig = plt.figure(figsize=(10, 3))
#     ax = fig.add_subplot(111, autoscale_on=False, xlim=(xr.min()-5, xr.max()+5), ylim=(nmax+1, -2))
#     plt.plot(xr, np.zeros_like(xr), 'ko', markersize=4)
#     if flag == "PoleDipole":
#         plt.plot(TxLoc[0][0], np.zeros(1), 'rv', markersize=10)
#         # print([TxLoc[0][0],0])
#         ax.annotate('A', xy=(TxLoc[0][0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
#     else:
#         plt.plot([TxLoc[0][0],TxLoc[1][0]], np.zeros(2), 'rv', markersize=10)
#         # print([[TxLoc[0][0],0],[TxLoc[1][0],0]])
#         ax.annotate('A', xy=(TxLoc[0][0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
#         ax.annotate('B', xy=(TxLoc[1][0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
#     # for i in range(ntx):
#     if i < ntx-nmax+1:
#         if flag == "PoleDipole":
#             txmid = TxLoc[0][0]
#         else:
#             txmid = (TxLoc[0][0] + TxLoc[1][0])*0.5

#         MLoc = RxLoc[0][j]
#         NLoc = RxLoc[1][j]
#         # plt.plot([MLoc[0],NLoc[0]], np.zeros(2), 'b^', markersize=10)
#         # ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
#         # ax.annotate('N', xy=(NLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
#         if flag == "DipolePole":
#             plt.plot(MLoc[0], np.zeros(1), 'bv', markersize=10)
#             ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
#             rxmid = MLoc[0]
#         else:
#             rxmid = (MLoc[0]+NLoc[0])*0.5
#             plt.plot(MLoc[0], np.zeros(1), 'bv', markersize=10)
#             plt.plot(NLoc[0], np.zeros(1), 'b^', markersize=10)
#             ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
#             ax.annotate('N', xy=(NLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
#         mid = (txmid+rxmid)*0.5
#         midSep = np.sqrt(np.square(txmid-rxmid))
#         plt.plot(txmid, np.zeros(1), 'ro')
#         plt.plot(rxmid, np.zeros(1), 'bo')
#         plt.plot(mid, midSep/2., 'go')
#         plt.plot(np.r_[txmid, mid], np.r_[0, midSep/2.], 'k:')
#         # for j in range(nmax):
#             # plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j+1], 'k:')
#         plt.plot(np.r_[rxmid, mid], np.r_[0, midSep/2.], 'k:')

#     else:
#         if flag == "PoleDipole":
#             txmid = TxLoc[0][0]
#         else:
#             txmid = (TxLoc[0][0] + TxLoc[1][0])*0.5


#         MLoc = RxLoc[0][j]
#         NLoc = RxLoc[1][j]
#         if flag == "DipolePole":
#             plt.plot(MLoc[0], np.zeros(1), 'bv', markersize=10)
#             ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
#             rxmid = MLoc[0]
#         else:
#             rxmid = (MLoc[0]+NLoc[0])*0.5
#             plt.plot(MLoc[0], np.zeros(1), 'bv', markersize=10)
#             plt.plot(NLoc[0], np.zeros(1), 'b^', markersize=10)
#             ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
#             ax.annotate('N', xy=(NLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
#         # plt.plot([MLoc[0],NLoc[0]], np.zeros(2), 'b^', markersize=10)
#         # ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
#         # ax.annotate('N', xy=(NLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')

#         # rxmid = xr[i+1:ntx+1]+dxr[i+1:ntx+1]*0.5
#         mid = (txmid+rxmid)*0.5
#         plt.plot((txmid+rxmid)*0.5, np.arange(mid.size)+1., 'bo')
#         plt.plot(rxmid, np.zeros(rxmid.size), 'go')
#         plt.plot(np.r_[txmid, mid[-1]], np.r_[0, mid.size], 'k:')
#         for j in range(ntx-i):
#             plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j+1], 'k:')
#     plt.xlabel("X (m)")
#     plt.ylabel("N-spacing")
#     plt.xlim(xr.min()-5, xr.max()+5)
#     plt.ylim(nmax*dx/2+dx, -2*dx)
#     plt.show()
#     return

# def DipoleDipolefun(i):
#     matplotlib.rcParams['font.size'] = 14
#     plt.figure(figsize=(10, 3))
#     nmax = 8
#     xr = np.linspace(-40, 40, 20)
#     ntx = xr.size-2
#     dxr = np.diff(xr)
#     plt.plot(xr[:-1]+dxr*0.5, np.zeros_like(xr[:-1]), 'ko')
#     plt.plot(xr[i]+dxr[i]*0.5, np.zeros(1), 'ro')
#     # for i in range(ntx):
#     if i < ntx-nmax+1:
#         txmid = xr[i]+dxr[i]*0.5
#         rxmid = xr[i+1:i+1+nmax]+dxr[i+1:i+1+nmax]*0.5
#         mid = (txmid+rxmid)*0.5
#         plt.plot(rxmid, np.zeros(rxmid.size), 'go')
#         plt.plot(mid, np.arange(nmax)+1., 'bo')
#         plt.plot(np.r_[txmid, mid[-1]], np.r_[0, nmax], 'k:')
#         for j in range(nmax):
#             plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j+1], 'k:')

#     else:
#         txmid = xr[i]+dxr[i]*0.5
#         rxmid = xr[i+1:ntx+1]+dxr[i+1:ntx+1]*0.5
#         mid = (txmid+rxmid)*0.5
#         plt.plot((txmid+rxmid)*0.5, np.arange(mid.size)+1., 'bo')
#         plt.plot(rxmid, np.zeros(rxmid.size), 'go')
#         plt.plot(np.r_[txmid, mid[-1]], np.r_[0, mid.size], 'k:')
#         for j in range(ntx-i):
#             plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j+1], 'k:')
#     plt.xlabel("X (m)")
#     plt.ylabel("N-spacing")
#     plt.xlim(xr.min(), xr.max())
#     plt.ylim(nmax+1, -1)
#     plt.show()
#     return

# def PseudoSectionWidget(survey,flag):
#     dx = 5
#     xr = np.arange(-40,41,dx)
#     if flag =="PoleDipole":
#         ntx, nmax = xr.size-2, 8
#         dxr = np.diff(xr)
#     elif flag =="DipolePole":
#         ntx, nmax = xr.size-1, 7
#         dxr = xr
#     elif flag =="DipoleDipole":
#         ntx, nmax = xr.size-3, 8
#         dxr = np.diff(xr)
#     xzlocs = getPseudoLocs(dxr, ntx, nmax,flag)
#     PseudoSectionPlot = lambda i,j: PseudoSectionPlotfnc(i,j,survey,flag)
#     return interact(PseudoSectionPlot, i=IntSlider(min=0, max=ntx-1, step = 1, value=0),j=IntSlider(min=0, max=nmax-1, step = 1, value=0))

# def MidpointPseudoSectionWidget():
#     ntx = 18
#     return interact(DipoleDipolefun, i=IntSlider(min=0, max=ntx-1, step = 1, value=0))

# def DC2Dfwdfun(mesh, survey, mapping, xr, xzlocs, rhohalf, rhoblk, xc, zc, dx, dz, rotAng, dobs, uncert, predmis, nmax=8, plotFlag=None):
#     matplotlib.rcParams['font.size'] = 14
#     sighalf, sigblk = 1./rhohalf, 1./rhoblk
#     m0 = sighalf*np.ones([mesh.nC,])
#     dini = survey.dpred(m0)
#     mtrue = createPlateMod(xc,zc,dx,dz,rotAng,sigplate,sighalf)
#     dpred  = survey.dpred(mtrue)
#     xi, yi = np.meshgrid(np.linspace(xr.min(), xr.max(), 120), np.linspace(1., nmax, 100))
#     #Cheat to compute a geometric factor define as G = dV_halfspace / rho_halfspace
#     appres = dpred/dini/sighalf
#     appresobs = dobs/dini/sighalf
#     pred = pylab.griddata(xzlocs[:,0], xzlocs[:,1], appres, xi, yi, interp='linear')

#     if plotFlag is not None:
#         fig = plt.figure(figsize = (12, 6))
#         ax1 = plt.subplot(211)
#         ax2 = plt.subplot(212)

#         dat1 = mesh.plotImage(np.log10(1./(mapping*mtrue)), ax=ax1, clim=(1, 3), grid=True, gridOpts={'color':'k', 'alpha':0.5})
#         cb1ticks = [1.,2.,3.]
#         cb1 = plt.colorbar(dat1[0], ax=ax1,ticks=cb1ticks)#,tickslabel =)  #, format="$10^{%4.1f}$")
#         cb1.ax.set_yticklabels(['{:.0f}'.format(10.**x) for x in cb1ticks])#, fontsize=16, weight='bold')
#         cb1.set_label("Resistivity (ohm-m)")
#         ax1.set_ylim(-20, 0.)
#         ax1.set_xlim(-40, 40)
#         ax1.set_xlabel("")
#         ax1.set_ylabel("Depth (m)")
#         ax1.set_aspect('equal')

#         dat2 = ax2.contourf(xi, yi, pred, 10)
#         ax2.contour(xi, yi, pred, 10, colors='k', alpha=0.5)
#         ax2.plot(xzlocs[:,0], xzlocs[:,1],'k.', ms = 3)
#         cb2 = plt.colorbar(dat2, ax=ax2)#, ticks=np.linspace(0, 3, 5))#format="$10^{%4.1f}$")
#         cb2.set_label("Apparent Resistivity \n (ohm-m)")
#         ax2.text(-38, 7, "Predicted")

#         ax2.set_ylim(nmax+1, 0.)
#         ax2.set_ylabel("N-spacing")
#         ax2.set_xlabel("Distance (m)")

#     else:
#         obs = pylab.griddata(xzlocs[:,0], xzlocs[:,1], appresobs, xi, yi, interp='linear')
#         fig = plt.figure(figsize = (12, 9))
#         ax1 = plt.subplot(311)
#         dat1 = mesh.plotImage(np.log10(1./(mapping*mtrue)), ax=ax1, clim=(1, 3), grid=True, gridOpts={'color':'k', 'alpha':0.5})
#         cb1ticks = [1.,2.,3.]
#         cb1 = plt.colorbar(dat1[0], ax=ax1,ticks=cb1ticks)#,tickslabel =)  #, format="$10^{%4.1f}$")
#         cb1.ax.set_yticklabels(['{:.0f}'.format(10.**x) for x in cb1ticks])#, fontsize=16, weight='bold')
#         cb1.set_label("Resistivity (ohm-m)")
#         ax1.set_ylim(-20, 0.)
#         ax1.set_xlim(-40, 40)
#         ax1.set_xlabel("")
#         ax1.set_ylabel("Depth (m)")
#         ax1.set_aspect('equal')

#         ax2 = plt.subplot(312)
#         dat2 = ax2.contourf(xi, yi, obs, 10)
#         ax2.contour(xi, yi, obs, 10, colors='k', alpha=0.5)
#         ax2.plot(xzlocs[:,0], xzlocs[:,1],'k.', ms = 3)
#         cb2 = plt.colorbar(dat2, ax=ax2)#, ticks=np.linspace(0, 3, 5),format="$10^{%4.1f}$")

#         cb2.set_label("Apparent Resistivity \n (ohm-m)")
#         ax2.set_ylim(nmax+1, 0.)
#         ax2.set_ylabel("N-spacing")
#         ax2.text(-38, 7, "Observed")

#         ax3 = plt.subplot(313)
#         if predmis=="pred":
#             dat3 = ax3.contourf(xi, yi, pred, 10)
#             ax3.contour(xi, yi, pred, 10, colors='k', alpha=0.5)
#             ax3.plot(xzlocs[:,0], xzlocs[:,1],'k.', ms = 3)
#             cb3 = plt.colorbar(dat3, ax=ax3, ticks=np.linspace(appres.min(), appres.max(), 5),format="%4.0f")
#             cb3.set_label("Apparent Resistivity \n (ohm-m)")
#             ax3.text(-38, 7, "Predicted")
#         elif predmis=="mis":
#             mis = (appresobs-appres)/(0.1*appresobs)
#             Mis = pylab.griddata(xzlocs[:,0], xzlocs[:,1], mis, xi, yi, interp='linear')
#             dat3 = ax3.contourf(xi, yi, Mis, 10)
#             ax3.contour(xi, yi, Mis, 10, colors='k', alpha=0.5)
#             ax3.plot(xzlocs[:,0], xzlocs[:,1],'k.', ms = 3)
#             cb3 = plt.colorbar(dat3, ax=ax3, ticks=np.linspace(mis.min(), mis.max(), 5), format="%4.2f")
#             cb3.set_label("Normalized misfit")
#             ax3.text(-38, 7, "Misifit")
#         ax3.set_ylim(nmax+1, 0.)
#         ax3.set_ylabel("N-spacing")
#         ax3.set_xlabel("Distance (m)")

#     plt.show()
#     return

# def DC2DPseudoWidgetWrapper(rhohalf, rhosph, xc, zc, dx, dz, rotAng, surveyType):
#     dobs, uncert, survey, xzlocs = DC2Dsurvey(surveyType)
#     DC2Dfwdfun(mesh, survey, mapping, xr, xzlocs, rhohalf, rhosph, xc, zc, dx, dz, rotAng, dobs, uncert, 'pred',plotFlag='PredOnly')
#     return None

# def DC2DPseudoWidget():
#     # print xzlocs
#     Q = interact(DC2DPseudoWidgetWrapper,
#          rhohalf = FloatSlider(min=10, max=1000, step=1, value = 1000, continuous_update=False),
#          rhosph = FloatSlider(min=10, max=1000, step=1, value = 50, continuous_update=False),
#          dx = FloatSlider(min=1.,max=20.,step=1.,value=10., continuous_update=False),
#          dz = FloatSlider(min=1.,max=20.,step=1.,value=10., continuous_update=False),
#          xc = FloatSlider(min=-30.,max=30.,step=1.,value=0., continuous_update=False),
#          zc = FloatSlider(min=-30.,max=0.,step=1.,value=-10., continuous_update=False),
#          rotAng = FloatSlider(min=-90.,max=90.,step=1.,value=0., continuous_update=False),
#          surveyType = ToggleButtons(options=['DipoleDipole','PoleDipole','DipolePole'])
#         )
#     return Q

# def DC2DfwdWrapper(rhohalf, rhosph,xc, zc, dx, dz, rotAng, predmis, surveyType):
#     dobs, uncert, survey, xzlocs = DC2Dsurvey(surveyType)
#     DC2Dfwdfun(mesh, survey, mapping, xr, xzlocs, rhohalf, rhosph, xc, zc, dx, dz, rotAng, dobs, uncert, predmis)
#     return None

# def DC2DfwdWidget():
#     # print xzlocs
#     Q = interact(DC2DfwdWrapper,
#          rhohalf = FloatSlider(min=10, max=1000, step=1, value = 1000, continuous_update=False),
#          rhosph = FloatSlider(min=10, max=1000, step=1, value = 50, continuous_update=False),
#          dx = FloatSlider(min=1.,max=20.,step=1.,value=10., continuous_update=False),
#          dz = FloatSlider(min=1.,max=20.,step=1.,value=10., continuous_update=False),
#          xc = FloatSlider(min=-30.,max=30.,step=1.,value=0., continuous_update=False),
#          zc = FloatSlider(min=-30.,max=0.,step=1.,value=-10., continuous_update=False),
#          rotAng = FloatSlider(min=-90.,max=90.,step=1.,value=0., continuous_update=False),
#          predmis = ToggleButtons(options=['pred','mis']),
#          surveyType = ToggleButtons(options=['DipoleDipole','PoleDipole','DipolePole'])
#         )
#     return Q
