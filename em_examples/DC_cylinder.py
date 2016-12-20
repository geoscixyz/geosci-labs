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

        x0 = [xc[0]-hx[0]*0.5, yc[0]-hy[0]*0.5]

        meshCore = Mesh.TensorMesh([hx, hy], x0=x0)

        actind = (mesh.gridCC[:,0]>xmin) & (mesh.gridCC[:,0]<xmax)

    elif mesh.dim == 2:
        xmin, xmax = xyzlim[0,0], xyzlim[0,1]
        ymin, ymax = xyzlim[1,0], xyzlim[1,1]

        xind = np.logical_and(mesh.vectorCCx>xmin, mesh.vectorCCx<xmax)
        yind = np.logical_and(mesh.vectorCCy>ymin, mesh.vectorCCy<ymax)

        xc = mesh.vectorCCx[xind]
        yc = mesh.vectorCCy[yind]

        hx = mesh.hx[xind]
        hy = mesh.hy[yind]

        x0 = [xc[0]-hx[0]*0.5, yc[0]-hy[0]*0.5]

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
        yc = mesh.vectorCCy[yind]
        zc = mesh.vectorCCz[zind]

        hx = mesh.hx[xind]
        hy = mesh.hy[yind]
        hz = mesh.hz[zind]

        x0 = [xc[0]-hx[0]*0.5, yc[0]-hy[0]*0.5, zc[0]-hz[0]*0.5]

        meshCore = Mesh.TensorMesh([hx, hy, hz], x0=x0)

        actind = (mesh.gridCC[:,0]>xmin) & (mesh.gridCC[:,0]<xmax) \
               & (mesh.gridCC[:,1]>ymin) & (mesh.gridCC[:,1]<ymax) \
               & (mesh.gridCC[:,2]>zmin) & (mesh.gridCC[:,2]<zmax)

    else:
        raise(Exception("Not implemented!"))


    return actind, meshCore

# Mesh, sigmaMap can be globals global
npad = 12
growrate = 3.
cs = 0.5
hx = [(cs,npad, -growrate),(cs,200),(cs,npad, growrate)]
hy = [(cs,npad, -growrate),(cs,100)]
mesh = Mesh.TensorMesh([hx, hy], "CN")
circmap = Maps.CircleMap(mesh)
circmap.slope = 1e5
sigmaMap = circmap
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


def cylinder_fields(A,B,r,sigcyl,sighalf,xc=0.,yc=-20.):

    mhalf = np.r_[np.log(sighalf), np.log(sighalf), xc, yc, r]
    mtrue = np.r_[np.log(sigcyl), np.log(sighalf), xc, yc, r]
    
    Mx = np.empty(shape=(0, 2))
    Nx = np.empty(shape=(0, 2))
    #rx = DC.Rx.Dipole_ky(Mx,Nx)
    rx = DC.Rx.Dipole(Mx,Nx)
    src = DC.Src.Dipole([rx], np.r_[A,0.], np.r_[B,0.])
    #survey = DC.Survey_ky([src])
    survey = DC.Survey([src])
    survey_prim = DC.Survey([src])
    #problem = DC.Problem2D_CC(mesh, sigmaMap = sigmaMap)
    problem = DC.Problem3D_CC(mesh, sigmaMap = sigmaMap)
    problem_prim = DC.Problem3D_CC(mesh, sigmaMap = sigmaMap)
    problem.Solver = SolverLU
    problem_prim.Solver = SolverLU
    problem.pair(survey)
    problem_prim.pair(survey_prim)

    primary_field = problem_prim.fields(mhalf)
    #phihalf = f[src, 'phi', 15]
    #ehalf = f[src, 'e']
    #jhalf = f[src, 'j']
    #charge = f[src, 'charge']

    total_field = problem.fields(mtrue)
    #phi = f[src, 'phi', 15]
    #e = f[src, 'e']
    #j = f[src, 'j']
    #charge = f[src, 'charge']

    return mtrue,mhalf, src, total_field, primary_field


def get_Surface_Potentials(src,field_dict):

    phi = field_dict[src,'phi']
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

def plot_Surface_Potentials(A,B,M,N,r,rhocyl,rhohalf,xc,yc,Field,Type):

    labelsize = 18.
    ticksize = 16.

    sigcyl = 1./rhocyl
    sighalf = 1./rhohalf

    mtrue, mhalf, src, total_field, primary_field = cylinder_fields(A,B,r,sigcyl,sighalf,xc,yc)

    #fig, ax = plt.subplots(3,1,figsize=(18,28),sharex=True)
    fig, ax = plt.subplots(2,1,figsize=(15,16),sharex=True)
    fig.subplots_adjust(right=0.8)

    xSurface, phiTotalSurface = get_Surface_Potentials(src,total_field)
    xSurface, phiPrimSurface = get_Surface_Potentials(src,primary_field)
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

    #Subplot 2: Fields
    ax[1].plot(np.arange(-r,r+r/10,r/10)+xc,np.sqrt(-np.arange(-r,r+r/10,r/10)**2.+r**2.)+yc,linestyle = 'dashed',color='k')
    ax[1].plot(np.arange(-r,r+r/10,r/10)+xc,-np.sqrt(-np.arange(-r,r+r/10,r/10)**2.+r**2.)+yc,linestyle = 'dashed',color='k')

    if Field == 'Model':
       
        label = 'Resisitivity (ohm-m)'
        xtype = 'CC'
        view = 'real'
        streamOpts = None
        ind = indCC

        formatter = None
        pcolorOpts = None


        if Type == 'Total':
            u = 1./(sigmaMap*mtrue)
        elif Type == 'Primary':
            u = 1./(sigmaMap*mhalf)
        elif Type == 'Secondary':
            u = 1./(sigmaMap*mtrue) - 1./(sigmaMap*mhalf)
    
    elif Field == 'Potential':
        
        label = 'Potential (V)'
        xtype = 'CC'
        view = 'real'
        streamOpts = None
        ind = indCC

        formatter = None
        pcolorOpts = None


        if Type == 'Total':
            u = total_field[src, 'phi']

        elif Type == 'Primary':
            u = primary_field[src,'phi']

        elif Type == 'Secondary':
            u = total_field[src, 'phi'] - primary_field[src, 'phi']

    elif Field == 'E':
        
        label = 'Electric Field (V/m)'
        xtype = 'F'
        view = 'vec'
        streamOpts = {'color':'w'}
        ind = indF

        formatter = LogFormatter(10, labelOnlyBase=False) 
        pcolorOpts = {'norm':matplotlib.colors.LogNorm()}
        
        if Type == 'Total':
            u = total_field[src, 'e']

        elif Type == 'Primary':
            u = primary_field[src,'e']
        
        elif Type == 'Secondary':
            u = total_field[src, 'e'] - primary_field[src, 'e']
    
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
        
        elif Type == 'Secondary':
            u = total_field[src,'j'] - primary_field[src,'j']

        elif Type == 'Primary':
            u = primary_field[src,'j']

    elif Field == 'Charge':
        
        label = 'Charge Density ($C/m^2$)'
        xtype = 'CC'
        view = 'real'
        streamOpts = None
        ind = indCC

        formatter = LogFormatter(10, labelOnlyBase=False) 
        pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=1e-11,linscale=1e-01)}
        
        if Type == 'Total':
            u = total_field[src,'charge']
        elif Type == 'Primary':
            u = primary_field[src,'charge']
        elif Type == 'Secondary':
            u = total_field[src,'charge']-primary_field[src,'charge']

    
    dat = meshcore.plotImage(u[ind], vType = xtype, ax=ax[1], grid=False,view=view, streamOpts=streamOpts, pcolorOpts = pcolorOpts) #gridOpts={'color':'k', 'alpha':0.5}
    
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



def cylinder_app():
    app = interact(plot_Surface_Potentials,
            rhohalf = FloatSlider(min=10.,max=1000.,step=10., value = 500., continuous_update=False),
            rhocyl = FloatSlider(min=10.,max=1000.,step=10., value = 500., continuous_update=False),
            r = FloatSlider(min=1.,max=20.,step=1.,value=10., continuous_update=False),
            xc = FloatSlider(min=-20.,max=20.,step=1.,value=0., continuous_update=False),
            yc = FloatSlider(min=-30.,max=0.,step=1.,value=-20., continuous_update=False),
            A = FloatSlider(min=-30.25,max=30.25,step=0.5,value=-30.25, continuous_update=False),
            B = FloatSlider(min=-30.25,max=30.25,step=0.5,value=30.25, continuous_update=False),
            M = FloatSlider(min=-30.25,max=30.25,step=0.5,value=-10.25, continuous_update=False),
            N = FloatSlider(min=-30.25,max=30.25,step=0.5,value=10.25, continuous_update=False),
            Field = ToggleButtons(options =['Model','Potential','E','J','Charge'],value='Model'),
            Type = ToggleButtons(options =['Total','Primary','Secondary'],value='Total')
            #Scale = ToggleButtons(options = ['Scalar','Log'],value='Scalar')
            )
    return app