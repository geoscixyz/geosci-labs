from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from SimPEG import Mesh, Maps, SolverLU, Utils
import numpy as np
from SimPEG.EM.Static import DC
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import LogFormatter
from matplotlib.path import Path
import matplotlib.patches as patches

from .Base import widgetify

import warnings
warnings.filterwarnings('ignore') # ignore warnings: only use this once you are sure things are working
from ipywidgets import (
    interactive, IntSlider, FloatSlider, FloatText, ToggleButtons, VBox
)


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

# Mesh, mapping can be globals global
npad = 8
cs = 0.5
hx = [(cs,npad, -1.3),(cs,200),(cs,npad, 1.3)]
hy = [(cs,npad, -1.3),(cs,100)]
mesh = Mesh.TensorMesh([hx, hy], "CN")
circmap = Maps.ParametricCircleMap(mesh)
circmap.slope = 1e5
mapping = circmap
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


def cylinder_fields(A,B,r,sigcyl,sighalf):
    xc, yc = 0.,-20.
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
    #problem = DC.Problem2D_CC(mesh, sigmaMap = mapping)
    problem = DC.Problem3D_CC(mesh, sigmaMap = mapping)
    problem_prim = DC.Problem3D_CC(mesh, sigmaMap = mapping)
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

def cylinder_wrapper(A,B,r,rhocyl,rhohalf,Field,Type):
    xc, yc = 0.,-20.
    sigcyl = 1./rhocyl
    sighalf = 1./rhohalf

    mtrue, mhalf,src, total_field, primary_field = cylinder_fields(A,B,r,sigcyl,sighalf)

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111,autoscale_on=False)
    ax.plot(A,1.,'+',markersize = 12, markeredgewidth = 3, color=[1.,0.,0])
    ax.plot(B,1.,'_',markersize = 12, markeredgewidth = 3, color=[0.,0.,1.])
    ax.plot(np.arange(-r,r+r/10,r/10),np.sqrt(-np.arange(-r,r+r/10,r/10)**2.+r**2.)+yc,linestyle = 'dashed',color='k')
    ax.plot(np.arange(-r,r+r/10,r/10),-np.sqrt(-np.arange(-r,r+r/10,r/10)**2.+r**2.)+yc,linestyle = 'dashed',color='k')

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
        pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=1e-4, linscale=0.01)}

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
        pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=1e-4, linscale=0.01)}


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
        pcolorOpts = {'norm':matplotlib.colors.SymLogNorm(linthresh=1e-12, linscale=0.01)}

        if Type == 'Total':
            u = total_field[src,'charge']
        elif Type == 'Primary':
            u = primary_field[src,'charge']
        elif Type == 'Secondary':
            u = total_field[src,'charge']-primary_field[src,'charge']


    dat = meshcore.plotImage(u[ind], vType = xtype, ax=ax, grid=False,view=view, streamOpts=streamOpts, pcolorOpts = pcolorOpts) #gridOpts={'color':'k', 'alpha':0.5}

    cb = plt.colorbar(dat[0], ax=ax,format = formatter)
    cb.set_label(label)
    ax.set_xlim([-40.,40.])
    ax.set_ylim([-40.,5.])
    ax.set_aspect('equal')
    plt.show()
    return


def cylinder_app():
    f = cylinder_wrapper
    app = interactive(f,
            rhohalf = FloatSlider(min=10.,max=1000.,step=10., value = 500., continuous_update=False),
            rhocyl = FloatSlider(min=10.,max=1000.,step=10., value = 500., continuous_update=False),
            r = FloatSlider(min=1.,max=20.,step=1.,value=10., continuous_update=False),
            A = FloatSlider(min=-40.,max=40.,step=1.,value=-30., continuous_update=False),
            B = FloatSlider(min=-40.,max=40.,step=1.,value=30., continuous_update=False),
            Field = ToggleButtons(options =['Model','Potential','E','J','Charge'],value='Model'),
            Type = ToggleButtons(options =['Total','Primary','Secondary'],value='Total')
            #Scale = ToggleButtons(options = ['Scalar','Log'],value='Scalar')
            )
    w = VBox(app.children)
    f.widget = w
    defaults=dict([(child.description, child.value) for child in app.children])
    iplot.on_displayed(f(**defaults))
    return w


def DC2Dsurvey(flag="PoleDipole"):

    if flag == "PoleDipole":
        ntx, nmax = xr.size-2, 8
    elif flag == "DipolePole":
        ntx, nmax = xr.size-2, 8
    elif flag == "DipoleDipole":
        ntx, nmax = xr.size-3, 8
    else:
        raise Exception('Not Implemented')
    xzlocs = getPseudoLocs(xr, ntx, nmax, flag)

    txList = []
    zloc = -2.5
    for i in range(ntx):
        if flag == "PoleDipole":
            A = np.r_[xr[i], zloc]
            B = np.r_[mesh.vectorCCx.min(), zloc]
            if i < ntx-nmax+1:
                M = np.c_[xr[i+1:i+1+nmax], np.ones(nmax)*zloc]
                N = np.c_[xr[i+2:i+2+nmax], np.ones(nmax)*zloc]
            else:
                M = np.c_[xr[i+1:ntx+1], np.ones(ntx-i)*zloc]
                N = np.c_[xr[i+2:i+2+nmax], np.ones(ntx-i)*zloc]
        elif flag == "DipolePole":
            A = np.r_[xr[i], zloc]
            B = np.r_[xr[i+1], zloc]
            if i < ntx-nmax+1:
                M = np.c_[xr[i+2:i+2+nmax], np.ones(nmax)*zloc]
                N = np.c_[np.ones(nmax)*mesh.vectorCCx.max(), np.ones(nmax)*zloc]
            else:
                M = np.c_[xr[i+2:ntx+2], np.ones(ntx-i)*zloc]
                N = np.c_[np.ones(ntx-i)*mesh.vectorCCx.max(), np.ones(ntx-i)*zloc]
        elif flag == "DipoleDipole":
            A = np.r_[xr[i], zloc]
            B = np.r_[xr[i+1], zloc]
            if i < ntx-nmax:
                M = np.c_[xr[i+2:i+2+nmax], np.ones(len(xr[i+2:i+2+nmax]))*zloc]
                N = np.c_[xr[i+3:i+3+nmax], np.ones(len(xr[i+3:i+3+nmax]))*zloc]
            else:
                M = np.c_[xr[i+2:len(xr)-1], np.ones(len(xr[i+2:len(xr)-1]))*zloc]
                N = np.c_[xr[i+3:len(xr)], np.ones(len(xr[i+3:len(xr)]))*zloc]

        rx = DC.Rx.Dipole(M, N)
        src = DC.Src.Dipole([rx], A, B)
        txList.append(src)

    survey = DC.Survey(txList)
    problem = DC.Problem3D_CC(mesh, sigmaMap = mapping)
    problem.pair(survey)

    sigblk, sighalf = 2e-2, 2e-3
    xc, yc, r = -15, -8, 4
    mtrue = np.r_[np.log(sigblk), np.log(sighalf), xc, yc, r]
    dtrue = survey.dpred(mtrue)
    perc = 0.1
    floor = np.linalg.norm(dtrue)*1e-3
    np.random.seed([1])
    uncert = np.random.randn(survey.nD)*perc + floor
    dobs = dtrue + uncert

    return dobs, uncert, survey, xzlocs

def getPseudoLocs(xr, ntx, nmax, flag = "PoleDipole"):
    xloc = []
    yloc = []
    for i in range(ntx):
        if i < ntx-nmax+1:

            if flag == 'DipoleDipole':
                txmid = xr[i]+dxr[i]*0.5
                rxmid = xr[i+1:i+1+nmax]+dxr[i+1:i+1+nmax]*0.5

            elif flag == 'PoleDipole':
                txmid = xr[i]
                rxmid = xr[i+1:i+1+nmax]+dxr[i+1:i+1+nmax]*0.5

            elif flag == 'DipolePole':
                txmid = xr[i]+dxr[i]*0.5
                rxmid = xr[i+1:i+1+nmax]

            mid = (txmid+rxmid)*0.5
            xloc.append(mid)
            yloc.append(np.arange(nmax)+1.)
        else:
            if flag == 'DipoleDipole':
                txmid = xr[i]+dxr[i]*0.5
                rxmid = xr[i+1:ntx+1]+dxr[i+1:ntx+1]*0.5

            elif flag == 'PoleDipole':
                txmid = xr[i]
                rxmid = xr[i+1:ntx+1]+dxr[i+1:ntx+1]*0.5

            elif flag == 'DipolePole':
                txmid = xr[i]+dxr[i]*0.5
                rxmid = xr[i+1:ntx+1]

            mid = (txmid+rxmid)*0.5
            xloc.append(mid)
            yloc.append(np.arange(mid.size)+1.)
    xlocvec = np.hstack(xloc)
    ylocvec = np.hstack(yloc)
    return np.c_[xlocvec, ylocvec]

def PseudoSectionPlotfnc(i,j,survey,flag="PoleDipole"):
    matplotlib.rcParams['font.size'] = 14
    nmax = 8
    dx = 5
    xr = np.arange(-40,41,dx)
    ntx = xr.size-2
    dxr = np.diff(xr)
    TxObj = survey.srcList
    TxLoc = TxObj[i].loc
    RxLoc = TxObj[i].rxList[0].locs
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(xr.min()-5, xr.max()+5), ylim=(nmax+1, -2))
    plt.plot(xr, np.zeros_like(xr), 'ko', markersize=4)
    if flag == "PoleDipole":
        plt.plot(TxLoc[0][0], np.zeros(1), 'rv', markersize=10)
        # print([TxLoc[0][0],0])
        ax.annotate('A', xy=(TxLoc[0][0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
    else:
        plt.plot([TxLoc[0][0],TxLoc[1][0]], np.zeros(2), 'rv', markersize=10)
        # print([[TxLoc[0][0],0],[TxLoc[1][0],0]])
        ax.annotate('A', xy=(TxLoc[0][0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
        ax.annotate('B', xy=(TxLoc[1][0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
    # for i in range(ntx):
    if i < ntx-nmax+1:
        if flag == "PoleDipole":
            txmid = TxLoc[0][0]
        else:
            txmid = (TxLoc[0][0] + TxLoc[1][0])*0.5

        MLoc = RxLoc[0][j]
        NLoc = RxLoc[1][j]
        # plt.plot([MLoc[0],NLoc[0]], np.zeros(2), 'b^', markersize=10)
        # ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
        # ax.annotate('N', xy=(NLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
        if flag == "DipolePole":
            plt.plot(MLoc[0], np.zeros(1), 'bv', markersize=10)
            ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
            rxmid = MLoc[0]
        else:
            rxmid = (MLoc[0]+NLoc[0])*0.5
            plt.plot(MLoc[0], np.zeros(1), 'bv', markersize=10)
            plt.plot(NLoc[0], np.zeros(1), 'b^', markersize=10)
            ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
            ax.annotate('N', xy=(NLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
        mid = (txmid+rxmid)*0.5
        midSep = np.sqrt(np.square(txmid-rxmid))
        plt.plot(txmid, np.zeros(1), 'ro')
        plt.plot(rxmid, np.zeros(1), 'bo')
        plt.plot(mid, midSep/2., 'go')
        plt.plot(np.r_[txmid, mid], np.r_[0, midSep/2.], 'k:')
        # for j in range(nmax):
            # plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j+1], 'k:')
        plt.plot(np.r_[rxmid, mid], np.r_[0, midSep/2.], 'k:')

    else:
        if flag == "PoleDipole":
            txmid = TxLoc[0][0]
        else:
            txmid = (TxLoc[0][0] + TxLoc[1][0])*0.5


        MLoc = RxLoc[0][j]
        NLoc = RxLoc[1][j]
        if flag == "DipolePole":
            plt.plot(MLoc[0], np.zeros(1), 'bv', markersize=10)
            ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
            rxmid = MLoc[0]
        else:
            rxmid = (MLoc[0]+NLoc[0])*0.5
            plt.plot(MLoc[0], np.zeros(1), 'bv', markersize=10)
            plt.plot(NLoc[0], np.zeros(1), 'b^', markersize=10)
            ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
            ax.annotate('N', xy=(NLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
        # plt.plot([MLoc[0],NLoc[0]], np.zeros(2), 'b^', markersize=10)
        # ax.annotate('M', xy=(MLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')
        # ax.annotate('N', xy=(NLoc[0], np.zeros(1)), xycoords='data', xytext=(-4.25, 7.5), textcoords='offset points')

        # rxmid = xr[i+1:ntx+1]+dxr[i+1:ntx+1]*0.5
        mid = (txmid+rxmid)*0.5
        plt.plot((txmid+rxmid)*0.5, np.arange(mid.size)+1., 'bo')
        plt.plot(rxmid, np.zeros(rxmid.size), 'go')
        plt.plot(np.r_[txmid, mid[-1]], np.r_[0, mid.size], 'k:')
        for j in range(ntx-i):
            plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j+1], 'k:')
    plt.xlabel("X (m)")
    plt.ylabel("N-spacing")
    plt.xlim(xr.min()-5, xr.max()+5)
    plt.ylim(nmax*dx/2+dx, -2*dx)
    plt.show()
    return

def DipoleDipolefun(i):
    matplotlib.rcParams['font.size'] = 14
    plt.figure(figsize=(10, 3))
    nmax = 8
    xr = np.linspace(-40, 40, 20)
    ntx = xr.size-2
    dxr = np.diff(xr)
    plt.plot(xr[:-1]+dxr*0.5, np.zeros_like(xr[:-1]), 'ko')
    plt.plot(xr[i]+dxr[i]*0.5, np.zeros(1), 'ro')
    # for i in range(ntx):
    if i < ntx-nmax+1:
        txmid = xr[i]+dxr[i]*0.5
        rxmid = xr[i+1:i+1+nmax]+dxr[i+1:i+1+nmax]*0.5
        mid = (txmid+rxmid)*0.5
        plt.plot(rxmid, np.zeros(rxmid.size), 'go')
        plt.plot(mid, np.arange(nmax)+1., 'bo')
        plt.plot(np.r_[txmid, mid[-1]], np.r_[0, nmax], 'k:')
        for j in range(nmax):
            plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j+1], 'k:')

    else:
        txmid = xr[i]+dxr[i]*0.5
        rxmid = xr[i+1:ntx+1]+dxr[i+1:ntx+1]*0.5
        mid = (txmid+rxmid)*0.5
        plt.plot((txmid+rxmid)*0.5, np.arange(mid.size)+1., 'bo')
        plt.plot(rxmid, np.zeros(rxmid.size), 'go')
        plt.plot(np.r_[txmid, mid[-1]], np.r_[0, mid.size], 'k:')
        for j in range(ntx-i):
            plt.plot(np.r_[rxmid[j], mid[j]], np.r_[0, j+1], 'k:')
    plt.xlabel("X (m)")
    plt.ylabel("N-spacing")
    plt.xlim(xr.min(), xr.max())
    plt.ylim(nmax+1, -1)
    plt.show()
    return

def PseudoSectionWidget(survey,flag):
    dx = 5
    xr = np.arange(-40,41,dx)
    if flag == "PoleDipole":
        ntx, nmax = xr.size-2, 8
        dxr = np.diff(xr)
    elif flag == "DipolePole":
        ntx, nmax = xr.size-1, 7
        dxr = xr
    elif flag == "DipoleDipole":
        ntx, nmax = xr.size-3, 8
        dxr = np.diff(xr)
    xzlocs = getPseudoLocs(dxr, ntx, nmax,flag)
    PseudoSectionPlot = lambda i,j: PseudoSectionPlotfnc(i,j,survey,flag)
    return widgetify(PseudoSectionPlot, i=IntSlider(min=0, max=ntx-1, step = 1, value=0),j=IntSlider(min=0, max=nmax-1, step = 1, value=0))

def MidpointPseudoSectionWidget():
    ntx = 18
    return widgetify(DipoleDipolefun, i=IntSlider(min=0, max=ntx-1, step = 1, value=0))

def DC2Dfwdfun(mesh, survey, mapping, xr, xzlocs, rhohalf, rhoblk, xc, yc, r, dobs, uncert, predmis, nmax=8, plotFlag=None):
    matplotlib.rcParams['font.size'] = 14
    sighalf, sigblk = 1./rhohalf, 1./rhoblk
    m0 = np.r_[np.log(sighalf), np.log(sighalf), xc, yc, r]
    dini = survey.dpred(m0)
    mtrue = np.r_[np.log(sigblk), np.log(sighalf), xc, yc, r]
    dpred  = survey.dpred(mtrue)
    xi, yi = np.meshgrid(np.linspace(xr.min(), xr.max(), 120), np.linspace(1., nmax, 100))
    #Cheat to compute a geometric factor define as G = dV_halfspace / rho_halfspace
    appres = dpred/dini/sighalf
    appresobs = dobs/dini/sighalf
    pred = pylab.griddata(xzlocs[:,0], xzlocs[:,1], appres, xi, yi, interp='linear')

    if plotFlag is not None:
        fig = plt.figure(figsize = (12, 6))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        dat1 = mesh.plotImage(np.log10(1./(mapping*mtrue)), ax=ax1, clim=(1, 3), grid=True, gridOpts={'color':'k', 'alpha':0.5})
        cb1ticks = [1.,2.,3.]
        cb1 = plt.colorbar(dat1[0], ax=ax1,ticks=cb1ticks)#,tickslabel =)  #, format="$10^{%4.1f}$")
        cb1.ax.set_yticklabels(['{:.0f}'.format(10.**x) for x in cb1ticks])#, fontsize=16, weight='bold')
        cb1.set_label("Resistivity (ohm-m)")
        ax1.set_ylim(-20, 0.)
        ax1.set_xlim(-40, 40)
        ax1.set_xlabel("")
        ax1.set_ylabel("Depth (m)")
        ax1.set_aspect('equal')

        dat2 = ax2.contourf(xi, yi, pred, 10)
        ax2.contour(xi, yi, pred, 10, colors='k', alpha=0.5)
        ax2.plot(xzlocs[:,0], xzlocs[:,1],'k.', ms = 3)
        cb2 = plt.colorbar(dat2, ax=ax2)#, ticks=np.linspace(0, 3, 5))#format="$10^{%4.1f}$")
        cb2.set_label("Apparent Resistivity \n (ohm-m)")
        ax2.text(-38, 7, "Predicted")

        ax2.set_ylim(nmax+1, 0.)
        ax2.set_ylabel("N-spacing")
        ax2.set_xlabel("Distance (m)")

    else:
        obs = pylab.griddata(xzlocs[:,0], xzlocs[:,1], appresobs, xi, yi, interp='linear')
        fig = plt.figure(figsize = (12, 9))
        ax1 = plt.subplot(311)
        dat1 = mesh.plotImage(np.log10(1./(mapping*mtrue)), ax=ax1, clim=(1, 3), grid=True, gridOpts={'color':'k', 'alpha':0.5})
        cb1ticks = [1.,2.,3.]
        cb1 = plt.colorbar(dat1[0], ax=ax1,ticks=cb1ticks)#,tickslabel =)  #, format="$10^{%4.1f}$")
        cb1.ax.set_yticklabels(['{:.0f}'.format(10.**x) for x in cb1ticks])#, fontsize=16, weight='bold')
        cb1.set_label("Resistivity (ohm-m)")
        ax1.set_ylim(-20, 0.)
        ax1.set_xlim(-40, 40)
        ax1.set_xlabel("")
        ax1.set_ylabel("Depth (m)")
        ax1.set_aspect('equal')

        ax2 = plt.subplot(312)
        dat2 = ax2.contourf(xi, yi, obs, 10)
        ax2.contour(xi, yi, obs, 10, colors='k', alpha=0.5)
        ax2.plot(xzlocs[:,0], xzlocs[:,1],'k.', ms = 3)
        cb2 = plt.colorbar(dat2, ax=ax2)#, ticks=np.linspace(0, 3, 5),format="$10^{%4.1f}$")

        cb2.set_label("Apparent Resistivity \n (ohm-m)")
        ax2.set_ylim(nmax+1, 0.)
        ax2.set_ylabel("N-spacing")
        ax2.text(-38, 7, "Observed")

        ax3 = plt.subplot(313)
        if predmis=="pred":
            dat3 = ax3.contourf(xi, yi, pred, 10)
            ax3.contour(xi, yi, pred, 10, colors='k', alpha=0.5)
            ax3.plot(xzlocs[:,0], xzlocs[:,1],'k.', ms = 3)
            cb3 = plt.colorbar(dat3, ax=ax3, ticks=np.linspace(appres.min(), appres.max(), 5),format="%4.0f")
            cb3.set_label("Apparent Resistivity \n (ohm-m)")
            ax3.text(-38, 7, "Predicted")
        elif predmis=="mis":
            mis = (appresobs-appres)/(0.1*appresobs)
            Mis = pylab.griddata(xzlocs[:,0], xzlocs[:,1], mis, xi, yi, interp='linear')
            dat3 = ax3.contourf(xi, yi, Mis, 10)
            ax3.contour(xi, yi, Mis, 10, colors='k', alpha=0.5)
            ax3.plot(xzlocs[:,0], xzlocs[:,1],'k.', ms = 3)
            cb3 = plt.colorbar(dat3, ax=ax3, ticks=np.linspace(mis.min(), mis.max(), 5), format="%4.2f")
            cb3.set_label("Normalized misfit")
            ax3.text(-38, 7, "Misifit")
        ax3.set_ylim(nmax+1, 0.)
        ax3.set_ylabel("N-spacing")
        ax3.set_xlabel("Distance (m)")

    plt.show()
    return

def DC2DPseudoWidgetWrapper(rhohalf,rhosph,xc,zc,r,surveyType):
    dobs, uncert, survey, xzlocs = DC2Dsurvey(surveyType)
    DC2Dfwdfun(mesh, survey, mapping, xr, xzlocs, rhohalf, rhosph, xc, zc, r, dobs, uncert, 'pred',plotFlag='PredOnly')
    return None

def DC2DPseudoWidget():
    return widgetify(
        DC2DPseudoWidgetWrapper,
        rhohalf = FloatSlider(min=10, max=1000, step=1, value = 1000, continuous_update=False),
        rhosph = FloatSlider(min=10, max=1000, step=1, value = 50, continuous_update=False),
        xc = FloatSlider(min=-40, max=40, step=1, value =  0, continuous_update=False),
        zc = FloatSlider(min= -20, max=0, step=1, value =  -10, continuous_update=False),
        r = FloatSlider(min= 0, max=15, step=0.5, value = 5, continuous_update=False),
        surveyType = ToggleButtons(options=['DipoleDipole','PoleDipole','DipolePole'])
    )

def DC2DfwdWrapper(rhohalf,rhosph,xc,zc,r,predmis,surveyType):
    dobs, uncert, survey, xzlocs = DC2Dsurvey(surveyType)
    DC2Dfwdfun(mesh, survey, mapping, xr, xzlocs, rhohalf, rhosph, xc, zc, r, dobs, uncert, predmis)
    return None

def DC2DfwdWidget():
    return widgetify(
        DC2DfwdWrapper,
        rhohalf = FloatSlider(min=10, max=1000, step=1, value = 1000, continuous_update=False),
        rhosph = FloatSlider(min=10, max=1000, step=1, value = 50, continuous_update=False),
        xc = FloatSlider(min=-40, max=40, step=1, value =  0, continuous_update=False),
        zc = FloatSlider(min= -20, max=0, step=1, value =  -10, continuous_update=False),
        r = FloatSlider(min= 0, max=15, step=0.5, value = 5, continuous_update=False),
        predmis = ToggleButtons(options=['pred','mis']),
        surveyType = ToggleButtons(options=['DipoleDipole','PoleDipole','DipolePole'])

    )
