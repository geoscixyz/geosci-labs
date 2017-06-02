from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from matplotlib.path import Path
import matplotlib.patches as patches


##############################################
#   PLOTTING FUNCTIONS FOR WIDGETS
##############################################

def fcn_FDEM_InductionSpherePlaneWidget(xtx,ytx,ztx,m,orient,x0,y0,z0,a,sig,mur,xrx,yrx,zrx,logf,Comp,Phase):

    sig = 10**sig
    f = 10**logf

    fvec = np.logspace(0,8,41)

    xmin, xmax, dx, ymin, ymax, dy = -30., 30., 0.3, -30., 30., 0.4
    X,Y = np.mgrid[xmin:xmax+dx:dx, ymin:ymax+dy:dy]
    X = np.transpose(X)
    Y = np.transpose(Y)

    Obj = SphereFEM(m,orient,xtx,ytx,ztx)

    Hx,Hy,Hz,Habs = Obj.fcn_ComputeFrequencyResponse(f,sig,mur,a,x0,y0,z0,X,Y,zrx)
    Hxi,Hyi,Hzi,Habsi = Obj.fcn_ComputeFrequencyResponse(fvec,sig,mur,a,x0,y0,z0,xrx,yrx,zrx)

    fig1 = plt.figure(figsize=(17,6))
    Ax1 = fig1.add_axes([0.04,0,0.43,1])
    Ax2 = fig1.add_axes([0.6,0,0.4,1])

    if Comp == 'x':
        Ax1 = plotAnomalyXYplane(Ax1,f,X,Y,ztx,Hx,Comp,Phase)
        Ax1 = plotPlaceTxRxSphereXY(Ax1,xtx,ytx,xrx,yrx,x0,y0,a)
        Ax2 = plotResponseFEM(Ax2,f,fvec,Hxi,Comp)
    elif Comp == 'y':
        Ax1 = plotAnomalyXYplane(Ax1,f,X,Y,ztx,Hy,Comp,Phase)
        Ax1 = plotPlaceTxRxSphereXY(Ax1,xtx,ytx,xrx,yrx,x0,y0,a)
        Ax2 = plotResponseFEM(Ax2,f,fvec,Hyi,Comp)
    elif Comp == 'z':
        Ax1 = plotAnomalyXYplane(Ax1,f,X,Y,ztx,Hz,Comp,Phase)
        Ax1 = plotPlaceTxRxSphereXY(Ax1,xtx,ytx,xrx,yrx,x0,y0,a)
        Ax2 = plotResponseFEM(Ax2,f,fvec,Hzi,Comp)
    elif Comp == 'abs':
        Ax1 = plotAnomalyXYplane(Ax1,f,X,Y,ztx,Habs,Comp,Phase)
        Ax1 = plotPlaceTxRxSphereXY(Ax1,xtx,ytx,xrx,yrx,x0,y0,a)
        Ax2 = plotResponseFEM(Ax2,f,fvec,Habsi,Comp)

    plt.show(fig1)


def fcn_FDEM_InductionSphereProfileWidget(xtx,ztx,m,orient,x0,z0,a,sig,mur,xrx,zrx,logf,Flag):

    sig = 10**sig
    f = 10**logf

    if orient == "Vert. Coaxial":
        orient = 'x'
    elif orient == "Horiz. Coplanar":
        orient = 'z'


    # Same global functions can be used but with ytx, y0, yrx, Y = 0.

    fvec = np.logspace(0,8,41)

    xmin, xmax, dx, zmin, zmax, dz = -30., 30., 0.3, -40., 20., 0.4
    X,Z = np.mgrid[xmin:xmax+dx:dx, zmin:zmax+dz:dz]
    X = np.transpose(X)
    Z = np.transpose(Z)

    Obj = SphereFEM(m,orient,xtx,0.,ztx)

    Hxi,Hyi,Hzi,Habsi = Obj.fcn_ComputeFrequencyResponse(fvec,sig,mur,a,x0,0.,z0,xrx,0.,zrx)
    Hxf,Hyf,Hzf = fcn_ComputePrimary(m,orient,xtx,0.,ztx,x0,0.,z0)

    fig1 = plt.figure(figsize=(17,6))
    Ax1 = fig1.add_axes([0.04,0,0.38,1])
    Ax2 = fig1.add_axes([0.6,0,0.4,1])

    Ax1 = plotProfileTxRxSphere(Ax1,xtx,ztx,x0,z0,a,xrx,zrx,X,Z,orient)

    if Flag == 'Hp':
        Hpx,Hpy,Hpz = fcn_ComputePrimary(m,orient,xtx,0.,ztx,X,0.,Z)
        Ax1 = plotProfileTxRxArrow(Ax1,x0,z0,Hxf,Hzf,Flag)
        Ax1 = plotProfileXZplane(Ax1,X,Z,Hpx,Hpz,Flag)
    elif Flag == 'Hs_real':
        Hx,Hy,Hz,Habs = Obj.fcn_ComputeFrequencyResponse(f,sig,mur,a,x0,0.,z0,X,0.,Z)
        Chi = fcn_ComputeExcitation_FEM(f,sig,mur,a)
        Ax1 = plotProfileTxRxArrow(Ax1,x0,z0,np.real(Chi)*Hxf,np.real(Chi)*Hzf,Flag)
        Ax1 = plotProfileXZplane(Ax1,X,Z,np.real(Hx),np.real(Hz),Flag)
    elif Flag == 'Hs_imag':
        Hx,Hy,Hz,Habs = Obj.fcn_ComputeFrequencyResponse(f,sig,mur,a,x0,0.,z0,X,0.,Z)
        Chi = fcn_ComputeExcitation_FEM(f,sig,mur,a)
        Ax1 = plotProfileTxRxArrow(Ax1,x0,z0,np.imag(Chi)*Hxf,np.imag(Chi)*Hzf,Flag)
        Ax1 = plotProfileXZplane(Ax1,X,Z,np.imag(Hx),np.imag(Hz),Flag)



    if orient == 'x':
        Ax2 = plotResponseFEM(Ax2,f,fvec,Hxi,orient)
    elif orient == 'z':
        Ax2 = plotResponseFEM(Ax2,f,fvec,Hzi,orient)

    plt.show(fig1)

def fcn_FDEM_InductionSphereProfileEM31Widget(xtx,ztx,L,m,orient,x0,z0,a,sig,mur,logf,Flag):

    xtx = xtx - L/2
    xrx = xtx + L
    zrx = ztx
    sig = 10**sig
    f = 10**logf

    if orient == "Vert. Coaxial":
        orient = 'x'
    elif orient == "Horiz. Coplanar":
        orient = 'z'

    # Same global functions can be used but with ytx, y0, yrx, Y = 0.

    fvec = np.logspace(0,8,41)

    xmin, xmax, dx, zmin, zmax, dz = -30., 30., 0.3, -40., 20., 0.4
    X,Z = np.mgrid[xmin:xmax+dx:dx, zmin:zmax+dz:dz]
    X = np.transpose(X)
    Z = np.transpose(Z)

    Obj = SphereFEM(m,orient,xtx,0.,ztx)

    Hxi,Hyi,Hzi,Habsi = Obj.fcn_ComputeFrequencyResponse(fvec,sig,mur,a,x0,0.,z0,xrx,0.,zrx)
    Hxf,Hyf,Hzf = fcn_ComputePrimary(m,orient,xtx,0.,ztx,x0,0.,z0)

    fig1 = plt.figure(figsize=(17,6))
    Ax1 = fig1.add_axes([0.04,0,0.38,1])
    Ax2 = fig1.add_axes([0.6,0,0.4,1])

    Ax1 = plotProfileTxRxSphere(Ax1,xtx,ztx,x0,z0,a,xrx,zrx,X,Z,orient)

    if Flag == 'Hp':
        Hpx,Hpy,Hpz = fcn_ComputePrimary(m,orient,xtx,0.,ztx,X,0.,Z)
        Ax1 = plotProfileTxRxArrow(Ax1,x0,z0,Hxf,Hzf,Flag)
        Ax1 = plotProfileXZplane(Ax1,X,Z,Hpx,Hpz,Flag)
    elif Flag == 'Hs_real':
        Hx,Hy,Hz,Habs = Obj.fcn_ComputeFrequencyResponse(f,sig,mur,a,x0,0.,z0,X,0.,Z)
        Chi = fcn_ComputeExcitation_FEM(f,sig,mur,a)
        Ax1 = plotProfileTxRxArrow(Ax1,x0,z0,np.real(Chi)*Hxf,np.real(Chi)*Hzf,Flag)
        Ax1 = plotProfileXZplane(Ax1,X,Z,np.real(Hx),np.real(Hz),Flag)
    elif Flag == 'Hs_imag':
        Hx,Hy,Hz,Habs = Obj.fcn_ComputeFrequencyResponse(f,sig,mur,a,x0,0.,z0,X,0.,Z)
        Chi = fcn_ComputeExcitation_FEM(f,sig,mur,a)
        Ax1 = plotProfileTxRxArrow(Ax1,x0,z0,np.imag(Chi)*Hxf,np.imag(Chi)*Hzf,Flag)
        Ax1 = plotProfileXZplane(Ax1,X,Z,np.imag(Hx),np.imag(Hz),Flag)



    if orient == 'x':
        Ax2 = plotResponseFEM(Ax2,f,fvec,Hxi,orient)
    elif orient == 'z':
        Ax2 = plotResponseFEM(Ax2,f,fvec,Hzi,orient)

    plt.show(fig1)

##############################################
#   GLOBAL FUNTIONS
##############################################

def fcn_ComputeExcitation_FEM(f,sig,mur,a):
    """Compute Excitation Factor (FEM)"""

    w = 2*np.pi*f
    mu = 4*np.pi*1e-7*mur
    alpha = a*np.sqrt(1j*w*mu*sig)

    chi = 1.5*(2*mur*(np.tanh(alpha) - alpha) + (alpha**2*np.tanh(alpha) - alpha + np.tanh(alpha)))/(mur*(np.tanh(alpha) - alpha) - (alpha**2*np.tanh(alpha) - alpha + np.tanh(alpha)))

    return chi

def fcn_ComputePrimary(m,orient,xtx,ytx,ztx,X,Y,Z):
        """Computes Inducing Field at Sphere"""

        R = np.sqrt((X-xtx)**2 + (Y-ytx)**2 + (Z-ztx)**2)

        if orient == "x":
            Hpx = (1/(4*np.pi))*(3*m*(X-xtx)*(X-xtx)/R**5 - m/R**3)
            Hpy = (1/(4*np.pi))*(3*m*(Y-ytx)*(X-xtx)/R**5)
            Hpz = (1/(4*np.pi))*(3*m*(Z-ztx)*(X-xtx)/R**5)
        elif orient == "y":
            Hpx = (1/(4*np.pi))*(3*m*(X-xtx)*(Y-ytx)/R**5)
            Hpy = (1/(4*np.pi))*(3*m*(Y-ytx)*(Y-ytx)/R**5 - m/R**3)
            Hpz = (1/(4*np.pi))*(3*m*(Z-ztx)*(Y-ytx)/R**5)
        elif orient == "z":
            Hpx = (1/(4*np.pi))*(3*m*(X-xtx)*(Z-ztx)/R**5)
            Hpy = (1/(4*np.pi))*(3*m*(Y-ytx)*(Z-ztx)/R**5)
            Hpz = (1/(4*np.pi))*(3*m*(Z-ztx)*(Z-ztx)/R**5 - m/R**3)

        return Hpx, Hpy, Hpz

##############################################
#   GLOBAL PLOTTING FUNTIONS
##############################################

def plotAnomalyXYplane(Ax,f,X,Y,Z,H,Comp,Phase):

    FS = 20

    if Phase == 'Real':
        H = np.real(H)
        Str = "Re("
        Str_title = "Real Component of "
    elif Phase == 'Imag':
        H = np.imag(H)
        Str = "Im("
        Str_title = "Imaginary Component of "

    tol = 1e5

    Sign = np.sign(H)
    H = np.abs(H)
    MAX = np.max(H)

    H = np.log10(tol*H/MAX)

    Sign[H<0] = 0.
    H[H<0] = 0.

    Cmap = 'RdYlBu'
    #Cmap = 'seismic_r'

    if Comp == 'abs':
        TickLabels = MAX*np.array([1.,1e-1,1e-2,1e-3,1e-4,0.,-1e-4,-1e-3,-1e-2,-1e-1,-1])
        TickLabels = ["%.1e" % x for x in TickLabels]
        Cplot = Ax.contourf(X,Y,Sign*H,50,cmap=Cmap, vmin=-5, vmax=5)
        cbar = plt.colorbar(Cplot, ax=Ax, pad=0.02, ticks=-np.linspace(-5,5,11))
    else:
        TickLabels = MAX*np.array([-1.,-1e-1,-1e-2,-1e-3,-1e-4,0.,1e-4,1e-3,1e-2,1e-1,1])
        TickLabels = ["%.1e" % x for x in TickLabels]
        Cplot = Ax.contourf(X,Y,Sign*H,50,cmap=Cmap, vmin=-5, vmax=5)
        cbar = plt.colorbar(Cplot, ax=Ax, pad=0.02, ticks=np.linspace(-5,5,11))

    if Comp == 'x':
        cbar.set_label(Str+'$\mathbf{Hx}$) [A/m]', rotation=270, labelpad = 25, size=FS+4)
        Ax.set_title(Str_title + "$\mathbf{Hx}$",fontsize=FS+6)
    elif Comp == 'y':
        cbar.set_label(Str+'$\mathbf{Hy}$) [A/m]', rotation=270, labelpad = 25, size=FS+4)
        Ax.set_title(Str_title + "$\mathbf{Hy}$",fontsize=FS+6)
    elif Comp == 'z':
        cbar.set_label(Str+'$\mathbf{Hz}$) [A/m]', rotation=270, labelpad = 25, size=FS+4)
        Ax.set_title(Str_title + "$\mathbf{Hz}$",fontsize=FS+6)
    elif Comp == 'abs':
        cbar.set_label(Str+'$\mathbf{|H|}$) [A/m]', rotation=270, labelpad = 25, size=FS+4)
        Ax.set_title(Str_title + "$\mathbf{|H|}$",fontsize=FS+6)

    cbar.set_ticklabels(TickLabels)
    cbar.ax.tick_params(labelsize=FS-2)

    Ax.set_xbound(np.min(X),np.max(X))
    Ax.set_ybound(np.min(Y),np.max(Y))
    Ax.set_xlabel('X [m]',fontsize=FS+2)
    Ax.set_ylabel('Y [m]',fontsize=FS+2,labelpad=-10)
    Ax.tick_params(labelsize=FS-2)

    return Ax

def plotPlaceTxRxSphereXY(Ax,xtx,ytx,xrx,yrx,x0,y0,a):


    Xlim = Ax.get_xlim()
    Ylim = Ax.get_ylim()

    FS = 20

    Ax.scatter(xtx,ytx,s=100,color='k')
    Ax.text(xtx-0.75,ytx+1.5,'$\mathbf{Tx}$',fontsize=FS+6)
    Ax.scatter(xrx,yrx,s=100,color='k')
    Ax.text(xrx-0.75,yrx-4,'$\mathbf{Rx}$',fontsize=FS+6)

    xs = x0 + a*np.cos(np.linspace(0,2*np.pi,41))
    ys = y0 + a*np.sin(np.linspace(0,2*np.pi,41))

    Ax.plot(xs,ys,ls=':',color='k',linewidth=3)

    Ax.set_xbound(Xlim)
    Ax.set_ybound(Ylim)

    return Ax

def plotResponseFEM(Ax,fi,f,H,Comp):

    FS = 20

    xTicks = (np.logspace(np.log(np.min(f)),np.log(np.max(f)),9))
    Ylim = np.array([np.min(np.real(H)),np.max(np.real(H))])

    Ax.grid('both', linestyle='-', linewidth=0.8, color=[0.8, 0.8, 0.8])
    Ax.semilogx(f,0*f,color='k',linewidth=2)
    Ax.semilogx(f,np.real(H),color='k',linewidth=4,label="Real")
    Ax.semilogx(f,np.imag(H),color='k',linewidth=4,ls='--',label="Imaginary")
    Ax.semilogx(np.array([fi,fi]),1.1*Ylim,linewidth=3,color='r')
    Ax.set_xbound(np.min(f),np.max(f))
    Ax.set_ybound(1.1*Ylim)
    Ax.set_xlabel('Frequency [Hz]',fontsize=FS+2)
    Ax.tick_params(labelsize=FS-2)
    Ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

    if Comp == 'x':
        Ax.set_ylabel('$\mathbf{Hx}$ [A/m]',fontsize=FS+4,labelpad=-5)
        Ax.set_title('$\mathbf{Hx}$ Response at $\mathbf{Rx}$',fontsize=FS+6)
    elif Comp == 'y':
        Ax.set_ylabel('$\mathbf{Hy}$ [A/m]',fontsize=FS+4,labelpad=-5)
        Ax.set_title('$\mathbf{Hy}$ Response at $\mathbf{Rx}$',fontsize=FS+6)
    elif Comp == 'z':
        Ax.set_ylabel('$\mathbf{Hz}$ [A/m]',fontsize=FS+4,labelpad=-5)
        Ax.set_title('$\mathbf{Hz}$ Response at $\mathbf{Rx}$',fontsize=FS+6)
    elif Comp == 'abs':
        Ax.set_ylabel('$\mathbf{|H|}$ [A/m]',fontsize=FS+4,labelpad=-5)
        Ax.set_title('$\mathbf{|H|}$ Response at $\mathbf{Rx}$',fontsize=FS+6)


    if np.max(np.real(H[-1])) > 0.:
        handles, labels = Ax.get_legend_handles_labels()
        Ax.legend(handles, labels, loc='upper left', fontsize=FS)
    elif np.max(np.real(H[-1])) < 0.:
        handles, labels = Ax.get_legend_handles_labels()
        Ax.legend(handles, labels, loc='lower left', fontsize=FS)

    return Ax


def plotProfileTxRxSphere(Ax,xtx,ztx,x0,z0,a,xrx,zrx,X,Z,orient):

    FS = 22

    phi = np.linspace(0,2*np.pi,41)
    psi = np.linspace(0,np.pi,21)

    if orient == 'x':
        Xtx = xtx + 0.5*np.cos(phi)
        Ztx = ztx + 2*np.sin(phi)
        Xrx = xrx + 0.5*np.cos(phi)
        Zrx = zrx + 2*np.sin(phi)
    elif orient == 'z':
        Xtx = xtx + 2*np.cos(phi)
        Ztx = ztx + 0.5*np.sin(phi)
        Xrx = xrx + 2*np.cos(phi)
        Zrx = zrx + 0.5*np.sin(phi)

    # Xs = x0 + a*np.cos(psi)
    # Zs1 = z0 + a*np.sin(psi)
    # Zs2 = z0 - a*np.sin(psi)

    XS = x0 + a*np.cos(phi)
    ZS = z0 + a*np.sin(phi)

    Ax.fill_between(np.array([np.min(X),np.max(X)]),np.array([0.,0.]),np.array([np.max(Z),np.max(Z)]),facecolor=(0.9,0.9,0.9))
    Ax.fill_between(np.array([np.min(X),np.max(X)]),np.array([0.,0.]),np.array([np.min(Z),np.min(Z)]),facecolor=(0.6,0.6,0.6),linewidth=2)
    # Ax.fill_between(Xs,Zs1,Zs2,facecolor=(0.4,0.4,0.4),linewidth=4)

    polyObj = plt.Polygon(np.c_[XS,ZS],closed=True,facecolor=((0.4,0.4,0.4)),edgecolor='k',linewidth=2)
    Ax.add_patch(polyObj)

    Ax.plot(Xtx,Ztx,'k',linewidth=4)
    Ax.plot(Xrx,Zrx,'k',linewidth=4)
    # Ax.plot(x0+a*np.cos(phi),z0+a*np.sin(phi),'k',linewidth=2)

    Ax.set_xbound(np.min(X),np.max(X))
    Ax.set_ybound(np.min(Z),np.max(Z))

    Ax.text(xtx-4,ztx+2,'$\mathbf{Tx}$',fontsize=FS)
    Ax.text(xrx,zrx+2,'$\mathbf{Rx}$',fontsize=FS)

    return Ax


def plotProfileXZplane(Ax,X,Z,Hx,Hz,Flag):

    FS = 20

    if Flag == 'Hp':
        Ax.streamplot(X,Z,Hx,Hz,color='b',linewidth=3.5,arrowsize=2)
        Ax.set_title('Primary Field',fontsize=FS+6)
    elif Flag == 'Hs_real':
        Ax.streamplot(X,Z,Hx,Hz,color='r',linewidth=3.5,arrowsize=2)
        Ax.set_title('Secondary Field (real)',fontsize=FS+6)
    elif Flag == 'Hs_imag':
        Ax.streamplot(X,Z,Hx,Hz,color='r',linewidth=3.5,arrowsize=2)
        Ax.set_title('Secondary Field (imaginary)',fontsize=FS+6)

    Ax.set_xbound(np.min(X),np.max(X))
    Ax.set_ybound(np.min(Z),np.max(Z))
    Ax.set_xlabel('X [m]',fontsize=FS+2)
    Ax.set_ylabel('Z [m]',fontsize=FS+2,labelpad=-10)
    Ax.tick_params(labelsize=FS-2)



def plotProfileTxRxArrow(Ax,x0,z0,Hxf,Hzf,Flag):

    Habsf = np.sqrt(Hxf**2 + Hzf**2)
    dx = Hxf/Habsf
    dz = Hzf/Habsf

    if Flag == 'Hp':
        Ax.arrow(x0-2.5*dx, z0-2.75*dz, 3*dx, 3*dz, fc=(0.,0.,0.8), ec="k",head_width=2.5, head_length=2.5,width=1,linewidth=2)
    else:
        Ax.arrow(x0-2.5*dx, z0-2.75*dz, 3*dx, 3*dz, fc=(0.8,0.,0.), ec="k",head_width=2.5, head_length=2.5,width=1,linewidth=2)

    return Ax



############################################
#   CLASS: SPHERE TOP VIEW
############################################

############################################
#   DEFINE CLASS

class SphereFEM():
    """Fucntionwhcihdf
    Input variables:

        Output variables:
    """

    def __init__(self,m,orient,xtx,ytx,ztx):
        """Defines Initial Attributes"""

        # INITIALIZES OBJECT

        # m: Transmitter dipole moment
        # orient: Transmitter dipole orentation 'x', 'y' or 'z'
        # xtx: Transmitter x location
        # ytx: Transmitter y location
        # ztx: Transmitter z location

        self.m = m
        self.orient = orient
        self.xtx = xtx
        self.ytx = ytx
        self.ztx = ztx

############################################
#   DEFINE METHODS

    def fcn_ComputeFrequencyResponse(self,f,sig,mur,a,x0,y0,z0,X,Y,Z):
        """Compute Single Frequency Response at (X,Y,Z)"""

        m = self.m
        orient = self.orient
        xtx = self.xtx
        ytx = self.ytx
        ztx = self.ztx

        chi = fcn_ComputeExcitation_FEM(f,sig,mur,a)
        Hpx,Hpy,Hpz = fcn_ComputePrimary(m,orient,xtx,ytx,ztx,x0,y0,z0)

        mx = 4*np.pi*a**3*chi*Hpx/3
        my = 4*np.pi*a**3*chi*Hpy/3
        mz = 4*np.pi*a**3*chi*Hpz/3
        R = np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)

        Hx = (1/(4*np.pi))*(3*(X-x0)*(mx*(X-x0) + my*(Y-y0) + mz*(Z-z0))/R**5 - mx/R**3)
        Hy = (1/(4*np.pi))*(3*(Y-y0)*(mx*(X-x0) + my*(Y-y0) + mz*(Z-z0))/R**5 - my/R**3)
        Hz = (1/(4*np.pi))*(3*(Z-z0)*(mx*(X-x0) + my*(Y-y0) + mz*(Z-z0))/R**5 - mz/R**3)
        Habs = np.sqrt(np.real(Hx)**2 + np.real(Hy)**2 + np.real(Hz)**2) + 1j*np.sqrt(np.imag(Hx)**2 + np.imag(Hy)**2 + np.imag(Hz)**2)

        return Hx, Hy, Hz, Habs













