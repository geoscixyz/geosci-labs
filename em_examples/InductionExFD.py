# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 10:43:37 2016

@author: Devin
"""
    
    # IMPORT PACKAGES
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt


def fcn_Widget(I,a1,a2,xRx,zRx,azm,R,L,f):

    xmin, xmax, dx, zmin, zmax, dz = -20., 20., 0.5, -20., 20., 0.5
    X,Z = np.mgrid[xmin:xmax+dx:dx, zmin:zmax+dz:dz]
    X = np.transpose(X)
    Z = np.transpose(Z)  

    Obj = IndExFD(I,a1,a2,xRx,zRx,azm,R,L)

    Obj.calc_PrimaryLoop()
    Hpx,Hpz,Habs = Obj.calc_PrimaryRegion(X,Z)
    EMF,Is = Obj.calc_InducedCurrent()
    EMFi,Isi = Obj.calc_InducedCurrenti(f)

    fig1 = plt.figure(figsize=(13,12))
    Ax11 = fig1.add_axes([0,0.55,0.52,0.43])
    Ax12 = fig1.add_axes([0.57,0.54,0.43,0.43])
    Ax2  = fig1.add_axes([0,0,1,0.45])

    Ax11,Cplot = Obj.plot_PrimaryRegion(X,Z,Hpx,Hpz,Habs,Ax11);
    Ax12 = Obj.plot_PrimaryLoop(Ax12,f)
    Ax2 = Obj.plot_InducedCurrent(Ax2,Is,f,EMFi,Isi)

    plt.show(fig1)




class IndExFD():
    """Fucntionwhcihdf
    Input variables:
        
        Output variables:
    """
    
    def __init__(self,I,a1,a2,x,z,azm,R,L):
        
        # INITIALIZES OBJECT
        
        # I: Transmitter loop Current
        # f: Transmitter frequency
        # a1: Transmitter Loop Radius
        # a2: Receiver loop Radius
        # x: Horizontal Receiver Loop Location
        # z: Vertical Receiver Loop Location
        # azm: Azimuthal angle for normal vector of receiver loop relative to up (-90,+90)
        # R: Resistance of receiver loop
        # L: Inductance of receiver loop
        
        self.I   = I
        self.a1  = a1
        self.a2  = a2
        self.x   = x
        self.z   = z
        self.azm = azm
        self.R   = R
        self.L   = L

    def calc_PrimaryRegion(self,X,Z):

        # CALCULATES INDUCING FIELD WITHIN REGION AND RETURNS AT LOCATIONS
        
        # Initiate Variables from object
        I   = self.I
        a1  = self.a1
        eps = 1e-7

        s = np.abs(X)   # Define Radial Distance

        k = 4*a1*s/(Z**2 + (a1+s)**2)

        Hpx  = np.sign(X)*(Z*I/(2*np.pi*s + eps))*(1/np.sqrt(Z**2 + (a1+s)**2))*(-sp.ellipk(k) + ((a1**2 + Z**2 + s**2)/(Z**2 + (s-a1)**2))*sp.ellipe(k))
        Hpz  =            (  I/(2*np.pi           ))*(1/np.sqrt(Z**2 + (a1+s)**2))*( sp.ellipk(k) + ((a1**2 - Z**2 - s**2)/(Z**2 + (s-a1)**2))*sp.ellipe(k))
        Hpx[(X>-1.025*a1) & (X<-0.975*a1) & (Z>-0.025*a1) & (Z<0.025*a1)] = 0.
        Hpx[(X<1.025*a1) & (X>0.975*a1) & (Z>-0.025*a1) & (Z<0.025*a1)] = 0.
        Hpz[(X>-1.025*a1) & (X<-0.975*a1) & (Z>-0.025*a1) & (Z<0.025*a1)] = 0.
        Hpz[(X<1.025*a1) & (X>0.975*a1) & (Z>-0.025*a1) & (Z<0.025*a1)] = 0.
        Habs = np.sqrt(Hpx**2 + Hpz**2)
        
        return Hpx,Hpz,Habs
        
    def plot_PrimaryRegion(self,X,Z,Hpx,Hpz,Habs,Ax):
        
        # INITIALIZE ATTRIBUTES
        a1  = self.a1
        a2  = self.a2
        xR  = self.x
        zR  = self.z
        azm = self.azm*np.pi/180
        
        FS = 20
        
        # LOOP ORIENTATIONS
        Phi = np.linspace(0,2*np.pi,101)
        xTx = a1*np.cos(Phi)
        zTx = 0.07*a1*np.sin(Phi)
        xRx = xR + a2*np.cos(Phi)*np.cos(azm) + 0.1*a2*np.sin(Phi)*np.sin(azm)
        zRx = zR - a2*np.cos(Phi)*np.sin(azm) + 0.1*a2*np.sin(Phi)*np.cos(azm)
        
        
        Ax.plot(xTx,zTx,color='black',linewidth=6)
        Ax.plot(xTx,zTx,color=((0.6,0.6,0.6)),linewidth=4)
        Ax.plot(xRx,zRx,color='black',linewidth=6)
        Ax.plot(xRx,zRx,color=((0.6,0.6,0.6)),linewidth=4)
        Cplot = Ax.contourf(X,Z,np.log10(Habs),40,cmap='ocean_r')
        cbar = plt.colorbar(Cplot, ax=Ax)
        cbar.set_label('log10($\mathbf{|H|}$)', rotation=270, labelpad = 20, size=FS)
        cbar.ax.tick_params(labelsize=FS-2)
        Ax.streamplot(X,Z,Hpx,Hpz,color=(0.2,0.2,0.2),linewidth=2)
        
        Ax.set_xbound(np.min(X),np.max(X))
        Ax.set_ybound(np.min(Z),np.max(Z))
        Ax.set_xlabel('X [m]',fontsize=FS+2)
        Ax.set_ylabel('Z [m]',fontsize=FS+2)
        Ax.tick_params(labelsize=FS-2)
        
        
        return Ax,Cplot
    
    def calc_PrimaryLoop(self):
        
        # CALCULATES INDUCING FIELD AT RX LOOP CENTER
        
        # Initiate Variables
        
        I   = self.I
        a1  = self.a1
        x   = self.x
        z   = self.z
        eps = 1e-7
        
        s = np.abs(x)   # Define Radial Distance

        k = 4*a1*s/(z**2 + (a1+s)**2)
        
        Hpx = np.sign(x)*(z*I/(2*np.pi*s + eps))*(1/np.sqrt(z**2 + (a1+s)**2))*(-sp.ellipk(k) + ((a1**2 + z**2 + s**2)/(z**2 + (s-a1)**2))*sp.ellipe(k))
        Hpz =            (  I/(2*np.pi           ))*(1/np.sqrt(z**2 + (a1+s)**2))*( sp.ellipk(k) + ((a1**2 - z**2 - s**2)/(z**2 + (s-a1)**2))*sp.ellipe(k))

        self.Hpx = Hpx
        self.Hpz = Hpz
    
    def plot_PrimaryLoop(self,Ax,f):
        
        FS = 20
        
        # INITIALIZE ATTRIBUTES
        azm = self.azm*np.pi/180
        a2  = self.a2
        Hpx = self.Hpx
        Hpz = self.Hpz
        
        mu0 = 4*np.pi*1e-7
        
        Phi = np.linspace(0,2*np.pi,101)
        xRx =   np.cos(Phi)*np.cos(azm) + 0.1*np.sin(Phi)*np.sin(azm)
        zRx = - np.cos(Phi)*np.sin(azm) + 0.1*np.sin(Phi)*np.cos(azm)
        dxH = 1.75*Hpx/np.sqrt(Hpx**2 + Hpz**2)
        dzH = 1.75*Hpz/np.sqrt(Hpx**2 + Hpz**2)
        dxn = np.sin(azm)
        dzn = np.cos(azm)
        
        Habs = np.sqrt(Hpx**2 + Hpz**2)
        Hnor = Hpx*np.sin(azm) + Hpz*np.cos(azm)
        Area = np.pi*a2**2
        EMF  = - 2*np.pi*mu0*f*Area*Hnor
        
        
        Ax.plot(xRx,zRx,color='black',linewidth=6)
        Ax.plot(xRx,zRx,color=((0.4,0.4,0.4)),linewidth=4)
        Ax.arrow(0., 0., dxH, dzH, fc="b", ec="k",head_width=0.3, head_length=0.3,width=0.08 )
        Ax.arrow(0., 0., dxn, dzn, fc="r", ec="k",head_width=0.3, head_length=0.3,width=0.08 )
        
        Ax.set_xbound(-3,3)
        Ax.set_ybound(-3.5,2.5)
        Ax.set_xticks([])
        Ax.set_yticks([])
        
        Ax.text(1.2*dxn,1.3*dzn,'$\mathbf{n}$',fontsize=FS+4,color='r')
        Ax.text(1.2*dxH,1.2*dzH,'$\mathbf{H_p}$',fontsize=FS+4,color='b')
        
        Habs_str = '{:.3e}'.format(Habs)
        Hn_str   = '{:.3e}'.format(Hnor)
        A_str    = '{:.3f}'.format(Area)
        f_str    = '{:.3e}'.format(f)
        EMF_str  = '{:.3e}j'.format(EMF)
        
        Ax.text(-2.9,-1.7,'$\mathbf{H_p}$ = '+Habs_str+' A/m',fontsize=20)
        Ax.text(-2.9,-2.1,'$\mathbf{H_{n}}$ = '+Hn_str+' A/m',fontsize=20)
        Ax.text(-2.9,-2.5,'Area = '+A_str+' m$^2$',fontsize=FS)
        Ax.text(-2.9,-2.9,'f = '+f_str+' Hz',fontsize=FS)
        Ax.text(-2.9,-3.3,'EMF = '+EMF_str+' V',fontsize=FS)
        
        return Ax
        
    
    def calc_InducedCurrent(self):
        
        #INITIALIZE ATTRIBUTES
        Hpx = self.Hpx
        Hpz = self.Hpz
        a2  = self.a2
        azm = np.pi*self.azm/180.
        R   = self.R
        L   = self.L
        
        w = 2*np.pi*np.logspace(0,8,101)
        mu = 4*np.pi*1e-7
        
        Ax = np.pi*a2**2*np.sin(azm)
        Az = np.pi*a2**2*np.cos(azm)
        
        Phi = mu*(Ax*Hpx + Az*Hpz)
        EMF = -1j*w*Phi
        Is = EMF/(R + 1j*w*L)
        
        return EMF,Is
        
    def calc_InducedCurrenti(self,f):
        
        #INITIALIZE ATTRIBUTES
        Hpx = self.Hpx
        Hpz = self.Hpz
        a2  = self.a2
        azm = np.pi*self.azm/180.
        R   = self.R
        L   = self.L
        
        w = 2*np.pi*f
        mu = 4*np.pi*1e-7
        
        Ax = np.pi*a2**2*np.sin(azm)
        Az = np.pi*a2**2*np.cos(azm)
        
        Phi = mu*(Ax*Hpx + Az*Hpz)
        EMF = -1j*w*Phi
        Is = EMF/(R + 1j*w*L)
        
        return EMF,Is
        
    def plot_InducedCurrent(self,Ax,Is,fi,EMFi,Isi):
        
        FS = 20
        
        R = self.R
        L = self.L
        
        Imax = np.max(-np.real(Is))
        
        f = np.logspace(0,8,101)
        

        
        Ax.semilogx(f,-np.real(Is),color='k',linewidth=4)
        Ax.semilogx(f,-np.imag(Is),color='k',ls='--',linewidth=4)
        Ax.semilogx(fi*np.array([1.,1.]),np.array([0,1.1*Imax]),color='r',ls='-',linewidth=3)
        
        Ax.set_xlabel('Frequency [Hz]',fontsize=FS+2)
        Ax.set_ylabel('-Current [A]',fontsize=FS+2)
        Ax.set_ybound(0,1.1*Imax)
        Ax.tick_params(labelsize=FS-2)
        
        R_str    = '{:.3e}'.format(R)
        L_str    = '{:.3e}'.format(L)
        f_str    = '{:.3e}'.format(fi)
        EMF_str  = '{:.3e} + {:.3e}j'.format(float(np.real(EMFi)),float(np.imag(EMFi)))
        I_str    = '{:.3e} + {:.3e}j'.format(float(np.real(Isi)),float(np.imag(Isi)))
        
        Ax.text(1.4,1.00*Imax,'R = '+R_str+' $\Omega$',fontsize=FS)
        Ax.text(1.4,0.92*Imax,'L = '+L_str+' H',fontsize=FS)
        Ax.text(1.4,0.84*Imax,'@ f = '+f_str+' Hz',fontsize=FS)
        Ax.text(1.4,0.76*Imax,'EMF = '+EMF_str+' V',fontsize=FS)
        Ax.text(1.4,0.68*Imax,'I = '+I_str+' A',fontsize=FS)
        
        return Ax
        
        
        
        
        
        
        
        
        
        
        
        


