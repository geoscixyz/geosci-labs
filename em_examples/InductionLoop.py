from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 09 10:43:37 2016

@author: Devin
"""
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter



##############################################
#   PLOTTING FUNCTIONS FOR WIDGETS
##############################################


def fcn_FDEM_Widget(I,a1,a2,xRx,zRx,azm,logR,logL,logf):

    R = 10**logR
    L = 10**logL
    f = 10**logf

    FS = 20

    xmin, xmax, dx, zmin, zmax, dz = -20., 20., 0.5, -20., 20., 0.5
    X,Z = np.mgrid[xmin:xmax+dx:dx, zmin:zmax+dz:dz]
    X = np.transpose(X)
    Z = np.transpose(Z)

    Obj = IndEx(I,a1,a2,xRx,zRx,azm,R,L)
    t_range = (4/f)*np.linspace(0,1,num=100)

    Obj.calc_PrimaryLoop()  # Calculate primary field at loop center
    Bpx,Bpz,Babs = Obj.calc_PrimaryRegion(X,Z)  # Calculates regional primary field
    EMF,Isf = Obj.calc_IndCurrent_FD_spectrum()
    Ire,Iim,Is,phi = Obj.calc_IndCurrent_cos_range(f,t_range)

    fig1 = plt.figure(figsize=(13,13))
    Ax11 = fig1.add_axes([0,0.62,0.46,0.37])
    Ax12 = fig1.add_axes([0.6,0.63,0.40,0.37])
    Ax21 = fig1.add_axes([0.1,0.31,0.8,0.25])
    Ax22 = fig1.add_axes([0.1,0,0.8,0.25])

    Ax11,Cplot = Obj.plot_PrimaryRegion(X,Z,Bpx,Bpz,Babs,Ax11)
    polyArray = np.array([[-20,10],[4,10],[4,20],[-20,20]])
    polyObj = plt.Polygon(polyArray,facecolor=((1,1,1)),edgecolor='k')
    Ax11.add_patch(polyObj)
    Ax12 = Obj.plot_InducedCurrent_FD(Ax12,Isf,f)
    Ax21,Ax21b,Ax22 = Obj.plot_InducedCurrent_cos(Ax21,Ax22,Ire,Iim,Is,phi,f,t_range)

    Babs_str = '{:.2e}'.format(1e9*Obj.Bpabs)
    Bn_str   = '{:.2e}'.format(1e9*Obj.Bpn)
    A_str    = '{:.2f}'.format(Obj.Area)

    Ax11.text(-19,17,'$\mathbf{|B_p|}$ = '+Babs_str+' nT',fontsize=FS,color='k')
    Ax11.text(-19,14,'$\mathbf{|B_n|}$ = '+Bn_str+' nT',fontsize=FS,color='k')
    Ax11.text(-19,11,'Area = '+A_str+' m$^2$',fontsize=FS,color='k')

    #f_str    = '{:.2e}'.format(f)
    #EMF_str  = '{:.2e}j'.format(EMFi.imag)
    #Ax12.text(-2.9,-1.0,'f = '+f_str+' Hz',fontsize=FS)
    #Ax12.text(-2.9,-1.4,'EMF = '+EMF_str+' V',fontsize=FS)

    plt.show(fig1)


def fcn_TDEM_Widget(I,a1,a2,xRx,zRx,azm,logR,logL,logt):

    R = 10**logR
    L = 10**logL
    t = 10**logt

    FS = 20

    xmin, xmax, dx, zmin, zmax, dz = -20., 20., 0.5, -20., 20., 0.5
    X,Z = np.mgrid[xmin:xmax+dx:dx, zmin:zmax+dz:dz]
    X = np.transpose(X)
    Z = np.transpose(Z)

    Obj = IndEx(I,a1,a2,xRx,zRx,azm,R,L)

    Obj.calc_PrimaryLoop()
    Bpx,Bpz,Babs = Obj.calc_PrimaryRegion(X,Z)
    V,Is = Obj.calc_IndCurrent_TD_offtime()
    EMFi,Isi = Obj.calc_IndCurrent_TD_i(t)

    fig1 = plt.figure(figsize=(13,5.8))
    Ax11 = fig1.add_axes([0,0,0.48,0.89])
    Ax12 = fig1.add_axes([0.61,0,0.40,0.89])

    Ax11,Cplot = Obj.plot_PrimaryRegion(X,Z,Bpx,Bpz,Babs,Ax11)
    polyArray = np.array([[-20,10],[4,10],[4,20],[-20,20]])
    polyObj = plt.Polygon(polyArray,facecolor=((1,1,1)),edgecolor='k')
    Ax11.add_patch(polyObj)
    Ax12 = Obj.plot_InducedCurrent_TD(Ax12,Is,t,EMFi,Isi)

    Babs_str = '{:.2e}'.format(1e9*Obj.Bpabs)
    Bn_str   = '{:.2e}'.format(1e9*Obj.Bpn)
    A_str    = '{:.2f}'.format(Obj.Area)

    Ax11.text(-19,17,'$\mathbf{|B_p|}$ = '+Babs_str+' nT',fontsize=FS,color='k')
    Ax11.text(-19,14,'$\mathbf{|B_n|}$ = '+Bn_str+' nT',fontsize=FS,color='k')
    Ax11.text(-19,11,'Area = '+A_str+' m$^2$',fontsize=FS,color='k')

    plt.show(fig1)


############################################
#   DEFINE CLASS
############################################

class IndEx():
    """Fucntionwhcihdf
    Input variables:

        Output variables:
    """

    def __init__(self,I,a1,a2,x,z,azm,R,L):
        """Defines Initial Attributes"""

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
        """Predicts magnitude and direction of primary field in region"""

        # CALCULATES INDUCING FIELD WITHIN REGION AND RETURNS AT LOCATIONS

        # Initiate Variables from object
        I   = self.I
        a1  = self.a1
        eps = 1e-6
        mu0 = 4*np.pi*1e-7   # 1e9*mu0

        s = np.abs(X)   # Define Radial Distance

        k = 4*a1*s/(Z**2 + (a1+s)**2)

        Bpx  = mu0*np.sign(X)*(Z*I/(2*np.pi*s + eps))*(1/np.sqrt(Z**2 + (a1+s)**2))*(-sp.ellipk(k) + ((a1**2 + Z**2 + s**2)/(Z**2 + (s-a1)**2))*sp.ellipe(k))
        Bpz  = mu0*           (  I/(2*np.pi           ))*(1/np.sqrt(Z**2 + (a1+s)**2))*( sp.ellipk(k) + ((a1**2 - Z**2 - s**2)/(Z**2 + (s-a1)**2))*sp.ellipe(k))
        Bpx[(X>-1.025*a1) & (X<-0.975*a1) & (Z>-0.025*a1) & (Z<0.025*a1)] = 0.
        Bpx[(X<1.025*a1) & (X>0.975*a1) & (Z>-0.025*a1) & (Z<0.025*a1)] = 0.
        Bpz[(X>-1.025*a1) & (X<-0.975*a1) & (Z>-0.025*a1) & (Z<0.025*a1)] = 0.
        Bpz[(X<1.025*a1) & (X>0.975*a1) & (Z>-0.025*a1) & (Z<0.025*a1)] = 0.
        Babs = np.sqrt(Bpx**2 + Bpz**2)

        return Bpx,Bpz,Babs

    def calc_PrimaryLoop(self):
        """Predicts magnitude and direction of primary field in loop center"""

        # CALCULATES INDUCING FIELD AT RX LOOP CENTER

        # Initiate Variables

        I   = self.I
        a1  = self.a1
        a2  = self.a2
        x   = self.x
        z   = self.z
        azm = self.azm
        eps = 1e-7
        mu0 = 4*np.pi*1e-7   # 1e9*mu0

        s = np.abs(x)   # Define Radial Distance

        k = 4*a1*s/(z**2 + (a1+s)**2)

        Bpx = mu0*np.sign(x)*(z*I/(2*np.pi*s + eps))*(1/np.sqrt(z**2 + (a1+s)**2))*(-sp.ellipk(k) + ((a1**2 + z**2 + s**2)/(z**2 + (s-a1)**2))*sp.ellipe(k))
        Bpz = mu0*           (  I/(2*np.pi           ))*(1/np.sqrt(z**2 + (a1+s)**2))*( sp.ellipk(k) + ((a1**2 - z**2 - s**2)/(z**2 + (s-a1)**2))*sp.ellipe(k))
        Bpabs = np.sqrt(Bpx**2 + Bpz**2)
        Bpn = np.sin(azm)*Bpx + np.cos(azm)*Bpz
        Area = np.pi*a2**2

        self.Bpx = Bpx
        self.Bpz = Bpz
        self.Bpabs = Bpabs
        self.Bpn = Bpn
        self.Area = Area

    def calc_IndCurrent_Cos_i(self,f,t):
        """Induced current at particular time and frequency"""

        Bpx = self.Bpx
        Bpz = self.Bpz
        a2  = self.a2
        azm = np.pi*self.azm/180.
        R   = self.R
        L   = self.L

        w = 2*np.pi*f

        Ax = np.pi*a2**2*np.sin(azm)
        Az = np.pi*a2**2*np.cos(azm)

        Phi = (Ax*Bpx + Az*Bpz)
        EMF = w*Phi*np.sin(w*t)
        Is = (Phi/(R**2 + (w*L)**2))*(-w**2*L*np.cos(w*t) + w*R*np.sin(w*t))

        return EMF,Is

    def calc_IndCurrent_cos_range(self,f,t):
        """Induced current over a range of times"""

        Bpx = self.Bpx
        Bpz = self.Bpz
        a2  = self.a2
        azm = np.pi*self.azm/180.
        R   = self.R
        L   = self.L

        w = 2*np.pi*f

        Ax = np.pi*a2**2*np.sin(azm)
        Az = np.pi*a2**2*np.cos(azm)

        Phi = (Ax*Bpx + Az*Bpz)
        phi = np.arctan(R/(w*L))-np.pi  # This is the phase and not phase lag
        Is  = -(w*Phi/(R*np.sin(phi) + w*L*np.cos(phi)))*np.cos(w*t + phi)
        Ire = -(w*Phi/(R*np.sin(phi) + w*L*np.cos(phi)))*np.cos(w*t)*np.cos(phi)
        Iim =  (w*Phi/(R*np.sin(phi) + w*L*np.cos(phi)))*np.sin(w*t)*np.sin(phi)

        return Ire,Iim,Is,phi

    def calc_IndCurrent_FD_i(self,f):
        """Give FD EMF and current for single frequency"""

        #INITIALIZE ATTRIBUTES
        Bpx = self.Bpx
        Bpz = self.Bpz
        a2  = self.a2
        azm = np.pi*self.azm/180.
        R   = self.R
        L   = self.L

        w = 2*np.pi*f

        Ax = np.pi*a2**2*np.sin(azm)
        Az = np.pi*a2**2*np.cos(azm)

        Phi = (Ax*Bpx + Az*Bpz)
        EMF = -1j*w*Phi
        Is = EMF/(R + 1j*w*L)

        return EMF,Is

    def calc_IndCurrent_FD_spectrum(self):
        """Gives FD induced current spectrum"""

        #INITIALIZE ATTRIBUTES
        Bpx = self.Bpx
        Bpz = self.Bpz
        a2  = self.a2
        azm = np.pi*self.azm/180.
        R   = self.R
        L   = self.L

        w = 2*np.pi*np.logspace(0,8,101)

        Ax = np.pi*a2**2*np.sin(azm)
        Az = np.pi*a2**2*np.cos(azm)

        Phi = (Ax*Bpx + Az*Bpz)
        EMF = -1j*w*Phi
        Is = EMF/(R + 1j*w*L)

        return EMF,Is

    def calc_IndCurrent_TD_i(self,t):
        """Give FD EMF and current for single frequency"""

        #INITIALIZE ATTRIBUTES
        Bpx = self.Bpx
        Bpz = self.Bpz
        a2  = self.a2
        azm = np.pi*self.azm/180.
        R   = self.R
        L   = self.L



        Ax = np.pi*a2**2*np.sin(azm)
        Az = np.pi*a2**2*np.cos(azm)

        Phi = (Ax*Bpx + Az*Bpz)
        Is = (Phi/L)*np.exp(-(R/L)*t)
#        V = (Phi*R/L)*np.exp(-(R/L)*t) - (Phi*R/L**2)*np.exp(-(R/L)*t)
        EMF = Phi

        return EMF,Is

    def calc_IndCurrent_TD_offtime(self):
        """Gives FD induced current spectrum"""

        #INITIALIZE ATTRIBUTES
        Bpx = self.Bpx
        Bpz = self.Bpz
        a2  = self.a2
        azm = np.pi*self.azm/180.
        R   = self.R
        L   = self.L

        t = np.logspace(-6,0,101)

        Ax = np.pi*a2**2*np.sin(azm)
        Az = np.pi*a2**2*np.cos(azm)

        Phi = (Ax*Bpx + Az*Bpz)
        Is = (Phi/L)*np.exp(-(R/L)*t)
        V = (Phi*R/L)*np.exp(-(R/L)*t) - (Phi*R/L**2)*np.exp(-(R/L)*t)

        return V,Is




       ###########################################
       #    PLOTTING FUNCTIONS
       ###########################################


    def plot_PrimaryRegion(self,X,Z,Bpx,Bpz,Babs,Ax):

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
        Ax.plot(xRx,zRx,color=((0.4,0.4,0.4)),linewidth=4)
        #Cplot = Ax.contourf(X,Z,np.log10(Babs),40,cmap='ocean_r')
        Cplot = Ax.contourf(X,Z,np.log10(1e9*Babs),40,cmap='viridis')
        cbar = plt.colorbar(Cplot, ax=Ax, pad=0.02)
        cbar.set_label('log$_{10}(\mathbf{|B_p|})$ [nT]', rotation=270, labelpad = 25, size=FS)
        cbar.ax.tick_params(labelsize=FS-2)
        #Ax.streamplot(X,Z,Bpx,Bpz,color=(0.2,0.2,0.2),linewidth=2)
        Ax.streamplot(X,Z,Bpx,Bpz,color=(1,1,1),linewidth=2)

        Ax.set_xbound(np.min(X),np.max(X))
        Ax.set_ybound(np.min(Z),np.max(Z))
        Ax.set_xlabel('X [m]',fontsize=FS+2)
        Ax.set_ylabel('Z [m]',fontsize=FS+2,labelpad=-10)
        Ax.tick_params(labelsize=FS-2)


        return Ax,Cplot

    def plot_PrimaryLoop(self,Ax):

        FS = 20

        # INITIALIZE ATTRIBUTES
        azm = self.azm*np.pi/180
        a2  = self.a2
        Bpx = self.Bpx
        Bpz = self.Bpz

        Phi = np.linspace(0,2*np.pi,101)
        xRx =   np.cos(Phi)*np.cos(azm) + 0.1*np.sin(Phi)*np.sin(azm)
        zRx = - np.cos(Phi)*np.sin(azm) + 0.1*np.sin(Phi)*np.cos(azm)
        dxB = 1.75*Bpx/np.sqrt(Bpx**2 + Bpz**2)
        dzB = 1.75*Bpz/np.sqrt(Bpx**2 + Bpz**2)
        dxn = np.sin(azm)
        dzn = np.cos(azm)

        Babs = np.sqrt(Bpx**2 + Bpz**2)
        Bnor = Bpx*np.sin(azm) + Bpz*np.cos(azm)
        Area = np.pi*a2**2
        #EMF  = - 2*np.pi*f*Area*Bnor


        Ax.plot(xRx,zRx,color='black',linewidth=6)
        Ax.plot(xRx,zRx,color=((0.4,0.4,0.4)),linewidth=4)
        Ax.arrow(0., 0., dxB, dzB, fc="b", ec="k",head_width=0.3, head_length=0.3,width=0.08 )
        Ax.arrow(0., 0., dxn, dzn, fc="r", ec="k",head_width=0.3, head_length=0.3,width=0.08 )

        Ax.set_xbound(-3,3)
        Ax.set_ybound(-1.5,4.5)
        Ax.set_xticks([])
        Ax.set_yticks([])

        Ax.text(1.2*dxn,1.3*dzn,'$\mathbf{n}$',fontsize=FS+4,color='r')
        Ax.text(1.2*dxB,1.2*dzB,'$\mathbf{B_p}$',fontsize=FS+4,color='b')

        Babs_str = '{:.3e}'.format(1e9*Babs)
        Bn_str   = '{:.3e}'.format(1e9*Bnor)
        A_str    = '{:.3f}'.format(Area)
        #f_str    = '{:.3e}'.format(f)
        #EMF_str  = '{:.3e}j'.format(EMF)

        Ax.text(-2.9,4.1,'$\mathbf{|B_p|}$ = '+Babs_str+' nT',fontsize=20)
        Ax.text(-2.9,3.7,'$\mathbf{|B_{n}|}$ = '+Bn_str+' nT',fontsize=20)
        Ax.text(-2.9,3.3,'Area = '+A_str+' m$^2$',fontsize=FS)
        #3Ax.text(-2.9,-2.1,'f = '+f_str+' Hz',fontsize=FS)
        #Ax.text(-2.9,-1.7,'EMF = '+EMF_str+' V',fontsize=FS)

        return Ax


    def plot_InducedCurrent_cos(self,Ax1,Ax2,Ire,Iim,Is,phi,f,t):

        FS = 20

        # Numerical Values
        w  = 2*np.pi*f
        I0 = self.I*np.cos(w*t)
        Ipmax = self.I
        Ismax = np.max(Is)
        Iremax= np.max(Ire)
        Iimmax= np.max(Iim)
        T = 1/f

        tL_phase = np.array([2*T,2*T])
        IL_phase = np.array([Ipmax,1.25*Ipmax])
        tR_phase = np.array([2*T-phi/w,2*T-phi/w])
        IR_phase = np.array([Ismax,4.1*Ismax])
        zero_line = 0*t


        xTicks  = (np.max(t)/8)*np.linspace(0,8,9)
        xLabels = ['0','T/2','T','3T/2','2T','5T/2','3T','7T/2','4T']

        Ax1.grid('both', linestyle='-', linewidth=0.8, color=[0.8, 0.8, 0.8])
        Ax1.plot(t,zero_line,color='k',linewidth=2)
        Ax1.plot(t,I0,color='k',linewidth=4)
        Ax1.plot(tL_phase,IL_phase,color='k',ls=':',linewidth=8)
        Ax1.set_xbound(0,np.max(t))
        Ax1.set_ybound(1.55*np.min(I0),1.55*np.max(I0))
        Ax1.set_xlabel('Time',fontsize=FS+2)
        Ax1.set_ylabel('Primary Current [A]',fontsize=FS+2)
        Ax1.tick_params(labelsize=FS-2)



        Ax1b = Ax1.twinx()
        Ax1b.plot(t,Is,color='g',linewidth=4)
        Ax1b.plot(tR_phase,IR_phase,color='k',ls=':',linewidth=8)
        Ax1b.set_xbound(0,np.max(t))
        Ax1b.set_ybound(5.01*np.min(Is),5.01*np.max(Is))
        Ax1b.set_ylabel('Secondary Current [A]',fontsize=FS+2,color='g')
        Ax1b.tick_params(labelsize=FS-2)
        Ax1b.tick_params(axis='y',colors='g')
        Ax1b.xaxis.set_ticks(xTicks)
        Ax1b.xaxis.set_ticklabels(xLabels)
        Ax1b.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

        T_str  = '{:.3e}'.format(T)
        Ip_str = '{:.3e}'.format(self.I)
        Is_str = '{:.3e}'.format(np.max(Is))
        phi_str= '{:.1f}'.format(-180*phi/np.pi)
        Ax1.text(0.05*T,1.3*Ipmax,'Period = '+T_str+' s',fontsize=FS-2)
        Ax1.text(0.05*T,-1.24*Ipmax,'$I_p$ Amplitude = '+Ip_str+' A',fontsize=FS-2)
        Ax1.text(0.05*T,-1.45*Ipmax,'$I_s$ Amplitude = '+Is_str+' A',fontsize=FS-2,color='g')
        Ax1.text(1.7*T,1.3*Ipmax,'Phase Lag ($\phi$) = '+phi_str+'$^o$',fontsize=FS,color='k')


        Ax2.grid('both', linestyle='-', linewidth=0.8, color=[0.8, 0.8, 0.8])
        Ax2.plot(t,zero_line,color='k',linewidth=2)
        Ax2.plot(t,Ire,color='b',linewidth=4)
        Ax2.plot(t,Iim,color='r',linewidth=4)
        Ax2.set_xbound(0,np.max(t))
        Ax2.set_ybound(1.61*np.min(Is),1.61*np.max(Is))
        Ax2.set_xlabel('Time',fontsize=FS+2)
        Ax2.set_ylabel('Secondary Current [A]',fontsize=FS+2)
        Ax2.tick_params(labelsize=FS-2)
        Ax2.xaxis.set_ticks(xTicks)
        Ax2.xaxis.set_ticklabels(xLabels)
        Ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

        Ire_str = '{:.3e}'.format(Iremax)
        Iim_str = '{:.3e}'.format(Iimmax)
        Ax2.text(0.05*T,-1.25*Ismax,'$I_{phase}$ Amplitude = '+Ire_str+' A',fontsize=FS-2,color='b')
        Ax2.text(0.05*T,-1.52*Ismax,'$I_{quad}$ Amplitude = '+Iim_str+' A',fontsize=FS-2,color='r')



        return Ax1, Ax1b, Ax2




    def plot_InducedCurrent_FD(self,Ax,Is,fi):

        FS = 20

        R = self.R
        L = self.L

        Imax = np.max(-np.real(Is))

        f = np.logspace(0,8,101)


        Ax.grid('both', linestyle='-', linewidth=0.8, color=[0.8, 0.8, 0.8])
        Ax.semilogx(f,-np.real(Is),color='k',linewidth=4,label="$I_{Re}$")
        Ax.semilogx(f,-np.imag(Is),color='k',ls='--',linewidth=4,label="$I_{Im}$")
        Ax.semilogx(fi*np.array([1.,1.]),np.array([0,1.1*Imax]),color='r',ls='-',linewidth=3)
        handles, labels = Ax.get_legend_handles_labels()
        Ax.legend(handles, labels, loc='upper left', fontsize=FS)

        Ax.set_xlabel('Frequency [Hz]',fontsize=FS+2)
        Ax.set_ylabel('$\mathbf{- \, I_s (\omega)}$ [A]',fontsize=FS+2,labelpad=-10)
        Ax.set_title('Frequency Response',fontsize=FS)
        Ax.set_ybound(0,1.1*Imax)
        Ax.tick_params(labelsize=FS-2)
        Ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

        #R_str    = '{:.3e}'.format(R)
        #L_str    = '{:.3e}'.format(L)
        #f_str    = '{:.3e}'.format(fi)
        #EMF_str  = '{:.2e}j'.format(EMFi.imag)
        #I_str    = '{:.2e} - {:.2e}j'.format(float(np.real(Isi)),np.abs(float(np.imag(Isi))))

        #Ax.text(1.4,1.01*Imax,'$R$ = '+R_str+' $\Omega$',fontsize=FS)
        #Ax.text(1.4,0.94*Imax,'$L$ = '+L_str+' H',fontsize=FS)
        #Ax.text(1.4,0.87*Imax,'$f$ = '+f_str+' Hz',fontsize=FS,color='r')
        #Ax.text(1.4,0.8*Imax,'$V$ = '+EMF_str+' V',fontsize=FS,color='r')
        #Ax.text(1.4,0.73*Imax,'$I_s$ = '+I_str+' A',fontsize=FS,color='r')

        return Ax

    def plot_InducedCurrent_TD(self,Ax,Is,ti,Vi,Isi):

        FS = 20

        R = self.R
        L = self.L

        Imax = np.max(Is)

        t = np.logspace(-6,0,101)

        Ax.grid('both', linestyle='-', linewidth=0.8, color=[0.8, 0.8, 0.8])
        Ax.semilogx(t,Is,color='k',linewidth=4)
        Ax.semilogx(ti*np.array([1.,1.]),np.array([0,1.3*Imax]),color='r',ls='-',linewidth=3)

        Ax.set_xlabel('Time [s]',fontsize=FS+2)
        Ax.set_ylabel('$\mathbf{I_s (\omega)}$ [A]',fontsize=FS+2,labelpad=-10)
        Ax.set_title('Transient Induced Current',fontsize=FS)
        Ax.set_ybound(0,1.2*Imax)
        Ax.tick_params(labelsize=FS-2)
        Ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

        #R_str    = '{:.3e}'.format(R)
        #L_str    = '{:.3e}'.format(L)
        #t_str    = '{:.3e}'.format(ti)
        #V_str    = '{:.3e}'.format(Vi)
        #I_str    = '{:.3e}'.format(Isi)

        #Ax.text(1.4e-6,1.12*Imax,'$R$ = '+R_str+' $\Omega$',fontsize=FS)
        #Ax.text(1.4e-6,1.04*Imax,'$L$ = '+L_str+' H',fontsize=FS)
        #Ax.text(4e-2,1.12*Imax,'$t$ = '+t_str+' s',fontsize=FS,color='r')
        #Ax.text(4e-2,1.04*Imax,'$V$ = '+V_str+' V',fontsize=FS,color='r')
        #Ax.text(4e-2,0.96*Imax,'$I_s$ = '+I_str+' A',fontsize=FS,color='r')

        return Ax














