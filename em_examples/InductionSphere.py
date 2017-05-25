import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter




##############################################
#   PLOTTING FUNCTIONS FOR WIDGETS
##############################################











############################################
#   DEFINE CLASSES
############################################

class IndSphere():
    """Fucntionwhcihdf
    Input variables:
        
        Output variables:
    """
    

    def __init__(self,m,m_dir,sig,mur,a,x0,y0,z0,xtx,ytx,ztx,xrx,yrx,zrx):
        """Defines Initial Attributes"""
        
        # INITIALIZES OBJECT
        
        # m: Transmitter dipole moment
        # m_dir: Transmitter dipole orentation 'x', 'y' or 'z'
        # h: Transmitter height
        # sig: Sphere conductivity
        # mur: Sphere relative permeability
        # a: Sphere radius
        # x0: Sphere x-location
        # y0: Sphere y-location
        # z0: Sphere z-location
        
        self.m = m
        self.m_dir = m_dir
        self.sig = sig
        self.mur = mur
        self.a = a
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.xtx = xtx
        self.ytx = ytx
        self.ztx = ztx
        self.xrx = xrx
        self.yrx = yrx
        self.zrx = zrx

############################################
#   DEFINE METHODS
############################################

    

    def fcn_ComputePrimary(self):
    	"""Computes Inducing Field at Sphere"""

    	m = self.m
        m_dir = self.m_dir
        xtx = self.xtx
        ytx = self.ytx
        ztx = self.ztx
        x0 = self.x0
        y0 = self.y0
        z0 = self.z0

        R = np.sqrt((x0-xtx)**2 + (y0-ytx)**2 + (z0-ztx)**2)

    	if m_dir == "x":
    		self.Hpx = (1/(4*np.pi))*(3*m*(x0-xtx)*(x0-xtx)/R**5 - m/R**3)
    		self.Hpy = (1/(4*np.pi))*(3*m*(y0-ytx)*(x0-xtx)/R**5)
    		self.Hpz = (1/(4*np.pi))*(3*m*(z0-ztx)*(x0-xtx)/R**5)
    	elif m_dir == "y":
    		self.Hpx = (1/(4*np.pi))*(3*m*(x0-xtx)*(y0-ytx)/R**5)
    		self.Hpy = (1/(4*np.pi))*(3*m*(y0-ytx)*(y0-ytx)/R**5 - m/R**3)
    		self.Hpz = (1/(4*np.pi))*(3*m*(z0-ztx)*(y0-ytx)/R**5)
    	elif m_dir == "z":
    		self.Hpx = (1/(4*np.pi))*(3*m*(x0-xtx)*(z0-ztx)/R**5)
    		self.Hpy = (1/(4*np.pi))*(3*m*(y0-ytx)*(z0-ztx)/R**5)
    		self.Hpz = (1/(4*np.pi))*(3*m*(z0-ztx)*(z0-ztx)/R**5 - m/R**3)

    global fcn_ComputeExcitation_FEM
    def fcn_ComputeExcitation_FEM(self,f):
    	"""Compute Excitation Factor (FEM)"""

        sig = self.sig
        mur = self.mur
        a = self.a
        w = 2*np.pi*f
        mu = 4*np.pi*1e-7*mur

        alpha = a*np.sqrt(1j*w*mu*sig)

        chi = 1.5*(2*mur*(np.tanh(alpha) - alpha) + (alpha**2*np.tanh(alpha) - alpha + np.tanh(alpha)))/(mur*(np.tanh(alpha) - alpha) - (alpha**2*np.tanh(alpha) - alpha + np.tanh(alpha)))

        return chi

    def fcn_ComputeFrequencyResponse(self,f):
    	"""Compute Spectrum at a single location (X,Y,Z)"""

        a = self.a
        x0 = self.x0
        y0 = self.y0
        z0 = self.z0
        xrx = self.xrx
        yrx = self.yrx
        zrx = self.zrx

        chi = fcn_ComputeExcitation_FEM(self,f)

        mx = 4*np.pi*a**3*chi*self.Hpx/3
        my = 4*np.pi*a**3*chi*self.Hpy/3
        mz = 4*np.pi*a**3*chi*self.Hpz/3
        R = np.sqrt((xrx-x0)**2 + (yrx-y0)**2 + (zrx-z0)**2)

        Hx = (1/(4*np.pi))*(3*(xrx-x0)*(mx*(xrx-x0) + my*(yrx-y0) + mz*(zrx-z0))/R**5 - mx/R**3)
        Hy = (1/(4*np.pi))*(3*(yrx-y0)*(mx*(xrx-x0) + my*(yrx-y0) + mz*(zrx-z0))/R**5 - my/R**3)
        Hz = (1/(4*np.pi))*(3*(zrx-z0)*(mx*(xrx-x0) + my*(yrx-y0) + mz*(zrx-z0))/R**5 - mz/R**3)
        Habs = np.sqrt(Hx**2 + Hy**2 + Hz**2)

        return chi, Hx, Hy, Hz, Habs

    def fcn_ComputeFrequencyResponsePlane(self,f,X,Y,Z):
    	"""Compute Single Frequency Response at (X,Y,Z)"""

        a = self.a
        x0 = self.x0
        y0 = self.y0
        z0 = self.z0

        chi = fcn_ComputeExcitation_FEM(self,f)

        mx = 4*np.pi*a**3*chi*self.Hpx/3
        my = 4*np.pi*a**3*chi*self.Hpy/3
        mz = 4*np.pi*a**3*chi*self.Hpz/3
        R = np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)

        Hx = (1/(4*np.pi))*(3*(X-x0)*(mx*(X-x0) + my*(Y-y0) + mz*(Z-z0))/R**5 - mx/R**3)
        Hy = (1/(4*np.pi))*(3*(Y-y0)*(mx*(X-x0) + my*(Y-y0) + mz*(Z-z0))/R**5 - my/R**3)
        Hz = (1/(4*np.pi))*(3*(Z-z0)*(mx*(X-x0) + my*(Y-y0) + mz*(Z-z0))/R**5 - mz/R**3)
        Habs = np.sqrt(Hx**2 + Hy**2 + Hz**2)

        return Hx, Hy, Hz, Habs

    



############################################
#   DEFINE PLOTTING FUNCTIONS
############################################

def plotResponseFEM(Ax,f,H,Comp):

	FS = 20

	xTicks = (np.logspace(np.log(np.min(f)),np.log(np.max(f)),9))
	
	Ax.grid('both', linestyle='-', linewidth=0.8, color=[0.8, 0.8, 0.8])
	Ax.semilogx(f,0*f,color='k',linewidth=2)
	Ax.semilogx(f,np.real(H),color='k',linewidth=4,label="Real")
	Ax.semilogx(f,np.imag(H),color='k',linewidth=4,ls='--',label="Imaginary")
	Ax.set_xbound(np.min(f),np.max(f))
	Ax.set_xlabel('Frequency [Hz]',fontsize=FS+2)
	Ax.tick_params(labelsize=FS-2)
	Ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

	if Comp == 'x':
		Ax.set_ylabel('$\mathbf{Hx}$ [A/m]',fontsize=FS+4,labelpad=-10)
	elif Comp == 'y':
		Ax.set_ylabel('$\mathbf{Hy}$ [A/m]',fontsize=FS+4,labelpad=-10)
	elif Comp == 'z':
		Ax.set_ylabel('$\mathbf{Hz}$ [A/m]',fontsize=FS+4,labelpad=-10)
	elif Comp == 'abs':
		Ax.set_ylabel('$\mathbf{|H|}$ [A/m]',fontsize=FS+4,labelpad=-10)
	elif Comp == 'chi':
		Ax.set_ylabel('$\mathbf{\chi}$ [(A/m)/(A/m)]',fontsize=FS+4,labelpad=-10)

	if np.max(np.real(H[-1])) > 0.:
		handles, labels = Ax.get_legend_handles_labels()
		Ax.legend(handles, labels, loc='upper left', fontsize=FS)
	elif np.max(np.real(H[-1])) < 0.:
		handles, labels = Ax.get_legend_handles_labels()
		Ax.legend(handles, labels, loc='lower left', fontsize=FS)

	return Ax


#def plotResponsePlaneFEM(Ax,f,X,Y,Z,H,Comp):

#	FS = 20

#	tol = 1e-4
#	Habs = np.abs(H)
#	MAX = np.max(np.abs(Habs))
#	Hsign = np.sign(H)












