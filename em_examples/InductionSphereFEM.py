import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter




##############################################
#   PLOTTING FUNCTIONS FOR WIDGETS
##############################################

def fcn_FDEM_InductionSpherePlaneWidget(xtx,ytx,ztx,m,m_dir,x0,y0,z0,a,sig,mur,xrx,yrx,zrx,f,Comp,Phase):

	fvec = np.logspace(0,8,49)

	xmin, xmax, dx, ymin, ymax, dy = -20., 20., 0.25, -20., 20., 0.25
	X,Y = np.mgrid[xmin:xmax+dx:dx, ymin:ymax+dy:dy]
	X = np.transpose(X)
	Y = np.transpose(Y)

	Obj = SpherePlaneFEM(m,m_dir)
	Obj.fcn_ComputePrimary(xtx,ytx,ztx,x0,y0,z0)

	Hx,Hy,Hz,Habs = Obj.fcn_ComputeFrequencyResponsePlane(f,sig,mur,a,x0,y0,z0,X,Y,zrx)
	Hxi,Hyi,Hzi,Habsi = Obj.fcn_ComputeFrequencyResponse(fvec,sig,mur,a,x0,y0,z0,xrx,yrx,zrx)

	fig1 = plt.figure(figsize=(17,6))
	Ax1 = fig1.add_axes([0.04,0,0.42,1])
	Ax2 = fig1.add_axes([0.6,0,0.4,1])

	if Comp == 'x':
		Ax1 = plotResponsePlaneFEM(Ax1,f,X,Y,ztx,Hx,Comp,Phase)
		Ax1 = plotPlaceTxRxSphere(Ax1,xtx,ytx,ztx,xrx,yrx,zrx,x0,y0,z0,a)
		Ax2 = plotResponseFEM(Ax2,fvec,Hxi,Comp)
	elif Comp == 'y':
		Ax1 = plotResponsePlaneFEM(Ax1,f,X,Y,ztx,Hy,Comp,Phase)
		Ax1 = plotPlaceTxRxSphere(Ax1,xtx,ytx,ztx,xrx,yrx,zrx,x0,y0,z0,a)
		Ax2 = plotResponseFEM(Ax2,fvec,Hyi,Comp)
	elif Comp == 'z':
		Ax1 = plotResponsePlaneFEM(Ax1,f,X,Y,ztx,Hz,Comp,Phase)
		Ax1 = plotPlaceTxRxSphere(Ax1,xtx,ytx,ztx,xrx,yrx,zrx,x0,y0,z0,a)
		Ax2 = plotResponseFEM(Ax2,fvec,Hzi,Comp)
	elif Comp == 'abs':
		Ax1 = plotResponsePlaneFEM(Ax1,f,X,Y,ztx,Habs,Comp,Phase)
		Ax1 = plotPlaceTxRxSphere(Ax1,xtx,ytx,ztx,xrx,yrx,zrx,x0,y0,z0,a)
		Ax2 = plotResponseFEM(Ax2,fvec,Habsi,Comp)


	plt.show(fig1)


##############################################
#   EXCITATION FUNCTIONS
##############################################

def fcn_ComputeExcitation_FEM(f,sig,mur,a):
    """Compute Excitation Factor (FEM)"""

    w = 2*np.pi*f
    mu = 4*np.pi*1e-7*mur
    alpha = a*np.sqrt(1j*w*mu*sig)

    chi = 1.5*(2*mur*(np.tanh(alpha) - alpha) + (alpha**2*np.tanh(alpha) - alpha + np.tanh(alpha)))/(mur*(np.tanh(alpha) - alpha) - (alpha**2*np.tanh(alpha) - alpha + np.tanh(alpha)))

    return chi


############################################
#   DEFINE CLASSES
############################################

class SpherePlaneFEM():
    """Fucntionwhcihdf
    Input variables:
        
        Output variables:
    """
    

    def __init__(self,m,m_dir):
        """Defines Initial Attributes"""
        
        # INITIALIZES OBJECT
        
        # m: Transmitter dipole moment
        # m_dir: Transmitter dipole orentation 'x', 'y' or 'z'
        
        self.m = m
        self.m_dir = m_dir

    def fcn_ComputePrimary(self,xtx,ytx,ztx,x0,y0,z0):
    	"""Computes Inducing Field at Sphere"""

    	m = self.m
    	m_dir = self.m_dir

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

    def fcn_ComputeFrequencyResponse(self,f,sig,mur,a,x0,y0,z0,xrx,yrx,zrx):
    	"""Compute Spectrum at a single location (X,Y,Z)"""

        chi = fcn_ComputeExcitation_FEM(f,sig,mur,a)

        mx = 4*np.pi*a**3*chi*self.Hpx/3
        my = 4*np.pi*a**3*chi*self.Hpy/3
        mz = 4*np.pi*a**3*chi*self.Hpz/3
        R = np.sqrt((xrx-x0)**2 + (yrx-y0)**2 + (zrx-z0)**2)

        Hx = (1/(4*np.pi))*(3*(xrx-x0)*(mx*(xrx-x0) + my*(yrx-y0) + mz*(zrx-z0))/R**5 - mx/R**3)
        Hy = (1/(4*np.pi))*(3*(yrx-y0)*(mx*(xrx-x0) + my*(yrx-y0) + mz*(zrx-z0))/R**5 - my/R**3)
        Hz = (1/(4*np.pi))*(3*(zrx-z0)*(mx*(xrx-x0) + my*(yrx-y0) + mz*(zrx-z0))/R**5 - mz/R**3)
        Habs = np.sqrt(np.real(Hx)**2 + np.real(Hy)**2 + np.real(Hz)**2) + 1j*np.sqrt(np.imag(Hx)**2 + np.imag(Hy)**2 + np.imag(Hz)**2)

        return Hx, Hy, Hz, Habs

    def fcn_ComputeFrequencyResponsePlane(self,f,sig,mur,a,x0,y0,z0,X,Y,Z):
    	"""Compute Single Frequency Response at (X,Y,Z)"""

        chi = fcn_ComputeExcitation_FEM(f,sig,mur,a)

        mx = 4*np.pi*a**3*chi*self.Hpx/3
        my = 4*np.pi*a**3*chi*self.Hpy/3
        mz = 4*np.pi*a**3*chi*self.Hpz/3
        R = np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)

        Hx = (1/(4*np.pi))*(3*(X-x0)*(mx*(X-x0) + my*(Y-y0) + mz*(Z-z0))/R**5 - mx/R**3)
        Hy = (1/(4*np.pi))*(3*(Y-y0)*(mx*(X-x0) + my*(Y-y0) + mz*(Z-z0))/R**5 - my/R**3)
        Hz = (1/(4*np.pi))*(3*(Z-z0)*(mx*(X-x0) + my*(Y-y0) + mz*(Z-z0))/R**5 - mz/R**3)
        Habs = np.sqrt(np.real(Hx)**2 + np.real(Hy)**2 + np.real(Hz)**2) + 1j*np.sqrt(np.imag(Hx)**2 + np.imag(Hy)**2 + np.imag(Hz)**2)

        return Hx, Hy, Hz, Habs


class SphereProfileFEM():
    """Fucntionwhcihdf
    Input variables:
        
        Output variables:
    """
    

    def __init__(self,m,orient):
        """Defines Initial Attributes"""
        
        # INITIALIZES OBJECT
        
        # m: Transmitter dipole moment
        # m_dir: Transmitter dipole orentation 'x', 'y' or 'z'
        
        self.m = m
        self.orient = orient





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


def plotResponsePlaneFEM(Ax,f,X,Y,Z,H,Comp,Phase):

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

	if Comp == 'abs':
		TickLabels = MAX*np.array([1.,1e-1,1e-2,1e-3,1e-4,0.,-1e-4,-1e-3,-1e-2,-1e-1,-1])
		TickLabels = ["%.1e" % x for x in TickLabels]
		Cplot = Ax.contourf(X,Y,Sign*H,50,cmap='seismic_r', vmin=-5, vmax=5)
		cbar = plt.colorbar(Cplot, ax=Ax, pad=0.02, ticks=-np.linspace(-5,5,11))
	else:
		TickLabels = MAX*np.array([-1.,-1e-1,-1e-2,-1e-3,-1e-4,0.,1e-4,1e-3,1e-2,1e-1,1])
		TickLabels = ["%.1e" % x for x in TickLabels]
		Cplot = Ax.contourf(X,Y,Sign*H,50,cmap='seismic_r', vmin=-5, vmax=5)
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

def plotPlaceTxRxSphere(Ax,xtx,ytx,ztx,xrx,yrx,zrx,x0,y0,z0,a):

	Xlim = Ax.get_xlim()
	Ylim = Ax.get_ylim()

	FS = 20

	Ax.scatter(xtx,ytx,s=100,color='k')
	Ax.text(xtx-0.5,ytx+1,'$\mathbf{Tx}$',fontsize=FS+6)
	Ax.scatter(xrx,yrx,s=100,color='k')
	Ax.text(xrx-0.5,yrx-2.5,'$\mathbf{Rx}$',fontsize=FS+6)

	xs = x0 + a*np.cos(np.linspace(0,2*np.pi,41))
	ys = y0 + a*np.sin(np.linspace(0,2*np.pi,41))

	Ax.plot(xs,ys,ls=':',color='k',linewidth=8)

	Ax.set_xbound(Xlim)
	Ax.set_ybound(Ylim)

	return Ax




