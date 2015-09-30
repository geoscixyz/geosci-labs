import numpy as np
try:
    from IPython.html.widgets import  interactive, IntSlider, widget, FloatText
    pass
except Exception, e:    
    from ipywidgets import interactive, IntSlider, widget, FloatText

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.size"] = 16
pi = np.pi
def getRicker(f,t):
    """
    Retrieves a Ricker wavelet with center frequency f.
    See: http://www.subsurfwiki.org/wiki/Ricker_wavelet
    """
    # assert len(f) == 1, 'Ricker wavelet needs 1 frequency as input'
    # f = f[0]
    pift = pi*f*t
    wav = (1 - 2*pift**2)*np.exp(-pift**2)
    return wav

def funDR1R2(x, v1, v2, v3, z1, z2):
# def funDR1R2(x, v1, v2, v3, z1, z2):
	"""
		Computes arrival time of direct, and two critically refracted waves from three layer models

		x: offset (array)
		v1: velocity of 1st layer (float)
		v2: velocity of 2nd layer (float)
		v3: velocity of 3rd layer (float)
		z1: thickness of 1st layer (float)
		z2: thickness of 2nd layer (float)

	"""
	direct = 1./v1*x
	theta1 = np.arcsin(v1/v2)    
	theta2 = np.arcsin(v2/v3)
	ti1 = 2*z1*np.cos(theta1)/v1
	ti2 = 2*z2*np.cos(theta2)/v2
	xc1 = 2*z1*np.tan(theta1)
	xc2 = 2*z2*np.tan(theta2)
	act1 = x > xc1
	act2 = x > xc2   
	ref1 = np.ones_like(x)*np.nan
	ref1[act1] = 1./v2*x[act1] + ti1    
	ref2 = np.ones_like(x)*np.nan
	ref2[act2] = 1./v3*x[act2] + ti2 + ti1
	return np.c_[direct, ref1, ref2]

def viewTXdiagram(x0, dx, v1, v2, v3, z1, z2):
    x = x0 + np.arange(12)*dx
    fig, ax = plt.subplots(1,1, figsize = (8, 6))
    dum = funDR1R2(x, v1, v2, v3, z1, z2)
    ax.plot(x, dum[:,0], 'ro')
    ax.plot(x, dum[:,1], 'bo')
    ax.plot(x, dum[:,2], 'go')
    ax.set_xlim(0., 120.)
    ax.set_ylim(0., 0.24)
    ax.invert_yaxis()
    ax.set_xlabel("Offset (m)")
    ax.set_ylabel("Time (s)")
    ax.grid(True)
    plt.show()
    return True


def viewWiggleTX(x0, dx, v1, v2, v3, z1, z2):
    x = x0 + np.arange(12)*dx
    dum = funDR1R2(x, v1, v2, v3, z1, z2)
    dt = 1e-3
    ntime = 241
    time = np.arange(ntime)*dt
    wave = getRicker(100., time)
    data = np.zeros((ntime, 12))
    for i in range(12):
        inds = []
        for j in range(3):
            temp = np.argmin(abs(time-dum[i,j]))
            if temp > 0:
                inds.append(temp)
        data[inds,i] = 1.
    data_convolved = np.zeros_like(data)
    for i in range(12):
        temp = np.convolve(wave,data[:,i])[:ntime]
        data_convolved[:,i] =  temp + np.random.randn(ntime)*0.2    
    fig, ax = plt.subplots(1, 1, figsize=(7, 8))
    wiggleVarx(data_convolved.T, x=x, sampr=dt, lwidth=1.,scale=0.2, ax=ax)
    ax.invert_yaxis()
    ax.set_xlim(-1., 120)
    ax.set_xlabel("Offset (m)")
    ax.set_ylabel("Time (s)")    
    plt.show()
    return ax

def makeinteractTXdiagram():
	v1 = 400.
	v2 = 1000. 
	v3 = 1500.
	z1, z2 = 5., 10.
	Q = interactive(lambda x0, dx: viewTXdiagram(x0, dx, v1=v1, v2=v2, v3=v3, z1=z1, z2=z2), 
	             x0=IntSlider(min=1, max=10, step=1, value=4),
	             dx=IntSlider(min=1, max=10, step=1,value=4))
	return Q

def makeinteractTXwigglediagram():
	v1 = 400.
	v2 = 1000. 
	v3 = 1500.
	z1, z2 = 5., 10.
	Q = interactive(lambda x0, dx: viewWiggleTX(x0, dx, v1=v1, v2=v2, v3=v3, z1=z1, z2=z2), 
	             x0=IntSlider(min=1, max=10, step=1, value=4),
	             dx=IntSlider(min=1, max=10, step=1,value=4))
	return Q

def veiwSeisRefracSurvey(x0, dx):
    x = x0 + np.arange(12)*dx
    z = np.r_[0., 5., 15.]
    fig, ax = plt.subplots(1,1, figsize = (7,3))
    ax.plot(0., 0., 'ro')
    ax.plot(x, np.zeros_like(x), 'bo')
    ax.legend(("shot", "geophone"), fontsize = 14, bbox_to_anchor=(1.4, 1.05))
    xtemp = np.r_[-1, 100.]
    for i in range(3):
        ax.plot(xtemp, np.ones_like(xtemp)*z[i],'k-', lw=2)    
    ax.set_xlim(-1, 100.)
    ax.set_ylim(20, -4.)
    ax.text(50, 3, "Layer 1, v1 = 400 m/s", fontsize = 14)
    ax.text(50, 11, "Layer 2, v2 = 1000 m/s", fontsize = 14)
    ax.text(50, 18, "Layer 3, v3 = 1500 m/s", fontsize = 14)

    ax.arrow(x[4], 1, dx=dx, dy=0.)
    ax.arrow(0, 1, dx=x0, dy=0.)
    ax.text(x[4], 3.5, ("dx=%i m")%(dx) , fontsize = 14)
    ax.text(0., 3.5, ("x0=%i m")%(x0) , fontsize = 14)
    ax.set_xlabel("Offset (m)")
    ax.set_ylabel("Depth (m)")
    plt.show()
    return True

def makeinteractSeisRefracSruvey():
	Q = interactive(veiwSeisRefracSurvey, 
	             x0=IntSlider(min=1, max=10, step=1, value=4),
	             dx=IntSlider(min=1, max=10, step=1,value=4))
	return Q
#========================================================================================================================
# Visualziation for wiggle plot
#========================================================================================================================

def wiggleVarx (traces, x, skipt=1,scale=1.,lwidth=.1,offsets=None,redvel=0., manthifts=None, tshift=0.,sampr=1.,clip=10., color='black',fill=True,line=True, ax=None):

  ns = traces.shape[1]
  ntr = traces.shape[0]
  t = np.arange(ns)*sampr
  timereduce = lambda offsets, redvel, shift: [float(offset) / redvel + shift for offset in offsets]

  if (offsets is not None):
    shifts = timereduce(offsets, redvel, tshift)
  elif (manthifts is not None):
    shifts = manthifts
  else:
    shifts = np.zeros((ntr,))

  for i in range(0, ntr, skipt):
    trace = traces[i].copy()
    trace[0] = 0
    trace[-1] = 0

    if ax == None:
      if (line):
        plt.plot(x[i] + clipsign(trace / scale, clip), t - shifts[i], color=color, linewidth=lwidth)
      if (fill):
        for j in range(ns):
          if (trace[j] < 0):
            trace[j] = 0
        plt.fill(x[i] + clipsign(trace / scale, clip), t - shifts[i], color=color, linewidth=0)
    else:
      if (line):
        ax.plot(x[i] + clipsign(trace / scale, clip), t - shifts[i], color=color, linewidth=lwidth)
      if (fill):
        for j in range(ns):
          if (trace[j] < 0):
            trace[j] = 0
        ax.fill(x[i] + clipsign(trace / scale, clip), t - shifts[i], color=color, linewidth=0)

def clipsign (value, clip):
  clipthese = abs(value) > clip
  return value * ~clipthese + np.sign(value)*clip*clipthese

