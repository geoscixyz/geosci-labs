
import scipy.io
from simpegseis import UtilsSeis
import numpy as np
import matplotlib.pyplot as plt

def NMOWidget(tintercept, v):
    data = scipy.io.loadmat('data_syn.mat')
    syndata = data['D'].T[:, :280]
    syndata = syndata.copy() + np.random.randn(syndata.shape[0], syndata.shape[1])*15.*abs(syndata).mean()
    np.random.randn()
    dx = 20
    xorig = np.arange(38)*dx
    time = HyperbolicFun(tintercept, xorig, v)
    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    kwargs = {
    'skipt':1,
    'scale': 0.05,
    'lwidth': 1.,
    'dx': dx,
    'sampr': 0.004,
    'clip' : dx*10.,
    }

    extent = [0., 38*dx, 1.2, 0.]
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    UtilsSeis.wiggle(syndata, ax = ax[0], **kwargs)
    toffset = np.sqrt(xorig**2/v**2+tintercept**2)-tintercept
    UtilsSeis.wiggle(syndata, ax = ax[1], manthifts=toffset, **kwargs)

    ax[0].axis(extent)
    ax[1].axis(extent)
    ax[0].plot(xorig, time, 'b', lw=2)
    ax[0].set_xlabel("Offset (m)")
    ax[1].set_xlabel("Offset (m)")
    ax[0].set_ylabel("Time (s)")
    ax[1].set_ylabel("Time (s)")

def HyperbolicFun(tintercept, x, velocity):
    time = np.sqrt(x**2/velocity**2+tintercept**2)
    return time

def stack(data, xorig, time, tintercept, v):
    toffset = np.sqrt(xorig**2/v**2+tintercept**2)
    Time = (time.reshape([1,-1])).repeat(data.shape[0], axis=0)
    Toffset = (toffset.reshape([-1,1])).repeat(data.shape[1], axis=1)
    nwindow = 60
    newdata = np.zeros_like(data)
    indmin = np.argmin(abs(Time-Toffset), axis=1)
    ind1 = np.arange(nwindow)+indmin[0]-int(0.5*nwindow)
    singletrace = np.zeros(data.shape[1])
    for i in range(indmin.size):
        ind_temp = np.arange(nwindow)+indmin[i]-int(0.5*nwindow)
        newdata[i, ind_temp] = data[i, ind_temp]
        singletrace[ind1] = data[i, ind_temp] + singletrace[ind1]
    fig, ax = plt.subplots(1, 1, figsize=(7, 8))
    kwargs = {
    'skipt':1,
    'scale': 1.,
    'lwidth': 1.,
    'sampr': 0.004,
    'ax': ax
    }
    extent = [singletrace.min(), singletrace.max(), time.max(), time.min()]
    ax.invert_yaxis()
    ax.axis(extent)
    UtilsSeis.wiggle(singletrace.reshape([1,-1]), **kwargs)
    ax.set_xlabel("Offset (m)")
    ax.set_ylabel("Time (s)")

def clipsign (value, clip):
  clipthese = abs(value) > clip
  return value * ~clipthese + np.sign(value)*clip*clipthese

def wiggle (traces, skipt=1,scale=1.,lwidth=.1,offsets=None,redvel=0., manthifts=None, tshift=0.,sampr=1.,clip=10., dx=1., color='black',fill=True,line=True, ax=None):

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
        plt.plot(i*dx + clipsign(trace / scale, clip), t - shifts[i], color=color, linewidth=lwidth)
      if (fill):
        for j in range(ns):
          if (trace[j] < 0):
            trace[j] = 0
        plt.fill(i*dx + clipsign(trace / scale, clip), t - shifts[i], color=color, linewidth=0)
    else:
      if (line):
        ax.plot(i*dx + clipsign(trace / scale, clip), t - shifts[i], color=color, linewidth=lwidth)
      if (fill):
        for j in range(ns):
          if (trace[j] < 0):
            trace[j] = 0
        ax.fill(i*dx + clipsign(trace / scale, clip), t - shifts[i], color=color, linewidth=0)

