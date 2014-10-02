from SimPEG.Utils import sub2ind, mkvc
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
def ViewWiggle(syndata):
    dx = 20
    fig, ax = plt.subplots(1, 1, figsize=(7, 8))
    kwargs = {
    'skipt':1,
    'scale': 0.05,
    'lwidth': 1.,
    'dx': dx,
    'sampr': 0.004,
    'clip' : dx*10.,
    }
    extent = [0., 38*dx, 1.0, 0.]

    ax.invert_yaxis()
    wiggle(syndata, ax = ax, **kwargs)
    ax.set_xlabel("Offset (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title("CMP gather")

def NoisyNMOWidget(tintercept, v1, v2, v3):
    syndata = np.load('obsdata1.npy')
    np.random.randn()
    dx = 20
    xorig = np.arange(38)*dx
    time1 = HyperbolicFun(tintercept, xorig, v1)
    time2 = HyperbolicFun(tintercept, xorig, v2)
    time3 = HyperbolicFun(tintercept, xorig, v3)

    fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    kwargs = {
    'skipt':1,
    'scale': 0.05,
    'lwidth': 1.,
    'dx': dx,
    'sampr': 0.004,
    'clip' : dx*10.,
    }

    extent = [0., 38*dx, 1.0, 0.]
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    wiggle(syndata, ax = ax[0], **kwargs)
    toffset = np.sqrt(xorig**2/v2**2+tintercept**2)-tintercept
    wiggle(syndata, ax = ax[1], manthifts=toffset, **kwargs)

    ax[0].axis(extent)
    ax[1].axis(extent)
    ax[0].plot(xorig, time1, 'b', lw=2)
    ax[0].plot(xorig, time2, 'r', lw=2)
    ax[0].plot(xorig, time3, 'g', lw=2)

    ax[0].set_xlabel("Offset (m)")
    ax[1].set_xlabel("Offset (m)")
    ax[0].set_ylabel("Time (s)")
    ax[1].set_ylabel("Time (s)")
    ax[0].set_title("CMP gather")
    ax[1].set_title("NMO corrected CMP gather")

def CleanNMOWidget(tintercept, v):
    syndata = np.load('syndata1.npy')
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

    extent = [0., 38*dx, 1.0, 0.]
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    wiggle(syndata, ax = ax[0], **kwargs)
    toffset = np.sqrt(xorig**2/v**2+tintercept**2)-tintercept
    wiggle(syndata, ax = ax[1], manthifts=toffset, **kwargs)

    ax[0].axis(extent)
    ax[1].axis(extent)
    ax[0].plot(xorig, time, 'b', lw=2)
    ax[0].set_xlabel("Offset (m)")
    ax[1].set_xlabel("Offset (m)")
    ax[0].set_ylabel("Time (s)")
    ax[1].set_ylabel("Time (s)")
    ax[0].set_title("CMP gather")
    ax[1].set_title("NMO corrected CMP gather")

def HyperbolicFun(tintercept, x, velocity):
    time = np.sqrt(x**2/velocity**2+tintercept**2)
    return time

def NMOstackthree(data, v1, v2, v3):
    dx = 20.
    xorig = np.arange(38)*dx
    time = np.load('time1.npy')
    traces = np.zeros((3,time.size))
    vtemp = np.r_[v1, v2, v3]
    for itry in range(3):
        traces[itry,:] = NMOstack(data, xorig, time, vtemp[itry])

    fig, ax = plt.subplots(1, 3, figsize=(10, 8))
    kwargs = {
    'skipt':1,
    'scale': 2.,
    'lwidth': 1.,
    'sampr': 0.004,
    'clip' : 10,
    }
    for i in range(3):
        extent = [traces[i,:].min(), traces[i,:].max(), time.max(), time.min()]
        ax[i].invert_yaxis()
        ax[i].axis(extent)
        wiggle(traces[i,:].reshape([1,-1]), ax=ax[i], **kwargs)
        ax[i].set_xlabel("Amplitude")
        ax[i].set_ylabel("Time (s)")
        ax[i].set_title(("Velocity = %6.1f")%(vtemp[i]))

def NMOstack(data, xorig, time, v):
    if np.isscalar(v):
        v = np.ones_like(time)*v
    Time = (time.reshape([1,-1])).repeat(data.shape[0], axis=0)
    singletrace = np.zeros(data.shape[1])
    for i in range(time.size):
        toffset = np.sqrt(xorig**2/v[i]**2+time[i]**2)
        Time = (time.reshape([1,-1])).repeat(data.shape[0], axis=0)
        Toffset = (toffset.reshape([-1,1])).repeat(data.shape[1], axis=1)
        indmin = np.argmin(abs(Time-Toffset), axis=1)
        singletrace[i] = (mkvc(data)[sub2ind(data.shape, np.c_[np.arange(data.shape[0]), indmin])]).sum()
    return singletrace

def NMOstackSingle(data, v):
    dx = 20.
    xorig = np.arange(38)*dx
    time = np.load('time1.npy')
    singletrace = NMOstack(data, xorig, time, v)

    fig, ax = plt.subplots(1, 1, figsize=(7, 8))
    kwargs = {
    'skipt':1,
    'scale': 2.,
    'lwidth': 1.,
    'sampr': 0.004,
    'ax': ax,
    'clip' : 10,
    }
    extent = [singletrace.min(), singletrace.max(), time.max(), time.min()]
    ax.invert_yaxis()
    ax.axis(extent)
    wiggle(singletrace.reshape([1,-1]), **kwargs)
    ax.set_xlabel("Amplitude")
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

