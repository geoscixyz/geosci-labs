from IPython.display import set_matplotlib_formats
import matplotlib
from SimPEG.utils import download, mkvc, sub2ind
import numpy as np
import scipy.io
from ipywidgets import (
    interact,
    interactive,
    IntSlider,
    widget,
    FloatText,
    FloatSlider,
    fixed,
)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ..base import wiggle

set_matplotlib_formats("png")
matplotlib.rcParams["savefig.dpi"] = 70  # Change this to adjust figure size


def ViewWiggle(syndat, obsdat):
    syndata = np.load(syndat)
    obsdata = np.load(obsdat)
    dx = 20
    _, ax = plt.subplots(1, 2, figsize=(14, 8))
    kwargs = {
        "skipt": 1,
        "scale": 0.05,
        "lwidth": 1.0,
        "dx": dx,
        "sampr": 0.004,
        "clip": dx * 10.0,
    }

    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    wiggle(syndata, ax=ax[0], **kwargs)
    wiggle(obsdata, ax=ax[1], **kwargs)
    ax[0].set_xlabel("Offset (m)")
    ax[0].set_ylabel("Time (s)")
    ax[0].set_title("Clean CMP gather")
    ax[1].set_xlabel("Offset (m)")
    ax[1].set_ylabel("Time (s)")
    ax[1].set_title("Noisy CMP gather")


def NoisyNMOWidget(t0, v, syndat, timdat):
    syndata = np.load(syndat)
    time_data = np.load(timdat)
    dx = 20
    xorig = np.arange(38) * dx
    time = HyperbolicFun(t0, xorig, v)
    plt.figure(figsize=(20, 8))
    kwargs = {
        "skipt": 1,
        "scale": 0.05,
        "lwidth": 1.0,
        "dx": dx,
        "sampr": 0.004,
        "clip": dx * 10.0,
    }
    gs1 = gridspec.GridSpec(1, 9)
    gs1.update(left=0.05, right=0.48, wspace=0.05)
    ax1 = plt.subplot(gs1[:, 0:3])
    ax2 = plt.subplot(gs1[:, 4:7])
    ax3 = plt.subplot(gs1[:, 8])

    extent = [0.0, 38 * dx, 1.0, 0.0]
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    wiggle(syndata, ax=ax1, **kwargs)
    t_reflector = 0.49
    toffset = np.sqrt(xorig ** 2 / v ** 2 + t0 ** 2) - t0
    wiggle(syndata, ax=ax2, manthifts=toffset + t0 - t_reflector, **kwargs)

    ax1.axis(extent)
    ax2.axis(extent)
    ax1.plot(xorig, time, "r", lw=2)

    ax1.set_xlabel("Offset (m)")
    ax2.set_xlabel("Offset (m)")
    ax1.set_ylabel("Time (s)")
    ax2.set_ylabel("Time (s)")
    ax1.set_title("CMP gather")
    ax2.set_title("NMO corrected CMP gather")

    singletrace = NMOstack(syndata, xorig, time_data, v)
    # singletrace = singletrace

    kwargs = {
        "skipt": 1,
        "scale": 2.0,
        "lwidth": 1.0,
        "sampr": 0.004,
        "ax": ax3,
        "clip": 10,
        "manthifts": np.r_[t0 - t_reflector],
    }
    extent = [singletrace.min(), singletrace.max(), time_data.max(), time_data.min()]
    ax3.invert_yaxis()
    ax3.axis(extent)
    wiggle(singletrace.reshape([1, -1]), **kwargs)
    ax3.set_xlabel("Amplitude")
    ax3.set_ylabel("Time (s)")
    ax3.set_xlim(-4.5, 4.5)
    ax3.set_xticks([-4.5, 0.0, 4.5])
    ax3.set_title("Stacked trace")

    plt.show()


def CleanNMOWidget(t0, v, syndat, timdat):
    syndata = np.load(syndat)
    time_data = np.load(timdat)
    np.random.randn()
    dx = 20
    xorig = np.arange(38) * dx
    time = HyperbolicFun(t0, xorig, v)
    # fig, ax = plt.subplots(1, 2, figsize=(14, 8))
    plt.figure(figsize=(20, 8))
    kwargs = {
        "skipt": 1,
        "scale": 0.05,
        "lwidth": 1.0,
        "dx": dx,
        "sampr": 0.004,
        "clip": dx * 10.0,
    }
    gs1 = gridspec.GridSpec(1, 9)
    gs1.update(left=0.05, right=0.48, wspace=0.05)
    ax1 = plt.subplot(gs1[:, 0:3])
    ax2 = plt.subplot(gs1[:, 4:7])
    ax3 = plt.subplot(gs1[:, 8])

    extent = [0.0, 38 * dx, 1.0, 0.0]
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    wiggle(syndata, ax=ax1, **kwargs)
    toffset = np.sqrt(xorig ** 2 / v ** 2 + t0 ** 2) - t0
    t_reflector = 0.39
    wiggle(syndata, ax=ax2, manthifts=toffset + t0 - t_reflector, **kwargs)

    ax1.axis(extent)
    ax2.axis(extent)
    ax1.plot(xorig, time, "r", lw=2)

    ax1.set_xlabel("Offset (m)")
    ax2.set_xlabel("Offset (m)")
    ax1.set_ylabel("Time (s)")
    ax2.set_ylabel("Time (s)")
    ax1.set_title("CMP gather")
    ax2.set_title("NMO corrected CMP gather")

    singletrace = NMOstack(syndata, xorig, time_data, v)
    # singletrace = singletrace
    kwargs = {
        "skipt": 1,
        "scale": 2.0,
        "lwidth": 1.0,
        "sampr": 0.004,
        "ax": ax3,
        "clip": 10,
        "manthifts": np.r_[t0 - t_reflector],
    }
    extent = [singletrace.min(), singletrace.max(), time_data.max(), time_data.min()]
    ax3.invert_yaxis()
    ax3.axis(extent)
    wiggle(singletrace.reshape([1, -1]), **kwargs)
    ax3.set_xlabel("Amplitude")
    ax3.set_ylabel("Time (s)")
    ax3.set_xlim(-4.5, 4.5)
    ax3.set_xticks([-4.5, 0.0, 4.5])
    ax3.set_title("Stacked trace")
    plt.show()


def HyperbolicFun(t0, x, velocity):
    time = np.sqrt(x ** 2 / velocity ** 2 + t0 ** 2)
    return time


def NMOstackthree(dat, tintercept, v1, v2, v3, timdat):
    data = np.load(dat)
    time = np.load(timdat)
    dx = 20.0
    xorig = np.arange(38) * dx
    traces = np.zeros((3, time.size))
    vtemp = np.r_[v1, v2, v3]
    for itry in range(3):
        traces[itry, :] = NMOstack(data, xorig, time, vtemp[itry])

    _, ax = plt.subplots(1, 3, figsize=(10, 8))
    t_reflector = 0.49
    kwargs = {
        "skipt": 1,
        "scale": 2.0,
        "lwidth": 1.0,
        "sampr": 0.004,
        "clip": 10,
        "manthifts": np.r_[t_reflector - tintercept],
    }
    for i in range(3):
        extent = [traces[i, :].min(), traces[i, :].max(), time.max(), time.min()]
        ax[i].invert_yaxis()
        ax[i].axis(extent)
        wiggle(traces[i, :].reshape([1, -1]), ax=ax[i], **kwargs)
        ax[i].set_xlabel("Amplitude")
        if i == 0:
            ax[i].set_ylabel("Time (s)")
        ax[i].set_title(("Velocity = %6.1f") % (vtemp[i]))


def NMOstack(data, xorig, time, v):
    if np.isscalar(v):
        v = np.ones_like(time) * v
    Time = (time.reshape([1, -1])).repeat(data.shape[0], axis=0)
    singletrace = np.zeros(data.shape[1])
    for i in range(time.size):
        toffset = np.sqrt(xorig ** 2 / v[i] ** 2 + time[i] ** 2)
        Time = (time.reshape([1, -1])).repeat(data.shape[0], axis=0)
        Toffset = (toffset.reshape([-1, 1])).repeat(data.shape[1], axis=1)
        indmin = np.argmin(abs(Time - Toffset), axis=1)
        singletrace[i] = (
            mkvc(data)[sub2ind(data.shape, np.c_[np.arange(data.shape[0]), indmin])]
        ).sum()
    return singletrace


def NMOstackSingle(data, tintercept, v, timeFile):
    dx = 20.0
    xorig = np.arange(38) * dx
    timdat = download(timeFile, verbose=False)
    time = np.load(timdat)
    singletrace = NMOstack(data, xorig, time, v)

    _, ax = plt.subplots(1, 1, figsize=(7, 8))
    kwargs = {
        "skipt": 1,
        "scale": 2.0,
        "lwidth": 1.0,
        "sampr": 0.004,
        "ax": ax,
        "clip": 10,
    }
    extent = [singletrace.min(), singletrace.max(), time.max(), time.min()]
    ax.invert_yaxis()
    ax.axis(extent)
    wiggle(singletrace.reshape([1, -1]), **kwargs)
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Time (s)")


def InteractClean(cleanDataFile, cleanTimeFile):
    clean = interactive(
        CleanNMOWidget,
        t0=FloatSlider(min=0.2, max=0.8, step=0.01, continuous_update=False),
        v=FloatSlider(min=1000.0, max=5000.0, step=100.0, continuous_update=False),
        syndat=fixed(cleanDataFile),
        timdat=fixed(cleanTimeFile),
    )
    return clean


def InteractNosiy(noisyDataFile, noisyTimeFile):
    noisy = interactive(
        NoisyNMOWidget,
        t0=FloatSlider(min=0.1, max=0.6, step=0.01, continuous_update=False),
        v=FloatSlider(min=800.0, max=2500.0, step=100.0, continuous_update=False),
        syndat=fixed(noisyDataFile),
        timdat=fixed(noisyTimeFile),
    )
    return noisy
