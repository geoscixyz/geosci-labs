import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from ipywidgets import interactive, IntSlider, widget, FloatText, FloatSlider

fontsize = 16
mpl.rcParams["font.size"] = fontsize

pi = np.pi

v1 = 600.0
v2 = 1200.0
v3 = 1700.0
z1, z2 = 5.0, 10.0

wavf = 100.0
t0 = 0.225 / wavf  # make it approx minimum phase


def getRicker(f, t, t0=t0):
    """
    Retrieves a Ricker wavelet with center frequency f.
    See: http://www.subsurfwiki.org/wiki/Ricker_wavelet
    """
    # assert len(f) == 1, 'Ricker wavelet needs 1 frequency as input'
    # f = f[0]
    pift = pi * f * (t - t0)
    wav = (1 - 2 * pift ** 2) * np.exp(-(pift ** 2))
    return wav


def plotWavelet(f=wavf, t=None, t0=t0, ax=None):
    if t is None:
        tmax = 1.5 / f
        t = np.linspace(0.0, tmax, 1000)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 4))
    wav = getRicker(f, t, t0)
    ax.plot(wav, t, color="k")
    ax.grid(which="both", linestyle="-", linewidth=0.5, color="k", alpha=0.3)
    ax.set_xlim([-1.2, 1.2])
    ax.axis([-1.2, 1.2, -0.1 * tmax, 1.1 * tmax])
    ax.invert_yaxis()
    ax.fill_betweenx(t, 0.0, wav, where=wav >= 0.0, color="k")
    ax.set_title("Source Wavelet", fontsize=fontsize)
    ax.set_xlabel("Signal Amplitude", fontsize=fontsize)
    ax.set_ylabel("time (s)", fontsize=fontsize)
    plt.show()
    return ax
    # ax.fill(t[i] + clipsign(trace / scale, clip), t - shifts[i], color=color, linewidth=0)


def direct(x, v1):
    """
    direct ray
    """
    return x / v1


def refraction1(x, v1, v2, z1):
    """
    refraction off of first interface
    """

    theta1 = np.arcsin(v1 / v2)
    ti1 = 2 * z1 * np.cos(theta1) / v1
    xc1 = 2 * z1 * np.tan(theta1)
    if np.isscalar(x):
        ref1 = 1.0 / v2 * x + ti1
    else:
        act1 = x > xc1
        ref1 = np.ones_like(x) * np.nan
        ref1[act1] = 1.0 / v2 * x[act1] + ti1
    return ref1


def refraction2(x, v1, v2, v3, z1, z2):
    theta1 = np.arcsin(v1 / v3)
    theta2 = np.arcsin(v2 / v3)
    ti1 = 2 * z1 * np.cos(theta1) / v1
    ti2 = 2 * z2 * np.cos(theta2) / v2
    xc2 = 2 * (z2) * np.tan(theta2) + 2 * (z1) * np.tan(theta1)
    if np.isscalar(x):
        ref2 = 1.0 / v3 * x + ti2 + ti1
    else:
        act2 = x > xc2
        ref2 = np.ones_like(x) * np.nan
        ref2[act2] = 1.0 / v3 * x[act2] + ti2 + ti1
    return ref2


def reflection1(x, v1, z1):
    t0 = 2.0 * z1 / v1
    refl1 = np.sqrt(t0 ** 2 + x ** 2 / v1 ** 2)
    return refl1


def viewTXdiagram(x0, dx, v1, v2, v3, z1, z2, ax=None, legend=True):
    x = x0 + np.arange(20) * dx
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 8))
    ax.plot(x, direct(x, v1), "-r", linewidth=2.0)
    ax.plot(x, refraction1(x, v1, v2, z1), "-b", linewidth=2.0)
    ax.plot(x, refraction2(x, v1, v2, v3, z1, z2), "-g", linewidth=2.0)
    ax.plot(x, reflection1(x, v1, z1), "-k", linewidth=2.0)
    if legend:
        ax.legend(["direct", "refraction1", "refraction2", "reflection1"], loc="best")
    majorxtick = np.arange(0.0, 131.0, 20)
    minorxtick = np.arange(0.0, 131, 5.0)
    majorytick = np.arange(0.0, 0.26, 0.05)
    minorytick = np.arange(0.0, 0.26, 0.01)
    ax.set_xticks(majorxtick)
    ax.set_xticks(minorxtick, minor=True)
    ax.set_yticks(majorytick)
    ax.set_yticks(minorytick, minor=True)
    ax.set_xlim(0.0, 130.0)
    ax.set_ylim(0.0, 0.25)
    ax.invert_yaxis()
    ax.set_xlabel("Offset (m)", fontsize=16)
    ax.set_ylabel("Time (s)", fontsize=16)
    ax.grid(
        which="both", axis="both", linestyle="-", linewidth=0.5, color="k", alpha=0.3
    )
    return ax


def plotWiggleTX(
    x0, dx, v1, v2, v3, z1, z2, tI=None, v=None, Fit=False, ax=None, noise=False
):
    x = x0 + np.arange(20) * dx
    if noise is True:
        noiseFact = 2.5
    else:
        noiseFact = 0.0
    # dum = funDR1R2(x, v1, v2, v3, z1, z2)
    wavs = np.c_[
        direct(x, v1),
        refraction1(x, v1, v2, z1),
        refraction2(x, v1, v2, v3, z1, z2),
        reflection1(x, v1, z1),
    ]
    dt = 1e-3
    ntime = 241
    time = np.arange(ntime) * dt
    wave = getRicker(wavf, time, t0)
    data = np.zeros((ntime, 20))
    for i in range(20):
        inds = []
        for j in range(4):
            temp = np.argmin(abs(time - wavs[i, j]))
            if temp > 0:
                inds.append(temp)
        data[inds, i] = 1.0
    data_convolved = np.zeros_like(data)
    if Fit is True:
        np.random.seed(seed=1)

    for i in range(20):
        temp = np.convolve(wave, data[:, i])[:ntime]
        data_convolved[:, i] = temp + noiseFact * np.random.randn(ntime) * time

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 8))

    wiggleVarx(data_convolved.T, x=x, sampr=dt, lwidth=1.0, scale=0.2, ax=ax)
    if Fit is True:
        temp = tI + 1.0 / v * x
        ax.plot(x, temp, "r", lw=2)
    ax.set_ylim(0, 0.25)
    ax.invert_yaxis()
    ax.set_xlim(-1.0, 130)
    ax.set_xlabel("Offset (m)", fontsize=16)
    ax.set_ylabel("Time (s)", fontsize=16)
    return ax


def showWiggleTX(
    x0, dx, v1, v2, v3, z1, z2, tI=None, v=None, Fit=False, ax=None, noise=False
):
    ax = plotWiggleTX(x0, dx, v1, v2, v3, z1, z2, tI, v, Fit, ax, noise)
    plt.show()
    return True


def makeinteractTXdiagram():
    Q = interactive(
        lambda x0, dx: viewTXdiagram(x0, dx, v1=v1, v2=v2, v3=v3, z1=z1, z2=z2),
        x0=IntSlider(min=1, max=10, step=1, value=4),
        dx=IntSlider(min=1, max=10, step=1, value=4),
    )

    return Q


def makeinteractTXwigglediagram():
    Q = interactive(
        lambda x0, dx, tI, v, Fit=False: showWiggleTX(
            x0, dx, v1=v1, v2=v2, v3=v3, z1=z1, z2=z2, tI=tI, v=v, Fit=Fit, noise=True
        ),
        x0=IntSlider(min=1, max=10, step=1, value=4),
        dx=IntSlider(min=1, max=10, step=1, value=4),
        tI=FloatSlider(min=0.0, max=0.25, step=0.0025, value=0.05),
        v=FloatSlider(min=400, max=2000, step=50, value=1000.0),
    )
    return Q


def viewSeisRefracSurvey(x0, dx):
    x = x0 + np.arange(12) * dx
    z = np.r_[0.0, 5.0, 15.0]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(0.0, 0.0, "r*", ms=15)
    ax.plot(x, np.zeros_like(x), "bo")

    ax.legend(("shot", "geophone"), fontsize=14, bbox_to_anchor=(1.4, 1.05))
    xtemp = np.r_[-1, 100.0]
    for i in range(3):
        ax.plot(xtemp, np.ones_like(xtemp) * z[i], "k-", lw=2)
    ax.set_xlim(-1, 100.0)
    ax.set_ylim(20, -4.0)
    ax.text(50, 3, "Layer 1, v1 = ?? m/s", fontsize=14)
    ax.text(50, 11, "Layer 2, v2 = ?? m/s", fontsize=14)
    ax.text(50, 18, "Layer 3, v3 = ?? m/s", fontsize=14)

    ax.arrow(0, 1, dx=x0, dy=0.0)

    ax.text(0.0, 3.5, ("x0=%i m") % (x0), fontsize=14)
    if np.logical_and(dx > 3, dx < 8):
        ax.text(x[4], 3.5, ("dx=%i m") % (dx), fontsize=14)
        ax.arrow(x[4], 1, dx=dx, dy=0.0)
    else:
        ax.text(x[2], 3.5, ("dx=%i m") % (dx), fontsize=14)
        ax.arrow(x[2], 1, dx=dx, dy=0.0)
    for i in range(12):
        ax.text(x[i] - 0.5, -1, str(i + 1), fontsize=14)
    ax.set_xlabel("Offset (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_yticks([])
    plt.show()
    return ax


def makeinteractSeisRefracSurvey():
    Q = interactive(
        viewSeisRefracSurvey,
        x0=IntSlider(min=0, max=10, step=1, value=0),
        dx=IntSlider(min=1, max=10, step=1, value=8),
    )
    return Q


# ========================================================================================================================
# Visualziation for wiggle plot
# ========================================================================================================================


def wiggleVarx(
    traces,
    x,
    skipt=1,
    scale=1.0,
    lwidth=0.1,
    offsets=None,
    redvel=0.0,
    manthifts=None,
    tshift=0.0,
    sampr=1.0,
    clip=10.0,
    color="black",
    fill=True,
    line=True,
    ax=None,
):

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ns = traces.shape[1]
    ntr = traces.shape[0]
    t = np.arange(ns) * sampr

    def timereduce(offsets, redvel, shift):
        return [float(offset) / redvel + shift for offset in offsets]

    if offsets is not None:
        shifts = timereduce(offsets, redvel, tshift)
    elif manthifts is not None:
        shifts = manthifts
    else:
        shifts = np.zeros((ntr,))

    for i in range(0, ntr, skipt):
        trace = traces[i].copy()
        trace[0] = 0
        trace[-1] = 0

        if line:
            ax.plot(
                x[i] + clipsign(trace / scale, clip),
                t - shifts[i],
                color=color,
                linewidth=lwidth,
            )
        if fill:
            for j in range(ns):
                if trace[j] < 0:
                    trace[j] = 0
            ax.fill(
                x[i] + clipsign(trace / scale, clip),
                t - shifts[i],
                color=color,
                linewidth=0,
            )

    return ax


def clipsign(value, clip):
    clipthese = abs(value) > clip
    return value * ~clipthese + np.sign(value) * clip * clipthese


if __name__ == "__main__":
    plotWavelet(wavf)
