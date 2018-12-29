from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from .FreqtoTime import transFilt

def ColeColePelton(f, sigmaInf, eta, tau, c, option):
    """

        .. math::
            \sigma(\omega) = \sigma_{\infty}\Big(1-\frac{\eta}{1+(1-\eta)(\imath\omega\tau)^c} \Big)

    """
    w = 2*np.pi*f
    sigma = sigmaInf*(1 - eta/(1 + (1-eta)*(1j*w*tau)**c))
    if option=="sigma":
        return sigma
    elif option=="resis":
        return 1./(sigma)
    else:
        raise Exception("Put only sigma or resis")

def vizColeCole(sigres="sigma", eta=0.1, tau=0.1, c=0.5, t1=800, t2=1400):
    frequency = np.logspace(-3, 6, 81)
    val = ColeColePelton(frequency, 1., eta, tau, c, option=sigres)
    datFcn = lambda f: ColeColePelton(f, 1., eta, tau, c, option=sigres)
    dt = 1e-3
    time = np.arange(10000)*dt + dt
    valT = transFilt(datFcn, time)
    fig= plt.figure(figsize = (16, 5))
    gs = matplotlib.gridspec.GridSpec(7, 7)
    axColeR = fig.add_subplot(gs[:, :3])     #Left
    axColeI = axColeR.twinx()
    axColeRT = fig.add_subplot( gs[:,4:])  #Right-Top
    valR = val.real
    valI = abs(val.imag)
    axColeR.semilogx(frequency, valR, color='k', linewidth=3)
    axColeI.semilogx(frequency, valI, color='r', linewidth=3)
    axColeR.set_xlabel("Frequency (Hz)")
    axColeRT.semilogx(time*1e3, valT, 'k', linewidth=3)
    axColeRT.set_xlabel("Time (msec)")
    axColeR.grid(True)
    tind = np.logical_and(time>=t1*1e-3, time<=t2*1e-3)
    if sigres == "sigma":
        axColeRT.plot(np.r_[1., 1e4], np.r_[0,0.], 'k:')
        axColeR.set_ylim(0., 1.)
        axColeI.set_ylim(0., 0.2)
        axColeR.set_ylabel("Real conductivity (S/m)")
        axColeI.set_ylabel("Imaginary conductivity (S/m)", color="r")
        axColeRT.set_ylabel("Step-off conductivity (S/m)")
        axColeRT.set_ylim(-0.5, 0.1)
    elif sigres == "resis":
        M = valT[tind].sum()*dt*1e3 / (1./(1.-eta))
        axColeRT.text(3e2, 0.8, ("M= %5.3f ms")%(M))
        axColeR.set_ylim(0., 2.)
        axColeI.set_ylim(0., 0.5)
        axColeR.set_ylabel("Real resistivity (ohm-m)")
        axColeI.set_ylabel("Imaginary resistivity (ohm-m)", color="r")
        axColeRT.set_ylabel("Step-off resistivity (ohm-m)")
        axColeRT.set_ylim(-0.1*2,0.5*2)
        axColeRT.plot(np.r_[t1, t1], np.r_[0,0.5*2], 'k:')
        axColeRT.plot(np.r_[t2, t2], np.r_[0,0.5*2], 'k:')
        axColeRT.plot(np.r_[1., 1e4], np.r_[0,0.], 'k:')
        axColeRT.fill_between(time[tind]*1e3, valT[tind], 0., color='k', alpha=0.5)
    for tl in axColeI.get_yticklabels():
        tl.set_color('r')
    plt.show()
#     return fig
