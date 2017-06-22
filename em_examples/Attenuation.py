from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0, epsilon_0

from ipywidgets import (
    interact, interactive, IntSlider, widget, FloatText, FloatSlider
)


def WaveVelSkind(frequency, epsr, sigma):
    omega = np.pi*np.complex128(frequency)
    k = np.sqrt(omega**2*mu_0*epsilon_0*epsr-1j*omega*mu_0*sigma)
    alpha = k.real
    beta = -k.imag
    return omega.real/alpha, 1./beta


def WaveVelandSkindWidget(epsr, sigma):
    frequency = np.logspace(1, 9, 61)
    vel, skind = WaveVelSkind(frequency, epsr, 10**sigma)
    figure, ax = plt.subplots(1, 2, figsize = (10, 4))
    ax[0].loglog(frequency, vel, 'b', lw=3)
    ax[1].loglog(frequency, skind, 'r', lw=3)
    ax[0].set_ylim(1e6, 1e9)
    ax[1].set_ylim(1e-1, 1e7)
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Velocity (m/s)')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Skin Depth (m)')
    ax[0].grid(True)
    ax[1].grid(True)

    plt.show()
    return


def WaveVelandSkindWidgetTBL(epsr, sigma):
    frequency = np.logspace(5, 10, 31)
    vel, skind = WaveVelSkind(frequency, epsr, 10**sigma)
    velsm, skindsm = WaveVelSkind(np.r_[25, 100, 1000]*1e6, epsr, 10**sigma)
    figure, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].loglog(frequency, vel, 'b-', lw=2)
    ax[1].loglog(frequency, skind, 'r-', lw=2)
    ax[0].loglog(np.r_[25., 25.]*1e6, np.r_[1e5, 1e8], 'k--', lw=1)
    ax[0].loglog(np.r_[100., 100.]*1e6, np.r_[1e5, 1e8], 'k--', lw=1)
    ax[0].loglog(np.r_[1000., 1000.]*1e6, np.r_[1e5, 1e8], 'k--', lw=1 )

    ax[1].loglog(np.r_[25., 25.]*1e6, np.r_[1e-1, 1e1], 'k--', lw=1)
    ax[1].loglog(np.r_[100., 100.]*1e6, np.r_[1e-1, 1e1], 'k--', lw=1)
    ax[1].loglog(np.r_[1000., 1000.]*1e6, np.r_[1e-1, 1e1], 'k--', lw=1 )

    ax[0].loglog(np.r_[25, 100, 1000]*1e6, velsm, 'ko', ms=5)
    # ax[0].text(25*1e6,   11*1e8, ("%3.1f m/s")%(velsm[0])   ,fontsize=10)
    # ax[0].text(100*1e6,  11*1e8, ("%3.1f m/s")%(velsm[1])  ,fontsize=10)
    # ax[0].text(1000*1e6, 11*1e8, ("%3.1f m/s")%(velsm[2]) ,fontsize=10)

    ax[1].loglog(np.r_[25, 100, 1000]*1e6, skindsm, 'ko', ms=5)
    ax[1].text(25*1e6, 11, ("%3.1f m")%(skindsm[0]), fontsize=10)
    ax[1].text(100*1e6, 11, ("%3.1f m")%(skindsm[1]), fontsize=10)
    ax[1].text(1000*1e6, 11, ("%3.1f m")%(skindsm[2]), fontsize=10)
    ax[0].set_xlim(10**5, 10**10)
    ax[1].set_xlim(10**5, 10**10)
    ax[0].set_ylim(1e5, 1e8)
    ax[1].set_ylim(1e-1, 1e1)
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Velocity (m/s)')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Skin Depth (m)')
    ax[0].grid(True)
    ax[1].grid(True)

    plt.show()
    return


def AttenuationWidgetTBL():
    i = interact(
        WaveVelandSkindWidgetTBL,
        epsr = FloatText(value=9.),
        sigma= FloatSlider(min=-4., max=1., step=0.5, value=-1.5)
    )
    return i
