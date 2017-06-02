from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
from SimPEG import EM
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.rcParams['font.size'] = 12
import warnings
warnings.filterwarnings("ignore")

from ipywidgets import *

from .Base import widgetify
from .View import DataView

def linefun(x1, x2, y1, y2, nx,tol=1e-3):
    dx = x2-x1
    dy = y2-y1

    if np.abs(dx)<tol:
        y = np.linspace(y1, y2,nx)
        x = np.ones_like(y)*x1
    elif np.abs(dy)<tol:
        x = np.linspace(x1, x2, nx)
        y = np.ones_like(x)*y1
    else:
        x = np.linspace(x1, x2, nx)
        slope = (y2-y1)/(x2-x1)
        y=slope*(x-x1)+y1
    return x, y


class DipoleWidget1D(object):
    """DipoleWidget"""

    x = None
    y = None
    z = None
    func = None
    sig = None
    freq = None
    obsLoc = None

    # Fixed spatial range in 3D
    xmin, xmax = -50., 50.
    ymin, ymax = -50., 50.
    zmin, zmax = -50., 50.
    sigmin, sigmax = -4.,4
    fmin, fmax = -4.,8.
    ns = 81
    nf = 121
    sigvec = np.linspace(sigmin,sigmax,ns)
    fvec = np.linspace(fmin,fmax,nf)

    def __init__(self):
        self.dataview = DataView()

    def SetDataview_1D(self, srcLoc,obsLoc, sigvec, fvec, orientation,normal, func):
        self.dataview.eval_loc(srcLoc,obsLoc, sigvec, fvec, orientation,normal, func)

    def InteractiveDipole1D(self):
        sigvec = self.sigvec
        fvec = self.fvec
        def Dipole1Dviz(orientation,component,view,normal,sigsl,freqsl,absloc,coordloc,mode):

            x = np.linspace(-50., 50., 100)
            y = np.arange(-50., 50., 100)

            srcLoc = np.r_[0., 0., 0.] # source location
            sig, f = 10.**sigsl, np.r_[10.**freqsl] # conductivity (S/m), frequency (Hz)

            if normal.upper() == "Z":
                obsLoc=np.c_[absloc,coordloc,np.r_[0.]]
                self.dataview.set_xyz(x,y, np.r_[0.], normal=normal) # set plane and locations ...

            elif normal.upper() == "Y":
                obsLoc=np.c_[absloc,np.r_[0.],coordloc]
                self.dataview.set_xyz(x,np.r_[0.],y, normal=normal) # set plane and locations ...

            elif normal.upper() == "X":
                obsLoc=np.c_[np.r_[0.],absloc,coordloc]
                self.dataview.set_xyz(np.r_[0.],x,y,normal=normal) # set plane and locations ...

            self.dataview.eval_loc(srcLoc,obsLoc, sigvec, fvec, orientation, normal, EM.Analytics.E_from_ElectricDipoleWholeSpace) # evaluate

            fig = plt.figure(figsize=(6.5*3, 5))
            ax0 = plt.subplot(121)
            ax2 = plt.subplot(122)

            ax1 = ax0.twinx()
            ax3 = ax2.twinx()

            if mode =="RI":
                ax0 = self.dataview.plot1D_FD(component="real",view=view,abscisse="Conductivity",slic=freqsl, logamp=True, ax=ax0, color = 'blue')
                ax1 = self.dataview.plot1D_FD(component="imag",view=view,abscisse="Conductivity",slic=freqsl, logamp=True, ax=ax1,legend=False, color = 'red')

                ax2 = self.dataview.plot1D_FD(component="real",view=view,abscisse="Frequency",slic=sigsl, logamp=True, ax=ax2, color = 'blue')
                ax3 = self.dataview.plot1D_FD(component="imag",view=view,abscisse="Frequency",slic=sigsl, logamp=True, ax=ax3,legend=False, color = 'red')

            elif mode =="AP":
                ax0 = self.dataview.plot1D_FD(component="Amplitude",view=view,abscisse="Conductivity",slic=freqsl, logamp=True, ax=ax0, color = 'blue')
                ax1 = self.dataview.plot1D_FD(component="Phase",view=view,abscisse="Conductivity",slic=freqsl, logamp=True, ax=ax1,legend=False, color = 'red')

                ax2 = self.dataview.plot1D_FD(component="Amplitude",view=view,abscisse="Frequency",slic=sigsl, logamp=True, ax=ax3, color = 'blue')
                ax3 = self.dataview.plot1D_FD(component="Phase",view=view,abscisse="Frequency",slic=sigsl, logamp=True, ax=ax3,legend=False, color = 'red')

            elif mode =="Phasor":
                ax0 = self.dataview.plot1D_FD(component="PHASOR",view=view,abscisse="Conductivity",slic=freqsl, logamp=True, ax=ax0, color = 'black')
                ax2 = self.dataview.plot1D_FD(component="PHASOR",view=view,abscisse="Frequency",slic=sigsl, logamp=True, ax=ax2, color = 'black')

            plt.tight_layout()

        out = widgetify(Dipole1Dviz
            ,mode = ToggleButtons(options=['RI','AP','Phasor'],value='RI') \
            ,view = ToggleButtons(options=['x','y','z'],value='x') \
            ,sigsl = FloatSlider(min=-4, max =4, step=0.1,value=0) \
            ,freqsl = FloatSlider(min=-4, max =8, step=0.1,value=-4) \
            ,absloc = FloatSlider(min=-50, max =50, step=1,value=25) \
            ,coordloc = FloatSlider(min=-50, max =50, step=1,value=0) \
            ,orientation= ToggleButtons(options=['x','y','z'],value='x') \
            ,component = ToggleButtons(options=['real','imag','amplitude','phase'],value='real') \
             ,normal = ToggleButtons(options=['x','y','z'],value='z') \
            )
        return out

