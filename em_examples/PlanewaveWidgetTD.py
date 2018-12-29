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
from ipywidgets import *
from scipy.constants import mu_0, epsilon_0

from .DipoleWidgetTD import DipoleWidgetTD, linefun, DisPosNegvalues
from .View import DataView
from .VolumeWidget import polyplane
from .TDEMPlanewave import *


def PlaneEHfield(z, t=0., sig=1., mu=mu_0,  epsilon=epsilon_0, E0=1.):
    """
        Plane wave propagating downward (negative z (depth))
    """
    bunja = -E0*(mu*sig)**0.5 * z * np.exp(-(mu*sig*z**2) / (4*t))
    bunmo = 2 * np.pi**0.5 * t**1.5
    Ex = bunja / bunmo
    Hy = E0 * np.sqrt(sig / (np.pi*mu*t))*np.exp(-(mu*sig*z**2) / (4*t))
    return Ex, Hy


class PlanewaveWidget(DipoleWidgetTD):
    """PlanewaveWidget"""

    x = None
    y = None
    z = None
    func = None

    # Fixed spatial range in 3D
    xmin, xmax = -500., 500.
    ymin, ymax = -500., 500.
    zmin, zmax = -1000.,0.

    def __init__(self):
        self.dataview = DataView()

    def SetDataview(self, srcLoc, sig, t, orientation, normal, functype, na=100, nb=100, loc=0.):

        self.srcLoc = srcLoc
        self.sig = sig
        self.t = t
        self.normal = normal
        self.SetGrid(normal, loc, na, nb)
        self.functype = functype
        self.dataview.set_xyz(self.x, self.y, self.z, normal=normal) # set plane and locations ...

        if self.functype == "E_from_SheetCurrent":
            self.func = E_field_from_SheetCurruent
        elif self.functype == "H_from_SheetCurrent":
            self.func = H_field_from_SheetCurruent
        elif self.functype == "J_from_SheetCurrent":
            self.func = J_field_from_SheetCurruent
        else:
            raise NotImplementedError()

        self.dataview.eval_2D_TD(srcLoc, sig, t, orientation, self.func) # evaluate

    def Planewave2Dviz(self, x1, y1, x2, y2, npts2D, npts, sig, t, srcLoc=0., orientation="x", view="x", normal="Z", functype="E_from_ED", loc=0., scale="log", dx=50.):
        nx, ny = npts2D, npts2D
        x, y = linefun(x1, x2, y1, y2, npts)
        if scale == "log":
            logamp = True
        elif scale == "linear":
            logamp = False
        else:
            raise NotImplementedError()
        self.SetDataview(srcLoc, sig, t, orientation, normal, functype, na=nx, nb=ny, loc=loc)
        plot1D = True
        if normal =="X" or normal=="x":
            xyz_line = np.c_[np.ones_like(x)*self.x, x, y]
            self.dataview.xyz_line =  xyz_line
        if normal =="Y" or normal=="y":
            xyz_line = np.c_[x, np.ones_like(x)*self.y, y]
            self.dataview.xyz_line =  xyz_line
        if normal =="Z" or normal=="z":
            xyz_line = np.c_[x, y, np.ones_like(x)*self.z]
            self.dataview.xyz_line =  xyz_line

        fig = plt.figure(figsize=(18*1.5,3.4*1.5))
        gs1 = gridspec.GridSpec(2, 7)
        gs1.update(left=0.05, right=0.48, wspace=0.05)
        ax1 = plt.subplot(gs1[:2, :3])
        ax1.axis("equal")

        ax1, dat1 = self.dataview.plot2D_TD(ax=ax1, view=view, colorbar=False, logamp=logamp)
        vmin, vmax = dat1.cvalues.min(), dat1.cvalues.max()
        if scale == "log":
            cb = plt.colorbar(dat1, ax=ax1, ticks=np.linspace(vmin, vmax, 5), format="$10^{%.1f}$")
        elif scale == "linear":
            cb = plt.colorbar(dat1, ax=ax1, ticks=np.linspace(vmin, vmax, 5), format="%.1e")

        tempstr = functype.split("_")

        title = tempstr[0]+view

        if tempstr[0] == "E":
            unit = " (V/m)"
            fieldname = "Electric field"
        elif tempstr[0] == "H":
            unit = " (A/m)"
            fieldname = "Magnetic field"
        else:
            raise NotImplementedError()

        label = fieldname + unit
        cb.set_label(label)
        ax1.set_title(title)


        if plot1D:
            ax1.plot(x, y, 'r.', ms=4)
            ax2 = plt.subplot(gs1[:, 4:6])
            val_line_x, val_line_y, val_line_z = self.dataview.eval_TD(xyz_line, srcLoc, np.r_[sig], np.r_[t], orientation, self.func)

            if view =="X" or view =="x":
                val_line = val_line_x
            elif view =="Y" or view =="y":
                val_line = val_line_y
            elif view =="Z" or view =="z":
                val_line = val_line_z

            distance = xyz_line[:, 2]

            if scale == "log":
                temp = val_line.copy()*np.nan
                temp[val_line>0.] = val_line[val_line>0.]
                ax2.plot(temp, distance, 'k.-')
                temp = val_line.copy()*np.nan
                temp[val_line<0.] = -val_line[val_line<0.]
                ax2.plot(temp, distance, 'k.--')
                ax2.set_xlim(abs(val_line).min(), abs(val_line).max())
                ax2.set_xscale(scale)

            elif scale == "linear":
                ax2.plot(val_line, distance, 'k.-')
                ax2.set_xlim(val_line.min(), val_line.max())
                ax2.set_xscale(scale)
                xticks = np.linspace(-abs(val_line).max(), abs(val_line).max(), 3)
                plt.plot(np.r_[0., 0.], np.r_[distance.min(), distance.max()], 'k-', lw=2)
                ax2.xaxis.set_ticks(xticks)
                ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))
                ax2.set_xlim(-abs(val_line).max(), abs(val_line).max())

            ax2.set_ylim(distance.min(), distance.max())
            ax2.set_ylabel("Profile (m)")

            if tempstr[0] == "E":
                label = "("+tempstr[0]+view+")-field (V/m) "
            elif tempstr[0] == "H":
                label = "("+tempstr[0]+view+")-field (A/m) "
            elif tempstr[0] == "J":
                label = "("+tempstr[0]+view+")-field (A/m$^2$) "
            else:
                raise NotImplementedError()

            ax2.set_title("EM data")
            ax2.set_xlabel(label)
            ax2.grid(True)
        plt.show()
        pass


    def InteractivePlaneWave(self, nRx=100, npts2D=50, scale="log", offset_plane=50., X1=-500, X2=500, Y1=-500, Y2=500, Z1=-1000, Z2=0):

        # x1, x2, y1, y2 = offset_rx, offset_rx, Z1, Z2
        self.xmin, self.xmax = X1, X2
        self.ymin, self.ymax = Y1, Y2
        self.zmin, self.zmax = Z1, Z2

        def foo(Field, Time, Sigma, Scale):
            t = np.r_[Time]
            sig = np.r_[Sigma]

            if Field == "Ex":
                normal = "Y"
                self.offset_rx = 0.
                Field = "E_from_SheetCurrent"
                Component = "x"

            elif Field == "Hy":
                normal = "X"
                self.offset_rx = 0.
                Field = "H_from_SheetCurrent"
                Component = "y"

            x1, x2, y1, y2 = self.offset_rx, self.offset_rx, Z1, Z2

            return self.Planewave2Dviz(x1, y1, x2, y2, npts2D, nRx, sig, t, srcLoc=0., orientation="X", view=Component, normal=normal, functype=Field, scale=Scale)

        out = widgets.interactive (foo
                        ,Field=widgets.ToggleButtons(options=["Ex", "Hy"])
                        ,Time=widgets.FloatSlider(min=0.01, max=1., step=0.01, value=0., description='$t$ (s)')
                        ,Sigma=widgets.FloatText(value=1, continuous_update=False)
                        ,Scale=widgets.ToggleButtons(options=['log','linear'], value="linear")
                        )
        return out


def InteractivePlaneProfile():
    srcLoc = 0.
    orientation = "X"
    nRx = 100

    def foo(Field, Sigma, Scale, Time):

        fig = plt.figure(figsize=(8,4))
        ax1 = plt.subplot(111)

        r = np.linspace(-1000., 0., nRx)
        val_ex, val_hy = PlaneEHfield(r, t=Time, sig=Sigma)

        if Field == "Ex":
            val = val_ex.flatten()
            label = "Ex-field (V/m)"

        elif Field == "Hy":
            val = val_hy.flatten()
            label = "Hy-field (A/m)"

        if Scale == "log":
            val_p, val_n = DisPosNegvalues(val)
            ax1.plot(r, val_p, 'k-', lw=2)
            ax1.plot(r, val_n, 'k--', lw=2)
            ax1.set_yscale(Scale)

        elif Scale == "linear":
            ax1.plot(r, val, 'k-', lw=2)
            ax1.set_yscale(Scale)
            y = ax1.yaxis.get_majorticklocs()
            yticksa = np.linspace(y.min(), y.max(), 3)
            ax1.yaxis.set_ticks(yticksa)
            ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))

        ax1.set_xlim(0, -1000)
        ax1.set_ylabel(label, color='k')
        ax1.set_xlabel("Z (m)")

        ax1.grid(True)
        plt.show()

    Q2 = widgets.interactive (foo
                    ,Field=widgets.ToggleButtons(options=['Ex','Hy'], value='Ex')
                    ,Sigma=widgets.FloatText(value=1, continuous_update=False, description='$\sigma$ (S/m)') \
                    ,Scale=widgets.ToggleButtons(options=['log','linear'], value="linear") \
                    ,Time=widgets.FloatSlider(min=0.01, max=1., step=0.01, value=0., description='$t$ (s)')
                    )
    return Q2
