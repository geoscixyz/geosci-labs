from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
from SimPEG import EM
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import matplotlib.gridspec as gridspec
from ipywidgets import *
from scipy.constants import mu_0, epsilon_0

from .DipoleWidgetFD import DipoleWidgetFD, linefun, DisPosNegvalues
from .VolumeWidget import polyplane
from .FDEMPlanewave import *
from .View import DataView

matplotlib.rcParams['font.size'] = 12


def PlaneEHfield(z, t=0., f=1., sig=1., mu=mu_0,  epsilon=epsilon_0, E0=1.):
    """
        Plane wave propagating downward (negative z (depth))
    """
    k = EM.Utils.k(f, sig, mu=mu, eps=epsilon)
    omega = EM.Utils.omega(f)
    Ex = E0*np.exp(1j*(k*z+omega*t))
    Z = omega*mu/k
    Hy = -E0/Z*np.exp(1j*(k*z+omega*t))
    return Ex, Hy

class PolarEllipse(object):


    def Planewave3D(self, itime):
        fig = plt.figure(figsize = (12*1.2, 5*1.2))
        ax = fig.add_subplot(111, projection='3d')
        t = np.linspace(0, 0.02, 1000.)
        X1, X2 = t.min(), t.max()
        Y1, Y2 = -1.2, 1.2
        Z1, Z2 = -1.2, 1.2

        Ex, Hy = PlaneEHfield(-10., t=t, f=100., sig=1e-2)
        ax.plot(t, Ex.real / Ex.real.max(), np.zeros_like(t), 'b', lw=1)
        ax.plot(np.r_[t[itime]], (Ex[itime]).real / Ex.real.max(), 0., 'bo', ms=5)
        ax.plot(t[itime]*np.ones(2), np.r_[0., (Ex[itime]).real / Ex.real.max()], np.ones(2)*(Hy[itime]).real / Hy.real.max(), 'r:')
        ax.plot(t, np.zeros_like(t), Hy.real / Hy.real.max(), 'r', lw=1)
        ax.plot(np.r_[t[itime]], 0., (Hy[itime]).real / Hy.real.max(), 'ro', ms=5)
        ax.plot(t[itime]*np.ones(2), np.ones(2)*(Ex[itime]).real / Ex.real.max() , np.r_[0., (Hy[itime]).real / Hy.real.max()], 'b:')

        ax.plot(np.ones_like(t)*t[itime], Ex.real / Ex.real.max(), Hy.real / Hy.real.max(), 'k-', lw=0.5)
        ax.plot(np.ones(1)*t[itime], Ex[itime].real / Ex.real.max(), Hy[itime].real / Hy.real.max(), 'ko', ms=5)
        ax.plot(t, np.zeros_like(t), np.zeros_like(t), 'k--')
        ax.plot(np.ones(2)*t[itime], np.r_[Y1, Y2], np.zeros(2), 'k--')
        ax.plot(np.ones(2)*t[itime], np.zeros(2), np.r_[Z1, Z2], 'k--')
        x = np.r_[X1,X2,X2,X1,X1]
        y = np.zeros(5)
        z = np.r_[Z1,Z1,Z2,Z2,Z1]
        verts = zip(x,y,z)
        polya = polyplane(verts, color="red", alpha=0.1)
        ax.plot(x, y, z, "r-", lw=1, alpha=0.2)
        x = np.r_[X1,X2,X2,X1,X1]
        y = np.r_[Y1,Y1,Y2,Y2,Y1]
        z = np.zeros(5)
        verts = zip(x, y,z)
        polyb = polyplane(verts, color="blue", alpha=0.1)
        ax.plot(x, y, z, "b-", lw=1, alpha=0.2)
        x = np.ones(5)*t[itime]
        y = np.r_[Y1,Y2,Y2,Y1,Y1]
        z = np.r_[Z1,Z1,Z2,Z2,Z1]
        verts = zip(x, y,z)
        polyc = polyplane(verts, color="grey", alpha=0.1)
        ax.plot(x, y, z, "k-", lw=1, alpha=0.2)

        ax.add_collection3d(polya)
        ax.add_collection3d(polyb)
        ax.add_collection3d(polyc)

        ax.set_zlim(1.2, -1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlim(t.min(), t.max())
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Ex (V/m)")
        ax.set_zlabel("Hy (A/m)")
        elev = 45
        azim = 290
        ax.view_init(elev, azim)
        plt.show()
        pass

    def Interactive(self):
        out = interactive(self.Planewave3D, itime=IntSlider(min=0, max=999, step=10))
        return out


class PlanewaveWidget(DipoleWidgetFD):
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

    def SetDataview(self, srcLoc, sig, f, orientation, normal, functype, na=100, nb=100, loc=0.,t=0.):

        self.srcLoc = srcLoc
        self.sig = sig
        self.f = f
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

        self.dataview.eval_2D(srcLoc, sig, f, orientation, self.func, t=t) # evaluate

    def Planewave2Dviz(self, x1, y1, x2, y2, npts2D, npts, sig, f, srcLoc=0., orientation="x", component="real", view="x", normal="Z", functype="E_from_ED", loc=0., scale="log", dx=50., t=0.):
        nx, ny = npts2D, npts2D
        x, y = linefun(x1, x2, y1, y2, npts)
        if scale == "log":
            logamp = True
        elif scale == "linear":
            logamp = False
        else:
            raise NotImplementedError()
        self.SetDataview(srcLoc, sig, f, orientation, normal, functype, na=nx, nb=ny, loc=loc, t=t)
        plot1D = True
        if normal =="X" or normal=="x":
            xyz_line = np.c_[np.ones_like(x)*self.x, x, y]
            self.dataview.xyz_line =  xyz_line
        if normal =="Y" or normal=="y":
            xyz_line = np.c_[x, np.ones_like(x)*self.y, y]
            self.dataview.xyz_line =  xyz_line
        if normal == "Z" or normal == "z":
            xyz_line = np.c_[x, y, np.ones_like(x)*self.z]
            self.dataview.xyz_line =  xyz_line

        fig = plt.figure(figsize=(18*1.5,3.4*1.5))
        gs1 = gridspec.GridSpec(2, 7)
        gs1.update(left=0.05, right=0.48, wspace=0.05)
        ax1 = plt.subplot(gs1[:2, :3])
        ax1.axis("equal")

        ax1, dat1 = self.dataview.plot2D_FD(ax=ax1, component=component,view=view, colorbar=False, logamp=logamp)
        vmin, vmax = dat1.cvalues.min(), dat1.cvalues.max()
        if scale == "log":
            cb = plt.colorbar(dat1, ax=ax1, ticks=np.linspace(vmin, vmax, 5), format="$10^{%.1f}$")
        elif scale == "linear":
            cb = plt.colorbar(dat1, ax=ax1, ticks=np.linspace(vmin, vmax, 5), format="%.1e")

        tempstr = functype.split("_")

        tname_end = ")"

        if component == "real":
            tname = "Re("
        elif component == "imag":
            tname = "Im("
        elif component == "amplitude":
            tname = "|"
            tname_end = "|"
        elif component == "phase":
            tname = "Phase("

        title = tname + tempstr[0]+view+tname_end

        if tempstr[0] == "E":
            unit = " (V/m)"
            fieldname = "Electric field"
        elif tempstr[0] == "H":
            unit = " (A/m)"
            fieldname = "Magnetic field"
        elif tempstr[0] == "J":
            unit =  " (A/m$^2$) "
            fieldname = "Current density"
        else:
            raise NotImplementedError()
        if component == "phase":
            unit = " (rad)"
        label = fieldname + unit
        cb.set_label(label)
        ax1.set_title(title)


        if plot1D:
            ax1.plot(x, y, 'r.', ms=4)
            ax2 = plt.subplot(gs1[:, 4:6])
            val_line_x, val_line_y, val_line_z = self.dataview.eval(xyz_line, srcLoc, np.r_[sig], np.r_[f], orientation, self.func, t=t)

            if view =="X" or view =="x":
                val_line = val_line_x
            elif view =="Y" or view =="y":
                val_line = val_line_y
            elif view =="Z" or view =="z":
                val_line = val_line_z
            elif view =="vec" or "amp":
                vecamp = lambda a, b, c: np.sqrt((a)**2+(b)**2+(c)**2)
                if component == "real":
                    val_line = vecamp(val_line_x.real, val_line_y.real, val_line_z.real)
                elif component == "imag":
                    val_line = vecamp(val_line_x.imag, val_line_y.imag, val_line_z.imag)
                elif component == "amplitude":
                    val_line = vecamp(abs(val_line_x), abs(val_line_y), abs(val_line_z))
                elif component == "phase":
                    val_line = vecamp(np.angle(val_line_x), np.angle(val_line_y), np.angle(val_line_z))

            distance = xyz_line[:, 2]

            if component == "real":
                val_line = val_line.real
            elif component == "imag":
                val_line = val_line.imag
            elif component == "amplitude":
                val_line = abs(val_line)
            elif component == "phase":
                val_line = np.angle(val_line)

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
                if view == "vec" or view== "amp":
                    label = "|"+tempstr[0]+"|-field (V/m) "
                else:
                    label = tname+tempstr[0]+view+")-field (V/m) "
            elif tempstr[0] == "H":
                if view == "vec" or view== "amp":
                    label = "|"+tempstr[0]+"|-field field (A/m) "
                else:
                    label = tname+tempstr[0]+view+")-field (A/m) "
            elif tempstr[0] == "J":
                if view == "vec" or view== "amp":
                    label = "|"+tempstr[0]+"|-field field (A/m$^2$) "
                else:
                    label = tname+tempstr[0]+view+")-field (A/m$^2$) "
            else:
                raise NotImplementedError()

            if component == "phase":
                label = tname+tempstr[0]+view+")-field (rad) "

            ax2.set_title("EM data")
            ax2.set_xlabel(label)

            # ax2.text(distance.min(), val_line.max(), 'A', fontsize = 16)
            # ax2.text(distance.max()*0.97, val_line.max(), 'B', fontsize = 16)
            # ax2.legend((component, ), bbox_to_anchor=(0.5, -0.3))
            ax2.grid(True)
        # plt.tight_layout()
        plt.show()
        pass


    def InteractivePlaneWave(self, nRx=100, npts2D=50, scale="log", offset_plane=50., X1=-500, X2=500, Y1=-500, Y2=500, Z1=-1000, Z2=0):

        # x1, x2, y1, y2 = offset_rx, offset_rx, Z1, Z2
        self.xmin, self.xmax = X1, X2
        self.ymin, self.ymax = Y1, Y2
        self.zmin, self.zmax = Z1, Z2

        def foo(Field, ComplexNumber, Frequency, Sigma, Scale, Time):
            f = np.r_[Frequency]
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

            if ComplexNumber == "Re":
                ComplexNumber = "real"
            elif ComplexNumber == "Im":
                ComplexNumber = "imag"
            elif ComplexNumber == "Amp":
                ComplexNumber = "amplitude"
            elif ComplexNumber == "Phase":
                ComplexNumber = "phase"

            return self.Planewave2Dviz(x1, y1, x2, y2, npts2D, nRx, sig, f, srcLoc=0., orientation="X", component=ComplexNumber, view=Component, normal=normal, functype=Field, scale=Scale, t=Time)

        out = widgets.interactive (foo
                        ,Field=widgets.ToggleButtons(options=["Ex", "Hy"])
                        ,ComplexNumber=widgets.ToggleButtons(options=['Re','Im','Amp', 'Phase'])
                        ,Frequency=widgets.FloatText(value=10., continuous_update=False)
                        ,Sigma=widgets.FloatText(value=1, continuous_update=False)
                        ,Scale=widgets.ToggleButtons(options=['log','linear'], value="linear")
                        ,Time=widgets.FloatSlider(min=0, max=0.2, step=0.01, value=0.))
        return out


def InteractivePlaneProfile():
    srcLoc = 0.
    orientation = "X"
    nRx = 100

    def foo(Field, Sigma, Scale, Fixed, Frequency, Time):

        fig = plt.figure(figsize=(8,4))
        ax1 = plt.subplot(111)
        ax2 = ax1.twinx()

        r = np.linspace(-1000., 0., nRx)
        val_ex, val_hy = PlaneEHfield(r, t=Time, f=Frequency, sig=Sigma)

        if Field == "Ex":
            valr = val_ex.real.flatten()
            vali = val_ex.imag.flatten()
            labelr = "Re (Ex)-field (V/m)"
            labeli = "Im (Ex)-field (V/m)"
        elif Field == "Hy":
            valr = val_hy.real.flatten()
            vali = val_hy.imag.flatten()
            labelr = "Re (Hy)-field (A/m)"
            labeli = "Im (Hy)-field (A/m)"

        elif Field == "Impedance":
            imp = - val_ex / val_hy
            valr = imp.real.flatten()
            vali = imp.imag.flatten()
            labelr = "Re (Z) (Ohm)"
            labeli = "Im (Z) (Ohm)"

        elif Field == "rhophi":
            imp = - val_ex / val_hy
            valr = abs(imp)**2 / (2*np.pi*Frequency*mu_0)
            vali = np.angle(imp, deg=True)
            labelr = "Apparent resistivity (Ohm-m)"
            labeli = "Phase of Impedance (degree)"

        if Scale == "log":
            valr_p, valr_n = DisPosNegvalues(valr)
            vali_p, vali_n = DisPosNegvalues(vali)
            if Field == "rhophi":
                ax1.plot(r, valr, 'k.')
            else:
                ax1.plot(r, valr_p, 'k-', lw=2)
            ax1.plot(r, valr_n, 'k--', lw=2)
            ax2.plot(r, vali_p, 'r-', lw=2)
            ax2.plot(r, vali_n, 'r--', lw=2)
            ax1.set_yscale(Scale)
            ax2.set_yscale(Scale)
            if Fixed:
                vmin1, vmax1 = ax1.get_ylim()
                vmin2, vmax2 = ax2.get_ylim()
                vmin = min(vmin1, vmin2)
                vmax = max(vmax1, vmax2)
                ax1.set_ylim(vmin, vmax)
                ax2.set_ylim(vmin, vmax)


        elif Scale == "linear":
            if Field == "rhophi":
                ax1.plot(r, valr, 'k.')
            else:
                ax1.plot(r, valr, 'k-', lw=2)
            ax2.plot(r, vali, 'r-', lw=2)
            ax1.set_yscale(Scale)
            ax2.set_yscale(Scale)
            y = ax1.yaxis.get_majorticklocs()
            yticksa = np.linspace(y.min(), y.max(), 3)
            ax1.yaxis.set_ticks(yticksa)

            if Fixed and Field is not "Impedance":
                vmax = np.r_[abs(valr), abs(vali)].max()
                vmin = -vmax
                ax1.set_ylim(vmin, vmax)
                ax2.set_ylim(vmin, vmax)
                # y = ax2.yaxis.get_majorticklocs()
                yticks = np.linspace(vmin, vmax, 3)
                ax1.yaxis.set_ticks(yticks)
                ax2.yaxis.set_ticks(yticks)

            ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
            ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))

        ax1.set_xlim(0, -1000)
        ax2.set_ylabel(labeli, color='r')
        ax1.set_ylabel(labelr, color='k')
        ax1.set_xlabel("Z (m)")


        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        ax1.grid(True)
        plt.show()

    Q2 = widgets.interactive (foo
                    ,Field=widgets.ToggleButtons(options=['Ex','Hy','Impedance','rhophi'], value='Ex')
                    ,Sigma=widgets.FloatText(value=1, continuous_update=False, description='$\sigma$ (S/m)') \
                    ,Scale=widgets.ToggleButtons(options=['log','linear'], value="linear") \
                    ,Fixed=widgets.widget_bool.Checkbox(value=False)
                    ,Frequency=widgets.FloatText(value=10., continuous_update=False, description='$f$ (Hz)') \
                    ,Time=widgets.FloatSlider(min=0, max=0.2, step=0.005, continuous_update=False, description='t (s)'))
    return Q2
