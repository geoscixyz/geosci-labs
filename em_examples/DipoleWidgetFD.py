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

from .View import DataView
from .Base import widgetify
from .FDEMDipolarfields import *


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


class DipoleWidgetFD(object):
    """DipoleWidget"""

    x = None
    y = None
    z = None
    func = None

    # Fixed spatial range in 3D
    xmin, xmax = -50., 50.
    ymin, ymax = -50., 50.
    zmin, zmax = -50., 50.

    def __init__(self):
        self.dataview = DataView()

    def SetDataview(self, srcLoc, sig, f, orientation, normal, functype, na=100, nb=100, loc=0.):

        self.srcLoc = srcLoc
        self.sig = sig
        self.f = f
        self.normal = normal
        self.SetGrid(normal, loc, na, nb)
        self.functype = functype
        self.dataview.set_xyz(self.x, self.y, self.z, normal=normal) # set plane and locations ...

        if self.functype == "E_from_ED":
            self.func = E_from_ElectricDipoleWholeSpace
        elif self.functype == "E_from_ED_galvanic":
            self.func = E_galvanic_from_ElectricDipoleWholeSpace
        elif self.functype == "E_from_ED_inductive":
            self.func = E_inductive_from_ElectricDipoleWholeSpace
        elif self.functype == "H_from_ED":
            self.func = H_from_ElectricDipoleWholeSpace
        elif self.functype == "J_from_ED":
            self.func = J_from_ElectricDipoleWholeSpace
        elif self.functype == "E_from_MD":
            self.func = E_from_MagneticDipoleWholeSpace
        elif self.functype == "H_from_MD":
            self.func = H_from_MagneticDipoleWholeSpace
        elif self.functype == "J_from_MD":
            self.func = J_from_MagneticDipoleWholeSpace
        else:
            raise NotImplementedError()

        self.dataview.eval_2D(srcLoc, sig, f, orientation, self.func) # evaluate

    def SetGrid(self, normal, loc, na, nb):
        # Assume we are seeing xy plane
        if normal =="X" or normal=="x":
            self.x = np.r_[loc]
            self.y = np.linspace(self.ymin, self.ymax, na)
            self.z = np.linspace(self.zmin, self.zmax, nb)
        if normal =="Y" or normal=="y":
            self.x = np.linspace(self.xmin, self.xmax, na)
            self.y = np.r_[loc]
            self.z = np.linspace(self.zmin, self.zmax, nb)
        if normal =="Z" or normal=="z":
            self.x = np.linspace(self.xmin, self.xmax, na)
            self.y = np.linspace(self.ymin, self.ymax, nb)
            self.z = np.r_[loc]

    def Dipole2Dviz(self, x1, y1, x2, y2, npts2D, npts, sig, f, srcLoc=np.r_[0., 0., 0.], orientation="x", component="real", view="x", normal="Z", functype="E_from_ED", loc=0., scale="log", dx=50., plot1D=False, plotTxProfile=False):
        nx, ny = npts2D, npts2D
        x, y = linefun(x1, x2, y1, y2, npts)
        if scale == "log":
            logamp = True
        elif scale == "linear":
            logamp = False
        else:
            raise NotImplementedError()

        self.SetDataview(srcLoc, sig, f, orientation, normal, functype, na=nx, nb=ny, loc=loc)
        # plot1D = False
        # plotTxProfile = False
        if normal =="X" or normal=="x":
            if abs(loc - 50) < 1e-5:
                plot1D = True
            xyz_line = np.c_[np.ones_like(x)*self.x, x, y]
            self.dataview.xyz_line =  xyz_line
        if normal =="Y" or normal=="y":
            if abs(loc - 0.) < 1e-5:
                plot1D = True
                plotTxProfile = True
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

        ax1, dat1 = self.dataview.plot2D_FD(ax=ax1, component=component,view=view, colorbar=False, logamp=logamp)
        vmin, vmax = dat1.cvalues.min(), dat1.cvalues.max()
        if scale == "log":
            cb = plt.colorbar(dat1, ax=ax1, ticks=np.linspace(vmin, vmax, 5), format="$10^{%.1f}$")
        elif scale == "linear":
            cb = plt.colorbar(dat1, ax=ax1, ticks=np.linspace(vmin, vmax, 5), format="%.1e")

        ax1.text(x[0], y[0], 'A', fontsize = 16, color='w')
        ax1.text(x[-1], y[-1]-5, 'B', fontsize = 16, color='w')
        tempstr = functype.split("_")
        if view == "vec":
            tname = "Vector "
            title = tname+tempstr[0]+"-field from "+tempstr[2]
        elif  view== "amp":
            tname = "|"
            title = tname+tempstr[0]+"|-field from "+tempstr[2]
        else:
            if component == "real":
                tname = "Re("
            elif component == "imag":
                tname = "Im("
            elif component == "amplitude":
                tname = "Amp("
            elif component == "phase":
                tname = "Phase("

            title = tname + tempstr[0]+view+")-field from "+tempstr[2]

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
        label_cb = tempstr[0]+view+"-field from "+tempstr[2]
        cb.set_label(label)
        ax1.set_title(title)


        if plotTxProfile:
            ax1.plot(np.r_[-20., 80.],np.zeros(2), 'b-', lw=1)
        if plot1D:
            ax1.plot(x,y, 'r.', ms=4)
            ax2 = plt.subplot(gs1[:, 4:6])
            val_line_x, val_line_y, val_line_z = self.dataview.eval(xyz_line, srcLoc, np.r_[sig], np.r_[f], orientation, self.func)

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
            distance = np.sqrt((x-x1)**2+(y-y1)**2) - dx # specific purpose


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
                xticks = np.linspace(val_line.min(), val_line.max(), 3)
                plt.plot(np.r_[0., 0.], np.r_[distance.min(), distance.max()], 'k-', lw=2)
                ax2.xaxis.set_ticks(xticks)
                ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))

            ax2.set_ylim(distance.min(), distance.max())
            ax2.set_ylabel("A-B profile (m)")

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

            ax2.set_title("EM data at Rx hole")
            ax2.set_xlabel(label)

            # ax2.text(distance.min(), val_line.max(), 'A', fontsize = 16)
            # ax2.text(distance.max()*0.97, val_line.max(), 'B', fontsize = 16)
            # ax2.legend((component, ), bbox_to_anchor=(0.5, -0.3))
            ax2.grid(True)
        plt.show()
        pass

    def InteractiveDipoleBH(self, nRx=20, npts2D=50, scale="log", offset_plane=50.,\
                            X1=-20, X2=80, Y1=-50, Y2=50, Z1=-50, Z2=50, \
                            plane="YZ", SrcType="ED", fieldvalue="E", compvalue="z"):

        # x1, x2, y1, y2 = offset_rx, offset_rx, Z1, Z2
        self.xmin, self.xmax = X1, X2
        self.ymin, self.ymax = Y1, Y2
        self.zmin, self.zmax = Z1, Z2

        def foo(Field, AmpDir, Component, ComplexNumber, Frequency, Sigma, Offset, Scale, Slider, FreqLog, SigLog, SrcType=SrcType):
            if Slider ==True:
                f = np.r_[10**FreqLog]
                sig = np.r_[10**SigLog]
            else:
                f = np.r_[Frequency]
                sig = np.r_[Sigma]

            if plane == "XZ":
                normal = "Y"
                self.offset_rx = 50.

            elif plane == "YZ":
                normal = "X"
                self.offset_rx = 0.
            x1, x2, y1, y2 = self.offset_rx, self.offset_rx, Z1, Z2

            if ComplexNumber == "Re":
                ComplexNumber = "real"
            elif ComplexNumber == "Im":
                ComplexNumber = "imag"
            elif ComplexNumber == "Amp":
                ComplexNumber = "amplitude"
            elif ComplexNumber == "Phase":
                ComplexNumber = "phase"

            if AmpDir == "Direction":
                # ComplexNumber = "real"
                Component = "vec"
            elif AmpDir == "Amp":
                # ComplexNumber = "real"
                Component = "amp"

            if SrcType == "ED":
                Field = Field+"_from_ED"
            elif SrcType == "MD":
                Field = Field+"_from_MD"

            return self.Dipole2Dviz(x1, y1, x2, y2, npts2D, nRx, sig, f, srcLoc=np.r_[0., 0., 0.], orientation="z", component=ComplexNumber, view=Component, normal=normal, functype=Field, loc=Offset, scale=Scale)

        out = widgetify(foo
                        ,Field=widgets.ToggleButtons(options=["E", "H", "J"], value=fieldvalue) \
                        ,AmpDir=widgets.ToggleButtons(options=['None','Amp','Direction'], value="Direction") \
                        ,Component=widgets.ToggleButtons(options=['x','y','z'], value=compvalue, description='Comp.') \
                        ,ComplexNumber=widgets.ToggleButtons(options=['Re','Im','Amp', 'Phase']) \
                        ,Frequency=widgets.FloatText(value=0., continuous_update=False, description='f (Hz)') \
                        ,Sigma=widgets.FloatText(value=0.01, continuous_update=False, description='$\sigma$ (S/m)') \
                        ,Offset=widgets.FloatText(value = offset_plane, continuous_update=False) \
                        ,Scale=widgets.ToggleButtons(options=['log','linear'], value="log") \
                        ,Slider=widgets.widget_bool.Checkbox(value=False)\
                        ,FreqLog=widgets.FloatSlider(min=-3, max=6, step=0.5, value=-3, continuous_update=False) \
                        ,SigLog=widgets.FloatSlider(min=-3, max=3, step=0.5, value=-3, continuous_update=False) \
                        ,SrcType = fixed(SrcType)
                        )
        return out

    def InteractiveDipole(self):
        def foo(orientation, normal, component, view, functype, flog, siglog, x1, y1, x2, y2, npts2D, npts, loc):
            f = np.r_[10**flog]
            sig = np.r_[10**siglog]
            return self.Dipole2Dviz(x1, y1, x2, y2, npts2D, npts, sig, f, srcLoc=np.r_[0., 0., 0.], orientation=orientation, component=component, view=view, normal=normal, functype=functype, loc=loc, dx=50.)

        out = widgetify(foo
            ,orientation=widgets.ToggleButtons(options=['x','y','z']) \
            ,normal=widgets.ToggleButtons(options=['X','Y','Z'], value="Z") \
            ,component=widgets.ToggleButtons(options=['real','imag','amplitude', 'phase']) \
            ,view=widgets.ToggleButtons(options=['x','y','z', 'vec']) \
            ,functype=widgets.ToggleButtons(options=["E_from_ED", "H_from_ED", "E_from_ED_galvanic", "E_from_ED_inductive"]) \
            ,flog=widgets.FloatSlider(min=-3, max=6, step=0.5, value=-3, continuous_update=False) \
            ,siglog=widgets.FloatSlider(min=-3, max=3, step=0.5, value=-3, continuous_update=False) \
            ,loc=widgets.FloatText(value=0.01) \
            ,x1=widgets.FloatText(value=-10) \
            ,y1=widgets.FloatText(value=0.01) \
            ,x2=widgets.FloatText(value=10) \
            ,y2=widgets.FloatText(value=0.01) \
            ,npts2D=widgets.IntSlider(min=4,max=200,step=2,value=40) \
            ,npts=widgets.IntSlider(min=4,max=200,step=2,value=40)
            )
        return out


def DisPosNegvalues(val):
    temp_p = val.copy()*np.nan
    temp_p[val>0.] = val[val>0.]
    temp_n = val.copy()*np.nan
    temp_n[val<0.] = -val[val<0.]
    return temp_p, temp_n


def InteractiveDipoleProfile(self, sig, Field, Scale):
    srcLoc = np.r_[0., 0., 0.]
    orientation = "z"
    nRx = 100.

    # def foo(Component, Profile, Scale, F1, F2, F3):
    def foo(Component, ComplexNumber, Sigma, Profile, F1, F2, F3, Scale, FixedScale=False):
    #     Scale = "log"
        orientation = "z"
        vals = []
        if Field =="E":
            unit = " (V/m)"
        elif Field =="H":
            unit = " (A/m)"
        elif Field =="J":
            unit = " (A/m $^2$)"

        if ComplexNumber == "ReIm":
            headerr, headeri = "Re(", "Im("
            textsep = ")"
        elif ComplexNumber == "AmpPhase":
            headerr, headeri = "|", "Phase("
            textsep = "|"

        labelr = headerr+Field+Component+textsep+"-field "+unit
        if ComplexNumber == "AmpPhase":
            unit = " (rad)"
        labeli = headeri+Field+Component+")-field " + unit

        F = [F1, F2, F3]
        if Component == "x":
            icomp = 0
        elif Component == "y":
            icomp = 1
        elif Component == "z":
            icomp = 2

        if Profile == "TxProfile":
            xyz_line = np.c_[np.linspace(-20., 80., nRx), np.zeros(nRx), np.zeros(nRx)]
            r = xyz_line[:,0]
            fig = plt.figure(figsize=(18*1.5,3.4*1.5))
            gs1 = gridspec.GridSpec(2, 7)
            gs1.update(left=0.05, right=0.48, wspace=0.05)
            ax1 = plt.subplot(gs1[:2, :3])
            ax2 = ax1.twinx()

        else:
            if Profile == "Rxhole":
                xyz_line = self.dataview.xyz_line.copy()
            elif Profile == "Txhole":
                xyz_line = self.dataview.xyz_line.copy()
                xyz_line[:,0] = 0.
            else:
                raise NotImplementedError()
            r = xyz_line[:,2]
            fig = plt.figure(figsize=(18*1.0,3.4*1.5))
            gs1 = gridspec.GridSpec(2, 7)
            gs1.update(left=0.05, right=0.48, wspace=0.05)
            ax1 = plt.subplot(gs1[:2, :3])
            ax2 = ax1.twiny()


        for ifreq, f in enumerate(F):
            Frequency = f
            vals.append(self.dataview.eval(xyz_line, srcLoc, np.r_[Sigma], np.r_[f], orientation, self.dataview.func2D))
            # for ifreq, f in enumerate(F):
            if ComplexNumber == "ReIm":
                valr = vals[ifreq][icomp].real.flatten()
                vali = vals[ifreq][icomp].imag.flatten()
            elif ComplexNumber == "AmpPhase":
                valr = abs(vals[ifreq][icomp]).flatten()
                vali = np.angle(vals[ifreq][icomp]).flatten()

            if Scale == "log":
                valr_p, valr_n = DisPosNegvalues(valr)
                vali_p, vali_n = DisPosNegvalues(vali)
                if Profile == "Rxhole" or Profile == "Txhole" :
                    ax1.plot(valr_p, r, 'k-')
                    ax1.plot(valr_n, r, 'k--')
                    if Frequency > 0.:
                        ax2.plot(vali_p, r, 'r-')
                        ax2.plot(vali_n, r, 'r--')
                elif Profile == "TxProfile":
                    ax1.plot(r, valr_p, 'k-')
                    ax1.plot(r, valr_n, 'k--')
                    if Frequency > 0.:
                        ax2.plot(r, vali_p, 'r-')
                        ax2.plot(r, vali_n, 'r--')

            elif Scale == "linear":
                if Profile == "Rxhole" or Profile == "Txhole" :
                    ax1.plot(valr, r, 'k-')
                    if Frequency > 0.:
                        ax1.plot(vali, r, 'r-')

                elif Profile == "TxProfile":
                    ax1.plot(r, valr, 'k-')
                    if Frequency > 0.:
                        ax1.plot(r, vali, 'r-')

        if Profile == "Rxhole" or Profile == "Txhole" :
            ax1.set_xscale(Scale)
            ax1.set_ylim(-50, 50)
            if Frequency > 0.:
                ax2.set_xscale(Scale)
                if FixedScale:
                    vmin1, vmax1 = ax1.get_xlim()
                    vmin2, vmax2 = ax2.get_xlim()
                    vmin = min(vmin1, vmin2)
                    vmax = max(vmax1, vmax2)
                    ax1.set_xlim(vmin, vmax)
                    ax2.set_xlim(vmin, vmax)
                ax2.set_xlabel(labeli, color='r')
            ax1.set_xlabel(labelr, color='k')
            ax1.set_ylabel("Z (m)")

        elif Profile == "TxProfile":
            ax1.set_yscale(Scale)
            ax1.set_xlim(-20, 80)
            if Frequency > 0.:
                ax2.set_yscale(Scale)
                if FixedScale:
                    vmin1, vmax1 = ax1.get_ylim()
                    vmin2, vmax2 = ax2.get_ylim()
                    vmin = min(vmin1, vmin2)
                    vmax = max(vmax1, vmax2)
                    ax1.set_ylim(vmin, vmax)
                    ax2.set_ylim(vmin, vmax)
                ax2.set_ylabel(labeli, color='r')
            ax1.set_ylabel(labelr, color='k')
            ax1.set_xlabel("X (m)")

        if Scale == "linear":
            if Profile == "Rxhole" or Profile == "Txhole" :
                # xticksa = np.linspace(valr.min(), valr.max(), 3)
                x = ax1.xaxis.get_majorticklocs()
                xticksa = np.linspace(x.min(), x.max(), 3)
                ax1.xaxis.set_ticks(xticksa)
                ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))
                if Frequency > 0.:
                    if FixedScale is not True:
                        x = ax2.xaxis.get_majorticklocs()
                    for tl in ax2.get_yticklabels():
                        tl.set_color('r')
                    xticksb = np.linspace(x.min(), x.max(), 3)
                    ax2.xaxis.set_ticks(xticksb)
                    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))

            elif Profile == "TxProfile":
                # yticksa = np.linspace(valr.min(), valr.max(), 3)
                y = ax1.yaxis.get_majorticklocs()
                yticksa = np.linspace(y.min(), y.max(), 3)
                ax1.yaxis.set_ticks(yticksa)
                ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))
                if Frequency > 0.:
                    if FixedScale is not True:
                        y = ax2.yaxis.get_majorticklocs()
                    yticksb = np.linspace(y.min(), y.max(), 3)
                    ax2.yaxis.set_ticks(yticksb)
                    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))

        if Frequency > 0.:
            if Profile == "Rxhole" or Profile == "Txhole":
                for tl in ax2.get_xticklabels():
                    tl.set_color('r')
            elif Profile == "TxProfile":
                for tl in ax2.get_yticklabels():
                    tl.set_color('r')
            else:
                raise NotImplementedError()

        ax1.grid(True)
    Q2 = widgetify(foo
        ,Profile=widgets.ToggleButtons(options=['Rxhole','Txhole','TxProfile'], value='Rxhole')
        ,Component=widgets.ToggleButtons(options=['x','y','z'], value='z', description='Comp.') \
        ,ComplexNumber=widgets.ToggleButtons(options=['ReIm','AmpPhase']) \
        ,Sigma=widgets.FloatText(value=sig, continuous_update=False, description='$\sigma$ (S/m)') \
        ,Scale=widgets.ToggleButtons(options=['log','linear'], value=Scale) \
        ,FixedScale=widgets.widget_bool.Checkbox(value=False, description='Fixed')
        ,F1=widgets.FloatText(value=0.1, continuous_update=False, description='$f_1$ (Hz)')
        ,F2=widgets.FloatText(value=100, continuous_update=False, description='$f_2$ (Hz)')\
        ,F3=widgets.FloatText(value=1000, continuous_update=False, description='$f_3$ (Hz)'))
    return Q2
