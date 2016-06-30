import numpy as np
from View import DataView
from SimPEG import EM
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['font.size'] = 12
import warnings
warnings.filterwarnings("ignore")
from ipywidgets import *

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


class DipoleWidget(object):
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
            self.func = EM.Analytics.E_from_ElectricDipoleWholeSpace
        elif self.functype == "E_from_ED_galvanic":
            self.func = EM.Analytics.E_galvanic_from_ElectricDipoleWholeSpace
        elif self.functype == "E_from_ED_inductive":
            self.func = EM.Analytics.E_inductive_from_ElectricDipoleWholeSpace
        elif self.functype == "H_from_ED":
            self.func = EM.Analytics.H_from_ElectricDipoleWholeSpace
        elif self.functype == "J_from_ED":
            self.func = EM.Analytics.J_from_ElectricDipoleWholeSpace

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

    def Dipole2Dviz(self, x1, y1, x2, y2, npts2D, npts, sig, f, srcLoc=np.r_[0., 0., 0.], orientation="x", component="real", view="x", normal="Z", functype="E_from_ED", loc=0., scale="log", dx=50.):
        import matplotlib.gridspec as gridspec
        nx, ny = npts2D, npts2D
        x, y = linefun(x1, x2, y1, y2, npts)
        if scale == "log":
            logamp = True
        elif scale == "linear":
            logamp = False
        else:
            raise NotImplementedError()

        self.SetDataview(srcLoc, sig, f, orientation, normal, functype, na=nx, nb=ny, loc=loc)

        plot1D = False
        if normal =="X" or normal=="x":
            if abs(loc - 50) < 1e-5:
                plot1D = True
            xyz_line = np.c_[np.ones_like(x)*self.x, x, y]
            self.dataview.xyz_line =  xyz_line
        if normal =="Y" or normal=="y":
            if abs(loc - 0.) < 1e-5:
                plot1D = True
            if loc < 0:
                plot1D = True
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
        title = tempstr[0]+view+"-field from "+tempstr[2]

        if tempstr[0] == "E":
            label = "Electric field (V/m) "
        elif tempstr[0] == "H":
            label = "Magnetic field (A/m) "
        elif tempstr[0] == "J":
            label = "Current density (A/m$^2$) "
        else:
            raise NotImplementedError()
        label_cb = tempstr[0]+view+"-field from "+tempstr[2]
        cb.set_label(label)
        ax1.set_title(title)


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

            if view == "vec" or view == "amp":
                viewstr = " vector"
            else:
                viewstr = view

                if component == "real":
                    val_line = val_line.real
                elif component == "imag":
                    val_line = val_line.imag
                elif component == "amplitude":
                    val_line = val_line
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
                label = tempstr[0]+viewstr+"-field (V/m) "
            elif tempstr[0] == "H":
                label = tempstr[0]+viewstr+"-field (A/m) "
            elif tempstr[0] == "J":
                label = tempstr[0]+viewstr+"-field (A/m$^2$) "
            else:
                raise NotImplementedError()


            ax2.set_title("EM data at Rx hole")
            ax2.set_xlabel(label)

            # ax2.text(distance.min(), val_line.max(), 'A', fontsize = 16)
            # ax2.text(distance.max()*0.97, val_line.max(), 'B', fontsize = 16)
            # ax2.legend((component, ), bbox_to_anchor=(0.5, -0.3))
            ax2.grid(True)
        plt.show()
        pass

    def InteractiveDipoleBH(self, nRx=20, npts2D=50, scale="log", offset_plane=50., X1=-20, X2=80, Y1=-50, Y2=50, Z1=-50, Z2=50, plane="YZ"):

        # x1, x2, y1, y2 = offset_rx, offset_rx, Z1, Z2
        self.xmin, self.xmax = X1, X2
        self.ymin, self.ymax = Y1, Y2
        self.zmin, self.zmax = Z1, Z2

        def foo(dtype, vector, rxdirection, fieldtype, frequency, sigma, offset, scale, flog, siglog, slider=False):
            if slider ==True:
                f = np.r_[10**flog]
                sig = np.r_[10**siglog]
            else:
                f = np.r_[frequency]
                sig = np.r_[sigma]

            if plane == "XZ":
                normal = "Y"
                self.offset_rx = 50.

            elif plane == "YZ":
                normal = "X"
                self.offset_rx = 0.
            x1, x2, y1, y2 = self.offset_rx, self.offset_rx, Z1, Z2

            if vector == "vector":
                dtype = "real"
                rxdirection = "vec"
            elif vector == "amplitude":
                dtype = "real"
                rxdirection = "amp"

            fieldtype = fieldtype+"_from_ED"

            return self.Dipole2Dviz(x1, y1, x2, y2, npts2D, nRx, sig, f, srcLoc=np.r_[0., 0., 0.], orientation="z", component=dtype, view=rxdirection, normal=normal, functype=fieldtype, loc=offset, scale=scale)

        out = widgets.interactive (foo
                        ,dtype=widgets.ToggleButtons(options=['real','imag','amplitude', 'phase']) \
                        ,vector=widgets.ToggleButtons(options=['scalar','vector','amplitude'], value="vector") \
                        ,rxdirection=widgets.ToggleButtons(options=['x','y','z'], value='z') \
                        ,fieldtype=widgets.ToggleButtons(options=["E", "H", "J"]) \
                        ,frequency=widgets.FloatText(value=0., continuous_update=False) \
                        ,flog=widgets.FloatSlider(min=-3, max=6, step=0.5, value=-3, continuous_update=False) \
                        ,sigma=widgets.FloatText(value=0.01, continuous_update=False) \
                        ,siglog=widgets.FloatSlider(min=-3, max=3, step=0.5, value=-3, continuous_update=False) \
                        ,offset=widgets.FloatText(value = offset_plane, continuous_update=False) \
                        ,scale=widgets.ToggleButtons(options=['log','linear'], value="log") \
                        )
        return out

    def InteractiveDipole(self):
        def foo(orientation, normal, component, view, functype, flog, siglog, x1, y1, x2, y2, npts2D, npts, loc):
            f = np.r_[10**flog]
            sig = np.r_[10**siglog]
            return self.Dipole2Dviz(x1, y1, x2, y2, npts2D, npts, sig, f, srcLoc=np.r_[0., 0., 0.], orientation=orientation, component=component, view=view, normal=normal, functype=functype, loc=loc, dx=50.)

        out = widgets.interactive (foo
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
