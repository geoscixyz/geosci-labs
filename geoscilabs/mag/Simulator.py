from . import Mag
from . import MagUtils

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import ipywidgets as widgets
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata, interp1d

from SimPEG.potential_fields import magnetics as mag
from SimPEG import utils, data
from scipy.constants import mu_0


def PFSimulator(prism, survey):
    def PFInteract(
        update,
        susc,
        comp,
        irt,
        Q,
        RemInc,
        RemDec,
        Profile_npt,
        Profile_azm,
        Profile_len,
        Profile_ctx,
        Profile_cty,
    ):

        # Get the line extent from the 2D survey for now
        sim = Mag.Simulation()
        sim.prism = prism.result
        sim.survey = survey.result

        return PlotFwrSim(
            sim,
            susc,
            comp,
            irt,
            Q,
            RemInc,
            RemDec,
            Profile_azm,
            Profile_len,
            Profile_npt,
            Profile_ctx,
            Profile_cty,
        )

    locs = survey.result.receiver_locations
    xlim = np.asarray([locs[:, 0].min(), locs[:, 0].max()])
    ylim = np.asarray([locs[:, 1].min(), locs[:, 1].max()])

    Lx = xlim[1] - xlim[0]
    Ly = ylim[1] - ylim[0]
    diag = (Lx ** 2.0 + Ly ** 2.0) ** 0.5 / 2.0

    ctx = np.mean(xlim)
    cty = np.mean(ylim)

    out = widgets.interactive(
        PFInteract,
        update=widgets.ToggleButton(description="Refresh", value=False),
        susc=widgets.FloatSlider(
            min=0, max=2, step=0.001, value=0.1, continuous_update=False
        ),
        comp=widgets.ToggleButtons(options=["tf", "bx", "by", "bz"]),
        irt=widgets.ToggleButtons(options=["induced", "remanent", "total"]),
        Q=widgets.FloatSlider(
            min=0.0, max=10, step=1, value=0, continuous_update=False
        ),
        RemInc=widgets.FloatSlider(
            min=-90.0, max=90, step=5, value=0, continuous_update=False
        ),
        RemDec=widgets.FloatSlider(
            min=-90.0, max=90, step=5, value=0, continuous_update=False
        ),
        Profile_npt=widgets.BoundedIntText(
            min=10, max=100, step=1, value=20, continuous_update=False
        ),
        Profile_azm=widgets.FloatSlider(
            min=-90, max=90, step=5, value=45.0, continuous_update=False
        ),
        Profile_len=widgets.FloatSlider(
            min=10, max=diag, step=10, value=Ly, continuous_update=False
        ),
        Profile_ctx=widgets.FloatSlider(
            value=ctx,
            min=xlim[0],
            max=xlim[1],
            step=0.1,
            continuous_update=False,
            color="black",
        ),
        Profile_cty=widgets.FloatSlider(
            value=cty,
            min=ylim[0],
            max=ylim[1],
            step=0.1,
            continuous_update=False,
            color="black",
        ),
    )
    return out

    # Create simulation


def PlotFwrSim(
    sim,
    susc,
    comp,
    irt,
    Q,
    rinc,
    rdec,
    Profile_azm,
    Profile_len,
    Profile_npt,
    Profile_ctx,
    Profile_cty,
):
    def MagSurvey2D(
        survey,
        dobj,
        Profile_ctx,
        Profile_cty,
        Profile_azm,
        Profile_len,
        Profile_npt,
        fig=None,
        ax=None,
        vmin=None,
        vmax=None,
        pred=None,
    ):

        # Get the line extent from the 2D survey for now
        Profile_azm /= 180.0 / np.pi
        Profile_len /= 2.0 * 0.98

        dx = np.cos(-Profile_azm) * Profile_len
        dy = np.sin(-Profile_azm) * Profile_len

        a = [Profile_ctx - dx, Profile_cty - dy]
        b = [Profile_ctx + dx, Profile_cty + dy]

        return plotMagSurvey2D(
            survey,
            dobj,
            a,
            b,
            Profile_npt,
            pred=None,
            fig=fig,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )

    def MagSurveyProfile(
        survey,
        dobj,
        Profile_ctx,
        Profile_cty,
        Profile_azm,
        Profile_len,
        Profile_npt,
        dpred=None,
        fig=None,
        ax=None,
    ):

        # Get the line extent from the 2D survey for now
        Profile_azm /= 180.0 / np.pi
        Profile_len /= 2.0 * 0.98

        dx = np.cos(-Profile_azm) * Profile_len
        dy = np.sin(-Profile_azm) * Profile_len

        a = [Profile_ctx - dx, Profile_cty - dy]
        b = [Profile_ctx + dx, Profile_cty + dy]

        xyz = survey.receiver_locations

        return plotProfile(xyz, dobj, a, b, Profile_npt, fig=fig, ax=ax)

    survey = sim.survey
    sim.Q, sim.rinc, sim.rdec = Q, rinc, rdec
    sim.uType, sim.mType = comp, irt
    sim.susc = susc

    # Compute fields from prism
    fields = sim.fields()

    dpred = np.zeros_like(fields[0])
    for b in fields:
        dpred += b

    f = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot()
    MagSurvey2D(
        survey,
        dpred,
        Profile_ctx,
        Profile_cty,
        Profile_azm,
        Profile_len,
        Profile_npt,
        fig=f,
        ax=ax1,
        pred=None,
    )

    f = plt.figure(figsize=(7, 3))
    ax2 = plt.subplot()
    MagSurveyProfile(
        survey,
        dpred,
        Profile_ctx,
        Profile_cty,
        Profile_azm,
        Profile_len,
        Profile_npt,
        dpred=None,
        fig=f,
        ax=ax2,
    )

    plt.show()


def ViewMagSurvey2D(survey, dobj):
    def MagSurvey2D(East, North, Width, Height, Azimuth, Length, Npts, Profile):

        # Get the line extent from the 2D survey for now
        Azimuth /= 180.0 / np.pi
        Length /= 2.0 * 0.98

        a = [East - np.cos(-Azimuth) * Length, North - np.sin(-Azimuth) * Length]

        b = [East + np.cos(-Azimuth) * Length, North + np.sin(-Azimuth) * Length]

        xlim = East + np.asarray([-Width / 2.0, Width / 2.0])
        ylim = North + np.asarray([-Height / 2.0, Height / 2.0])

        # Re-sample the survey within the region
        rxLoc = survey.receiver_locations

        ind = np.all(
            [
                rxLoc[:, 0] > xlim[0],
                rxLoc[:, 0] < xlim[1],
                rxLoc[:, 1] > ylim[0],
                rxLoc[:, 1] < ylim[1],
            ],
            axis=0,
        )

        rxLoc = mag.receivers.Point(rxLoc[ind, :])
        srcField = mag.sources.SourceField(
            receiver_list=[rxLoc], parameters=survey.source_field.parameters
        )
        surveySim = mag.Survey(srcField)

        fig = plt.figure(figsize=(6, 9))
        ax1 = plt.subplot(2, 1, 1)
        plotMagSurvey2D(surveySim, dobj.dobs[ind], a, b, Npts, fig=fig, ax=ax1)

        if Profile:

            ax2 = plt.subplot(2, 1, 2)

            xyz = surveySim.receiver_locations
            plotProfile(xyz, dobj.dobs[ind], a, b, Npts, pred=None, fig=fig, ax=ax2)

        return surveySim

    locs = survey.receiver_locations
    xlim = np.asarray([locs[:, 0].min(), locs[:, 0].max()])
    ylim = np.asarray([locs[:, 1].min(), locs[:, 1].max()])

    Lx = xlim[1] - xlim[0]
    Ly = ylim[1] - ylim[0]
    diag = (Lx ** 2.0 + Ly ** 2.0) ** 0.5 / 2.0

    East = np.mean(xlim)
    North = np.mean(ylim)
    cntr = [East, North]

    out = widgets.interactive(
        MagSurvey2D,
        East=widgets.FloatSlider(
            min=cntr[0] - Lx,
            max=cntr[0] + Lx,
            step=10,
            value=cntr[0],
            continuous_update=False,
        ),
        North=widgets.FloatSlider(
            min=cntr[1] - Ly,
            max=cntr[1] + Ly,
            step=10,
            value=cntr[1],
            continuous_update=False,
        ),
        Width=widgets.FloatSlider(
            min=10, max=Lx * 1.05, step=10, value=Lx * 1.05, continuous_update=False
        ),
        Height=widgets.FloatSlider(
            min=10, max=Ly * 1.05, step=10, value=Ly * 1.05, continuous_update=False
        ),
        Azimuth=widgets.FloatSlider(
            min=-90, max=90, step=5, value=0, continuous_update=False
        ),
        Length=widgets.FloatSlider(
            min=10, max=diag, step=10, value=Ly, continuous_update=False
        ),
        Npts=widgets.BoundedIntText(
            min=10, max=100, step=1, value=20, continuous_update=False
        ),
        Profile=widgets.ToggleButton(description="Profile", value=False),
    )

    return out


def plotMagSurvey2D(
    survey, dobj, a, b, npts, pred=None, fig=None, ax=None, vmin=None, vmax=None
):
    """
    Plot the data and line profile inside the spcified limits
    """

    # So you can provide a vector
    if isinstance(dobj, data.Data):
        dobj = dobj.dobs

    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = plt.subplot(1, 2, 1)

    x, y = linefun(a[0], b[0], a[1], b[1], npts)
    rxLoc = survey.receiver_locations

    utils.plot_utils.plot2Ddata(rxLoc, dobj, ax=ax)

    ax.plot(x, y, "w.", ms=10)
    ax.text(x[0], y[0], "A", fontsize=16, color="w", ha="left")
    ax.text(x[-1], y[-1], "B", fontsize=16, color="w", ha="right")
    ax.grid(True)

    if pred is not None:
        ax2 = plt.subplot(1, 2, 2)

        utils.plot_utils.plot2Ddata(rxLoc, pred, ax=ax2, clim=[pred.min(), pred.max()])
        ax2.plot(x, y, "w.", ms=10)
        ax2.text(x[0], y[0], "A", fontsize=16, color="w", ha="left")
        ax2.text(x[-1], y[-1], "B", fontsize=16, color="w", ha="right")
        ax2.set_yticks([])
        ax2.set_yticklabels("")
        ax2.grid(True)

    plt.show()
    return


def plotProfile(xyz, dobj, a, b, npts, pred=None, fig=None, ax=None, dType="3D"):
    """
    Plot the data and line profile inside the spcified limits
    """

    # So you can provide a vector
    if isinstance(dobj, data.Data):
        dobj = dobj.dobs

    if fig is None:
        fig = plt.figure(figsize=(6, 4))

        plt.rcParams.update({"font.size": 14})

    if ax is None:
        ax = plt.subplot()

    rxLoc = xyz

    x, y = linefun(a[0], b[0], a[1], b[1], npts)

    distance = np.sqrt((x - a[0]) ** 2.0 + (y - a[1]) ** 2.0)

    if dType == "2D":
        distance = rxLoc[:, 1]
        dline = dobj

    else:
        dline = griddata(rxLoc[:, :2], dobj, (x, y), method="linear")

    ax.plot(distance, dline, "b.-")

    if pred is not None:

        if dType == "2D":
            distance = rxLoc[:, 1]
            dline = pred

        else:
            dline = griddata(rxLoc[:, :2], pred, (x, y), method="linear")

        ax.plot(distance, dline, "r.-")

    ax.set_xlim(distance.min(), distance.max())

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Magnetic field (nT)")

    # ax.text(distance.min(), dline.max()*0.8, 'A', fontsize = 16)
    # ax.text(distance.max()*0.97, out_linei.max()*0.8, 'B', fontsize = 16)
    ax.legend(("survey", "simulated"), bbox_to_anchor=(1, 1))
    ax.grid(True)
    plt.show()

    return True


def linefun(x1, x2, y1, y2, nx, tol=1e-3):
    dx = x2 - x1
    dy = y2 - y1

    if np.abs(dx) < tol:
        y = np.linspace(y1, y2, nx)
        x = np.ones_like(y) * x1
    elif np.abs(dy) < tol:
        x = np.linspace(x1, x2, nx)
        y = np.ones_like(x) * y1
    else:
        x = np.linspace(x1, x2, nx)
        slope = (y2 - y1) / (x2 - x1)
        y = slope * (x - x1) + y1
    return x, y


def ViewPrism(survey):
    def Prism(
        update,
        dx,
        dy,
        dz,
        x0,
        y0,
        elev,
        prism_inc,
        prism_dec,
        View_dip,
        View_azm,
        View_lim,
    ):

        prism = definePrism()

        # TODO: this is a temporary fix for
        # X-North, Y-East...

        prism.dx, prism.dy, prism.dz, prism.z0 = dy, dx, dz, elev
        prism.x0, prism.y0 = y0, x0
        prism.pinc, prism.pdec = prism_inc, prism_dec

        # Display the prism and survey points
        plotObj3D(prism, survey, View_dip, View_azm, View_lim)

        return prism

    rxLoc = survey.receiver_locations
    cntr = np.mean(rxLoc[:, :2], axis=0)

    xlim = rxLoc[:, 0].max() - rxLoc[:, 0].min()
    ylim = rxLoc[:, 1].max() - rxLoc[:, 1].min()

    lim = np.max([xlim, ylim]) / 2.0

    out = widgets.interactive(
        Prism,
        update=widgets.ToggleButton(description="Refresh", value=False),
        dx=widgets.FloatSlider(
            min=0.01, max=1000.0, step=0.01, value=lim / 4, continuous_update=False
        ),
        dy=widgets.FloatSlider(
            min=0.01, max=1000.0, step=0.01, value=lim / 4, continuous_update=False
        ),
        dz=widgets.FloatSlider(
            min=0.01, max=1000.0, step=0.01, value=lim / 4, continuous_update=False
        ),
        x0=widgets.FloatSlider(
            min=cntr[1] - 1000,
            max=cntr[1] + 1000,
            step=1.0,
            value=cntr[1],
            continuous_update=False,
        ),
        y0=widgets.FloatSlider(
            min=cntr[0] - 1000,
            max=cntr[0] + 1000,
            step=1.0,
            value=cntr[0],
            continuous_update=False,
        ),
        elev=widgets.FloatSlider(
            min=-1000.0, max=1000.0, step=1.0, value=0.0, continuous_update=False
        ),
        prism_inc=(-90.0, 90.0, 5.0),
        prism_dec=(-90.0, 90.0, 5.0),
        View_dip=widgets.FloatSlider(
            min=0, max=90, step=1, value=30, continuous_update=False
        ),
        View_azm=widgets.FloatSlider(
            min=0, max=360, step=1, value=220, continuous_update=False
        ),
        View_lim=widgets.FloatSlider(
            min=1, max=2 * lim, step=1, value=lim, continuous_update=False
        ),
    )

    return out


def plotObj3D(
    prism, survey, View_dip, View_azm, View_lim, fig=None, axs=None, title=None
):

    """
    Plot the prism in 3D
    """

    x1, x2 = prism.xn[0] - prism.xc, prism.xn[1] - prism.xc
    y1, y2 = prism.yn[0] - prism.yc, prism.yn[1] - prism.yc
    z1, z2 = prism.zn[0] - prism.zc, prism.zn[1] - prism.zc
    pinc, pdec = prism.pinc, prism.pdec

    if isinstance(survey, mag.Survey) is False:
        survey = survey[0]

    rxLoc = survey.receiver_locations

    if fig is None:
        fig = plt.figure(figsize=(7, 7))

    if axs is None:
        axs = fig.add_subplot(111, projection="3d")

    if title is not None:
        axs.set_title(title)

    # plt.rcParams.update({'font.size': 13})

    cntr = [prism.x0, prism.y0]
    axs.set_xlim3d(-View_lim + cntr[0], View_lim + cntr[0])
    axs.set_ylim3d(-View_lim + cntr[1], View_lim + cntr[1])
    axs.set_zlim3d(rxLoc[:, 2].max() * 1.1 - View_lim * 2, rxLoc[:, 2].max() * 1.1)

    # Create a rectangular prism, rotate and plot
    block_xyz = np.asarray(
        [
            [x1, x1, x2, x2, x1, x1, x2, x2],
            [y1, y2, y2, y1, y1, y2, y2, y1],
            [z1, z1, z1, z1, z2, z2, z2, z2],
        ]
    )

    R = MagUtils.rotationMatrix(pinc, pdec)

    xyz = R.dot(block_xyz).T

    # Offset the prism to true coordinate
    offx = prism.xc
    offy = prism.yc
    offz = prism.zc

    # print xyz
    # Face 1
    axs.add_collection3d(
        Poly3DCollection(
            [list(zip(xyz[:4, 0] + offx, xyz[:4, 1] + offy, xyz[:4, 2] + offz))]
        )
    )

    # Face 2
    axs.add_collection3d(
        Poly3DCollection(
            [list(zip(xyz[4:, 0] + offx, xyz[4:, 1] + offy, xyz[4:, 2] + offz))],
            facecolors="w",
        )
    )

    # Face 3
    axs.add_collection3d(
        Poly3DCollection(
            [
                list(
                    zip(
                        xyz[[0, 1, 5, 4], 0] + offx,
                        xyz[[0, 1, 5, 4], 1] + offy,
                        xyz[[0, 1, 5, 4], 2] + offz,
                    )
                )
            ]
        )
    )

    # Face 4
    axs.add_collection3d(
        Poly3DCollection(
            [
                list(
                    zip(
                        xyz[[3, 2, 6, 7], 0] + offx,
                        xyz[[3, 2, 6, 7], 1] + offy,
                        xyz[[3, 2, 6, 7], 2] + offz,
                    )
                )
            ]
        )
    )

    # Face 5
    axs.add_collection3d(
        Poly3DCollection(
            [
                list(
                    zip(
                        xyz[[0, 4, 7, 3], 0] + offx,
                        xyz[[0, 4, 7, 3], 1] + offy,
                        xyz[[0, 4, 7, 3], 2] + offz,
                    )
                )
            ]
        )
    )

    # Face 6
    axs.add_collection3d(
        Poly3DCollection(
            [
                list(
                    zip(
                        xyz[[1, 5, 6, 2], 0] + offx,
                        xyz[[1, 5, 6, 2], 1] + offy,
                        xyz[[1, 5, 6, 2], 2] + offz,
                    )
                )
            ]
        )
    )

    axs.set_xlabel("East (Y; m)")
    axs.set_ylabel("North (X; m)")
    axs.set_zlabel("Depth (Z; m)")
    axs.scatter(rxLoc[:, 0], rxLoc[:, 1], zs=rxLoc[:, 2], s=1, alpha=0.5)
    axs.view_init(View_dip, View_azm)
    plt.show()

    return True


class definePrism(object):
    """
        Define a prism and its attributes

        Prism geometry:
            - dx, dy, dz: width, length and height of prism
            - depth : depth to top of prism
            - susc : susceptibility of prism
            - x0, y0 : center of prism in horizontal plane
            - pinc, pdec : inclination and declination of prism
    """

    x0, y0, z0, dx, dy, dz = 0.0, 0.0, 0.0, 1.0, 1.0, 1.0
    pinc, pdec = 0.0, 0.0

    # Define the nodes of the prism
    @property
    def xn(self):
        xn = np.asarray([-self.dx / 2.0 + self.x0, self.dx / 2.0 + self.x0])

        return xn

    @property
    def yn(self):
        yn = np.asarray([-self.dy / 2.0 + self.y0, self.dy / 2.0 + self.y0])

        return yn

    @property
    def zn(self):
        zn = np.asarray([-self.dz + self.z0, self.z0])

        return zn

    @property
    def xc(self):
        xc = (self.xn[0] + self.xn[1]) / 2.0

        return xc

    @property
    def yc(self):
        yc = (self.yn[0] + self.yn[1]) / 2.0

        return yc

    @property
    def zc(self):
        zc = (self.zn[0] + self.zn[1]) / 2.0

        return zc


def fitline(prism, survey, dobj):
    def profiledata(Binc, Bdec, Bigrf, depth, susc, comp, irt, Q, rinc, rdec, update):

        # Get the line extent from the 2D survey for now
        sim = Mag.Simulation()
        sim.prism = prism.result

        xyzLoc = survey.receiver_locations.copy()
        xyzLoc[:, 2] += depth

        rxLoc = mag.receivers.Point(xyzLoc)
        srcField = mag.sources.SourceField(
            receiver_list=[rxLoc], parameters=(Bigrf, -Binc, Bdec)
        )
        survey2D = mag.Survey(srcField)
        sim.survey = survey2D

        sim.Q, sim.rinc, sim.rdec = Q, -rinc, rdec
        sim.uType, sim.mType = comp, irt
        sim.susc = susc

        # Compute fields from prism
        fields = sim.fields()

        dpred = np.zeros_like(fields[0])
        for b in fields:
            dpred += b

        dpred += +Bigrf
        a = np.r_[xyzLoc[:, 0].min(), 0]
        b = np.r_[xyzLoc[:, 0].max(), 0]
        return plotProfile(xyzLoc, dobj, a, b, 10, pred=dpred, dType="2D")

    Q = widgets.interactive(
        profiledata,
        Binc=widgets.FloatSlider(
            min=-90.0, max=90, step=5, value=90, continuous_update=False
        ),
        Bdec=widgets.FloatSlider(
            min=-90.0, max=90, step=5, value=0, continuous_update=False
        ),
        Bigrf=widgets.FloatSlider(
            min=54000.0, max=55000, step=10, value=54500, continuous_update=False
        ),
        depth=widgets.FloatSlider(min=0.0, max=5.0, step=0.05, value=0.5),
        susc=widgets.FloatSlider(min=0.0, max=800.0, step=5.0, value=1.0),
        comp=widgets.ToggleButtons(options=["tf", "bx", "by", "bz"]),
        irt=widgets.ToggleButtons(options=["induced", "remanent", "total"]),
        Q=widgets.FloatSlider(min=0.0, max=10.0, step=0.1, value=0.0),
        rinc=widgets.FloatSlider(min=-180.0, max=180.0, step=1.0, value=0.0),
        rdec=widgets.FloatSlider(min=-180.0, max=180.0, step=1.0, value=0.0),
        update=widgets.ToggleButton(description="Refresh", value=False),
    )
    return Q
