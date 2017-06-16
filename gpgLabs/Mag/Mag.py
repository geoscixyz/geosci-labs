import simpegPF as MAG
import simpegCoordUtils as Utils

from scipy.constants import mu_0
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from IPython.html.widgets import *
# import ipywidgets as widgets
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# monFile = "data2015/StudentData2015_Monday.csv"
# monData = pd.DataFrame(pd.read_csv(filename, header = 0))

# filename = "data2014/HZrebarProfile.csv"
# data = pd.DataFrame(pd.read_csv(filename, header = 0))
# loc = data["Distance"].values

diameter = 1.4e-2
length = 3.
xlim = np.r_[5., 25.]
rx_h = 1.9

Bincd = 70.205
Bdecd = 16.63
Bigrfd = 54450

# Depth of burial: Monday was 35cm. I believe that Wednesday was ~45cm


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

    x0, y0, z0, dx, dy, dz = 0., 0., 0., 1., 1., 1.
    pinc, pdec = 0., 0.


    # Define the nodes of the prism
    @property
    def xn(self):
        xn = np.asarray([-self.dx/2. + self.x0, self.dx/2. + self.x0])

        return xn

    @property
    def yn(self):
        yn = np.asarray([-self.dy/2. + self.y0, self.dy/2. + self.y0])

        return yn

    @property
    def zn(self):
        zn = np.asarray([-self.dz + self.z0, self.z0])

        return zn

    @property
    def xc(self):
        xc = (self.xn[0] + self.xn[1]) / 2.

        return xc

    @property
    def yc(self):
        yc = (self.yn[0] + self.yn[1]) / 2.

        return yc

    @property
    def zc(self):
        zc = (self.zn[0] + self.zn[1]) / 2.

        return zc


def plotProfile(prob2D, x0, data, Binc, Bdec, Bigrf, susc, Q, rinc, rdec):
    if data is 'MonSt':
        filename = "../assets/Mag/data/Lab1_monday_TA.csv"
    elif data is 'WedSt':
        filename = "../assets/Mag/data/Lab1_Wednesday_student.csv"
    elif data is 'WedTA':
        filename = "../assets/Mag/data/Lab1_Wednesday_TA.csv"

    dat = pd.DataFrame(pd.read_csv(filename, header = 0))
    tf  = dat["MAG_MEAN"].values
    std = dat["STDEV"].values
    loc = dat["station"].values
    #teams = dat["Team"].values

    tfa = tf - Bigrf

    p = prob2D.prism

    # nx, ny = 100, 1
    # shape = (nx, ny)

    dx = x0 - prob2D.survey.xylim

    prob2D.survey.profile
    if prob2D.survey.profile == "EW":
        x1, x2, y1, y2 = prob2D.survey.xr[0]-dx, prob2D.survey.xr[-1]-dx, 0., 0.
    elif prob2D.survey.profile == "NS":
        x1, x2, y1, y2 = 0., 0., prob2D.survey.yr[0]-dx, prob2D.survey.yr[-1]-dx
    elif prob2D.survey.profile == "45N":
        x1, x2, y1, y2 = prob2D.survey.xr[0], prob2D.survey.xr[-1], prob2D.survey.yr[0], prob2D.survey.yr[-1]


    x, y = linefun(x1, x2, y1, y2, 100)
    xyz_line = np.c_[x, y, np.ones_like(x)*prob2D.survey.rx_h]

    distance = np.sqrt((x-x1)**2.+(y-y1)**2.)

    xlim = [0,distance[-1]]
    prob1D = MAG.problem()
    srvy1D = MAG.survey()
    srvy1D._rxLoc = xyz_line

    prob1D.prism = p
    prob1D.survey = srvy1D

    prob1D.Bdec, prob1D.Binc, prob1D.Bigrf = Bdec, Binc, Bigrf
    prob1D.Q, prob1D.rinc, prob1D.rdec = Q, rinc, rdec
    prob1D.uType, prob1D.mType = 'tf', 'total'
    prob1D.susc = susc

    # Compute fields from prism
    magi, magr = prob1D.fields()

    #out_linei, out_liner = getField(p, xyz_line, comp, 'total')
    #out_linei = getField(p, xyz_line, comp,'induced')
    #out_liner = getField(p, xyz_line, comp,'remanent')

    # distance = np.sqrt((x-x1)**2.+(y-y1)**2.)

    f = plt.figure(figsize = (10, 5))
    gs = gridspec.GridSpec(2, 1,height_ratios=[2,1])

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax1.plot(x0, p.z0, 'ko')
    ax1.text(x0+0.5, p.z0, 'Rebar', color='k')
    ax1.text(xlim[0]+1.,-1.2, 'Magnetometer height (1.9 m)', color='b')
    ax1.plot(xlim, np.r_[rx_h, rx_h], 'b--')

    # magi,magr = getField(p, rxLoc, 'bz', 'total')

    ax1.plot(xlim, np.r_[0., 0.], 'k--')
    ax1.set_xlim(xlim)
    ax1.set_ylim(-2.5, 2.5)

    ax0.scatter(loc,tfa)
    ax0.errorbar(loc,tfa,yerr=std,linestyle = "None",color="k")
    ax0.set_xlim(xlim)
    ax0.grid(which="both")

    ax0.plot(distance, magi, 'b', label='induced')
    ax0.plot(distance, magr, 'r', label='remnant')
    ax0.plot(distance, magi+magr, 'k', label='total')
    ax0.legend(loc=2)
    # ax[1].plot(loc-8, magnT[::-1], )

    ax1.set_xlabel("Northing (m)")
    ax1.set_ylabel("Depth (m)")

    ax0.set_ylabel("Total field anomaly (nT)")

    ax0.grid(True)
    ax1.grid(True)

    if prob2D.survey.profile == "EW":
        ax1.set_xlabel("Easting (m)")
        ax0.set_xlabel("Easting (m)")

    elif prob2D.survey.profile == "NS":
        ax1.set_xlabel("Northing (m)")
        ax0.set_xlabel("Northing (m)")

    elif prob2D.survey.profile == "45N":
        ax1.set_xlabel("Distance SW-NE (m)")
        ax0.set_xlabel("Distance SW-NE (m)")

            # ax1.invert_yaxis()

    plt.tight_layout()
    plt.show()

    return True


def plotObj3D(p, rx_h, elev, azim, npts2D, xylim,
              profile=None, x0=15., y0=0., fig=None, axs=None, plotSurvey=True):

    # define the survey area
    surveyArea = (-xylim, xylim, -xylim, xylim)
    shape = (npts2D, npts2D)

    xr = np.linspace(-xylim, xylim, shape[0])
    yr = np.linspace(-xylim, xylim, shape[1])
    X, Y = np.meshgrid(xr, yr)
    Z = np.ones(np.shape(X))*rx_h

    rxLoc = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]

    depth = p.z0
    x1, x2 = p.xn[0]-p.xc, p.xn[1]-p.xc
    y1, y2 = p.yn[0]-p.yc, p.yn[1]-p.yc
    z1, z2 = p.zn[0]-p.zc, p.zn[1]-p.zc
    pinc, pdec = p.pinc, p.pdec

    if fig is None:
        fig = plt.figure(figsize=(7, 7))

    if axs is None:
        axs = fig.add_subplot(111, projection='3d')
    plt.rcParams.update({'font.size': 13})

    axs.set_xlim3d(surveyArea[:2])
    axs.set_ylim3d(surveyArea[2:])
#     axs.set_zlim3d(depth+np.array(surveyArea[:2]))
    axs.set_zlim3d(-surveyArea[-1]*1.5, 3)

    # Create a rectangular prism, rotate and plot
    block_xyz = np.asarray([[x1, x1, x2, x2, x1, x1, x2, x2],
                           [y1, y2, y2, y1, y1, y2, y2, y1],
                           [z1, z1, z1, z1, z2, z2, z2, z2]])

    # rot = Utils.mkvc(Utils.dipazm_2_xyz(pinc, pdec))

    # xyz = Utils.rotatePointsFromNormals(block_xyz.T, np.r_[0., 1., 0.], rot,
    #                                     np.r_[p.xc, p.yc, p.zc])

    R = Utils.rotationMatrix(pinc, pdec)

    xyz = R.dot(block_xyz).T

    #print xyz
    # Face 1
    axs.add_collection3d(Poly3DCollection([zip(xyz[:4, 0]+p.xc,
                                               xyz[:4, 1]+p.yc,
                                               xyz[:4, 2]+p.zc)], facecolors='w'))

    # Face 2
    axs.add_collection3d(Poly3DCollection([zip(xyz[4:, 0]+p.xc,
                                               xyz[4:, 1]+p.yc,
                                               xyz[4:, 2]+p.zc)], facecolors='w'))

    # Face 3
    axs.add_collection3d(Poly3DCollection([zip(xyz[[0, 1, 5, 4], 0]+p.xc,
                                               xyz[[0, 1, 5, 4], 1]+p.yc,
                                               xyz[[0, 1, 5, 4], 2]+p.zc)], facecolors='w'))

    # Face 4
    axs.add_collection3d(Poly3DCollection([zip(xyz[[3, 2, 6, 7], 0]+p.xc,
                                               xyz[[3, 2, 6, 7], 1]+p.yc,
                                               xyz[[3, 2, 6, 7], 2]+p.zc)], facecolors='w'))

    # Face 5
    axs.add_collection3d(Poly3DCollection([zip(xyz[[0, 4, 7, 3], 0]+p.xc,
                                               xyz[[0, 4, 7, 3], 1]+p.yc,
                                               xyz[[0, 4, 7, 3], 2]+p.zc)], facecolors='w'))

    # Face 6
    axs.add_collection3d(Poly3DCollection([zip(xyz[[1, 5, 6, 2], 0]+p.xc,
                                               xyz[[1, 5, 6, 2], 1]+p.yc,
                                               xyz[[1, 5, 6, 2], 2]+p.zc)], facecolors='w'))

    axs.set_xlabel('Easting (X; m)')
    axs.set_ylabel('Northing (Y; m)')
    axs.set_zlabel('Depth (Z; m)')
    # axs.invert_zaxis()
    # axs.invert_yaxis()

    if plotSurvey:
        axs.plot(rxLoc[:, 0], rxLoc[:, 1], rxLoc[:, 2], '.g', alpha=0.5)

    if profile == "EW":
        axs.plot(np.r_[surveyArea[:2]], np.r_[0., 0.], np.r_[rx_h, rx_h], 'r-')
    elif profile == "NS":
        axs.plot(np.r_[0., 0.], np.r_[surveyArea[2:]], np.r_[rx_h, rx_h], 'r-')
    elif profile == "45N":
        axs.plot(np.r_[surveyArea[:2]], np.r_[surveyArea[2:]], np.r_[rx_h, rx_h], 'r-')
        # axs.plot(np.r_[surveyArea[2:]], np.r_[0., 0.], np.r_[rx_h, rx_h], 'r-')

    axs.view_init(elev, azim)
    plt.show()

    return True


def linefun(x1, x2, y1, y2, nx, tol=1e-3):
    dx = x2-x1
    dy = y2-y1

    if np.abs(dx) < tol:
        y = np.linspace(y1, y2, nx)
        x = np.ones_like(y)*x1
    elif np.abs(dy) < tol:
        x = np.linspace(x1, x2, nx)
        y = np.ones_like(x)*y1
    else:
        x = np.linspace(x1, x2, nx)
        slope = (y2-y1)/(x2-x1)
        y = slope*(x-x1)+y1
    return x, y


def plogMagSurvey2D(prob2D, susc, Einc, Edec, Bigrf, comp, irt,  Q, rinc, rdec, fig=None, axs1=None, axs2=None):

    import matplotlib.gridspec as gridspec

    # The MAG problem created is stored in result[1]
    # prob2D = Box.result[1]

    if fig is None:
        fig = plt.figure(figsize=(18*1.5,3.4*1.5))

        plt.rcParams.update({'font.size': 14})
        gs1 = gridspec.GridSpec(2, 7)
        gs1.update(left=0.05, right=0.48, wspace=0.05)

    if axs1 is None:
        axs1 = plt.subplot(gs1[:2, :3])

    if axs2 is None:
        axs2 = plt.subplot(gs1[0, 4:])

    axs1.axis("equal")

    prob2D.Bdec, prob2D.Binc, prob2D.Bigrf = Edec, Einc, Bigrf
    prob2D.Q, prob2D.rinc, prob2D.rdec = Q, rinc, rdec
    prob2D.uType, prob2D.mType = comp, 'total'
    prob2D.susc = susc

    # Compute fields from prism
    b_ind, b_rem = prob2D.fields()

    if irt == 'total':
        out = b_ind + b_rem

    elif irt == 'induced':
        out = b_ind

    else:
        out = b_rem

    X, Y = np.meshgrid(prob2D.survey.xr, prob2D.survey.yr)

    dat = axs1.contourf(X,Y, np.reshape(out, (X.shape)).T, 25)
    cb = plt.colorbar(dat, ax=axs1, ticks=np.linspace(out.min(), out.max(), 5))
    cb.set_label("nT")

    axs1.plot(X, Y, '.k')

    # Compute fields on the line by creating a similar mag problem
    prob2D.survey.profile
    if prob2D.survey.profile == "EW":
        x1, x2, y1, y2 = prob2D.survey.xr[0], prob2D.survey.xr[-1], 0., 0.
    elif prob2D.survey.profile == "NS":
        x1, x2, y1, y2 = 0., 0., prob2D.survey.yr[0], prob2D.survey.yr[-1]
    elif prob2D.survey.profile == "45N":
        x1, x2, y1, y2 = prob2D.survey.xr[0], prob2D.survey.xr[-1], prob2D.survey.yr[0], prob2D.survey.yr[-1]

    x, y = linefun(x1, x2, y1, y2, prob2D.survey.npts2D)
    xyz_line = np.c_[x, y, np.ones_like(x)*prob2D.survey.rx_h]
    # Create problem
    prob1D = MAG.problem()
    srvy1D = MAG.survey()
    srvy1D._rxLoc = xyz_line

    prob1D.prism = prob2D.prism
    prob1D.survey = srvy1D

    prob1D.Bdec, prob1D.Binc, prob1D.Bigrf = Edec, Einc, Bigrf
    prob1D.Q, prob1D.rinc, prob1D.rdec = Q, rinc, rdec
    prob1D.uType, prob1D.mType = comp, 'total'
    prob1D.susc = susc

    # Compute fields from prism
    out_linei, out_liner = prob1D.fields()

    #out_linei, out_liner = getField(p, xyz_line, comp, 'total')
    #out_linei = getField(p, xyz_line, comp,'induced')
    #out_liner = getField(p, xyz_line, comp,'remanent')

    out_linet = out_linei+out_liner

    distance = np.sqrt((x-x1)**2.+(y-y1)**2.)


    axs1.plot(x,y, 'w.', ms=3)

    axs1.text(x[0], y[0], 'A', fontsize=16, color='w')
    axs1.text(x[-1], y[-1], 'B', fontsize=16,
             color='w', horizontalalignment='right')

    axs1.set_xlabel('Easting (X; m)')
    axs1.set_ylabel('Northing (Y; m)')
    axs1.set_xlim(X.min(), X.max())
    axs1.set_ylim(Y.min(), Y.max())
    axs1.set_title(irt+' '+comp)

    axs2.plot(distance, out_linei, 'b.-')
    axs2.plot(distance, out_liner, 'r.-')
    axs2.plot(distance, out_linet, 'k.-')
    axs2.set_xlim(distance.min(), distance.max())

    axs2.set_xlabel("Distance (m)")
    axs2.set_ylabel("Magnetic field (nT)")

    axs2.text(distance.min(), out_linei.max()*0.8, 'A', fontsize = 16)
    axs2.text(distance.max()*0.97, out_linei.max()*0.8, 'B', fontsize = 16)
    axs2.legend(("induced", "remanent", "total"), bbox_to_anchor=(0.5, -0.3))
    axs2.grid(True)
    plt.show()

    return True


def fitline(Box):

    def profiledata(data, Binc, Bdec, Bigrf, x0, depth, susc, Q, rinc, rdec, update):

        prob = Box.result[1]
        prob.prism.z0 = -depth

        return plotProfile(prob, x0, data, Binc, Bdec, Bigrf, susc, Q, rinc, rdec)

    Q = widgets.interactive(profiledata, data=widgets.ToggleButtons(options=['MonSt','WedTA','WedSt']),\
             Binc=widgets.FloatSlider(min=-90.,max=90,step=5,value=90,continuous_update=False),\
             Bdec=widgets.FloatSlider(min=-90.,max=90,step=5,value=0,continuous_update=False),\
             Bigrf=widgets.FloatSlider(min=54000.,max=55000,step=10,value=54500,continuous_update=False),\
             x0=widgets.FloatSlider(min=5., max=25., step=0.1, value=15.), \
             depth=widgets.FloatSlider(min=0.,max=2.,step=0.05,value=0.5), \
             susc=widgets.FloatSlider(min=0., max=800.,step=5., value=1.),\
             Q=widgets.FloatSlider(min=0., max=10.,step=0.1, value=0.),\
             rinc=widgets.FloatSlider(min=-180., max=180.,step=1., value=0.),\
             rdec=widgets.FloatSlider(min=-180., max=180.,step=1., value=0.),
             update=widgets.ToggleButton(description='Refresh', value=False) \
             )
    return Q


def ViewMagSurvey2DInd(Box):


    def MagSurvey2DInd(susc, Einc, Edec, Bigrf, comp, irt, Q, rinc, rdec, update):

        # Get the line extent from the 2D survey for now
        prob = Box.result[1]

        return plogMagSurvey2D(prob, susc, Einc, Edec, Bigrf, comp, irt, Q, rinc, rdec)

    out = widgets.interactive (MagSurvey2DInd
                    ,susc=widgets.FloatSlider(min=0,max=200,step=0.1,value=0.1,continuous_update=False) \
                    ,Einc=widgets.FloatSlider(min=-90.,max=90,step=5,value=90,continuous_update=False) \
                    ,Edec=widgets.FloatSlider(min=-90.,max=90,step=5,value=0,continuous_update=False) \
                    ,Bigrf=widgets.FloatSlider(min=53000.,max=55000,step=10,value=54500,continuous_update=False) \
                    ,comp=widgets.ToggleButtons(options=['tf','bx','by','bz'])
                    ,irt=widgets.ToggleButtons(options=['induced','remanent', 'total'])
                    ,Q=widgets.FloatSlider(min=0.,max=10,step=1,value=0,continuous_update=False) \
                    ,rinc=widgets.FloatSlider(min=-90.,max=90,step=5,value=0,continuous_update=False) \
                    ,rdec=widgets.FloatSlider(min=-90.,max=90,step=5,value=0,continuous_update=False) \
                    ,update=widgets.ToggleButton(description='Refresh', value=False) \
                    )
    return out


def Prism(dx, dy, dz, depth, pinc, pdec, npts2D, xylim, rx_h, profile, View_elev, View_azim):
    #p = definePrism(dx, dy, dz, depth,pinc=pinc, pdec=pdec, susc = 1., Einc=90., Edec=0., Bigrf=1e-6)
    p = definePrism()
    p.dx, p.dy, p.dz, p.z0 = dx, dy, dz, -depth
    p.pinc, p.pdec = pinc, pdec

    srvy = MAG.survey()
    srvy.rx_h, srvy.npts2D, srvy.xylim, srvy.profile = rx_h, npts2D, xylim, profile

    # Create problem
    prob = MAG.problem()
    prob.prism = p
    prob.survey = srvy

    return plotObj3D(p, rx_h, View_elev, View_azim, npts2D, xylim, profile=profile), prob

def ViewPrism(dx, dy, dz, depth, xylim=3.):
    elev, azim = 20, 250
    npts2D = 20
    Q = widgets.interactive(Prism \
                            , dx=widgets.FloatSlider(min=1e-4, max=5., step=0.02, value=dx, continuous_update=False) \
                            , dy=widgets.FloatSlider(min=1e-4, max=5., step=0.02, value=dy, continuous_update=False) \
                            , dz=widgets.FloatSlider(min=1e-4, max=100., step=0.02, value=dz, continuous_update=False) \
                            , depth=widgets.FloatSlider(min=0., max=10., step=0.1, value=-depth, continuous_update=False)\
                            , pinc=(-90., 90., 5.) \
                            , pdec=(-90., 90., 5.) \
                            , npts2D=widgets.FloatSlider(min=5, max=100, step=5, value=npts2D, continuous_update=False) \
                            , xylim=widgets.FloatSlider(min=1, max=10, step=1, value=xylim, continuous_update=False) \
                            , rx_h=widgets.FloatSlider(min=0.1, max=2.5, step=0.1, value=rx_h, continuous_update=False) \
                            , profile=widgets.ToggleButtons(options=['EW', 'NS', '45N'])
                            , View_elev=widgets.FloatSlider(min=-90, max=90, step=5, value=elev, continuous_update=False) \
                            , View_azim=widgets.FloatSlider(min=0, max=360, step=5, value=azim, continuous_update=False)
                            )

    return Q
