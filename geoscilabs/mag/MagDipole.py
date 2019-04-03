import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from ipywidgets import (
    interactive,
    IntSlider,
    widget,
    FloatText,
    FloatSlider,
    ToggleButton,
    VBox,
    HBox,
    Output,
    interactive_output,
    Layout,
    FloatRangeSlider,
    Checkbox
)
def MagneticMonopoleField(obsloc, poleloc=(0.0, 0.0, 0.0), Q=1):
    # relative obs. loc. to pole, assuming pole at origin
    dx, dy, dz = obsloc[0] - poleloc[0], obsloc[1] - poleloc[1], obsloc[2] - poleloc[2]
    r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    Bx = Q * 1e-7 / r ** 2 * dx
    By = Q * 1e-7 / r ** 2 * dy
    Bz = Q * 1e-7 / r ** 2 * dz
    return Bx, By, Bz


def VerticalMagneticLongDipoleLine(
    radius, L, stepsize=0.1, nstepmax=1000, dist_tol=0.5
):
    yloc, zloc = [radius], [0.0]
    dist2pole = np.sqrt(yloc[0] ** 2 + (zloc[0] - L / 2) ** 2)
    # loop to get the lower half
    count = 1
    while (dist2pole > dist_tol) & (count < nstepmax):
        _, By1, Bz1 = MagneticMonopoleField(
            (0.0, yloc[-1], zloc[-1]), (0.0, 0.0, L / 2), Q=1
        )
        _, By2, Bz2 = MagneticMonopoleField(
            (0.0, yloc[-1], zloc[-1]), (0.0, 0.0, -L / 2), Q=-1
        )
        By, Bz = By1 + By2, Bz1 + Bz2
        B = np.sqrt(By ** 2 + Bz ** 2)
        By, Bz = By / B * stepsize, Bz / B * stepsize
        yloc = np.append(yloc, yloc[-1] + By)
        zloc = np.append(zloc, zloc[-1] + Bz)
        dist2pole = np.sqrt(yloc[-1] ** 2 + (zloc[-1] - L / 2) ** 2)
        count += 1
    # mirror to get the upper half
    yloc = np.append(yloc[-1:0:-1], yloc)
    zloc = np.append(-zloc[-1:0:-1], zloc)
    return yloc, zloc


def MagneticLongDipoleLine(dipoleloc, dipoledec, dipoleinc, dipoleL, radii, Nazi=10):
    x0, y0, z0 = dipoleloc[0], dipoleloc[1], dipoleloc[2]

    # rotation matrix
    theta, alpha = -np.pi * (dipoleinc + 90.0) / 180.0, -np.pi * dipoledec / 180.0
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ]
    )
    Rz = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0.0],
            [np.sin(alpha), np.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R = np.dot(Rz, Rx)

    azimuth = np.linspace(0.0, 2 * np.pi, num=Nazi, endpoint=False)
    xloc, yloc, zloc = [], [], []
    for r in radii:
        hloc, vloc = VerticalMagneticLongDipoleLine(r, dipoleL, stepsize=0.5)
        for a in azimuth:
            x, y, z = np.sin(a) * hloc, np.cos(a) * hloc, vloc
            xyz = np.dot(R, np.vstack((x, y, z)))
            xloc.append(xyz[0] + x0)
            yloc.append(xyz[1] + y0)
            zloc.append(xyz[2] + z0)
    return xloc, yloc, zloc


def MagneticLongDipoleField(
    dipoleloc, dipoledec, dipoleinc, dipoleL, obsloc, dipolemoment=1.0
):
    dec, inc, L = np.radians(dipoledec), np.radians(dipoleinc), dipoleL
    x1 = L / 2 * np.cos(inc) * np.sin(dec)
    y1 = L / 2 * np.cos(inc) * np.cos(dec)
    z1 = L / 2 * -np.sin(inc)
    x2, y2, z2 = -x1, -y1, -z1
    Q = dipolemoment * 4e-7 * np.pi / L
    Bx1, By1, Bz1 = MagneticMonopoleField(
        obsloc, (x1 + dipoleloc[0], y1 + dipoleloc[1], z1 + dipoleloc[2]), Q=Q
    )
    Bx2, By2, Bz2 = MagneticMonopoleField(
        obsloc, (x2 + dipoleloc[0], y2 + dipoleloc[1], z2 + dipoleloc[2]), Q=-Q
    )
    return Bx1 + Bx2, By1 + By2, Bz1 + Bz2

def DrawMagneticDipole3D(
    dipoleLoc_X=0.,dipoleLoc_Y=0.,dipoleLoc_Z=-5., dipoledec=0., dipoleinc=0., dipoleL=1., dipolemoment=1.0, B0=53600e-9 , Binc=90., Bdec=0,
    xStart=-6, xEnd=6, yStart=-6,yEnd=6,
    showField =True,showCurve=True,showLocation=True,showStrength=True,
    ifUpdate=True
):
    if(ifUpdate==False):
        return 0;
    dipoleloc=(dipoleLoc_X,dipoleLoc_Y,dipoleLoc_Z);
    B0x = B0*np.cos(np.radians(Binc))*np.sin(np.radians(Bdec))
    B0y = B0*np.cos(np.radians(Binc))*np.cos(np.radians(Bdec))
    B0z = -B0*np.sin(np.radians(Binc))

    # set observation grid
    z =  1. # x, y bounds and elevation
    radii = (2., 5.) # how many layers of field lines for plotting
    ymin=xmin = min(xStart,yStart,z-max(radii)*2)
    ymax=xmax = max(xEnd,yEnd,max(radii)*2)
    profile_x = 0. # x-coordinate of y-profile
    profile_y = 0. # y-coordinate of x-profile
    h = 0.2 # grid interval
    Naz = 10 # number of azimuth

    # get field lines
    linex, liney, linez = MagneticLongDipoleLine(dipoleloc,dipoledec,dipoleinc,dipoleL,radii,Naz)

    # get map
    xi, yi = np.meshgrid(np.r_[xStart:xEnd+h:h], np.r_[yStart:yEnd+h:h])
    x1, y1 = xi.flatten(), yi.flatten()
    z1 = np.full(x1.shape,z)
    Bx, By, Bz = np.zeros(len(x1)), np.zeros(len(x1)), np.zeros(len(x1))

    for i in np.arange(len(x1)):
        Bx[i], By[i], Bz[i] = MagneticLongDipoleField(dipoleloc,dipoledec,dipoleinc,dipoleL,(x1[i],y1[i],z1[i]),dipolemoment)
    Ba1 = np.dot(np.r_[B0x,B0y,B0z], np.vstack((Bx,By,Bz)))

    # get x-profile
    x2 = np.r_[xStart:xEnd+h:h]
    y2, z2 = np.full(x2.shape,profile_y), np.full(x2.shape,z)
    Bx, By, Bz = np.zeros(len(x2)), np.zeros(len(x2)), np.zeros(len(x2))
    for i in np.arange(len(x2)):
        Bx[i], By[i], Bz[i] = MagneticLongDipoleField(dipoleloc,dipoledec,dipoleinc,dipoleL,(x2[i],y2[i],z2[i]),dipolemoment)
    Ba2 = np.dot(np.r_[B0x,B0y,B0z], np.vstack((Bx,By,Bz)))

    # get y-profile
    y3 = np.r_[yStart:yEnd+h:h]
    x3, z3 = np.full(y3.shape,profile_x), np.full(y3.shape,z)
    Bx, By, Bz = np.zeros(len(x3)), np.zeros(len(x3)), np.zeros(len(x3))
    for i in np.arange(len(x3)):
        Bx[i], By[i], Bz[i] = MagneticLongDipoleField(dipoleloc,dipoledec,dipoleinc,dipoleL,(x3[i],y3[i],z3[i]),dipolemoment)
    Ba3 = np.dot(np.r_[B0x,B0y,B0z], np.vstack((Bx,By,Bz)))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    if showField:
        # plot field lines
        for lx,ly,lz in zip(linex,liney,linez):
            ax.plot(lx,ly,lz,'-',markersize=1, zorder=100)

    if showLocation:
        ax.scatter(x1,y1,z1,s=2,alpha=0.3)
    
    if showStrength:
        # plot map
        Bt = Ba1.reshape(xi.shape)*1e9 # contour and color scale in nT
        c = ax.contourf(xi,yi,Bt,alpha=1,zdir='z',offset=xmin,cmap='jet',
                        levels=np.linspace(Bt.min(),Bt.max(),50,endpoint=True),zorder = 0)
        fig.colorbar(c)

    if showCurve:
         # auto-scaling for profile plot
        ptpmax = np.max((Ba2.ptp(),Ba3.ptp())) # dynamic range
        autoscaling = np.max(radii) / ptpmax
        # plot x-profile
        ax.scatter(x2,y2,z2,s=2,c='black',alpha=0.3)
        ax.plot(x2,Ba2*autoscaling,zs=ymax,c='black',zdir='y')

        # plot y-profile
        ax.scatter(x3,y3,z3,s=2,c='black',alpha=0.3)
        ax.plot(y3,Ba3*autoscaling,zs=xmin,c='black',zdir='x')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    # ax.set_zlim(-(xmax-xmin)/2, (xmax-xmin)/2)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(xmin,xmax)
    ax.set_zlim(xmin,xmax)
    

# draw the widgets
def interact_pic():
    dipoleLoc_X=FloatText(
            value=0,
            description='DipoleX',
            disabled=False
    )
    dipoleLoc_Y=FloatText(
        value=0,
        description='DipoleY',
        disabled=False
    )
    dipoleLoc_Z=FloatText(
        value=-5.,
        description='DipoleZ',
        disabled=False
    )
    dipoleL=FloatText(
        value=1.,
        description='Length',
        disabled=False
    )
    dipoleDec=FloatText(
        value=0,
        description='dipoleDec',
        disabled=False
    )
    dipoleInc=FloatText(
        value=0.,
        description='dipoleInc',
        disabled=False
    )
    dipolemoment=FloatText(
        value=1.,
        description='Moment',
        disabled=False
    )
    B0 = FloatText(
        value=53600e-9,
        description=r"$B_0$",
        disabled=False
    )
    Binc = FloatText(
        value=90,
        description="Binc",
        disabled=False
    )
    Bdec = FloatText(
        value=0,
        description="Bdec",
        disabled=False
    )

    xStart = FloatText(
        value=-6,
        description="xStart",
        disabled=False
    )
    xEnd = FloatText(
        value=6,
        description="xEnd",
        disabled=False
    )
    yStart = FloatText(
        value=-6,
        description="yStart",
        disabled=False
    )
    yEnd = FloatText(
        value=6,
        description="yEnd",
        disabled=False
    )

    showField = Checkbox(
        value=True,
        description='Show the Field',
        disabled=False
    )
    showCurve = Checkbox(
        value=True,
        description='show the profiles',
        disabled=False
    )
    showLocation = Checkbox(
        value=True,
        description='show the locations of measurement',
        disabled=False
    )
    showStrength = Checkbox(
        value=True,
        description='Show the Field Strength Figure',
        disabled=False
    )
    ifUpdate = Checkbox(
        value=True,
        description='Update Fig Real Time',
        disabled=False
    )
    # ifUpdate = ToggleButton(
    #     value=True,
    #     description="UpdateRealTime",
    #     disabled=False,
    #     button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
    #     tooltip="Click me"
    # )
    

    out1 = HBox([dipoleLoc_X,dipoleLoc_Y,dipoleLoc_Z])
    out2 = HBox([dipoleDec, dipoleInc])
    out3 = HBox([dipoleL])
    out4 = HBox([dipolemoment])
    out5 = HBox([B0,Binc,Bdec])
    out6 = HBox([xStart,xEnd])
    out7 = HBox([yStart,yEnd])
    out8 = VBox([showField,showCurve, showLocation, showStrength,ifUpdate])
    out = interactive_output(
        DrawMagneticDipole3D,
        {
            "dipoleLoc_X":dipoleLoc_X,
            "dipoleLoc_Y":dipoleLoc_Y,
            "dipoleLoc_Z":dipoleLoc_Z,
            "dipoledec": dipoleDec,
            "dipoleinc": dipoleInc,
            "dipoleL": dipoleL,
            "dipolemoment": dipolemoment,
            "B0":B0,
            "Binc":Binc,
            "Bdec":Bdec,
            "xStart":xStart,
            "xEnd":xEnd,
            "yStart":yStart,
            "yEnd":yEnd,
            "showField":showField,
            "showCurve":showCurve,
            "showLocation":showLocation,
            "showStrength":showStrength,
            "ifUpdate":ifUpdate
        },
    )
    return VBox([out1,out2,out3,out4,out5,out6,out7,HBox([out, out8])])