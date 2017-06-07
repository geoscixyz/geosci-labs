from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
from ipywidgets import *
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from .VolumeWidget import Arrow3D, polyplane
from .Base import widgetify


def plotObj3D(
    fig=None, ax=None, offset_plane=0., offset_rx=50., elev=20, azim=300,
    X1=-500., X2=500, Y1=-500, Y2=500, Z1=-1000, Z2=0, nRx=10, plane="XZ",
    **kwargs
):
    plt.rcParams.update({'font.size': 13})

    # define the survey area
    if fig is None:
        fig = plt.figure(figsize=(7, 7))

    if ax is None:
        ax = fig.add_subplot(111, projection='3d')

    # fixed
    xoffset_rx = offset_rx
    yoffset_rx = 0.

    # XZ plane
    x = np.r_[X1, X2, X2, X1, X1]
    y = np.ones(5)*offset_plane
    z = np.r_[Z1, Z1, Z2, Z2, Z1]
    verts = zip(x, y, z)
    polya = polyplane(verts, color="red")
    plt.plot(x, y, z, "k:", lw=1)

    # YZ plane
    x = np.ones(5)*offset_plane
    y = np.r_[Y1, Y2, Y2, Y1, Y1]
    z = np.r_[Z1, Z1, Z2, Z2, Z1]
    verts = zip(x, y, z)
    polyb = polyplane(verts, color="blue")
    plt.plot(x, y, z, "k:", lw=1)

    x = np.r_[X1, X2, X2, X1, X1]
    y = np.r_[Y1, Y1, Y2, Y2, Y1]
    z = np.ones(5)*0.
    verts = zip(x, y, z)

    polyc = polyplane(verts, color="grey", alpha=0.4)

    ax.add_collection3d(polya)
    ax.add_collection3d(polyb)
    ax.add_collection3d(polyc)

    # ax.plot(np.ones(2)*xoffset_rx, np.ones(2)*yoffset_rx, np.r_[Z1, Z2], 'k-', lw=1)
    ax.plot(np.ones(2)*0., np.ones(2)*0., np.r_[Z1, Z2], 'k--', lw=2)
    # ax.plot(xoffset_rx*np.ones(nRx), yoffset_rx*np.ones(nRx), np.linspace(Z1, Z2, nRx), "r.", ms=4)
    # ax.plot(np.linspace(X1, X2, nRx), np.zeros(nRx), np.zeros(nRx), "b-", ms=4)

    a = Arrow3D(
        [0, 0], [0, 0], [-200, -500], mutation_scale=10,
        lw=3, arrowstyle="->", color="k"
    )
    jx = Arrow3D(
        [-200, 200], [0, 0], [0, 0], mutation_scale=10,
        lw=3, arrowstyle="->", color="r"
    )
    ex = Arrow3D(
        [0, 400], [0, 0], [-200, -200], mutation_scale=10,
        lw=2, arrowstyle="->", color="r"
    )
    hy = Arrow3D(
        [0, 0], [0, 400], [-200, -200], mutation_scale=10,
        lw=2, arrowstyle="->", color="b"
    )

    ax.add_artist(a)
    ax.add_artist(jx)
    ax.add_artist(ex)
    ax.add_artist(hy)

    ax.text(0, 0+100., -500, "Wave propagation")
    ax.text(400, 0, -200, "$E_x$", color="red")
    ax.text(0, 400, -200, "$H_y$", color="blue")
    ax.text(0, -300, Z2+200, "Current sheet ($I_x$)", color="red")
    # ax.text(xoffset_rx, yoffset_rx, Z2, "Rx hole")
    # ax.text(X2, 0, 0, "Tx profile")

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    ax.set_xlim3d(X1, X2)
    ax.set_ylim3d(Y1, Y2)
    ax.set_zlim3d(Z1, Z2+100.)

    ax.view_init(elev, azim)

    plt.show()
    return ax


def InteractivePlanes():
    def foo(Plane, Offset, nRx):
        X0, Y0, Z0 = -20, -50, -50
        X2, Y2, Z2 = X0+100., Y0+100., Z0+100.
        return plotObj3D(
            offset_plane=Offset, X1=X0, X2=X2, Y1=Y0, Y2=Y2, Z1=Z0, Z2=Z2,
            nRx=nRx, plane=Plane
        )

    out = widgetify(
        foo,
        Offset=widgets.FloatSlider(
            min=-100, max=100, step=5., value=50., continuous_update=False
        ),
        # ,X0=widgets.FloatText(value=-20) \
        # ,Y0=widgets.FloatText(value=-50.) \
        # ,Z0=widgets.FloatText(value=-50.) \
        nRx=widgets.IntSlider(
            min=4, max=200, step=2, value=40, continuous_update=False
        ),
        Plane=widgets.ToggleButtons(options=['XZ', 'YZ'], value="YZ")
    )
    return out

if __name__ == '__main__':

    plotObj3D()
