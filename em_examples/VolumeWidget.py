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

from .Base import widgetify


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def polyplane(verts, alpha=0.2, color="green"):
    poly = Poly3DCollection([list(verts)])
    poly.set_alpha(alpha)
    poly.set_facecolor(color)
    return poly


def plotObj3D(
    fig=None, ax=None, offset_plane=0., offset_rx=50., elev=20, azim=300,
    X1=-20, X2=80, Y1=-50, Y2=50, Z1=-50, Z2=50, nRx=10, plane="XZ", **kwargs
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

    if plane == "XZ":
        x = np.r_[X1, X2, X2, X1, X1]
        y = np.ones(5)*offset_plane
        z = np.r_[Z1, Z1, Z2, Z2, Z1]
        verts = zip(x, y, z)

    elif plane == "YZ":
        x = np.ones(5)*offset_plane
        y = np.r_[Y1, Y2, Y2, Y1, Y1]
        z = np.r_[Z1, Z1, Z2, Z2, Z1]
        verts = zip(x, y, z)

    polya = polyplane(verts)

    x = np.r_[X1, X2, X2, X1, X1]
    y = np.r_[Y1, Y1, Y2, Y2, Y1]
    z = np.ones(5)*0.
    verts = zip(x, y, z)

    polyb = polyplane(verts, color="grey")

    x = np.ones(5)*50.
    y = np.r_[Y1, Y2, Y2, Y1, Y1]
    z = np.r_[Z1, Z1, Z2, Z2, Z1]
    plt.plot(x, y, z, "k:", lw=1)

    x = np.r_[X1, X2, X2, X1, X1]
    y = np.ones(5)*0.
    z = np.r_[Z1, Z1, Z2, Z2, Z1]
    plt.plot(x, y, z, "k:", lw=1)

    ax.add_collection3d(polya)
    ax.add_collection3d(polyb)

    ax.plot(np.ones(2)*xoffset_rx, np.ones(2)*yoffset_rx, np.r_[Z1, Z2], 'k-', lw=1)
    ax.plot(np.ones(2)*0., np.ones(2)*0., np.r_[Z1, Z2], 'k-', lw=1)
    ax.plot(xoffset_rx*np.ones(nRx), yoffset_rx*np.ones(nRx), np.linspace(Z1, Z2, nRx), "r.", ms=4)
    ax.plot(np.linspace(X1, X2, nRx), np.zeros(nRx), np.zeros(nRx), "b-", ms=4)

    a = Arrow3D(
        [0, 0], [0, 0], [8, -8], mutation_scale=10,
        lw=2, arrowstyle="<->", color="r"
    )
    ax.add_artist(a)

    ax.text(0, 0, Z2, "Tx hole")
    ax.text(xoffset_rx, yoffset_rx, Z2, "Rx hole")
    ax.text(X2, 0, 0, "Tx profile")

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    ax.set_xlim3d(X1, X2)
    ax.set_ylim3d(Y1, Y2)
    ax.set_zlim3d(Z1, Z2)

    ax.view_init(elev, azim)

    plt.show()
    return ax


def InteractivePlanes(planevalue="XZ", offsetvalue=0.):
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
            min=-100, max=100, step=5., value=offsetvalue,
            continuous_update=False
        ),
        # ,X0=widgets.FloatText(value=-20) \
        # ,Y0=widgets.FloatText(value=-50.) \
        # ,Z0=widgets.FloatText(value=-50.) \
        nRx=widgets.IntSlider(
            min=4, max=200, step=2, value=40, continuous_update=False
        ),
        Plane=widgets.ToggleButtons(options=['XZ', 'YZ'], value=planevalue)
    )
    return out

if __name__ == '__main__':

    plotObj3D()
