import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from ipywidgets import interactive, IntSlider, widget, FloatText, FloatSlider, Checkbox


def fempipeWidget(alpha, pipedepth):
    respEW, respNS, X, Y = fempipe(alpha, pipedepth)

    plt.figure(figsize=(12, 9))
    ax0 = plt.subplot2grid((2, 2), (0, 0))
    ax1 = plt.subplot2grid((2, 2), (0, 1))
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    dat0 = ax0.imshow(respEW.real * 100, extent=[X.min(), X.max(), Y.min(), Y.max()])
    dat1 = ax1.imshow(respNS.real * 100, extent=[X.min(), X.max(), Y.min(), Y.max()])
    plt.colorbar(dat0, ax=ax0)
    plt.colorbar(dat1, ax=ax1)
    ax0.set_title("In-phase EW boom (%)", fontsize=12)
    ax1.set_title("In-phase NS boom (%)", fontsize=12)
    ax0.set_xlabel("Easting (m)", fontsize=12)
    ax1.set_xlabel("Easting (m)", fontsize=12)
    ax0.set_ylabel("Northing (m)", fontsize=12)
    ax1.set_ylabel("Northing (m)", fontsize=12)
    ax0.plot(np.r_[0.0, 0.0], np.r_[-10.0, 10.0], "k--", lw=2)
    ax1.plot(np.r_[0.0, 0.0], np.r_[-10.0, 10.0], "k--", lw=2)

    ax2.plot(Y[:, 20], respEW[:, 20].real, "k.-")
    ax2.plot(Y[:, 20], respEW[:, 20].imag, "k--")
    ax2.plot(Y[:, 20], respNS[:, 20].real, "r.-")
    ax2.plot(Y[:, 20], respNS[:, 20].imag, "r--")
    ax2.legend(
        (
            "In-phase EW boom",
            "Out-of-phase EW boom",
            "In-phase NS boom",
            "Out-of-phase NS boom",
        ),
        loc=4,
    )
    ax2.grid(True)
    ax2.set_ylabel("Hs/Hp (%)", fontsize=16)
    ax2.set_xlabel("Northing (m)", fontsize=16)
    ax2.set_title("Northing profile line at Easting 0 m", fontsize=16)

    plt.tight_layout()
    plt.show()


def fempipe(a, pipedepth):
    """
        EOSC350 forward modeling of EM-31 responses with pipeline model
        Only two adjustable parameters: alpha and depth of pipe below surface
        Pipeline oriented W-E (many small loops lined up)
        forward model EW ans NS boom configurations
        Plot in-phase maps of EW and NS boom
        Plot NS profile
    """

    freq = 9800
    L = 0.1
    s = 3.6
    R = 2 * np.pi * freq * L / a
    # fa = (1j * a) / (1 + 1j * a)
    # tau = L / R
    boomheight = 1.0
    Npipe = 20
    xmax = 10.0
    npts = 100

    pipeloc = np.c_[
        np.linspace(-10, 10, Npipe), np.zeros(Npipe), np.zeros(Npipe) - pipedepth
    ]
    pipeloc = np.vstack((pipeloc, pipeloc))
    pipeangle1 = np.c_[np.zeros(Npipe) + 90, np.zeros(Npipe) + 0]
    # pipeangle2 = np.c_[np.zeros(Npipe) + 90, np.zeros(Npipe) + 90]  # .. what's this?
    pipeangle3 = np.c_[np.zeros(Npipe) + 0, np.zeros(Npipe) + 0]
    pipeangle = np.vstack((pipeangle1, pipeangle3))

    x = np.linspace(-xmax, xmax, num=npts)
    y = x.copy()

    X, Y = np.meshgrid(x, y)
    XY = np.c_[X.flatten(), Y.flatten()]

    loop1loc_NS = np.c_[XY[:, 0], XY[:, 1] - s / 2, boomheight * np.ones(XY.shape[0])]
    loop3loc_NS = np.c_[XY[:, 0], XY[:, 1] + s / 2, boomheight * np.ones(XY.shape[0])]
    loop1angle = np.c_[np.ones(XY.shape[0]) * 0.0, np.ones(XY.shape[0]) * 0.0]
    loop3angle = np.c_[np.ones(XY.shape[0]) * 0.0, np.ones(XY.shape[0]) * 0.0]
    loop1loc_EW = np.c_[XY[:, 0] - s / 2, XY[:, 1], boomheight * np.ones(XY.shape[0])]
    loop3loc_EW = np.c_[XY[:, 0] + s / 2, XY[:, 1], boomheight * np.ones(XY.shape[0])]
    respEW = 0j
    respNS = 0j
    for q in range(pipeloc.shape[0]):
        loop2loc = np.c_[
            np.ones(XY.shape[0]) * pipeloc[q, 0],
            np.ones(XY.shape[0]) * pipeloc[q, 1],
            np.ones(XY.shape[0]) * pipeloc[q, 2],
        ]
        loop2angle = np.c_[
            np.ones(XY.shape[0]) * pipeangle[q, 0],
            np.ones(XY.shape[0]) * pipeangle[q, 1],
        ]
        respEW += HsHp(
            loop1loc_EW,
            loop1angle,
            loop2loc,
            loop2angle,
            loop3loc_EW,
            loop3angle,
            freq,
            L,
            R,
        )
        respNS += HsHp(
            loop1loc_NS,
            loop1angle,
            loop2loc,
            loop2angle,
            loop3loc_NS,
            loop3angle,
            freq,
            L,
            R,
        )

    return respEW.reshape((npts, npts)), respNS.reshape((npts, npts)), X, Y


def Lij(loopiloc, loopiangle, loopjloc, loopjangle):
    """

        Calculate mnutual inductance of two loops (simplified to magnetic dipole)
        SEG EM Volume II (Page 14): ... Lij as the amount of magnetic flux that
        cuts circuit i due to a unit current in loop j.
        Since we use magnetic dipole model here, the magnetic flux will be the
        magnetic intensity B obtained by Biot-Savart Law.
        Angles in degree
        Inductance in T*m^2/A; Here the current and loop area are both unit.

    """
    xi = loopiloc[:, 0]
    yi = loopiloc[:, 1]
    zi = loopiloc[:, 2]
    xj = loopjloc[:, 0]
    yj = loopjloc[:, 1]
    zj = loopjloc[:, 2]
    thetai = loopiangle[:, 0]
    alphai = loopiangle[:, 1]
    thetaj = loopjangle[:, 0]
    alphaj = loopjangle[:, 1]

    thetai = thetai / 180 * np.pi  # degtorad(thetai)
    alphai = alphai / 180 * np.pi  # degtorad(alphai)
    thetaj = thetaj / 180 * np.pi  # degtorad(thetaj)
    alphaj = alphaj / 180 * np.pi  # degtorad(alphaj)

    # http://en.wikipedia.org/wiki/Magnetic_moment#Magnetic_flux_density_due_to_an_arbitrary_oriented_dipole_moment_at_the_origin
    # assume the dipole at origin, the observation is now at
    x = xi - xj
    y = yi - yj
    z = zi - zj
    # orthogonal decomposition of dipole moment
    p = np.cos(thetaj)
    # vertical
    n = np.sin(thetaj) * np.cos(alphaj)  # y
    m = np.sin(thetaj) * np.sin(alphaj)  # x

    Hx = (
        (
            3.0
            * (m * x + n * y + p * z)
            * x
            / ((x ** 2 + y ** 2 + z ** 2) ** (5.0 / 2))
            - m / ((x ** 2 + y ** 2 + z ** 2) ** (3.0 / 2))
        )
        / 4.0
        / np.pi
    )
    Hy = (
        (
            3.0
            * (m * x + n * y + p * z)
            * y
            / ((x ** 2 + y ** 2 + z ** 2) ** (5.0 / 2))
            - n / ((x ** 2 + y ** 2 + z ** 2) ** (3.0 / 2))
        )
        / 4.0
        / np.pi
    )
    Hz = (
        (
            3.0
            * (m * x + n * y + p * z)
            * z
            / ((x ** 2 + y ** 2 + z ** 2) ** (5.0 / 2))
            - p / ((x ** 2 + y ** 2 + z ** 2) ** (3.0 / 2))
        )
        / 4.0
        / np.pi
    )
    H = np.c_[Hx, Hy, Hz]
    # project B field to normal direction of loop i
    L = (
        H
        * np.c_[
            np.sin(thetai) * np.sin(alphai),
            np.sin(thetai) * np.cos(alphai),
            np.cos(thetai),
        ]
    )
    return L.sum(axis=1)


def HsHp(loop1loc, loop1angle, loop2loc, loop2angle, loop3loc, loop3angle, freq, L, R):

    """

        EM response of 3-loop model
        response = Hs/Hp = - (L12*L23/L22/L13) * (i*a/(1+i*a))

    """

    a = 2.0 * np.pi * freq * L / R

    L12 = L * Lij(loop1loc, loop1angle, loop2loc, loop2angle)
    L23 = L * Lij(loop2loc, loop2angle, loop3loc, loop3angle)
    L13 = Lij(loop1loc, loop1angle, loop3loc, loop3angle)

    response = -(L12 * L23 / L13 / L) * ((1j * a) / (1 + 1j * a))

    return response


def interact_femPipe():
    Q = interactive(
        fempipeWidget,
        alpha=FloatSlider(
            min=0.1, max=5.0, step=0.1, value=1.0, continuous_update=False
        ),
        pipedepth=FloatSlider(
            min=0.5, max=4.0, step=0.1, value=1.0, continuous_update=False
        ),
    )
    return Q


if __name__ == "__main__":
    a = 1.0
    pipedepth = 1.0
    respEW, respNS, X, Y = fempipe(a, pipedepth)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].pcolor(X, Y, respEW.real, 40)
    ax[1].pcolor(X, Y, respNS.real, 40)
    plt.show()
