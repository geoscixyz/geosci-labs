import numpy as np


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
