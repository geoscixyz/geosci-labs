import numpy as np


def rotationMatrix(inc, dec, normal=True):
    """
        Take an inclination and declination angle and return a rotation matrix

    """

    phi = -np.deg2rad(np.asarray(inc))
    theta = -np.deg2rad(np.asarray(dec))

    Rx = np.asarray(
        [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
    )

    Rz = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    if normal:
        R = Rz.dot(Rx)
    else:
        R = Rx.dot(Rz)

    return R


def dipazm_2_xyz(dip, azm_N):
    """
    dipazm_2_xyz(dip,azm_N)

    Function converting degree angles for dip and azimuth from north to a
    3-components in cartesian coordinates.

    INPUT
    dip     : Value or vector of dip from horizontal in DEGREE
    azm_N   : Value or vector of azimuth from north in DEGREE

    OUTPUT
    M       : [n-by-3] Array of xyz components of a unit vector in cartesian

    Created on Dec, 20th 2015

    @author: dominiquef
    """

    # Modify azimuth from North to Cartesian-X
    azm_X = (450.0 - azm_N) % 360.0

    D = np.deg2rad(np.asarray(dip))
    I = np.deg2rad(azm_X)

    M = np.zeros(3)
    M[0] = np.cos(D) * np.cos(I)
    M[1] = np.cos(D) * np.sin(I)
    M[2] = np.sin(D)

    return M
