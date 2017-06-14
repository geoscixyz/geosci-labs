import numpy as np

def crossProd(v0,v1):
    """
        Cross product of 2 vectors

        :param numpy.array v0: vector of length 3
        :param numpy.array v1: vector of length 3
        :rtype: numpy.array
        :return: cross product of v0,v1
    """
    # ensure both n0, n1 are vectors of length 1
    assert len(v0) == 3, "Length of v0 should be 3"
    assert len(v1) == 3, "Length of v1 should be 3"

    v2 = np.zeros(3,dtype=float)

    v2[0] = v0[1]*v1[2] - v1[1]*v0[2]
    v2[1] = v1[0]*v0[2] - v0[0]*v1[2]
    v2[2] = v0[0]*v1[1] - v1[0]*v0[1]

    return v2

def rotationMatrixFromNormals(v0,v1,tol=1e-20):
    """
        Performs the minimum number of rotations to define a rotation from the direction indicated by the vector n0 to the direction indicated by n1.
        The axis of rotation is n0 x n1
        https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

        :param numpy.array v0: vector of length 3
        :param numpy.array v1: vector of length 3
        :param tol = 1e-20: tolerance. If the norm of the cross product between the two vectors is below this, no rotation is performed
        :rtype: numpy.array, 3x3
        :return: rotation matrix which rotates the frame so that n0 is aligned with n1

    """

    # ensure both n0, n1 are vectors of length 1
    assert len(v0) == 3, "Length of n0 should be 3"
    assert len(v1) == 3, "Length of n1 should be 3"

    # ensure both are true normals
    n0 = v0*1./np.linalg.norm(v0)
    n1 = v1*1./np.linalg.norm(v1)

    n0dotn1 = n0.dot(n1)

    # define the rotation axis, which is the cross product of the two vectors
    rotAx = crossProd(n0,n1)

    if np.linalg.norm(rotAx) < tol:
        return np.eye(3,dtype=float)

    rotAx *= 1./np.linalg.norm(rotAx)

    cosT = n0dotn1/(np.linalg.norm(n0)*np.linalg.norm(n1))
    sinT = np.sqrt(1.-n0dotn1**2)

    ux = np.array([[0., -rotAx[2], rotAx[1]], [rotAx[2], 0., -rotAx[0]], [-rotAx[1], rotAx[0], 0.]],dtype=float)

    return np.eye(3,dtype=float) + sinT*ux + (1.-cosT)*(ux.dot(ux))


def rotatePointsFromNormals(XYZ,n0,n1,x0=np.r_[0.,0.,0.]):
    """
        rotates a grid so that the vector n0 is aligned with the vector n1

        :param numpy.array n0: vector of length 3, should have norm 1
        :param numpy.array n1: vector of length 3, should have norm 1
        :param numpy.array x0: vector of length 3, point about which we perform the rotation
        :rtype: numpy.array, 3x3
        :return: rotation matrix which rotates the frame so that n0 is aligned with n1
    """

    R = rotationMatrixFromNormals(n0, n1)

    assert XYZ.shape[1] == 3, "Grid XYZ should be 3 wide"
    assert len(x0) == 3, "x0 should have length 3"

    return (XYZ - x0).dot(R.T) + x0

def mkvc(x, numDims=1):
    """Creates a vector with the number of dimension specified

    e.g.::

        a = np.array([1, 2, 3])

        mkvc(a, 1).shape
            > (3, )

        mkvc(a, 2).shape
            > (3, 1)

        mkvc(a, 3).shape
            > (3, 1, 1)

    """
    if type(x) == np.matrix:
        x = np.array(x)

    if hasattr(x, 'tovec'):
        x = x.tovec()

    assert isinstance(x, np.ndarray), "Vector must be a numpy array"

    if numDims == 1:
        return x.flatten(order='F')
    elif numDims == 2:
        return x.flatten(order='F')[:, np.newaxis]
    elif numDims == 3:
        return x.flatten(order='F')[:, np.newaxis, np.newaxis]


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

    M = np.zeros((1, 3))

    # Modify azimuth from North to Cartesian-X
    azm_X = (450. - np.asarray(azm_N)) % 360.

    # The inclination is a clockwise rotation
    # around x-axis to honor the convention of positive down
    I = -np.deg2rad(np.asarray(dip))
    D = np.deg2rad(azm_X)

    M[:, 0] = np.cos(I) * np.cos(D)
    M[:, 1] = np.cos(I) * np.sin(D)
    M[:, 2] = np.sin(I)

    return M.T


def rotationMatrix(inc, dec, normal=True):
    """
        Take an inclination and declination angle and return a rotation matrix

    """

    phi = -np.deg2rad(np.asarray(inc))
    theta = -np.deg2rad(np.asarray(dec))

    Rx = np.asarray([[1, 0, 0],
                    [0, np.cos(phi), -np.sin(phi)],
                    [0, np.sin(phi), np.cos(phi)]])

    Rz = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])

    if normal:
        R = Rz.dot(Rx)
    else:
        R = Rx.dot(Rz)

    return R
