from scipy.constants import mu_0
import re
import numpy as np
import simpegCoordUtils as Utils


class survey(object):

    rx_h = 1.9
    npts2D = 20
    xylim = 5.
    rxLoc = None

    @property
    def rxLoc(self):
        if getattr(self, '_rxLoc', None) is None:
            # Create survey locations
            X, Y = np.meshgrid(self.xr, self.yr)
            Z = np.ones(np.shape(X))*self.rx_h

            self._rxLoc = np.c_[Utils.mkvc(X), Utils.mkvc(Y), Utils.mkvc(Z)]

        return self._rxLoc

    @property
    def xr(self):
        nx = self.npts2D
        self._xr = np.linspace(-self.xylim, self.xylim, nx)

        return self._xr

    @property
    def yr(self):
        ny = self.npts2D
        self._yr = np.linspace(-self.xylim, self.xylim, ny)

        return self._yr


class problem(object):
    """
            Earth's field:
            - Binc, Bdec : inclination and declination of Earth's mag field
            - Bigrf : amplitude of earth's field in units of nT

        Remnance:
            - Q : Koenigsberger ratio
            - Rinc, Rdec : inclination and declination of remnance in block

    """
    Bdec, Binc, Bigrf = 90., 0., 50000.
    Q, rinc, rdec = 0., 0., 0.
    uType, mType = 'tf', 'induced'
    susc = 1.
    prism = None
    survey = None

    @property
    def Mind(self):
        # Define magnetization direction as sum of induced and remanence
        mind = Utils.dipazm_2_xyz(self.Binc, self.Bdec)
        R = Utils.rotationMatrix(-self.prism.pinc, -self.prism.pdec, normal=False)
        Mind = self.susc*self.Higrf*R.dot(mind)
        # Mind = self.susc*self.Higrf*Utils.dipazm_2_xyz(self.Binc - self.prism.pinc,
        #                                                self.Bdec - self.prism.pdec)
        return Mind

    @property
    def Mrem(self):

        mrem = Utils.dipazm_2_xyz(self.rinc, self.rdec)
        R = Utils.rotationMatrix(-self.prism.pinc, -self.prism.pdec, normal=False)
        Mrem = self.Q*self.susc*self.Higrf * R.dot(mrem)

        # Mrem = self.Q*self.susc*self.Higrf * \
        #        Utils.dipazm_2_xyz(self.rinc - self.prism.pinc, self.rdec - self.prism.pdec)

        return Mrem

    @property
    def Higrf(self):
        Higrf = self.Bigrf * 1e-9 / mu_0

        return Higrf

    @property
    def G(self):

        if getattr(self, '_G', None) is None:
            #print "Computing G"

            # rot = Utils.mkvc(Utils.dipazm_2_xyz(self.prism.pinc, self.prism.pdec))

            # rxLoc = Utils.rotatePointsFromNormals(self.survey.rxLoc, rot, np.r_[0., 1., 0.],
            #                                      np.r_[0, 0, 0])


            xLoc = self.survey.rxLoc[:, 0] - self.prism.xc
            yLoc = self.survey.rxLoc[:, 1] - self.prism.yc
            zLoc = self.survey.rxLoc[:, 2] - self.prism.zc

            R = Utils.rotationMatrix(-self.prism.pinc, -self.prism.pdec, normal=False)

            rxLoc = R.dot(np.c_[xLoc, yLoc, zLoc].T).T

            rxLoc = np.c_[rxLoc[:, 0] + self.prism.xc, rxLoc[:, 1] + self.prism.yc, rxLoc[:, 2] + self.prism.zc]

            # Create the linear forward system
            self._G = Intrgl_Fwr_Op(self.prism.xn, self.prism.yn, self.prism.zn, rxLoc)

        return self._G

    def fields(self):

        if (self.mType == 'induced') or (self.mType == 'total'):

            b = self.G.dot(self.Mind)
            self.fieldi = self.extractFields(b)

        if (self.mType == 'remanent') or (self.mType == 'total'):

            b = self.G.dot(self.Mrem)

            self.fieldr = self.extractFields(b)

        if self.mType == 'induced':
            return self.fieldi
        elif self.mType == 'remanent':
            return self.fieldr
        elif self.mType == 'total':
            return self.fieldi, self.fieldr

    def extractFields(self, bvec):

        nD = bvec.shape[0]/3
        bvec = np.reshape(bvec, (3, nD))

        # rot = Utils.mkvc(Utils.dipazm_2_xyz(-self.prism.pinc, -self.prism.pdec))

        # bvec = Utils.rotatePointsFromNormals(bvec.T, rot, np.r_[0., 1., 0.],
        #                                      np.r_[0, 0, 0]).T

        R = Utils.rotationMatrix(self.prism.pinc, self.prism.pdec)
        bvec = R.dot(bvec)

        if self.uType == 'bx':
            u = Utils.mkvc(bvec[0, :])

        if self.uType == 'by':
            u = Utils.mkvc(bvec[1, :])

        if self.uType == 'bz':
            u = Utils.mkvc(bvec[2, :])

        if self.uType == 'tf':
            # Projection matrix
            Ptmi = Utils.dipazm_2_xyz(self.Binc, self.Bdec).T

            u = Utils.mkvc(Ptmi.dot(bvec))

        return u


def Intrgl_Fwr_Op(xn, yn, zn, rxLoc):

    """

    Magnetic forward operator in integral form

    flag        = 'ind' | 'full'

      1- ind : Magnetization fixed by user

      3- full: Full tensor matrix stored with shape([3*ndata, 3*nc])

    Return
    _G = Linear forward modeling operation

     """

    yn2, xn2, zn2 = np.meshgrid(yn[1:], xn[1:], zn[1:])
    yn1, xn1, zn1 = np.meshgrid(yn[0:-1], xn[0:-1], zn[0:-1])

    Yn = np.c_[Utils.mkvc(yn1), Utils.mkvc(yn2)]
    Xn = np.c_[Utils.mkvc(xn1), Utils.mkvc(xn2)]
    Zn = np.c_[Utils.mkvc(zn1), Utils.mkvc(zn2)]

    ndata = rxLoc.shape[0]

    # Pre-allocate forward matrix
    G = np.zeros((int(3*ndata), 3))

    for ii in range(ndata):

        tx, ty, tz = get_T_mat(Xn, Yn, Zn, rxLoc[ii, :])

        G[ii, :] = tx / 1e-9 * mu_0
        G[ii+ndata, :] = ty / 1e-9 * mu_0
        G[ii+2*ndata, :] = tz / 1e-9 * mu_0

    return G


def get_T_mat(Xn, Yn, Zn, rxLoc):
    """
    Load in the active nodes of a tensor mesh and computes the magnetic tensor
    for a given observation location rxLoc[obsx, obsy, obsz]

    INPUT:
    Xn, Yn, Zn: Node location matrix for the lower and upper most corners of
                all cells in the mesh shape[nC,2]
    M
    OUTPUT:
    Tx = [Txx Txy Txz]
    Ty = [Tyx Tyy Tyz]
    Tz = [Tzx Tzy Tzz]

    where each elements have dimension 1-by-nC.
    Only the upper half 5 elements have to be computed since symetric.
    Currently done as for-loops but will eventually be changed to vector
    indexing, once the topography has been figured out.

    Created on Oct, 20th 2015

    @author: dominiquef

     """

    eps = 1e-10  # add a small value to the locations to avoid /0

    nC = Xn.shape[0]

    # Pre-allocate space for 1D array
    Tx = np.zeros((1, 3*nC))
    Ty = np.zeros((1, 3*nC))
    Tz = np.zeros((1, 3*nC))

    dz2 = rxLoc[2] - Zn[:, 0] + eps
    dz1 = rxLoc[2] - Zn[:, 1] + eps

    dy2 = rxLoc[1] - Yn[:, 1]  + eps
    dy1 = rxLoc[1] - Yn[:, 0] + eps

    dx2 = rxLoc[0] - Xn[:, 1]  + eps
    dx1 = rxLoc[0] - Xn[:, 0]  + eps

    R1 = (dy2**2 + dx2**2)
    R2 = (dy2**2 + dx1**2)
    R3 = (dy1**2 + dx2**2)
    R4 = (dy1**2 + dx1**2)

    arg1 = np.sqrt(dz2**2 + R2)
    arg2 = np.sqrt(dz2**2 + R1)
    arg3 = np.sqrt(dz1**2 + R1)
    arg4 = np.sqrt(dz1**2 + R2)
    arg5 = np.sqrt(dz2**2 + R3)
    arg6 = np.sqrt(dz2**2 + R4)
    arg7 = np.sqrt(dz1**2 + R4)
    arg8 = np.sqrt(dz1**2 + R3)

    Tx[0, 0:nC] = np.arctan2(dy1 * dz2, (dx2 * arg5)) +\
        - np.arctan2(dy2 * dz2, (dx2 * arg2)) +\
        np.arctan2(dy2 * dz1, (dx2 * arg3)) +\
        - np.arctan2(dy1 * dz1, (dx2 * arg8)) +\
        np.arctan2(dy2 * dz2, (dx1 * arg1)) +\
        - np.arctan2(dy1 * dz2, (dx1 * arg6)) +\
        np.arctan2(dy1 * dz1, (dx1 * arg7)) +\
        - np.arctan2(dy2 * dz1, (dx1 * arg4))

    Ty[0, 0:nC] = np.log((dz2 + arg2) / (dz1 + arg3)) +\
        -np.log((dz2 + arg1) / (dz1 + arg4)) +\
        np.log((dz2 + arg6) / (dz1 + arg7)) +\
        -np.log((dz2 + arg5) / (dz1 + arg8))

    Ty[0, nC:2*nC] = np.arctan2(dx1 * dz2, (dy2 * arg1)) +\
        - np.arctan2(dx2 * dz2, (dy2 * arg2)) +\
        np.arctan2(dx2 * dz1, (dy2 * arg3)) +\
        - np.arctan2(dx1 * dz1, (dy2 * arg4)) +\
        np.arctan2(dx2 * dz2, (dy1 * arg5)) +\
        - np.arctan2(dx1 * dz2, (dy1 * arg6)) +\
        np.arctan2(dx1 * dz1, (dy1 * arg7)) +\
        - np.arctan2(dx2 * dz1, (dy1 * arg8))

    R1 = (dy2**2 + dz1**2)
    R2 = (dy2**2 + dz2**2)
    R3 = (dy1**2 + dz1**2)
    R4 = (dy1**2 + dz2**2)

    Ty[0, 2*nC:] = np.log((dx1 + np.sqrt(dx1**2 + R1)) /
                          (dx2 + np.sqrt(dx2**2 + R1))) +\
        -np.log((dx1 + np.sqrt(dx1**2 + R2)) / (dx2 + np.sqrt(dx2**2 + R2))) +\
        np.log((dx1 + np.sqrt(dx1**2 + R4)) / (dx2 + np.sqrt(dx2**2 + R4))) +\
        -np.log((dx1 + np.sqrt(dx1**2 + R3)) / (dx2 + np.sqrt(dx2**2 + R3)))

    R1 = (dx2**2 + dz1**2)
    R2 = (dx2**2 + dz2**2)
    R3 = (dx1**2 + dz1**2)
    R4 = (dx1**2 + dz2**2)

    Tx[0, 2*nC:] = np.log((dy1 + np.sqrt(dy1**2 + R1)) /
                          (dy2 + np.sqrt(dy2**2 + R1))) +\
        -np.log((dy1 + np.sqrt(dy1**2 + R2)) / (dy2 + np.sqrt(dy2**2 + R2))) +\
        np.log((dy1 + np.sqrt(dy1**2 + R4)) / (dy2 + np.sqrt(dy2**2 + R4))) +\
        -np.log((dy1 + np.sqrt(dy1**2 + R3)) / (dy2 + np.sqrt(dy2**2 + R3)))

    Tz[0, 2*nC:] = -(Ty[0, nC:2*nC] + Tx[0, 0:nC])
    Tz[0, nC:2*nC] = Ty[0, 2*nC:]
    Tx[0, nC:2*nC] = Ty[0, 0:nC]
    Tz[0, 0:nC] = Tx[0, 2*nC:]

    Tx = Tx/(4*np.pi)
    Ty = Ty/(4*np.pi)
    Tz = Tz/(4*np.pi)

    return Tx, Ty, Tz


def plot_obs_2D(rxLoc, d=None, varstr='TMI Obs',
                vmin=None, vmax=None, levels=None, fig=None, axs=None):
    """ Function plot_obs(rxLoc,d)
    Generate a 2d interpolated plot from scatter points of data

    INPUT
    rxLoc       : Observation locations [x,y,z]
    d           : Data vector

    OUTPUT
    figure()

    Created on Dec, 27th 2015

    @author: dominiquef

    """

    from scipy.interpolate import griddata
    import pylab as plt

    # Plot result
    if fig is None:
        fig = plt.figure()

    if axs is None:
        axs = plt.subplot()

    plt.sca(axs)
    plt.scatter(rxLoc[:, 0], rxLoc[:, 1], c='k', s=10)

    if d is not None:

        if (vmin is None):
            vmin = d.min()

        if (vmax is None):
            vmax = d.max()

        # Create grid of points
        x = np.linspace(rxLoc[:, 0].min(), rxLoc[:, 0].max(), 100)
        y = np.linspace(rxLoc[:, 1].min(), rxLoc[:, 1].max(), 100)

        X, Y = np.meshgrid(x, y)

        # Interpolate
        d_grid = griddata(rxLoc[:, 0:2], d, (X, Y), method='linear')

        plt.imshow(d_grid, extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower', vmin=vmin, vmax=vmax, cmap="plasma")
        plt.colorbar(fraction=0.02)

        if levels is None:
            plt.contour(X, Y, d_grid, 10, vmin=vmin, vmax=vmax, cmap="plasma")
        else:
            plt.contour(X, Y, d_grid, levels=levels, colors='r',
                        vmin=vmin, vmax=vmax, cmap="plasma")

    plt.title(varstr)
    plt.gca().set_aspect('equal', adjustable='box')

    return fig

