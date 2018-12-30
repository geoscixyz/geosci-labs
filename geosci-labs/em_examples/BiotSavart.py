from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as sp
from SimPEG import Utils
from scipy.constants import mu_0


def BiotSavartFun(mesh, r_pts, component = 'z'):
    """
        Compute systematrix G using Biot-Savart Law


        G = np.vstack((G1,G2,G3..,Gnpts)

        .. math::

    """
    if r_pts.ndim == 1:
        npts = 1
    else:
        npts = r_pts.shape[0]
    e = np.ones((mesh.nC, 1))
    o = np.zeros((mesh.nC, 1))
    const = mu_0/4/np.pi
    G = np.zeros((npts, mesh.nC*3))

    for i in range(npts):
        if npts == 1:
            r_rx = np.repeat(Utils.mkvc(r_pts).reshape([1,-1]), mesh.nC, axis = 0)
        else:
            r_rx = np.repeat(r_pts[i,:].reshape([1,-1]), mesh.nC, axis = 0)
        r_CC = mesh.gridCC
        r = r_rx-r_CC
        r_abs = np.sqrt((r**2).sum(axis = 1))
        rxind = r_abs==0.
        # r_abs[rxind] = mesh.vol.min()**(1./3.)*0.5
        r_abs[rxind] = 1e20
        Sx = const*Utils.sdiag(mesh.vol*r[:,0]/r_abs**3)
        Sy = const*Utils.sdiag(mesh.vol*r[:,1]/r_abs**3)
        Sz = const*Utils.sdiag(mesh.vol*r[:,2]/r_abs**3)

        # G_temp = sp.vstack((sp.hstack(( o.T,     e.T*Sz, -e.T*Sy)), \
        #                       sp.hstack((-e.T*Sz,  o.T,     e.T*Sx)), \
        #                       sp.hstack((-e.T*Sy,  e.T*Sx,  o.T   ))))
        if component == 'x':
            G_temp = np.hstack(( o.T,     e.T*Sz, -e.T*Sy))
        elif component == 'y':
            G_temp = np.hstack((-e.T*Sz,  o.T,     e.T*Sx))
        elif component == 'z':
            G_temp = np.hstack(( e.T*Sy, -e.T*Sx,  o.T   ))
        G[i,:] = G_temp

    return G

