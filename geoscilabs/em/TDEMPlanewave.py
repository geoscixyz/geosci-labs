from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
import numpy as np
from scipy.constants import mu_0, pi, epsilon_0
import numpy as np
from SimPEG import Utils

def E_field_from_SheetCurruent(XYZ, srcLoc, sig, t, E0=1., orientation='X', kappa=0., epsr=1.):
    """
        Computing Analytic Electric fields from Plane wave in a Wholespace
        TODO:
            Add description of parameters
    """

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & t.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single frequency can be specified.")

    mu = mu_0*(1+kappa)

    if orientation == "X":
        z = XYZ[:, 2]
        bunja = -E0*(mu*sig)**0.5 * z * np.exp(-(mu*sig*z**2) / (4*t))
        bunmo = 2 * np.pi**0.5 * t**1.5
        Ex = bunja / bunmo
        Ey = np.zeros_like(z)
        Ez = np.zeros_like(z)
        return Ex, Ey, Ez
    else:
        raise NotImplementedError()

def H_field_from_SheetCurruent(XYZ, srcLoc, sig, t, E0=1., orientation='X', kappa=0., epsr=1.):
    """
        Plane wave propagating downward (negative z (depth))
    """

    XYZ = Utils.asArray_N_x_Dim(XYZ, 3)
    # Check
    if XYZ.shape[0] > 1 & t.shape[0] > 1:
        raise Exception("I/O type error: For multiple field locations only a single frequency can be specified.")

    mu = mu_0*(1+kappa)
    if orientation == "X":
        z = XYZ[:, 2]
        Hx = np.zeros_like(z)
        Hy = E0 * np.sqrt(sig / (np.pi*mu*t))*np.exp(-(mu*sig*z**2) / (4*t))
        Hz = np.zeros_like(z)
        return Hx, Hy, Hz
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    pass
