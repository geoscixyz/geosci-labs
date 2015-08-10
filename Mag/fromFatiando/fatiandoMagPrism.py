from __future__ import division
import numpy as np
import fatiandoUtils as utils
"""
The following code is adapted from the Fatiando software package (http://www.fatiando.org/). We are using the non-cython version of the code (slower, but less instillations required. For faster version, see original Fatiando package.)

Uieda, L, Oliveira Jr, V C, Ferreira, A, Santos, H B; Caparica Jr, J F (2014), Fatiando a Terra: a Python package for modeling and inversion in geophysics. figshare. doi:10.6084/m9.figshare.1115194

"""


"""
Calculate the potential fields of the 3D right rectangular prism.

.. note:: All input units are SI. Output is in conventional units: SI for the
    gravitatonal potential, mGal for gravity, Eotvos for gravity gradients, nT
    for magnetic total field anomalies.

.. note:: The coordinate system of the input parameters is x -> North,
    y -> East and z -> Down.

**Gravity**

The gravitational fields are calculated using the formula of Nagy et al.
(2000). Available functions are:

* :func:`~fatiando.gravmag.prism.potential`
* :func:`~fatiando.gravmag.prism.gx`
* :func:`~fatiando.gravmag.prism.gy`
* :func:`~fatiando.gravmag.prism.gz`
* :func:`~fatiando.gravmag.prism.gxx`
* :func:`~fatiando.gravmag.prism.gxy`
* :func:`~fatiando.gravmag.prism.gxz`
* :func:`~fatiando.gravmag.prism.gyy`
* :func:`~fatiando.gravmag.prism.gyz`
* :func:`~fatiando.gravmag.prism.gzz`

.. warning::

    The gxy, gxz, and gyz components have singularities when the computation
    point is aligned with the corners of the prism on the bottom, east, and
    north sides, respectively. In these cases, the above functions will move
    the computation point slightly to avoid these singularities. Unfortunately,
    this means that the result will not be as accurate **on those points**.


**Magnetic**

Available fields are the total-field anomaly (using the formula of
Bhattacharyya, 1964) and x, y, z components of the magnetic induction:

* :func:`~fatiando.gravmag.prism.tf`
* :func:`~fatiando.gravmag.prism.bx`
* :func:`~fatiando.gravmag.prism.by`
* :func:`~fatiando.gravmag.prism.bz`

**Auxiliary Functions**

Calculates the second derivatives of the function

.. math::

    \phi(x,y,z) = \int\int\int \frac{1}{r}
                  \mathrm{d}\nu \mathrm{d}\eta \mathrm{d}\zeta

with respect to the variables :math:`x`, :math:`y`, and :math:`z`.
In this equation,

.. math::

    r = \sqrt{(x - \nu)^2 + (y - \eta)^2 + (z - \zeta)^2}

and :math:`\nu`, :math:`\eta`, :math:`\zeta` are the Cartesian
coordinates of an element inside the volume of a 3D prism.
These second derivatives are used to calculate
the total field anomaly and the gravity gradient tensor
components.

* :func:`~fatiando.gravmag.prism.kernelxx`
* :func:`~fatiando.gravmag.prism.kernelxy`
* :func:`~fatiando.gravmag.prism.kernelxz`
* :func:`~fatiando.gravmag.prism.kernelyy`
* :func:`~fatiando.gravmag.prism.kernelyz`
* :func:`~fatiando.gravmag.prism.kernelzz`

**References**

Bhattacharyya, B. K. (1964), Magnetic anomalies due to prism-shaped bodies with
arbitrary polarization, Geophysics, 29(4), 517, doi: 10.1190/1.1439386.

Nagy, D., G. Papp, and J. Benedek (2000), The gravitational potential and its
derivatives for the prism: Journal of Geodesy, 74, 552--560,
doi: 10.1007/s001900000116.

----
"""


CM = 10. ** (-7) #: Proportionality constant used in the magnetic method in henry/m (SI)
T2NT = 10. ** (9) #: Conversion factor from tesla to nanotesla


def safe_atan2(y, x):
    """
    Correct the value of the angle returned by arctan2 to match the sign of the
    tangent. Also return 0 instead of 2Pi for 0 tangent.
    """
    res = np.arctan2(y, x)
    res[y == 0] = 0
    res[(y > 0) & (x < 0)] -= np.pi
    res[(y < 0) & (x < 0)] += np.pi
    return res


def safe_log(x):
    """
    Return 0 for log(0) because the limits in the formula terms tend to 0
    (see Nagy et al., 2000)
    """
    res = np.log(x)
    res[x == 0] = 0
    return res

def tf(xp, yp, zp, prisms, inc, dec, pmag=None):
    res = np.zeros_like(xp)
    # Calculate the 3 components of the unit vector in the direction of the
    # regional field
    fx, fy, fz = utils.dircos(inc, dec)
    if pmag is not None:
        if isinstance(pmag, float) or isinstance(pmag, int):
            pintensity = pmag
            pmx, pmy, pmz = fx, fy, fz
        else:
            pintensity = np.linalg.norm(pmag)
            pmx, pmy, pmz = np.array(pmag) / pintensity
    for prism in prisms:
        if prism is None or ('magnetization' not in prism.props
                             and pmag is None):
            continue
        if pmag is None:
            mag = prism.props['magnetization']
            if isinstance(mag, float) or isinstance(mag, int):
                intensity = mag
                mx, my, mz = fx, fy, fz
            else:
                intensity = np.linalg.norm(mag)
                mx, my, mz = np.array(mag) / intensity
        else:
            intensity = pintensity
            mx, my, mz = pmx, pmy, pmz
        # First thing to do is make the computation point P the origin of the
        # coordinate system
        x = [prism.x2 - xp, prism.x1 - xp]
        y = [prism.y2 - yp, prism.y1 - yp]
        z = [prism.z2 - zp, prism.z1 - zp]
        # Now calculate the total field anomaly
        for k in range(2):
            intensity *= -1
            z_sqr = z[k]**2
            for j in range(2):
                y_sqr = y[j]**2
                for i in range(2):
                    x_sqr = x[i]**2
                    xy = x[i]*y[j]
                    r_sqr = x_sqr + y_sqr + z_sqr
                    r = np.sqrt(r_sqr)
                    zr = z[k]*r
                    res += ((-1.)**(i + j))*intensity*(
                        0.5*(my*fz + mz*fy) *
                        safe_log((r - x[i]) / (r + x[i]))
                        + 0.5*(mx*fz + mz*fx) *
                        safe_log((r - y[j]) / (r + y[j]))
                        - (mx*fy + my*fx)*safe_log(r + z[k])
                        - mx*fx*safe_atan2(xy, x_sqr + zr + z_sqr)
                        - my*fy*safe_atan2(xy, r_sqr + zr - x_sqr)
                        + mz*fz*safe_atan2(xy, zr))
    res *= CM*T2NT
    return res


def bx(xp, yp, zp, prisms, pmag=None):
    if pmag is not None:
        mx, my, mz = pmag
    bx = np.zeros_like(xp)
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mx, my, mz = prism.props['magnetization']
        v1 = kernelxx(xp, yp, zp, prism)
        v2 = kernelxy(xp, yp, zp, prism)
        v3 = kernelxz(xp, yp, zp, prism)
        bx += (v1*mx + v2*my + v3*mz)
    bx *= CM*T2NT
    return bx


def by(xp, yp, zp, prisms, pmag=None):
    if pmag is not None:
        mx, my, mz = pmag
    by = np.zeros_like(xp)
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mx, my, mz = prism.props['magnetization']
        v2 = kernelxy(xp, yp, zp, prism)
        v4 = kernelyy(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        by += (v2*mx + v4*my + v5*mz)
    by *= CM*T2NT
    return by


def bz(xp, yp, zp, prisms, pmag=None):
    if pmag is not None:
        mx, my, mz = pmag
    bz = np.zeros_like(xp)
    for prism in prisms:
        if (prism is None or
                ('magnetization' not in prism.props and pmag is None)):
            continue
        if pmag is None:
            mx, my, mz = prism.props['magnetization']
        v3 = kernelxz(xp, yp, zp, prism)
        v5 = kernelyz(xp, yp, zp, prism)
        v6 = kernelzz(xp, yp, zp, prism)
        bz += (v3*mx + v5*my + v6*mz)
    bz *= CM*T2NT
    return bz


def kernelxx(xp, yp, zp, prism):
    res = np.zeros(len(xp), dtype=np.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = -safe_atan2(z[k]*y[j], x[i]*r)
                res += ((-1.)**(i + j + k))*kernel
    return res


def kernelyy(xp, yp, zp, prism):
    res = np.zeros(len(xp), dtype=np.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = -safe_atan2(z[k]*x[i], y[j]*r)
                res += ((-1.)**(i + j + k))*kernel
    return res


def kernelzz(xp, yp, zp, prism):
    res = np.zeros(len(xp), dtype=np.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = -safe_atan2(y[j]*x[i], z[k]*r)
                res += ((-1.)**(i + j + k))*kernel
    return res


def kernelxy(xp, yp, zp, prism):
    res = np.zeros(len(xp), dtype=np.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = safe_log(z[k] + r)
                res += ((-1.)**(i + j + k))*kernel
    return res


def kernelxz(xp, yp, zp, prism):
    res = np.zeros(len(xp), dtype=np.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = safe_log(y[j] + r)
                res += ((-1.)**(i + j + k))*kernel
    return res


def kernelyz(xp, yp, zp, prism):
    res = np.zeros(len(xp), dtype=np.float)
    # First thing to do is make the computation point P the origin of the
    # coordinate system
    x = [prism.x2 - xp, prism.x1 - xp]
    y = [prism.y2 - yp, prism.y1 - yp]
    z = [prism.z2 - zp, prism.z1 - zp]
    # Evaluate the integration limits
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                kernel = safe_log(x[i] + r)
                res += ((-1.)**(i + j + k))*kernel
    return res
