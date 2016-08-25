"""
The following code is adapted from the Fatiando software package (http://www.fatiando.org/)

Uieda, L, Oliveira Jr, V C, Ferreira, A, Santos, H B; Caparica Jr, J F (2014), Fatiando a Terra: a Python package for modeling and inversion in geophysics. figshare. doi:10.6084/m9.figshare.1115194

"""

from __future__ import division
import numpy as np


# GRIDDER 
def regular(area, shape, z=None):
    """
    Create a regular grid.

    The x directions is North-South and y East-West. Imagine the grid as a
    matrix with x varying in the lines and y in columns.

    Returned arrays will be flattened to 1D with ``numpy.ravel``.

    .. warning::

        As of version 0.4, the ``shape`` argument was corrected to be
        ``shape = (nx, ny)`` instead of ``shape = (ny, nx)``.


    Parameters:

    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * shape
        Shape of the regular grid, ie ``(nx, ny)``.
    * z
        Optional. z coordinate of the grid points. If given, will return an
        array with the value *z*.

    Returns:

    * ``[x, y]``
        Numpy arrays with the x and y coordinates of the grid points
    * ``[x, y, z]``
        If *z* given. Numpy arrays with the x, y, and z coordinates of the grid
        points

    Examples::

        >>> x, y = regular((0, 10, 0, 5), (5, 3))
        >>> x
        array([  0. ,   0. ,   0. ,   2.5,   2.5,   2.5,   5. ,   5. ,   5. ,
                 7.5,   7.5,   7.5,  10. ,  10. ,  10. ])
        >>> x.reshape((5, 3))
        array([[  0. ,   0. ,   0. ],
               [  2.5,   2.5,   2.5],
               [  5. ,   5. ,   5. ],
               [  7.5,   7.5,   7.5],
               [ 10. ,  10. ,  10. ]])
        >>> y.reshape((5, 3))
        array([[ 0. ,  2.5,  5. ],
               [ 0. ,  2.5,  5. ],
               [ 0. ,  2.5,  5. ],
               [ 0. ,  2.5,  5. ],
               [ 0. ,  2.5,  5. ]])
        >>> x, y = regular((0, 0, 0, 5), (1, 3))
        >>> x.reshape((1, 3))
        array([[ 0.,  0.,  0.]])
        >>> y.reshape((1, 3))
        array([[ 0. ,  2.5,  5. ]])
        >>> x, y, z = regular((0, 10, 0, 5), (5, 3), z=-10)
        >>> z.reshape((5, 3))
        array([[-10., -10., -10.],
               [-10., -10., -10.],
               [-10., -10., -10.],
               [-10., -10., -10.],
               [-10., -10., -10.]])


    """
    nx, ny = shape
    x1, x2, y1, y2 = area
    assert x1 <= x2, \
        "Invalid area dimensions {}, {}. x1 must be < x2.".format(x1, x2)
    assert y1 <= y2, \
        "Invalid area dimensions {}, {}. y1 must be < y2.".format(y1, y2)
    xs = np.linspace(x1, x2, nx)
    ys = np.linspace(y1, y2, ny)
    # Must pass ys, xs in this order because meshgrid uses the first argument
    # for the columns
    arrays = np.meshgrid(ys, xs)[::-1]
    if z is not None:
        arrays.append(z*np.ones(nx*ny, dtype=np.float))
    return [i.ravel() for i in arrays]


# MESHER
class GeometricElement(object):

    """
    Base class for all geometric elements.
    """

    def __init__(self, props):
        self.props = {}
        if props is not None:
            for p in props:
                self.props[p] = props[p]

    def addprop(self, prop, value):
        """
        Add a physical property to this geometric element.

        If it already has the property, the given value will overwrite the
        existing one.

        Parameters:

        * prop : str
            Name of the physical property.
        * value : float
            The value of this physical property.

        """
        self.props[prop] = value


class Prism(GeometricElement):

    """
    Create a 3D right rectangular prism.

    .. note:: The coordinate system used is x -> North, y -> East and z -> Down

    Parameters:

    * x1, x2 : float
        South and north borders of the prism
    * y1, y2 : float
        West and east borders of the prism
    * z1, z2 : float
        Top and bottom of the prism
    * props : dict
        Physical properties assigned to the prism.
        Ex: ``props={'density':10, 'magnetization':10000}``

    Examples:

        >>> from fatiando.mesher import Prism
        >>> p = Prism(1, 2, 3, 4, 5, 6, {'density':200})
        >>> p.props['density']
        200
        >>> print p.get_bounds()
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        >>> print p
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6 | density:200
        >>> p = Prism(1, 2, 3, 4, 5, 6)
        >>> print p
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6
        >>> p.addprop('density', 2670)
        >>> print p
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6 | density:2670

    """

    def __init__(self, x1, x2, y1, y2, z1, z2, props=None):
        GeometricElement.__init__(self, props)
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.y1 = float(y1)
        self.y2 = float(y2)
        self.z1 = float(z1)
        self.z2 = float(z2)

    def __str__(self):
        """Return a string representation of the prism."""
        names = [('x1', self.x1), ('x2', self.x2), ('y1', self.y1),
                 ('y2', self.y2), ('z1', self.z1), ('z2', self.z2)]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)

    def get_bounds(self):
        """
        Get the bounding box of the prism (i.e., the borders of the prism).

        Returns:

        * bounds : list
            ``[x1, x2, y1, y2, z1, z2]``, the bounds of the prism

        Examples:

            >>> p = Prism(1, 2, 3, 4, 5, 6)
            >>> print p.get_bounds()
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        """
        return [self.x1, self.x2, self.y1, self.y2, self.z1, self.z2]

    def center(self):
        """
        Return the coordinates of the center of the prism.

        Returns:

        * coords : list = [xc, yc, zc]
            Coordinates of the center

        Example:

            >>> prism = Prism(1, 2, 1, 3, 0, 2)
            >>> print prism.center()
            [ 1.5  2.   1. ]

        """
        xc = 0.5 * (self.x1 + self.x2)
        yc = 0.5 * (self.y1 + self.y2)
        zc = 0.5 * (self.z1 + self.z2)
        return np.array([xc, yc, zc])
