# -*- coding: utf-8 -*-
"""
Gamut Boundary Descriptor (GDB) - Morovic and Luo (2000)
========================================================

Defines the * Morovic and Luo (2000)* *Gamut Boundary Descriptor (GDB)*
computation objects:

-   :func:`colour.gamut.gamut_boundary_descriptor_Morovic2000`

See Also
--------
`Gamut Boundary Descriptor Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/gamut/boundary.ipynb>`_

References
----------
-   :cite:`` :
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import cartesian_to_spherical
from colour.utilities import as_int_array, as_float_array, tsplit

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['gamut_boundary_descriptor_Morovic2000']


def gamut_boundary_descriptor_Morovic2000(Jab,
                                          E=np.array([50, 0, 0]),
                                          m=16,
                                          n=16):
    Jab = as_float_array(Jab)
    E = as_float_array(E)

    r_theta_alpha = cartesian_to_spherical(
        np.roll(np.reshape(Jab, [-1, 3]) - E, 2, -1))
    r, theta, alpha = tsplit(r_theta_alpha)

    GDB_m = np.full([m, n, 3], np.nan)

    theta_i = theta / np.pi * m
    theta_i = as_int_array(np.clip(np.floor(theta_i), 0, m - 1))

    alpha_i = (alpha + np.pi) / (2 * np.pi) * n
    alpha_i = as_int_array(np.clip(np.floor(alpha_i), 0, n - 1))

    for i in np.arange(m):
        for j in np.arange(n):
            i_j = np.intersect1d(
                np.argwhere(theta_i == i), np.argwhere(alpha_i == j))

            if i_j.size == 0:
                continue

            GDB_m[i, j] = r_theta_alpha[i_j[np.argmax(r[i_j])]]

    # Naive non-vectorised implementation kept for reference.
    # :math:`r_m` is used to keep track of the maximum :math:`r` value.
    # r_m = np.full([m, n, 1], np.nan)
    # for i, r_theta_alpha_i in enumerate(r_theta_alpha):
    #     p_i, a_i = theta_i[i], alpha_i[i]
    #     r_i_j = r_m[p_i, a_i]
    #
    #     if r[i] > r_i_j or np.isnan(r_i_j):
    #         GDB_m[p_i, a_i] = r_theta_alpha_i
    #         r_m[p_i, a_i] = r[i]

    return GDB_m
