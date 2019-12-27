# -*- coding: utf-8 -*-
"""
Common Gamut Boundary Descriptor (GDB) Utilities
================================================

Defines various *Gamut Boundary Descriptor (GDB)* common utilities.

-   :func:`colour.interpolate_gamut_boundary_descriptor`

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
import scipy.interpolate

from colour.algebra import (spherical_to_cartesian, Extrapolator,
                            CubicSplineInterpolator, LinearInterpolator)
from colour.constants import DEFAULT_INT_DTYPE
from colour.models import Jab_to_JCh
from colour.utilities import (as_int_array, is_trimesh_installed,
                              linear_conversion, orient, tsplit, tstack,
                              warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'Jab_to_spherical', 'spherical_to_Jab', 'close_gamut_boundary_descriptor',
    'interpolate_gamut_boundary_descriptor',
    'tessellate_gamut_boundary_descriptor'
]


def Jab_to_spherical(Jab):
    return cartesian_to_spherical(np.roll(Jab, 2, -1))


def spherical_to_Jab(rho_phi_theta):
    return np.roll(spherical_to_cartesian(rho_phi_theta), 1, -1)


def close_gamut_boundary_descriptor(GDB_m):
    GDB_m = spherical_to_Jab(GDB_m)

    # Index of first row with values.
    r_t_i = np.argmin(np.all(np.any(np.isnan(GDB_m), axis=-1), axis=1))

    # First row with values.
    GDB_m_t = GDB_m[r_t_i, ...]
    GDB_m_t = GDB_m_t[~np.any(np.isnan(GDB_m_t), axis=-1)]
    if not np.allclose(GDB_m_t, GDB_m_t[0]) or GDB_m_t.shape[0] == 1:
        Jab_t = np.nanmean(
            GDB_m_t[GDB_m_t[..., 0] == np.max(GDB_m_t[..., 0])], axis=0)

        if r_t_i != 0:
            r_t_i -= 1

        warning('Closing top of GBD at row {0} with value {1}'.format(
            r_t_i, Jab_t))

        GDB_m[r_t_i] = np.tile(Jab_t, [1, GDB_m.shape[1], 1])

    # Index of last row with values.
    r_b_i = np.argmin(
        np.flip(np.all(np.any(np.isnan(GDB_m), axis=-1), axis=1)))
    r_b_i = GDB_m.shape[0] - 1 - r_b_i

    # Last row with values.
    GDB_m_b = GDB_m[r_b_i, ...]
    GDB_m_b = GDB_m_b[~np.any(np.isnan(GDB_m_b), axis=-1)]
    if not np.allclose(GDB_m_b, GDB_m_b[0]) or GDB_m_b.shape[0] == 1:
        Jab_b = np.nanmean(
            GDB_m_b[GDB_m_b[..., 0] == np.min(GDB_m_b[..., 0])], axis=0)

        if r_b_i != GDB_m.shape[0] - 1:
            r_b_i += 1

        warning('Closing bottom of GBD at row {0} with value {1}'.format(
            r_b_i, Jab_b))

        GDB_m[r_b_i] = np.tile(Jab_b, [1, GDB_m.shape[1], 1])

    GDB_m = Jab_to_spherical(GDB_m)

    return GDB_m


def interpolate_gamut_boundary_descriptor(GDB_m):
    GDB_m = spherical_to_Jab(GDB_m)

    GDB_m_i = np.copy(GDB_m)

    shape_r, shape_c = GDB_m.shape[0], GDB_m.shape[1]
    r_slice = np.s_[0:shape_r]
    c_slice = np.s_[0:shape_c]

    # If bounding columns have NaN, :math:`GDB_m` matrix is tiled
    # horizontally so that right values interpolate with left values and
    # vice-versa.
    if np.any(np.isnan(GDB_m_i[..., 0])) or np.any(np.isnan(GDB_m_i[..., -1])):
        warning(
            'Gamut boundary descriptor matrix bounding columns contains NaN '
            'and will be horizontally tiled for interpolation!')
        c_slice = np.s_[shape_c:shape_c * 2]
        GDB_m_i = np.hstack([GDB_m_i] * 3)

    # If bounding rows have NaN, :math:`GDB_m` matrix is reflected vertically
    # so that top and bottom values are replicated via interpolation, i.e.
    # equivalent to nearest-neighbour interpolation.
    if np.any(np.isnan(GDB_m_i[0, ...])) or np.any(np.isnan(GDB_m_i[-1, ...])):
        warning('Gamut boundary descriptor matrix bounding rows contains NaN '
                'and will be vertically reflected for interpolation!')
        r_slice = np.s_[shape_r:shape_r * 2]
        GDB_m_f = orient(GDB_m_i, 'Flop')
        GDB_m_i = np.vstack([GDB_m_f, GDB_m_i, GDB_m_f])

    shape_r, shape_c = GDB_m_i.shape[0], GDB_m_i.shape[1]
    mask = np.any(~np.isnan(GDB_m_i), axis=-1)

    # Vertical/columns interpolation.
    GDB_m_i_v = np.copy(GDB_m_i)
    y = np.linspace(0, 1, shape_r)
    for i in range(shape_c):
        mask_i = mask[..., i]

        if not np.any(mask_i):
            continue

        for j in range(3):
            GDB_m_i_v[..., i, j] = Extrapolator(
                LinearInterpolator(y[mask_i], GDB_m_i_v[..., i, j][mask_i]))(y)
    GDB_m_i_v_s = Jab_to_spherical(GDB_m_i_v)

    # Horizontal/rows interpolation.
    GDB_m_i_h = np.copy(GDB_m_i)
    x = np.linspace(0, 1, shape_c)
    for i in range(shape_r):
        mask_i = mask[i]

        if not np.any(mask_i):
            continue

        for j in range(3):
            GDB_m_i_h[i, ..., j] = Extrapolator(
                LinearInterpolator(x[mask_i], GDB_m_i_h[i, ..., j][mask_i]))(x)
    GDB_m_i_h_s = Jab_to_spherical(GDB_m_i_h)

    # GDB_m_i[..., 0] = np.maximum(GDB_m_i_v_s[..., 0], GDB_m_i_h_s[..., 0])
    # GDB_m_i[..., 1:] = (GDB_m_i_v_s[..., 1:] + GDB_m_i_h_s[..., 1:]) / 2

    # Interpolation tends to flatten and pull the surface toward its
    # barycenter, cherry-picking the point with the maximum chroma value as the
    # correct one.
    GDB_m_i[~mask] = np.where(
        (GDB_m_i_v_s[..., 0] > GDB_m_i_h_s[..., 0])[..., np.newaxis],
        GDB_m_i_v,
        GDB_m_i_h,
    )[~mask]

    GDB_m_i = GDB_m_i[r_slice, c_slice, :]

    return GDB_m_i


def interpolate_gamut_boundary_descriptor_2(GDB_m):
    import cv2 as cv

    GDB_m = spherical_to_Jab(GDB_m)

    GDB_m_i = np.copy(GDB_m)

    shape_r, shape_c = GDB_m.shape[0], GDB_m.shape[1]

    r_slice = np.s_[shape_r:shape_r * 2]
    GDB_m_f = orient(GDB_m_i, 'Flop')
    GDB_m_i = np.vstack([GDB_m_f, GDB_m_i, GDB_m_f])

    shape_r, shape_c = GDB_m_i.shape[0], GDB_m_i.shape[1]
    c_slice = np.s_[shape_c:shape_c * 2]
    GDB_m_i = np.hstack([GDB_m_i] * 3)

    mask = as_int_array(np.any(np.isnan(GDB_m_i), axis=-1)).astype(np.uint8)
    for i in range(3):
        GDB_m_i[..., i] = cv.inpaint(GDB_m_i[..., i].astype(np.float32), mask,
                                     7, cv.INPAINT_NS)

    GDB_m_i = Jab_to_spherical(GDB_m_i[r_slice, c_slice, :])

    return GDB_m_i


def resize_gamut_boundary_descriptor(GDB_m, m, n):
    GDB_m = spherical_to_Jab(GDB_m)

    GDB_m_i = np.zeros([m, n, 3])

    x = np.linspace(0, 1, GDB_m.shape[0])
    y = np.linspace(0, 1, GDB_m.shape[1])
    x_g, y_g = np.meshgrid(x, y, indexing='ij')

    x_s = np.linspace(0, 1, m)
    y_s = np.linspace(0, 1, n)
    x_s_g, y_s_g = np.meshgrid(x_s, y_s, indexing='ij')
    for i in range(3):
        GDB_m_i[..., i] = scipy.interpolate.griddata(
            (np.ravel(x_g), np.ravel(y_g)),
            np.ravel(GDB_m[..., i]), (x_s_g, y_s_g),
            method='linear')

    GDB_m_i = Jab_to_spherical(GDB_m_i)

    return GDB_m_i


def tessellate_gamut_boundary_descriptor(GDB_m):
    if is_trimesh_installed():
        import trimesh

        vertices = spherical_to_Jab(GDB_m)

        # Wrapping :math:`GDB_m` to create faces between the outer columns.
        vertices = np.insert(
            vertices, vertices.shape[1], vertices[:, 0, :], axis=1)

        shape_r, shape_c = vertices.shape[0], vertices.shape[1]

        faces = []
        for i in np.arange(shape_r - 1):
            for j in np.arange(shape_c - 1):
                a = [i, j]
                b = [i, j + 1]
                c = [i + 1, j]
                d = [i + 1, j + 1]

                # Avoiding overlapping triangles when tessellating the bottom.
                if not i == 0:
                    faces.append([a, c, b])

                # Avoiding overlapping triangles when tessellating the top.
                if not i == shape_r - 2:
                    faces.append([c, d, b])

        indices = np.ravel_multi_index(
            np.transpose(as_int_array(faces)), [shape_r, shape_c])

        GDB_t = trimesh.Trimesh(
            vertices=vertices.reshape([-1, 3]),
            faces=np.transpose(indices),
            validate=True)

        if not GDB_t.is_watertight:
            warning('Tessellated mesh has holes!')

        return GDB_t


if __name__ == '__main__':
    # 9c91accdd8ea9c39437694bb3265fa6b09fd87d2
    # ccf400f7ce1eab488592517e5564c482ed8406d4
    # rd
    # source Environments/plotly/bin/activate
    # export PYTHONPATH=$PYTHONPATH:/Users/kelsolaar/Documents/Development/colour-science/colour:/Users/kelsolaar/Documents/Development/colour-science/trimesh
    # python /Users/kelsolaar/Documents/Development/colour-science/colour/colour/gamut/boundary/common.py

    import sys
    import trimesh
    import trimesh.smoothing
    import plotly.graph_objects as go

    import colour
    import colour.plotting
    from colour.plotting import render
    from colour.gamut import gamut_boundary_descriptor_Morovic2000
    from colour.algebra import cartesian_to_spherical

    np.set_printoptions(
        formatter={'float': '{:0.3f}'.format},
        linewidth=2048,
        suppress=True,
        threshold=sys.maxsize)

    print('*' * 79)
    print('*' * 79)
    print('*' * 79)

    def to_spherical_Morovic2000(Jab, E=np.array([50, 0, 0])):
        Jab_o = Jab - E
        J, a, b = tsplit(Jab_o)

        r = np.linalg.norm(Jab_o, axis=-1)
        alpha = np.arctan2(b, a)
        theta = np.arctan2(J, np.linalg.norm(tstack([a, b]), axis=-1))

        return tstack([r, theta, alpha])

    def indexes_gamut_boundary_descriptor_Morovic2000(r_theta_alpha, m, n):
        r, theta, alpha = tsplit(r_theta_alpha)

        theta_i = theta / np.pi * m
        theta_i = as_int_array(np.clip(np.floor(theta_i), 0, m - 1))
        alpha_i = (alpha + np.pi) / (2 * np.pi) * n
        alpha_i = as_int_array(np.clip(np.floor(alpha_i), 0, n - 1))

        return tstack([theta_i, alpha_i])

    def create_plane(width=1,
                     height=1,
                     width_segments=1,
                     height_segments=1,
                     direction='+z'):

        x_grid = width_segments
        y_grid = height_segments

        x_grid1 = x_grid + 1
        y_grid1 = y_grid + 1

        # Positions, normals and texcoords.
        positions = np.zeros(x_grid1 * y_grid1 * 3)
        normals = np.zeros(x_grid1 * y_grid1 * 3)
        texcoords = np.zeros(x_grid1 * y_grid1 * 2)

        y = np.arange(y_grid1) * height / y_grid - height / 2
        x = np.arange(x_grid1) * width / x_grid - width / 2

        positions[::3] = np.tile(x, y_grid1)
        positions[1::3] = -np.repeat(y, x_grid1)

        normals[2::3] = 1

        texcoords[::2] = np.tile(np.arange(x_grid1) / x_grid, y_grid1)
        texcoords[1::2] = np.repeat(1 - np.arange(y_grid1) / y_grid, x_grid1)

        # Faces and outline.
        faces, outline = [], []
        for i_y in range(y_grid):
            for i_x in range(x_grid):
                a = i_x + x_grid1 * i_y
                b = i_x + x_grid1 * (i_y + 1)
                c = (i_x + 1) + x_grid1 * (i_y + 1)
                d = (i_x + 1) + x_grid1 * i_y

                faces.extend(((a, b, d), (b, c, d)))
                outline.extend(((a, b), (b, c), (c, d), (d, a)))

        positions = np.reshape(positions, (-1, 3))
        texcoords = np.reshape(texcoords, (-1, 2))
        normals = np.reshape(normals, (-1, 3))

        faces = np.reshape(faces, (-1, 3)).astype(np.uint32)
        outline = np.reshape(outline, (-1, 2)).astype(np.uint32)

        direction = direction.lower()
        if direction in ('-x', '+x'):
            shift, neutral_axis = 1, 0
        elif direction in ('-y', '+y'):
            shift, neutral_axis = -1, 1
        elif direction in ('-z', '+z'):
            shift, neutral_axis = 0, 2

        sign = -1 if '-' in direction else 1

        positions = np.roll(positions, shift, -1)
        normals = np.roll(normals, shift, -1) * sign
        colors = np.ravel(positions)
        colors = np.hstack((np.reshape(
            np.interp(colors, (np.min(colors), np.max(colors)), (0, 1)),
            positions.shape), np.ones((positions.shape[0], 1))))
        colors[..., neutral_axis] = 0

        vertices = np.zeros(
            positions.shape[0],
            [('position', np.float32, 3), ('texcoord', np.float32, 2),
             ('normal', np.float32, 3), ('color', np.float32, 4)])

        vertices['position'] = positions
        vertices['texcoord'] = texcoords
        vertices['normal'] = normals
        vertices['color'] = colors

        return vertices, faces, outline

    def create_box(width=1,
                   height=1,
                   depth=1,
                   width_segments=1,
                   height_segments=1,
                   depth_segments=1,
                   planes=None):
        planes = (('+x', '-x', '+y', '-y', '+z', '-z')
                  if planes is None else [d.lower() for d in planes])

        w_s, h_s, d_s = width_segments, height_segments, depth_segments

        planes_m = []
        if '-z' in planes:
            planes_m.append(create_plane(width, depth, w_s, d_s, '-z'))
            planes_m[-1][0]['position'][..., 2] -= height / 2
        if '+z' in planes:
            planes_m.append(create_plane(width, depth, w_s, d_s, '+z'))
            planes_m[-1][0]['position'][..., 2] += height / 2

        if '-y' in planes:
            planes_m.append(create_plane(height, width, h_s, w_s, '-y'))
            planes_m[-1][0]['position'][..., 1] -= depth / 2
        if '+y' in planes:
            planes_m.append(create_plane(height, width, h_s, w_s, '+y'))
            planes_m[-1][0]['position'][..., 1] += depth / 2

        if '-x' in planes:
            planes_m.append(create_plane(depth, height, d_s, h_s, '-x'))
            planes_m[-1][0]['position'][..., 0] -= width / 2
        if '+x' in planes:
            planes_m.append(create_plane(depth, height, d_s, h_s, '+x'))
            planes_m[-1][0]['position'][..., 0] += width / 2

        positions = np.zeros((0, 3), dtype=np.float32)
        texcoords = np.zeros((0, 2), dtype=np.float32)
        normals = np.zeros((0, 3), dtype=np.float32)

        faces = np.zeros((0, 3), dtype=np.uint32)
        outline = np.zeros((0, 2), dtype=np.uint32)

        offset = 0
        for vertices_p, faces_p, outline_p in planes_m:
            positions = np.vstack((positions, vertices_p['position']))
            texcoords = np.vstack((texcoords, vertices_p['texcoord']))
            normals = np.vstack((normals, vertices_p['normal']))

            faces = np.vstack((faces, faces_p + offset))
            outline = np.vstack((outline, outline_p + offset))
            offset += vertices_p['position'].shape[0]

        vertices = np.zeros(
            positions.shape[0],
            [('position', np.float32, 3), ('texcoord', np.float32, 2),
             ('normal', np.float32, 3), ('color', np.float32, 4)])

        colors = np.ravel(positions)
        colors = np.hstack((np.reshape(
            np.interp(colors, (np.min(colors), np.max(colors)), (0, 1)),
            positions.shape), np.ones((positions.shape[0], 1))))

        vertices['position'] = positions
        vertices['texcoord'] = texcoords
        vertices['normal'] = normals
        vertices['color'] = colors

        return vertices, faces, outline

    # *************************************************************************

    t = 9
    theta = np.tile(np.radians(np.linspace(0, 180, t)), (t, 1))
    theta = np.squeeze(np.transpose(theta).reshape(-1, 1))
    phi = np.tile(
        np.radians(np.linspace(-180, 180, t)) + np.radians(360 / 8 / 4), t)
    rho = np.ones(t * t) * 50
    rho_theta_phi = tstack([rho, theta, phi])
    Jab_p = np.roll(spherical_to_cartesian(rho_theta_phi), 1, -1)
    Jab_p += [50, 0, 0]
    sp_p = cartesian_to_spherical(
        np.roll(np.reshape(Jab_p, [-1, 3]) - [50, 0, 0], 2, -1))
    q_p = indexes_gamut_boundary_descriptor_Morovic2000(sp_p, 8, 8)
    print(np.unique(q_p, axis=0).shape)
    print(np.hstack([Jab_p, q_p]))
    # print('Jab_p')
    # print(Jab_p.shape)
    # print(np.unique(Jab_p, axis=0).shape)
    # print(Jab_p)
    #

    # print('Spherical')
    # J, a, b = tsplit(Jab_p - [50, 0, 0])
    # cs = ((cartesian_to_spherical(np.reshape(tstack([a, b, J]), [-1, 3])) +
    #        [0, 0, np.pi]) / [1, np.pi, 2 * np.pi] * [1, 8, 8])
    # print(np.unique(cs, axis=0).shape)
    # print(cs)
    # print('^')
    #
    # print('to_spherical_Morovic2000(Jab_p)')
    # sp_M2000 = to_spherical_Morovic2000(Jab_p)
    # print(sp_M2000)
    # print(np.unique(sp_M2000, axis=0).shape)
    #

    # *************************************************************************

    # t = 16
    # theta_s = np.squeeze(
    #     np.transpose(np.tile(np.radians(np.linspace(0, 180, t)),
    #                          (t + 1, 1))).reshape(-1, 1))
    # phi_s = np.tile(np.radians(np.linspace(-180, 180, t + 1)), t)
    # rho_s = np.ones(t * (t +1)) * 50
    # rho_theta_phi_s = tstack([rho_s, theta_s, phi_s])
    # print(rho_theta_phi_s)
    #
    # Jab_p_s = np.roll(spherical_to_cartesian(rho_theta_phi_s), 1, -1)
    # Jab_p_s += [50, 0, 0]
    # print(Jab_p_s.shape)
    #
    # X_i_s, Y_i_s, Z_i_s = tsplit(Jab_p_s)
    #
    # X_i, Y_i, Z_i = tsplit(Jab_p)
    #
    # fig = go.Figure(data=[
    #     go.Scatter3d(
    #         x=np.ravel(Y_i_s),
    #         y=np.ravel(Z_i_s),
    #         z=np.ravel(X_i_s),
    #         mode='markers',
    #         marker=dict(size=3),
    #     ),
    #     go.Scatter3d(
    #         x=np.ravel(Y_i),
    #         y=np.ravel(Z_i),
    #         z=np.ravel(X_i),
    #         mode='markers',
    #         marker=dict(size=6),
    #     ),
    # ])
    #
    # fig.show()

    # *************************************************************************

    vertices, faces, outline = create_box(1, 1, 1, 128, 128, 128)
    RGB_r = vertices['position'] + 0.5
    Jab_r = colour.convert(
        RGB_r, 'RGB', 'CIE Lab', verbose={'describe': 'short'}) * 100
    mesh_r = trimesh.Trimesh(
        vertices=Jab_r.reshape([-1, 3]), faces=faces, validate=True)
    mesh_r.fix_normals()
    # mesh_r.show()

    np.random.seed(16)
    RGB = np.random.random([200, 200, 3])

    # *************************************************************************

    # s = 32
    # RGB_t = colour.plotting.geometry.cube(
    #     width_segments=s, height_segments=s, depth_segments=s)
    # print(RGB_t)

    RGB_t = RGB_r

    Jab_t = colour.convert(
        RGB_t, 'RGB', 'CIE Lab', verbose={'describe': 'short'}) * 100

    # *************************************************************************

    # XYZ = np.random.rand(1000, 3)
    # XYZ = XYZ[colour.is_within_visible_spectrum(XYZ)]
    #
    # Jab_x = colour.convert(
    #     XYZ, 'CIE XYZ', 'CIE Lab', verbose={'describe': 'short'}) * 100

    # *************************************************************************

    # Jab_p[..., 0] = np.clip(Jab_p[..., 0], 45, 50)
    # Jab = Jab_p
    Jab = Jab_t
    # Jab = Jab_x

    # *************************************************************************

    # v_r = 8
    # h_r = 8
    v_r = 32
    h_r = 32
    # v_r = 18
    # h_r = 16
    # segments = (v_r, v_r + 2, v_r + 4,  v_r + 6,  v_r + 8)
    segments = (v_r, )
    slc_v = v_r
    slc_h = h_r

    GDB_m_s = []
    for v in segments:
        print('^' * 79)
        print('v, hr', v, h_r)

        GDB_m = gamut_boundary_descriptor_Morovic2000(Jab, [50, 0, 0], v, h_r)
        print('GBD', GDB_m.shape)
        print(GDB_m[..., 0][:slc_v, :slc_h])
        print(GDB_m[..., 1][:slc_v, :slc_h])
        print(GDB_m[..., 2][:slc_v, :slc_h])

        if True:
            GDB_m = close_gamut_boundary_descriptor(GDB_m)
            print('Close', GDB_m.shape)
            print(GDB_m[..., 0][:slc_v, :slc_h])
            print(GDB_m[..., 1][:slc_v, :slc_h])
            print(GDB_m[..., 2][:slc_v, :slc_h])

        if True:
            GDB_m = interpolate_gamut_boundary_descriptor(GDB_m)
            print('Interpolated', GDB_m.shape)
            print(GDB_m[..., 0][:slc_v, :slc_h])
            print(GDB_m[..., 1][:slc_v, :slc_h])
            print(GDB_m[..., 2][:slc_v, :slc_h])

        if False:
            d = GDB_m.shape[0] - v_r
            if v != v_r:
                print('!!! Resizing !!!')
                GDB_m = resize_gamut_boundary_descriptor(GDB_m, v_r + d, h_r)
            print('Resize', GDB_m.shape)
            print(GDB_m[..., 0][:slc_v, :slc_h])
            print(GDB_m[..., 1][:slc_v, :slc_h])
            print(GDB_m[..., 2][:slc_v, :slc_h])

        # GDB_m_i[GDB_m_i.shape[0] // 2:, ...] += (
        #     np.random.rand(*(GDB_m_i[GDB_m_i.shape[0] // 2:, ...].shape)) * 25)
        GDB_m_s.append(GDB_m)

    import matplotlib.pyplot as plt

    def plot_gamut_boundary_descriptors(GDB_m, columns=None, **kwargs):
        GDB_m = [Jab_to_JCh(spherical_to_Jab(GDB_m_c)) for GDB_m_c in GDB_m]

        assert len(np.unique([GDB_m_c.shape for GDB_m_c in GDB_m])) <= 3, (
            'Gamut boundary descriptor matrices have incompatible shapes!')

        shape_r, shape_c, = GDB_m[0].shape[0], GDB_m[0].shape[1]

        if columns is None:
            columns = shape_c

        figure, axes_a = plt.subplots(
            DEFAULT_INT_DTYPE(np.ceil(shape_c / columns)),
            columns,
            sharex='col',
            sharey='row',
            gridspec_kw={
                'hspace': 0,
                'wspace': 0
            })

        axes_a = np.ravel(axes_a)

        for i in range(shape_c):
            for j in range(len(GDB_m)):
                axes_a[i].plot(
                    GDB_m[j][..., i, 1],
                    GDB_m[j][..., i, 0],
                    label='{0:d} $^\\degree$'.format(
                        int(i / GDB_m[0].shape[1] * 360)))

                axes_a[i].legend()

        render(**kwargs)

    plot_gamut_boundary_descriptors(GDB_m_s, 6)

    GDB_m_s_z = np.copy(GDB_m_s[0])
    ranges = [[-50, 50], [0, np.pi], [-np.pi, np.pi]]
    for i in range(3):
        GDB_m_s_z[..., i] = linear_conversion(
            GDB_m_s_z[..., i], (ranges[i][0], ranges[i][1]), (0, 1))

    colour.plotting.plot_image(GDB_m_s_z)

    GDB_t_s = [
        tessellate_gamut_boundary_descriptor(GDB_m) for GDB_m in GDB_m_s
    ]

    # trimesh.smoothing.filter_laplacian(GDB_t, iterations=25)
    # trimesh.repair.broken_faces(GDB_t, color=(255, 0, 0, 255))

    GDB_t_r = GDB_t_s[0]
    for i, GDB_t in enumerate(GDB_t_s[1:]):
        # GDB_t.vertices += [(i + 1) * 100, 0, 0]
        GDB_t_r = GDB_t_r + GDB_t

    # mesh_r.vertices -= [50, 150, 0]
    # GDB_t_r = GDB_t_r + mesh_r

    # trimesh.smoothing.filter_laplacian(GDB_t_r, iterations=25)

    GDB_t_r.export('/Users/kelsolaar/Downloads/mesh.obj', 'obj')

    GDB_t_r.show()
