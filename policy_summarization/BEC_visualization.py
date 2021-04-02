import matplotlib.pyplot as plt
import numpy as np
import policy_summarization.BEC_helpers as BEC_helpers
from scipy.spatial import geometric_slerp
import matplotlib.tri as mtri

def cart2sph(x, y, z):
    '''
    Return corresponding spherical coordinates (azimuth, elevation) of a Cartesian point (x, y, z)
    '''
    azi = np.arctan2(y, x)
    ele = np.arccos(z/np.sqrt(x**2 + y**2 + z**2))

    return azi, ele

def visualize_spherical_polygon(poly, fig=None, ax=None, alpha=0.2, plot_ref_sphere=True):
    '''
    Visualize the spherical polygon created by the intersection between the constraint polyhedron and a unit sphere
    '''
    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca(projection='3d')

    vertices = np.array(poly.vertices())
    vertex_adj = poly.vertex_adjacency_matrix()
    vertex_adj_triu = np.triu(vertex_adj)

    # pull out vertices of the constraint polyhedron that lie on a constraint plane (as opposed to vertices that are
    # formed due to boundary constraints, where the latter simply clutter the visualization)
    spherical_polygon_vertices = {}
    for vertex_idx, vertex in enumerate(vertices):
        if (vertex != np.array([0, 0, 0])).any() and BEC_helpers.lies_on_constraint_plane(poly, vertex):
            vertex_normed = vertex / np.linalg.norm(vertex)
            spherical_polygon_vertices[vertex_idx] = vertex_normed

            ax.scatter(vertex_normed[0], vertex_normed[1], vertex_normed[2], marker='o', c='g', s=50)

    t_vals = np.linspace(0, 1, 50)

    # plot the spherical BEC polygon
    for spherical_polygon_vertex_idx in spherical_polygon_vertices.keys():
        vertex_normed = spherical_polygon_vertices[spherical_polygon_vertex_idx]

        # plot the spherically interpolated lines between adjacent vertices that lie on a constraint plane
        adj_vertex_idxs = vertex_adj_triu[spherical_polygon_vertex_idx]
        for adj_vertex_idx, is_adj in enumerate(adj_vertex_idxs):
            # if this vertex is adjacent, not at the origin, and lies on a constraint plane ...
            if is_adj == 1 and (vertices[adj_vertex_idx] != np.array([0, 0, 0])).any() and (adj_vertex_idx in spherical_polygon_vertices.keys()):
                adj_vertex_normed = spherical_polygon_vertices[adj_vertex_idx]
                result = geometric_slerp(vertex_normed, adj_vertex_normed, t_vals)
                ax.plot(result[:, 0], result[:, 1], result[:, 2], c='k')

    if plot_ref_sphere:
        # plot full unit sphere for reference
        u = np.linspace(0, 2 * np.pi, 30, endpoint=True)
        v = np.linspace(0, np.pi, 30, endpoint=True)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='y', alpha=alpha)

    # plot only the potion of the sphere corresponding to the valid solid angle
    hrep = np.array(poly.Hrepresentation())
    boundary_facet_idxs = np.where(hrep[:, 0] != 0)
    # first remove boundary constraint planes (those with a non-zero offset)
    min_constraints = np.delete(hrep, boundary_facet_idxs, axis=0)
    min_constraints = min_constraints[:, 1:]

    def sample_valid_region(constraints, min_azi, max_azi, min_ele, max_ele, n_azi, n_ele):
        # sample along the sphere
        u = np.linspace(min_azi, max_azi, n_azi, endpoint=True)
        v = np.linspace(min_ele, max_ele, n_ele, endpoint=True)

        x = np.outer(np.cos(u), np.sin(v)).reshape(1, -1)
        y = np.outer(np.sin(u), np.sin(v)).reshape(1, -1)
        z = np.outer(np.ones(np.size(u)), np.cos(v)).reshape(1, -1)
        sph_points = np.vstack((x, y, z))

        # see which points on the sphere obey all constraints
        dist_to_plane = constraints.dot(sph_points)
        n_constraints_satisfied = np.sum(dist_to_plane >= 0, axis=0)
        n_min_constraints = constraints.shape[0]

        idx_valid_sph_points = np.where(n_constraints_satisfied == n_min_constraints)[0]
        valid_sph_x = np.take(x, idx_valid_sph_points)
        valid_sph_y = np.take(y, idx_valid_sph_points)
        valid_sph_z = np.take(z, idx_valid_sph_points)

        return valid_sph_x, valid_sph_y, valid_sph_z

    # obtain x, y, z coordinates on the sphere that obey the constraints
    valid_sph_x, valid_sph_y, valid_sph_z = sample_valid_region(min_constraints, 0, 2 * np.pi, 0, np.pi, 1000, 1000)

    # resample coordinates on the sphere within the valid region (for higher density)
    sph_polygon_azi = []
    sph_polygon_ele = []

    for x in range(len(valid_sph_x)):
        azi, ele = cart2sph(valid_sph_x[x], valid_sph_y[x], valid_sph_z[x])
        sph_polygon_azi.append(azi)
        sph_polygon_ele.append(ele)

    # obtain (the higher density of) x, y, z coordinates on the sphere that obey the constraints
    valid_sph_x, valid_sph_y, valid_sph_z = \
        sample_valid_region(min_constraints, min(sph_polygon_azi), max(sph_polygon_azi), min(sph_polygon_ele), max(sph_polygon_ele), 50, 50)

    # create a triangulation mesh on which to interpolate using spherical coordinates
    valid_azi = []
    valid_ele = []
    for x in range(len(valid_sph_x)):
        azi, ele = cart2sph(valid_sph_x[x], valid_sph_y[x], valid_sph_z[x])
        valid_azi.append(azi)
        valid_ele.append(ele)
    tri = mtri.Triangulation(valid_azi, valid_ele)

    # reject triangles that are too large (which often result from connecting non-neighboring vertices) w/ a corresp mask
    dev_x = np.ptp(valid_sph_x[tri.triangles], axis=1) > 5 * np.mean(np.ptp(valid_sph_x[tri.triangles], axis=1))
    dev_y = np.ptp(valid_sph_y[tri.triangles], axis=1) > 5 * np.mean(np.ptp(valid_sph_y[tri.triangles], axis=1))
    dev_z = np.ptp(valid_sph_z[tri.triangles], axis=1) > 5 * np.mean(np.ptp(valid_sph_z[tri.triangles], axis=1))
    first_or = np.logical_or(dev_x, dev_y)
    second_or = np.logical_or(first_or, dev_z)
    tri.set_mask(second_or)

    # plot valid x, y, z coordinates on sphere as a mesh of valid triangles
    ax.plot_trisurf(valid_sph_x, valid_sph_y, valid_sph_z, triangles=tri.triangles, mask=second_or, color='y', alpha=1)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

def visualize_planes(constraints, fig=None, ax=None, alpha=0.5):
    '''
    Plot the planes associated with the normal vectors contained within 'constraints'
    '''
    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.gca(projection='3d')

    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    z = np.linspace(-1, 1, 10)

    X_xy, Y_xy, = np.meshgrid(x, y)

    for constraint in constraints:
        if constraint[0, 2] != 0:
            Z = (-constraint[0, 0] * X_xy - constraint[0, 1] * Y_xy) / constraint[0, 2]
            ax.plot_surface(X_xy, Y_xy, Z, alpha=alpha)
        elif constraint[0, 1] != 0:
            X_xz, Z_xz, = np.meshgrid(x, z)
            Y = (-constraint[0, 0] * X_xz - constraint[0, 2] * Z_xz) / constraint[0, 1]
            ax.plot_surface(X_xz, Y, Z_xz, alpha=alpha)
        else:
            Y_yz, Z_yz, = np.meshgrid(y, z)
            X = (-constraint[0, 1] * Y_yz - constraint[0, 2] * Z_yz) / constraint[0, 0]
            ax.plot_surface(X, Y_yz, Z_yz, alpha=alpha)
