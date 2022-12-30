import matplotlib.pyplot as plt
import numpy as np
import policy_summarization.BEC_helpers as BEC_helpers
import policy_summarization.computational_geometry as cg
from scipy.spatial import geometric_slerp
import matplotlib.tri as mtri
from termcolor import colored

def visualize_spherical_polygon(poly, fig=None, ax=None, alpha=1.0, color='y', plot_ref_sphere=True):
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

            # ax.scatter(vertex_normed[0], vertex_normed[1], vertex_normed[2], marker='o', c='g', s=50)

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
                # ax.plot(result[:, 0], result[:, 1], result[:, 2], c='k')

    if plot_ref_sphere:
        # plot full unit sphere for reference
        u = np.linspace(0, 2 * np.pi, 30, endpoint=True)
        v = np.linspace(0, np.pi, 30, endpoint=True)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='y', alpha=0.2)

    # plot only the potion of the sphere corresponding to the valid solid angle
    hrep = np.array(poly.Hrepresentation())
    boundary_facet_idxs = np.where(hrep[:, 0] != 0)
    # first remove boundary constraint planes (those with a non-zero offset)
    min_constraints = np.delete(hrep, boundary_facet_idxs, axis=0)
    min_constraints = min_constraints[:, 1:]

    # obtain x, y, z coordinates on the sphere that obey the constraints
    valid_sph_x, valid_sph_y, valid_sph_z = cg.sample_valid_region(min_constraints, 0, 2 * np.pi, 0, np.pi, 1000, 1000)

    if len(valid_sph_x) == 0:
        print(colored("Was unable to sample valid points for visualizing the BEC (which is likely too small).",
                      'red'))
        return

    # resample coordinates on the sphere within the valid region (for higher density)
    sph_polygon = cg.cart2sph(np.array([valid_sph_x, valid_sph_y, valid_sph_z]).T)
    sph_polygon_ele = sph_polygon[:, 0]
    sph_polygon_azi = sph_polygon[:, 1]

    # obtain (the higher density of) x, y, z coordinates on the sphere that obey the constraints
    valid_sph_x, valid_sph_y, valid_sph_z = \
        cg.sample_valid_region(min_constraints, min(sph_polygon_azi), max(sph_polygon_azi), min(sph_polygon_ele), max(sph_polygon_ele), 50, 50)

    # create a triangulation mesh on which to interpolate using spherical coordinates
    sph_polygon = cg.cart2sph(np.array([valid_sph_x, valid_sph_y, valid_sph_z]).T)
    valid_ele = sph_polygon[:, 0]
    valid_azi = sph_polygon[:, 1]

    tri = mtri.Triangulation(valid_azi, valid_ele)

    # reject triangles that are too large (which often result from connecting non-neighboring vertices) w/ a corresp mask
    dev_x = np.ptp(valid_sph_x[tri.triangles], axis=1) > 5 * np.mean(np.ptp(valid_sph_x[tri.triangles], axis=1))
    dev_y = np.ptp(valid_sph_y[tri.triangles], axis=1) > 5 * np.mean(np.ptp(valid_sph_y[tri.triangles], axis=1))
    dev_z = np.ptp(valid_sph_z[tri.triangles], axis=1) > 5 * np.mean(np.ptp(valid_sph_z[tri.triangles], axis=1))
    first_or = np.logical_or(dev_x, dev_y)
    second_or = np.logical_or(first_or, dev_z)
    tri.set_mask(second_or)

    # plot valid x, y, z coordinates on sphere as a mesh of valid triangles
    ax.plot_trisurf(valid_sph_x, valid_sph_y, valid_sph_z, triangles=tri.triangles, mask=second_or, color=color, alpha=alpha)

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


def visualize_projection(constraints, weights, proj_type, plot_lim=[(-1, 1), (-1, 1)], scale=1.0, title=None, xlabel=None, ylabel=None, fig_name=None, just_save=False):
    '''
    Summary: Visualize the constraints. Use scale to determine whether to show L1 normalized weights and constraints or
    weights and constraints where the step cost is 1.
    '''
    # This visualization function is currently specialized to handle plotting problems with two unknown weights or two
    # unknown and one known weight. For higher dimensional constraints, this visualization function must be updated.

    plt.xlim(plot_lim[0][0] * scale, plot_lim[0][1] * scale)
    plt.ylim(plot_lim[1][0] * scale, plot_lim[1][1] * scale)

    wt_marker_size = 200

    wt_shading = 1. / len(constraints)

    if proj_type == 'xy':
        for constraint in constraints:
            if constraint[0, 1] == 0.:
                # completely vertical line
                pt = (-weights[0, -1] * constraint[0, 2]) / constraint[0, 0]
                plt.plot(np.array([pt, pt]) * scale, np.array([-1, 1]) * scale)

                # use (1, 0) as a test point to decide which half space to color
                if constraint[0, 0] + (weights[0, -1] * constraint[0, 2]) >= 0:
                    # color the right side of the line
                    plt.axvspan(pt * scale, 1 * scale, alpha=wt_shading, color='blue')
                else:
                    # color the left side of the line
                    plt.axvspan(-1 * scale, pt * scale, alpha=wt_shading, color='blue')
            else:
                pt_1 = (constraint[0, 0] - (weights[0, -1] * constraint[0, 2])) / constraint[0, 1]
                pt_2 = (-constraint[0, 0] - (weights[0, -1] * constraint[0, 2])) / constraint[0, 1]
                plt.plot(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale)

                # use (0, 1) as a test point to decide which half space to color
                if constraint[0, 1] + (weights[0, -1] * constraint[0, 2]) >= 0:
                    plt.fill_between(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale, np.array([1, 1]) * scale, alpha=wt_shading, color='blue')
                else:
                    plt.fill_between(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale, np.array([-1, -1]) * scale, alpha=wt_shading, color='blue')

        # plot ground truth weight
        plt.scatter(weights[0, 0] * scale, weights[0, 1] * scale, s=wt_marker_size, color='red', zorder=2)
        # labels
        if xlabel is not None:
            plt.xlabel(xlabel)
        else:
            plt.xlabel(r'$w_0$')
        if ylabel is not None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel(r'$w_1$')
    elif proj_type == 'xz':
        for constraint in constraints:
            if constraint[0, 2] == 0.:
                # completely vertical line
                pt = (-weights[0, 1] * constraint[0, 1]) / constraint[0, 0]
                plt.plot(np.array([pt, pt]) * scale, np.array([-1, 1]) * scale)

                # use (1, 0) as a test point to decide which half space to color
                if constraint[0, 0] + (weights[0, 1] * constraint[0, 1]) >= 0:
                    # color the right side of the line
                    plt.axvspan(pt * scale, 1 * scale, alpha=wt_shading, color='blue')
                else:
                    # color the left side of the line
                    plt.axvspan(-1 * scale, pt * scale, alpha=wt_shading, color='blue')
            else:
                pt_1 = (constraint[0, 0] - (weights[0, 1] * constraint[0, 1])) / constraint[0, 2]
                pt_2 = (-constraint[0, 0] - (weights[0, 1] * constraint[0, 1])) / constraint[0, 2]
                plt.plot(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale)

                # use (0, 1) as a test point to decide which half space to color
                if weights[0, 1] * constraint[0, 1] + constraint[0, 2] >= 0:
                    plt.fill_between(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale, np.array([1, 1]) * scale, alpha=wt_shading, color='blue')
                else:
                    plt.fill_between(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale, np.array([-1, -1]) * scale, alpha=wt_shading, color='blue')

        # plot ground truth weight
        plt.scatter(weights[0, 0] * scale, weights[0, 2] * scale, s=wt_marker_size, color='red', zorder=2)
        # labels
        if xlabel is not None:
            plt.xlabel(xlabel)
        else:
            plt.xlabel(r'$w_0$')
        if ylabel is not None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel(r'$w_2$')
    else:
        # yz
        for constraint in constraints:
            if constraint[0, 2] == 0.:
                # completely vertical line
                pt = (-weights[0, 0] * constraint[0, 0]) / constraint[0, 1]
                plt.plot(np.array([pt, pt]) * scale, np.array([-1, 1]) * scale)

                # use (1, 0) as a test point to decide which half space to color
                if (weights[0, 0] * constraint[0, 0]) + constraint[0, 1] >= 0:
                    # color the right side of the line
                    plt.axvspan(pt * scale, 1 * scale, alpha=wt_shading, color='blue')
                else:
                    # color the left side of the line
                    plt.axvspan(-1 * scale, pt * scale, alpha=wt_shading, color='blue')
            else:
                pt_1 = (constraint[0, 1] - (weights[0, 0] * constraint[0, 0])) / constraint[0, 2]
                pt_2 = (-constraint[0, 1] - (weights[0, 0] * constraint[0, 0])) / constraint[0, 2]
                plt.plot(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale)

                # use (0, 1) as a test point to decide which half space to color
                if weights[0, 0] * constraint[0, 0] + constraint[0, 2] >= 0:
                    plt.fill_between(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale, np.array([1, 1]) * scale, alpha=wt_shading, color='blue')
                else:
                    plt.fill_between(np.array([-1, 1]) * scale, np.array([pt_1, pt_2]) * scale, np.array([-1, -1]) * scale, alpha=wt_shading, color='blue')

        # plot ground truth weight
        plt.scatter(weights[0, 1] * scale, weights[0, 2] * scale, s=wt_marker_size, color='red', zorder=2)
        # labels
        if xlabel is not None:
            plt.xlabel(xlabel)
        else:
            plt.xlabel(r'$w_1$')
        if ylabel is not None:
            plt.ylabel(ylabel)
        else:
            plt.ylabel(r'$w_2$')

    if title is not None:
        plt.title(title)
    else:
        plt.title(proj_type)
    plt.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name, dpi=200, transparent=True)
    if not just_save:
        plt.show()
    plt.close()
