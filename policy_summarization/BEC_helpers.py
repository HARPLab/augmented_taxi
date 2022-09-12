import numpy as np
from pypoman import compute_polygon_hull, indicate_violating_constraints
from scipy.optimize import linprog
import sage.all
import sage.geometry.polyhedron.base as Polyhedron
from termcolor import colored
import difflib
from sklearn import metrics
import itertools
import pickle

from policy_summarization import computational_geometry as cg

def normalize_constraints(constraints):
    '''
    Summary: Normalize all constraints such that the L1 norm is equal to 1
    '''
    normalized_constraints = []
    zero_constraint = np.zeros(constraints[0].shape)
    for constraint in constraints:
        if not equal_constraints(constraint, zero_constraint):
            normalized_constraints.append(constraint / np.linalg.norm(constraint, ord=1))

    return normalized_constraints

def remove_duplicate_constraints(constraints):
    '''
    Summary: Remove any duplicate constraints
    '''
    nonredundant_constraints = []
    zero_constraint = np.zeros(constraints[0].shape)

    for query in constraints:
        add_it = True
        for comp in nonredundant_constraints:
            # don't keep any duplicate constraints or degenerate zero constraints
            if equal_constraints(query, comp) or equal_constraints(query, zero_constraint):
                add_it = False
                break
        if add_it:
            nonredundant_constraints.append(query)

    return nonredundant_constraints

def equal_constraints(c1, c2):
    '''
    Summary: Check for equality between two constraints c1 and c2
    '''
    if np.sum(abs(c1 / np.linalg.norm(c1) - c2 / np.linalg.norm(c2))) <= 1e-05:
        return True
    else:
        return False

def clean_up_constraints(constraints, weights, step_cost_flag):
    '''
    Summary: Normalize constraints, remove duplicates, and remove redundant constraints
    '''
    normalized_constraints = normalize_constraints(constraints)
    if len(normalized_constraints) > 0:
        nonduplicate_constraints = remove_duplicate_constraints(normalized_constraints)
        if len(nonduplicate_constraints) > 1:
            min_subset_constraints = remove_redundant_constraints(nonduplicate_constraints, weights, step_cost_flag)
        else:
            min_subset_constraints = nonduplicate_constraints
    else:
        min_subset_constraints = normalized_constraints

    return min_subset_constraints

def remove_redundant_constraints_lp(constraints, weights, step_cost_flag):
    '''
    Summary: Remove redundant constraint that do not change the underlying BEC region (without consideration for
    whether how it intersects the L1 constraints)
    '''
    # these lists are effectively one level deep so a shallow copy should suffice. copy over the original constraints
    # and remove redundant constraints one by one
    nonredundant_constraints = constraints.copy()
    redundundant_constraints = []

    for query_constraint in constraints:
        # create a set of constraints the excludes the current constraint in question (query_constraint)
        constraints_other = []
        for nonredundant_constraint in nonredundant_constraints:
            if not equal_constraints(query_constraint, nonredundant_constraint):
                constraints_other.append(list(-nonredundant_constraint[0]))

        # if there are other constraints left to compare to
        if len(constraints_other) > 0:
            # solve linear program
            # min_x a^Tx, st -Ax >= -b (note that scipy traditionally accepts bounds as Ax <= b, hence the negative multiplier to the constraints)
            a = np.ndarray.tolist(query_constraint[0])
            b = [0] * len(constraints_other)
            if step_cost_flag:
                # the last weight is the step cost, which is assumed to be known by the learner. adjust the bounds accordingly
                res = linprog(a, A_ub=constraints_other, b_ub=b, bounds=[(-1, 1), (-1, 1), (weights[0, -1], weights[0, -1])])
            else:
                res = linprog(a, A_ub=constraints_other, b_ub=b, bounds=[(-1, 1)] * constraints[0].shape[1])

            # if query_constraint * res.x^T >= 0, then this constraint is redundant. copy over everything except this constraint
            if query_constraint.dot(res.x.reshape(-1, 1))[0][0] >= -1e-05: # account for slight numerical instability
                copy_array = []
                for nonredundant_constraint in nonredundant_constraints:
                    if not equal_constraints(query_constraint, nonredundant_constraint):
                        copy_array.append(nonredundant_constraint)
                nonredundant_constraints = copy_array
                redundundant_constraints.append(query_constraint)
        else:
            break

    return nonredundant_constraints, redundundant_constraints

def remove_redundant_constraints(constraints, weights, step_cost_flag):
    '''
    Summary: Remove redundant constraints
    '''
    if step_cost_flag:
        # Remove redundant constraint that do not change the underlying intersection between the BEC region and the
        # L1 constraints
        try:
            BEC_length_all_constraints, nonredundant_constraint_idxs = calculate_BEC_length(constraints, weights,
                                                                                            step_cost_flag)
        except:
            # a subset of these constraints aren't numerically stable (e.g. you can have a constraint that's ever so slightly
            # over the ground truth reward weight and thus fail to yield a proper polygonal convex hull. remove the violating constraints
            A, b = constraints_to_halfspace_matrix(constraints, weights, step_cost_flag)
            violating_idxs = indicate_violating_constraints(A, b)

            for violating_idx in sorted(violating_idxs[0], reverse=True):
                del constraints[violating_idx]

            BEC_length_all_constraints, nonredundant_constraint_idxs = calculate_BEC_length(constraints, weights,
                                                                                            step_cost_flag)

        nonredundant_constraints = [constraints[x] for x in nonredundant_constraint_idxs]

        for query_idx, query_constraint in enumerate(constraints):
            if query_idx not in nonredundant_constraint_idxs:
                pass
            else:
                # see if this is truly non-redundant or crosses an L1 constraint exactly where another constraint does
                constraints_other = []
                for constraint_idx, constraint in enumerate(nonredundant_constraints):
                    if not equal_constraints(query_constraint, constraint):
                        constraints_other.append(constraint)
                if len(constraints_other) > 0:
                    BEC_length = calculate_BEC_length(constraints_other, weights, step_cost_flag)[0]

                    # simply remove the first redundant constraint. can also remove the redundant constraint that's
                    # 1) conveyed by the fewest environments, 2) conveyed by a higher minimum complexity environment,
                    # 3) doesn't work as well with visual similarity of other nonredundant constraints
                    if np.isclose(BEC_length, BEC_length_all_constraints):
                        nonredundant_constraints = constraints_other
    else:
        # remove constraints that don't belong in the minimal H-representation of the corresponding polyhedron (not
        # including the boundary constraints/facets)
        ieqs = constraints_to_halfspace_matrix_sage(constraints)
        poly = Polyhedron.Polyhedron(ieqs=ieqs)
        hrep = np.array(poly.Hrepresentation())

        # remove boundary constraints/facets from consideration
        boundary_facet_idxs = np.where(hrep[:, 0] != 0)
        hrep_constraints = np.delete(hrep, boundary_facet_idxs, axis=0)
        # remove the first column since these constraints goes through the origin
        nonredundant_constraints = hrep_constraints[:, 1:]
        # reshape so that each element is a valid weight vector
        nonredundant_constraints = nonredundant_constraints.reshape(nonredundant_constraints.shape[0], 1, nonredundant_constraints.shape[1])

    return list(nonredundant_constraints)

def perform_BEC_constraint_bookkeeping_flattened(BEC_constraints, min_subset_constraints_record):
    '''
    Summary: For each constraint in min_subset_constraints_record, see if it matches one of the BEC_constraints
    (assumes a flattened list of min_subset_constraints_record with no division across environments, which is deprecated)
    '''
    BEC_constraint_bookkeeping = []

    # keep track of which demo conveys which of the BEC constraints
    for constraints in min_subset_constraints_record:
        covers = []
        for BEC_constraint_idx in range(len(BEC_constraints)):
            contains_BEC_constraint = False
            for constraint in constraints:
                if equal_constraints(constraint, BEC_constraints[BEC_constraint_idx]):
                    contains_BEC_constraint = True
            if contains_BEC_constraint:
                covers.append(1)
            else:
                covers.append(0)

        BEC_constraint_bookkeeping.append(covers)

    BEC_constraint_bookkeeping = np.array(BEC_constraint_bookkeeping)

    return BEC_constraint_bookkeeping

def perform_BEC_constraint_bookkeeping(BEC_constraints, min_subset_constraints_record):
    '''
    Summary: For each constraint in min_subset_constraints_record, see if it matches one of the BEC_constraints
    '''
    BEC_constraint_bookkeeping = [[] for i in range(len(BEC_constraints))]

    # keep track of which demo conveys which of the BEC constraints
    for env_idx, constraints_env in enumerate(min_subset_constraints_record):
        for traj_idx, constraints_traj in enumerate(constraints_env):
            covers = []
            for BEC_constraint_idx in range(len(BEC_constraints)):
                contains_BEC_constraint = False
                for constraint in constraints_traj:
                    if equal_constraints(constraint, BEC_constraints[BEC_constraint_idx]):
                        contains_BEC_constraint = True
                if contains_BEC_constraint:
                    BEC_constraint_bookkeeping[BEC_constraint_idx].append((env_idx, traj_idx))

    return BEC_constraint_bookkeeping

'''
reward weight in 2D with known step cost
'''
def constraints_to_halfspace_matrix(constraints, weights, step_cost_flag):
    '''
    Summary: convert the half space representation of a convex polygon (Ax < b) into the corresponding polytope vertices
    '''
    if step_cost_flag:

        n_boundary_constraints = 4
        A = np.zeros((len(constraints) + n_boundary_constraints, len(constraints[0][0]) - 1))
        b = np.zeros(len(constraints) + n_boundary_constraints)

        for j in range(len(constraints)):
            A[j, :] = np.array([-constraints[j][0][0], -constraints[j][0][1]])
            b[j] = constraints[j][0][2] * weights[0, -1]

        # add the L1 boundary constraints
        A[len(constraints), :] = np.array([1, 0])
        b[len(constraints)] = 1
        A[len(constraints) + 1, :] = np.array([-1, 0])
        b[len(constraints) + 1] = 1
        A[len(constraints) + 2, :] = np.array([0, 1])
        b[len(constraints) + 2] = 1
        A[len(constraints) + 3, :] = np.array([0, -1])
        b[len(constraints) + 3] = 1
    else:
        raise Exception("Not yet implemented.")

    return A, b

def compute_L1_intersections(constraints, weights, step_cost_flag):
    '''
    :param constraints (list of constraints, corresponding to the A of the form Ax >= 0): constraints that comprise the
        BEC region
    :param weights (numpy array): Ground truth reward weights used by agent to derive its optimal policy
    :param step_cost_flag (bool): Indicates that the last weight element is a known step cost
    :return: L1_intersections: List of L1 constraint line segments that intersect with the constraints that comprise
        the BEC region
    '''
    A, b = constraints_to_halfspace_matrix(constraints, weights, step_cost_flag)

    # compute the vertices of the convex polygon formed by the BEC constraints (BEC polygon), in counterclockwise order
    vertices, simplices = compute_polygon_hull(A, b)
    # clean up the indices of the constraints that gave rise to the polygon hull
    polygon_hull_constraints = np.unique(simplices)
    # don't consider the L1 boundary constraints
    polygon_hull_constraints = polygon_hull_constraints[polygon_hull_constraints < len(constraints)]
    polygon_hull_constraint_idxs = polygon_hull_constraints.astype(np.int64)

    # clockwise order
    vertices.reverse()

    # L1 constraints in 2D
    L1_constraints = [[[-1 + abs(weights[0, -1]), 0], [0, 1 - abs(weights[0, -1])]], [[0, 1 - abs(weights[0, -1])], [1 - abs(weights[0, -1]), 0]],
                      [[1 - abs(weights[0, -1]), 0], [0, -1 + abs(weights[0, -1])]], [[0, -1 + abs(weights[0, -1])], [-1 + abs(weights[0, -1]), 0]]]

    # intersect the L1 constraints with the BEC polygon
    L1_intersections = cg.cyrus_beck_2D(np.array(vertices), L1_constraints)

    return L1_intersections, polygon_hull_constraint_idxs

def calculate_BEC_length(constraints, weights, step_cost_flag, feature=None):
    '''
    :param constraints (list of constraints, corresponding to the A of the form Ax >= 0): constraints that comprise the
        BEC region
    :param weights (numpy array): Ground truth reward weights used by agent to derive its optimal policy
    :param step_cost_flag (bool): Indicates that the last weight element is a known step cost
    :param return_midpt (bool): Whether to return the midpoint of the valid BEC area as an estimate of the human's
           current understanding of the reward weights
    :param feature (int): Whether the intersection length should be computed across all feature dimensions (default)
           or only across a specified feature dimension
    :return: total_intersection_length: total length of the intersection between the BEC region and the L1 constraints
    '''
    L1_intersections, polygon_hull_constraint_idxs = compute_L1_intersections(constraints, weights, step_cost_flag)

    # compute the total length of all intersections
    intersection_lengths = cg.compute_lengths(L1_intersections, query_dim=feature)
    total_intersection_length = np.sum(intersection_lengths)

    return total_intersection_length, polygon_hull_constraint_idxs

def compute_BEC_midpt(constraints, weights, step_cost_flag):
    '''
    :param constraints (list of constraints, corresponding to the A of the form Ax >= 0): constraints that comprise the
        BEC region
    :param weights (numpy array): Ground truth reward weights used by agent to derive its optimal policy
    :param step_cost_flag (bool): Indicates that the last weight element is a known step cost
    :return: midpt: Midpoint of the valid BEC area
    '''
    L1_intersections, polygon_hull_constraint_idxs = compute_L1_intersections(constraints, weights, step_cost_flag)

    # compute the total length of all intersections
    intersection_lengths = cg.compute_lengths(L1_intersections)
    total_intersection_length = np.sum(intersection_lengths)

    # estimate the human's reward weight as the mean of the current BEC area (note that this hasn't been tested for
    # non-continguous BEC areas
    d = total_intersection_length / 2
    d_traveled = 0

    for idx, intersection in enumerate(L1_intersections):
        # travel fully along this constraint line
        if d > d_traveled + intersection_lengths[idx]:
            d_traveled += intersection_lengths[idx]
        else:
            t = (d - d_traveled) / intersection_lengths[idx]
            midpt = L1_intersections[idx][0] + t * (L1_intersections[idx][1] - L1_intersections[idx][0])
            midpt = np.append(midpt, -(1 - np.sum(abs(midpt)))) # add in the step cost, currently hardcoded
            break
    midpt = midpt.reshape(1, -1)

    return midpt

def obtain_extreme_vertices(constraints, weights, step_cost_flag):
    '''
    :param constraints (list of constraints, corresponding to the A of the form Ax >= 0): constraints that comprise the
        BEC region
    :param weights (numpy array): Ground truth reward weights used by agent to derive its optimal policy
    :param step_cost_flag (bool): Indicates that the last weight element is a known step cost
    :return: unique_extreme_vertices: List of extreme vertices (vertices comprising the convex hull that have a
        extreme (max or min) value in at least a single axis)
    '''
    L1_intersections, polygon_hull_constraint_idxs = compute_L1_intersections(constraints, weights, step_cost_flag)

    # convert the list of vertices into a numpy array
    vertices = np.concatenate(L1_intersections, axis=0)
    unique_vertices = np.unique(vertices, axis=0)

    extreme_vertices = []

    # find vertices with the maximum and minimum values for each dimension
    for dim_idx in range(unique_vertices.shape[1]):
        extreme_vertices_idxs = np.where(unique_vertices[:, dim_idx] == unique_vertices[:, dim_idx].min())[0]
        extreme_vertices.extend(extreme_vertices_idxs)

        extreme_vertices_idxs = np.where(unique_vertices[:, dim_idx] == unique_vertices[:, dim_idx].max())[0]
        extreme_vertices.extend(extreme_vertices_idxs)

    unique_extreme_vertices_idxs = np.unique(extreme_vertices)

    unique_extreme_vertices = []
    for idx in unique_extreme_vertices_idxs:
        unique_vertex = unique_vertices[idx, :]
        # back out the step cost using the L1 constraint and add it if called for
        if step_cost_flag:
            unique_vertex = np.append(unique_vertex, -(1 - np.sum(abs(unique_vertices[0, :])))) # negative since it's a cost
        unique_extreme_vertices.append(unique_vertex.reshape(1, -1))

    return unique_extreme_vertices

'''
reward weight in 3D with known step cost
'''
def constraints_to_halfspace_matrix_sage(constraints):
    '''
    Summary: convert list of halfspace constraints into an array of halfspace constraints. Add bounding cube

    Halfspace representation of a convex polygon (Ax < b):
    [-1,7,3,4] represents the inequality 7x_1 + 3x_2 + 4x_3 >= 1
    '''
    constraints_stacked = np.vstack(constraints)
    constraints_stacked = np.insert(constraints_stacked, 0, np.zeros((constraints_stacked.shape[0]), dtype='int'), axis=1)
    constraints_stacked = np.vstack((constraints_stacked, np.array([1, 1, 0, 0]), np.array([1, -1, 0, 0]), np.array([1, 0, 1, 0]), np.array([1, 0, -1, 0]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, -1])))
    ieqs = constraints_stacked

    return ieqs

def calc_dihedral_supp(plane1, plane2):
    '''
    Calculate the supplement to the dihedral angle between two planes
    Plane equation: Ax + By + Cz + D = 0

    :param plane1: [D, A, B, C]
    :param plane2: [D, A, B, C]
    :return: supplement
    '''
    angle = np.arccos((plane1[1] * plane2[1] + plane1[2] * plane2[2] + plane1[3] * plane2[3]) /
              (np.sqrt(plane1[1] ** 2 + plane1[2] ** 2 + plane1[3] ** 2) * np.sqrt(
                  plane2[1] ** 2 + plane2[2] ** 2 + plane2[3] ** 2)))

    # take the supplement of the dihedral angle
    supplement = np.pi - angle

    return supplement

def calc_solid_angles(constraint_sets):
    '''
    Use the spherical excess formula to calculate the area of the spherical polygon
    '''
    solid_angles = []

    for constraint_set in constraint_sets:
        if len(constraint_set) == 1:
            # hemisphere
            solid_angles.append(2 * np.pi)
        elif len(constraint_set) == 2:
            # lune / diangle, whose area is 2 * theta (https://www.math.csi.cuny.edu/abhijit/623/spherical-triangle.pdf)
            constraints_stacked = np.vstack(constraint_set)
            constraints_stacked = np.insert(constraints_stacked, 0,
                                            np.zeros((constraints_stacked.shape[0]), dtype='int'), axis=1)
            theta = calc_dihedral_supp(constraints_stacked[0, :], constraints_stacked[1, :])
            solid_angles.append(2 * theta)
        else:
            ieqs = constraints_to_halfspace_matrix_sage(constraint_set)
            poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation

            hrep = np.array(poly.Hrepresentation())

            facet_adj_triu = np.triu(poly.facet_adjacency_matrix())  # upper triangular matrix of fact adjacencies
            boundary_facet_idxs = np.where(hrep[:, 0] != 0)[0]       # boundary facets have a non-zero offset (D in plane eq)

            dihedral_angles = []
            for curr_facet_idx in range(facet_adj_triu.shape[0]):
                # no need to consider adjacent facets to a boundary facet as you're removed from the sphere center
                if curr_facet_idx in boundary_facet_idxs:
                    continue
                adj_facet_idxs = np.where(facet_adj_triu[curr_facet_idx, :] > 0)[0]
                for adj_facet_idx in adj_facet_idxs:
                    # no need to consider the dihedral angles to a bounding facet
                    if adj_facet_idx not in boundary_facet_idxs:
                        # calculate the dihedral angle
                        dihedral_angles.append(calc_dihedral_supp(hrep[curr_facet_idx, :], hrep[adj_facet_idx, :]))

            # spherical excess formula / Girard's theorem
            solid_angles.append(sum(dihedral_angles) - (len(dihedral_angles) - 2) * np.pi)

    return solid_angles

def lies_on_constraint_plane(poly, point):
    '''
    See if this point lies on one of the constraint planes comprising the polyhedron
    '''
    hrep = np.array(poly.Hrepresentation())
    boundary_facet_idxs = np.where(hrep[:, 0] != 0)[0]

    for constraint_idx, constraint in enumerate(hrep):
        if constraint_idx not in boundary_facet_idxs:
            # constraint should go through the origin, so compare to 0
            if np.isclose(point[0] * constraint[1] + point[1] * constraint[2] + point[2] * constraint[3], 0):
                return True

    return False

def obtain_sph_polygon_vertices(polyhedron, add_noise=False):
    '''
    If a vertex of a polyhedron doesn't lie at the origin or isn't vertex of a bounding box (i.e. it actually
    lies on a constraint plane), the project it onto the sphere and return it
    '''
    vertices = np.array(polyhedron.vertices())
    spherical_polygon_vertices = []
    for vertex_idx, vertex in enumerate(vertices):
        if (vertex != np.array([0, 0, 0])).any() and lies_on_constraint_plane(polyhedron, vertex):
            vertex_normed = vertex / np.linalg.norm(vertex)
            if add_noise:
                spherical_polygon_vertices.append(vertex_normed + np.random.sample(3) * 0.001)
            else:
                spherical_polygon_vertices.append(vertex_normed)

    return spherical_polygon_vertices

def sample_human_models_random(constraints, n_models):
    '''
    Summary: sample representative weights that the human could currently attribute to the agent accordingly to a
    uniformly random distribution
    '''

    sample_human_models = []

    if len(constraints) > 0:
        constraints_matrix = np.vstack(constraints)

        # obtain x, y, z coordinates on the sphere that obey the constraints
        valid_sph_x, valid_sph_y, valid_sph_z = cg.sample_valid_region(constraints_matrix, 0, 2 * np.pi, 0, np.pi, 1000, 1000)

        if len(valid_sph_x) == 0:
            print(colored("Was unable to sample valid human models within the BEC (which is likely too small).",
                        'red'))
            return sample_human_models

        # resample coordinates on the sphere within the valid region (for higher density)
        sph_polygon_azi = []
        sph_polygon_ele = []

        for x in range(len(valid_sph_x)):
            azi, ele = cg.cart2sph(valid_sph_x[x], valid_sph_y[x], valid_sph_z[x])
            sph_polygon_azi.append(azi)
            sph_polygon_ele.append(ele)

        min_azi = min(sph_polygon_azi)
        max_azi = max(sph_polygon_azi)
        min_ele = min(sph_polygon_ele)
        max_ele = max(sph_polygon_ele)

        # sample according to the inverse CDF of the uniform distribution along the sphere
        u_low = min_azi / (2 * np.pi)
        u_high = max_azi / (2 * np.pi)
        v_low = (1 - np.cos(min_ele)) / 2
        v_high = (1 - np.cos(max_ele)) / 2

        while len(sample_human_models) < n_models:
            theta = 2 * np.pi * np.random.uniform(low=u_low, high=u_high, size=2 * n_models)
            phi = np.arccos(1 - 2 * np.random.uniform(low=v_low, high=v_high, size=2 * n_models))

            # reject points that fall outside of the desired area

            # see which points on the sphere obey all constraints
            theta_grid, phi_grid = np.meshgrid(theta, phi)
            sph_points = np.array(cg.sph2cat(theta_grid.flatten(), phi_grid.flatten())).T
            dist_to_plane = constraints_matrix.dot(sph_points.T)
            n_constraints_satisfied = np.sum(dist_to_plane >= 0, axis=0)
            n_min_constraints = constraints_matrix.shape[0]

            idx_valid_sph_points = np.where(n_constraints_satisfied == n_min_constraints)[0]
            valid_sph_points = sph_points[idx_valid_sph_points, :]

            # reshape so that each element is a valid weight vector
            valid_sph_points = valid_sph_points.reshape(valid_sph_points.shape[0], 1, valid_sph_points.shape[1])

            sample_human_models.extend(valid_sph_points)

        # trim so that you only return the needed number of models
        sample_human_models = sample_human_models[:n_models]

    else:
        theta = 2 * np.pi * np.random.uniform(low=0, high=1, size=n_models)
        phi = np.arccos(1 - 2 * np.random.uniform(low=0, high=1, size=n_models))

        valid_sph_points = np.array(list(map(cg.sph2cat, theta, phi)))
        # reshape so that each element is a valid weight vector
        valid_sph_points = valid_sph_points.reshape(valid_sph_points.shape[0], 1, valid_sph_points.shape[1])

        sample_human_models.extend(valid_sph_points)

    return sample_human_models

def sample_average_model(constraints, sample_rate=1000):
    '''
    Summary: sample a representative average of the weights the human could currently attribute to the agent
    '''

    sample_human_models = []

    if len(constraints) > 0:
        constraints_matrix = np.vstack(constraints)

        # obtain x, y, z coordinates on the sphere that obey the constraints
        valid_sph_x, valid_sph_y, valid_sph_z = cg.sample_valid_region(constraints_matrix, 0, 2 * np.pi, 0, np.pi, sample_rate, sample_rate)

        # resample coordinates on the sphere within the valid region (for higher density)
        sph_polygon_azi = []
        sph_polygon_ele = []

        for x in range(len(valid_sph_x)):
            azi, ele = cg.cart2sph(valid_sph_x[x], valid_sph_y[x], valid_sph_z[x])
            sph_polygon_azi.append(azi)
            sph_polygon_ele.append(ele)

        min_azi = min(sph_polygon_azi)
        max_azi = max(sph_polygon_azi)
        min_ele = min(sph_polygon_ele)
        max_ele = max(sph_polygon_ele)

        # sample according to the inverse CDF of the uniform distribution along the sphere
        u_low = min_azi / (2 * np.pi)
        u_high = max_azi / (2 * np.pi)
        v_low = (1 - np.cos(min_ele)) / 2
        v_high = (1 - np.cos(max_ele)) / 2

        theta = 2 * np.pi * np.random.uniform(low=u_low, high=u_high, size=sample_rate)
        phi = np.arccos(1 - 2 * np.random.uniform(low=v_low, high=v_high, size=sample_rate))

        # reject points that fall outside of the desired area

        # see which points on the sphere obey all constraints
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        sph_points = np.array(cg.sph2cat(theta_grid.flatten(), phi_grid.flatten())).T
        dist_to_plane = constraints_matrix.dot(sph_points.T)
        n_constraints_satisfied = np.sum(dist_to_plane >= 0, axis=0)
        n_min_constraints = constraints_matrix.shape[0]

        idx_valid_sph_points = np.where(n_constraints_satisfied == n_min_constraints)[0]
        valid_sph_points = sph_points[idx_valid_sph_points, :]

        center = np.mean(valid_sph_points, axis=0)
        center_tiled = np.tile(center, (valid_sph_points.shape[0], 1))
        distances = np.linalg.norm(valid_sph_points - center_tiled, axis=1)

        closest_center_point = valid_sph_points[np.argmin(distances), :]

        # reshape so that each element is a valid weight vector
        closest_center_point = closest_center_point.reshape(1, -1)

        sample_human_models.append(closest_center_point)
    else:
        theta = 2 * np.pi * np.random.uniform(low=0, high=1, size=1)
        phi = np.arccos(1 - 2 * np.random.uniform(low=0, high=1, size=1))

        valid_sph_points = np.array(cg.sph2cat(theta, phi))
        # reshape so that each element is a valid weight vector
        valid_sph_points = valid_sph_points.reshape(1, -1)

        sample_human_models.append(valid_sph_points)

    return sample_human_models

def sample_human_models_uniform(constraints, n_models):
    '''
    Summary: sample representative weights that the human could currently attribute to the agent, by greedily selecting
    points that minimize the maximize distance to any other point (k-centers problem)
    '''

    sample_human_models = []

    if len(constraints) > 0:
        constraints_matrix = np.vstack(constraints)

        # obtain x, y, z coordinates on the sphere that obey the constraints
        valid_sph_x, valid_sph_y, valid_sph_z = cg.sample_valid_region(constraints_matrix, 0, 2 * np.pi, 0, np.pi, 1000, 1000)

        if len(valid_sph_x) == 0:
            print(colored("Was unable to sample valid human models within the BEC (which is likely too small).",
                        'red'))
            return sample_human_models

        # resample coordinates on the sphere within the valid region (for higher density)
        sph_polygon_azi = []
        sph_polygon_ele = []

        for x in range(len(valid_sph_x)):
            azi, ele = cg.cart2sph(valid_sph_x[x], valid_sph_y[x], valid_sph_z[x])
            sph_polygon_azi.append(azi)
            sph_polygon_ele.append(ele)

        min_azi = min(sph_polygon_azi)
        max_azi = max(sph_polygon_azi)
        min_ele = min(sph_polygon_ele)
        max_ele = max(sph_polygon_ele)

        # sample according to the inverse CDF of the uniform distribution along the sphere
        u_low = min_azi / (2 * np.pi)
        u_high = max_azi / (2 * np.pi)
        v_low = (1 - np.cos(min_ele)) / 2
        v_high = (1 - np.cos(max_ele)) / 2

        n_discrete_samples = 100
        while len(sample_human_models) < n_models:
            n_discrete_samples += 20
            theta = 2 * np.pi * np.linspace(u_low, u_high, n_discrete_samples)
            phi = np.arccos(1 - 2 * np.linspace(v_low, v_high, n_discrete_samples))

            # reject points that fall outside of the desired area

            # see which points on the sphere obey all constraints
            theta_grid, phi_grid = np.meshgrid(theta, phi)
            sph_points = np.array(cg.sph2cat(theta_grid.flatten(), phi_grid.flatten())).T
            dist_to_plane = constraints_matrix.dot(sph_points.T)
            n_constraints_satisfied = np.sum(dist_to_plane >= 0, axis=0)
            n_min_constraints = constraints_matrix.shape[0]

            idx_valid_sph_points = np.where(n_constraints_satisfied == n_min_constraints)[0]
            valid_sph_points = sph_points[idx_valid_sph_points, :]

            # greedily select k 'centers' such that the maximum distance from any point to a center is minimized
            # solution is never worse than twice the optimal solution (2-approximation greedy algorithm)
            # https://www.geeksforgeeks.org/k-centers-problem-set-1-greedy-approximate-algorithm/
            if len(valid_sph_points) == n_models:
                sample_human_models.extend(valid_sph_points)
            else:
                pairwise = metrics.pairwise.euclidean_distances(valid_sph_points)
                select_idxs = selectKcities(pairwise.shape[0], pairwise, n_models)
                select_sph_points = valid_sph_points[select_idxs]
                # reshape so that each element is a valid weight vector
                select_sph_points = select_sph_points.reshape(select_sph_points.shape[0], 1, select_sph_points.shape[1])
                sample_human_models.extend(select_sph_points)
    else:
        theta = 2 * np.pi * np.linspace(0, 1, int(np.ceil(np.sqrt(n_models))))
        phi = np.arccos(1 - 2 * np.linspace(0, 1, int(np.ceil(np.sqrt(n_models)))))

        theta_grid, phi_grid = np.meshgrid(theta, phi)

        valid_sph_points = np.array(cg.sph2cat(theta_grid.flatten(), phi_grid.flatten())).T

        # reshape so that each element is a valid weight vector
        valid_sph_points = valid_sph_points.reshape(valid_sph_points.shape[0], 1, valid_sph_points.shape[1])
        valid_sph_points = valid_sph_points[:n_models]

        sample_human_models.extend(valid_sph_points)

    return sample_human_models

def selectKcities(n, weights, k):
    dist = [float('inf')] * n
    centers = []

    # index of city having the maximum distance to it's closest center
    max = 0
    for i in range(k):
        centers.append(max)
        for j in range(n):
            # updating the distance of the cities to their closest centers
            dist[j] = min(dist[j], weights[max][j])

        # updating the index of the city with the maximum distance to it's closest center
        max = np.argmax(dist)

    # maximum distance of a city to a center that is our answer
    # print(dist[max])

    # cities that were chosen to be made centers
    # for i in centers:
    #     print(i, end=" ")

    return centers

def calculate_information_gain(previous_constraints, new_constraints, weights, step_cost_flag):
    if len(previous_constraints) > 0 and len(new_constraints) > 0:
        hypothetical_constraints = new_constraints.copy()
        hypothetical_constraints.extend(previous_constraints)
        hypothetical_constraints = remove_redundant_constraints(hypothetical_constraints, weights, step_cost_flag)

        old_BEC = calc_solid_angles([previous_constraints])[0]
        new_BEC = calc_solid_angles([hypothetical_constraints])[0]

        ig = old_BEC / new_BEC
    elif len(new_constraints) > 0:
        old_BEC = 4 * np.pi
        new_BEC = calc_solid_angles([new_constraints])[0]

        ig = old_BEC / new_BEC
    else:
        ig = 0

    return ig

def calculate_counterfactual_overlap_pct(human_traj, agent_traj):
    # consider both actions and states when finding overlap between trajectories
    # find all actions that are common and in the same sequence between the agent and human actions
    matcher = difflib.SequenceMatcher(None, human_traj, agent_traj, autojunk=False)
    matches = matcher.get_matching_blocks()
    overlap = 0
    for match in matches:
        overlap += match[2]

    # percentage of the human counterfactual that overlaps with the agent's trajectory
    try:
        overlap_pct = overlap / len(human_traj)
    except:
        # the length of the human trajectory may be zero because the human model never converged
        overlap_pct = 0

    return overlap_pct

def update_variable_filter(nonzero_counter):
    '''
    Update the filter to mask the next variable with least number of nonzero constraints along that variable
    '''

    if np.isinf(nonzero_counter).all():
        # if none of the minimum BEC constraints have 0's along any of the variables to begin with, initialize
        # the variable filter to None
        variable_filter = None
    else:
        # if there are still 0's in the minimum BEC constraints, try to filter across those variables and update the
        # zero counter accordingly
        variable_filter = np.zeros((1, nonzero_counter.shape[0]))
        variable_filter[0, np.argmin(nonzero_counter)] = 1
        nonzero_counter[np.argmin(nonzero_counter)] = float('inf')

    return variable_filter, nonzero_counter

def combine_counterfactual_constraints(args):
    '''
    Combine combine the human counterfactual (constraints_env) and one-step deviation (min_subset_constraints) constraints
    '''
    data_loc, env_idx, min_subset_constraints, n_human_models, counterfactual_folder_idx, weights, step_cost_flag = args

    constraints_env_across_models = []
    for model_idx in range(n_human_models):
        with open('models/' + data_loc + '/counterfactual_data_' + str(counterfactual_folder_idx) + '/model' + str(
                model_idx) + '/cf_data_env' + str(
            env_idx).zfill(5) + '.pickle', 'rb') as f:
            best_human_trajs_record_env, constraints_env, human_rewards_env = pickle.load(f)

        # only consider the first best human trajectory (no need to consider partial trajectories)
        constraints_env_across_models.append(constraints_env)

    # reorder such that each subarray is a comparison amongst the models
    constraints_env_across_models_per_traj = [list(itertools.chain.from_iterable(i)) for i in
                                              zip(*constraints_env_across_models)]

    new_min_subset_constraints = []
    for traj_idx, constraints_across_models in enumerate(constraints_env_across_models_per_traj):
        joint_constraints = []
        joint_constraints.extend(constraints_across_models)
        joint_constraints.extend(min_subset_constraints[traj_idx])
        joint_constraints = remove_redundant_constraints(joint_constraints, weights, step_cost_flag)
        new_min_subset_constraints.append(joint_constraints)

    return new_min_subset_constraints