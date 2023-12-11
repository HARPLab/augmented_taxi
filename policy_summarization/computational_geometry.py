import numpy as np
import matplotlib.pyplot as plt
from spherical_geometry import great_circle_arc

'''
2-sphere geometry
'''
def generate_equidistributed_points_on_sphere(num_points):
    """
    Generate `num_points` equidistributed points on the surface of a unit sphere using algorithm suggested in
    "How to generate equidistributed points on the surface of a sphere" by Markus Deserno, 2004.

    Note that this algorithm may return slightly fewer than the number of points requested.
    """
    print('num_points: ', num_points)
    N_count = 0
    a = 4 * np.pi / num_points
    d = np.sqrt(a)
    M_theta = int(np.round(np.pi / d))
    d_theta = np.pi / M_theta
    d_phi = a / d_theta
    points = np.empty((num_points, 3))
    for m in range(M_theta):
        theta = np.pi * (m + 0.5) / M_theta
        M_phi = int(np.round(2 * np.pi * np.sin(theta) / d_phi))
        for n in range(M_phi):
            phi = 2 * np.pi * n / M_phi
            points[N_count, 0] = np.sin(theta) * np.cos(phi)
            points[N_count, 1] = np.sin(theta) * np.sin(phi)
            points[N_count, 2] = np.cos(theta)
            N_count += 1
            if N_count == num_points:
                break

    points = points[0:N_count]

    return points

# following code from https://skeptric.com/calculate-centroid-on-sphere/#Attempting-to-Implement-Algorithm-A1
def distance(x, y, axis=0):
    return np.sqrt((np.power(x - y, 2)).sum(axis=axis))

def geodist(x, y, eps=1e-6):
    dotprod = y.T @ x
    assert ((-1 - eps) <= dotprod).all() and (dotprod <= (1 + eps)).all()
    dotprod = dotprod.clip(-1, 1)
    return np.arccos(dotprod)

def avg_distance(ps, m):
    # Allow m to be a vector *or* a matrix
    if len(m.shape) == 1:
        m = m[:, None]
    return geodist(ps, m).mean(axis=1)

def improve_centroid(c, ps, weights, eps=1e-10):
    '''
    :param c: old centroid
    :param ps: points
    :return: new centroid
    '''
    # weight every particle equally
    # ans = (ps / np.sqrt(1 - np.power(c@ps, 2))).sum(axis=-1)

    # weight different particles differently
    ans = (ps / np.sqrt(1 - np.power(c @ ps, 2) + eps)) * np.tile(weights, (3, 1)) # add a little noise to prevent a degenerate sqrt
    ans = ans.sum(axis=-1)

    norm = np.sqrt(ans @ ans)
    return ans / norm

def fixpoint(f, x0, eps=1e-5, maxiter=1000, **kwargs):
    for _ in range(maxiter):
        x = f(x0, **kwargs)
        if geodist(x, x0) < eps:
            return x
        x0 = x
    raise Exception("Did not converge")

def spherical_centroid(ps, weights, eps=1e-5, maxiter=10000):
    return fixpoint(improve_centroid, np.zeros((3,)), ps=ps, weights=weights, eps=eps, maxiter=maxiter)

def cart2sph(cartesian):
    '''
    Return corresponding spherical coordinates (elevation, azimuth) of a Cartesian point (x, y, z)
    Azimuth (range: 0, 2pi) and elevation (range: 0, pi) axes align with x and z axes respectively
    '''
    if len(cartesian.shape) == 1:
        cartesian = cartesian.reshape(1, -1)

    spherical = np.empty((cartesian.shape[0], 2))
    xy = cartesian[:, 0] ** 2 + cartesian[:, 1] ** 2
    spherical[:, 0] = np.arctan2(np.sqrt(xy), cartesian[:, 2])  # for elevation angle defined from Z-axis down
    spherical[:, 1] = np.arctan2(cartesian[:, 1], cartesian[:, 0]) % (2 * np.pi)  # azimuth

    return spherical

def sph2cart(spherical):
    '''
    Return corresponding Cartesian point (x, y, z) of spherical coordinates (azimuth, elevation)
    Azimuth (range: 0, 2pi) and elevation (range: 0, pi) axes align with x and z axes respectively
    '''
    if len(spherical.shape) == 1:
        spherical = spherical.reshape(1, -1)

    elevations, azimuths = spherical[:, 0], spherical[:, 1]

    cartesian = np.empty((spherical.shape[0], 3))

    cartesian[:, 0] = np.cos(azimuths) * np.sin(elevations)  # x
    cartesian[:, 1] = np.sin(azimuths) * np.sin(elevations)  # y
    cartesian[:, 2] = np.cos(elevations)  # z

    return cartesian

def cart2latlong(coords):
    """
    Convert an array of 3D Cartesian coordinates (x, y, z) to spherical coordinates (latitude, longitude).
    The input `coords` should be a numpy array of shape (N, 3), where N is the number of points and each row contains the x, y, and z coordinates of a point.
    The output is a numpy array of shape (N, 2), where each row contains the latitude and longitude of a point.
    Latitude is between -pi/2 and pi/2 (with 0 being the xy plane), and longitude is between -pi and pi (with 0 being the x axis).
    """
    x, y, z = np.transpose(coords)
    longitude = np.arctan2(y, x)
    r_xy = np.sqrt(x**2 + y**2)
    latitude = np.arctan2(z, r_xy)

    return np.column_stack((latitude, longitude))

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

def sort_points_by_angle(points, center):
    '''
    Sort points in a clockwise order around a center point (when viewing the sphere from the outside)
    and a reference direction (i.e. north)
    Taken from the Spherical Geometry python library: https://github.com/spacetelescope/spherical_geometry
    '''

    north = [0., 0., 1.]
    ang = great_circle_arc.angle(north, center, points)
    pt = [points[i, :] for i in range(points.shape[0])]

    duo = list(zip(pt, ang))
    duo = sorted(duo, key=lambda d: d[1], reverse=True)
    points = np.asarray([d[0] for d in duo])

    return list(points)

def compute_average_point(points):
    average_point = points.mean(axis=0)
    return average_point / np.linalg.norm(average_point)


'''
2D polar geometry
'''

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

'''
2D convex polygon geometry
'''
def _find_clipped_line(n, line, polygon, normals):
    # calculate P1 - P0
    P0 = np.array([line[0][0], line[0][1]])
    P1 = np.array([line[1][0], line[1][1]])
    P1_P0 = np.array([P1[0] - P0[0], P1[1] - P0[1]])

    # calculate the values of P0 - PEi for all edges
    P0_PE = np.zeros(polygon.shape)
    for edge_idx in range(n):
        P0_PE[edge_idx] = [P0[0] - polygon[edge_idx][0], P0[1] - polygon[edge_idx][1]]

    # numerator and denominator for solving for t values, the portion along the P1_P0 line segment that crosses the edges
    numerator = []
    denominator = []

    # find the intersection t value for the line and each of the edges
    for edge_idx in range(n):
        denominator_val = np.dot(normals[edge_idx], P1_P0)

        # if the line is parallel to the edge in question, see if it's also collinear. if not, then simply move
        # on to the next edge
        if denominator_val == 0:

            # need to check if the lines are collinear
            tri_a = np.array([[polygon[edge_idx][0], polygon[edge_idx][1], 1],
                              [polygon[(edge_idx + 1) % n][0], polygon[(edge_idx + 1) % n][1], 1],
                              [P0[0], P0[1], 1]])
            tri_b = np.array([[polygon[edge_idx][0], polygon[edge_idx][1], 1],
                              [polygon[(edge_idx + 1) % n][0], polygon[(edge_idx + 1) % n][1], 1],
                              [P1[0], P1[1], 1]])
            # collinear
            if abs(np.linalg.det(tri_a)) <= 1e-05 and abs(np.linalg.det(tri_b)) <= 1e-05:
                # the line is parallel and to one of the polygon edges. find the intersection between these two lines, if it exists
                minv = np.dot((P0[0], P0[1]), P1_P0)
                maxv = np.dot((P1[0], P1[1]), P1_P0)
                q0 = np.dot(polygon[edge_idx], P1_P0)
                q1 = np.dot(polygon[(edge_idx + 1) % n], P1_P0)
                minq = min(q0, q1)
                maxq = max(q0, q1)

                if (maxq < minv or minq > maxv):
                    # there is no overlap with this edge
                    continue
                else:
                    if minv > minq:
                        ov0 = [P0[0], P0[1]]
                    else:
                        if q0 < q1:
                            ov0 = polygon[edge_idx]
                        else:
                            ov0 = polygon[(edge_idx + 1) % n]

                    if maxv < maxq:
                        ov1 = [P1[0], P1[1]]
                    else:
                        if q0 > q1:
                            ov1 = polygon[edge_idx]
                        else:
                            ov1 = polygon[(edge_idx + 1) % n]
                return np.array([ov0, ov1])

            # check to see if the line is inside or outside the polygon
            t = -np.dot(P1_P0, P0_PE[edge_idx]) / np.dot(P1_P0, P1_P0)

            # if the line is outside, then simply return. if the line is inside, the other edges will clip this line
            if np.dot(normals[edge_idx], P0 + t * P1_P0 - polygon[edge_idx]) > 0:
                return
        else:
            denominator.append(denominator_val)
            numerator.append(np.dot(normals[edge_idx], P0_PE[edge_idx]))

    t = []
    t_enter = []
    t_exit = []
    for edge_idx in range(len(denominator)):
        t.append(-numerator[edge_idx] / denominator[edge_idx])

        # t value for exiting the polygon
        if denominator[edge_idx] > 0:
            t_exit.append(t[edge_idx])
        # t value for entering the polygon
        else:
            t_enter.append(t[edge_idx])

    # find the plane that you entered last
    t_enter.append(0.)
    lb = max(t_enter)
    # find the plane that you exit first
    t_exit.append(1.)
    ub = min(t_exit)

    if lb > ub:
        # this line is outside of the polygon
        clipped_line = None
    else:
        clipped_line = np.array([[line[0][0] + lb * P1_P0[0], line[0][1] + lb * P1_P0[1]],
                        [line[0][0] + ub * P1_P0[0], line[0][1] + ub * P1_P0[1]]])

    return clipped_line

def cyrus_beck_2D(polygon, lines):
    '''
    :param polygon (list of [x,y] of floats, one for each polygonal vertex in clockwise order)
    :param lines (nested list of [[x1, y1], [x2, y2]], one for each L1 constraint)
    :return: clipped_lines (list of [[x1, y1], [x2, y2]])

    Summary: Run the Cyrus Beck algorithm for determining the intersection between a convex polygon and a line
    '''
    # note: the vertices of the polygon needs to be in clockwise order
    n = len(polygon)
    clipped_lines = []

    # calculate the outward facing normals of each of the polygon edges
    normals = np.zeros(polygon.shape)
    for vert_idx in range(n):
        normals[vert_idx] = [polygon[vert_idx][1] - polygon[(vert_idx + 1) % n][1], polygon[(vert_idx + 1) % n][0] - polygon[vert_idx][0]]

    # see if the polygon clips any of the edges
    for line in lines:
        clipped_line = _find_clipped_line(n, line, polygon, normals)

        if clipped_line is not None:
            clipped_lines.append(clipped_line)

    return clipped_lines

def compute_lengths(lines, query_dim=None):
    lengths = np.zeros(len(lines))
    n = 0
    for line in lines:
        if query_dim is None:
            lengths[n] = np.linalg.norm(line[1] - line[0])
        else:
            # limit the computed length to only the dimension corresponding to the desired feature
            lengths[n] = abs(line[1][query_dim] - line[0][query_dim])
        n += 1

    return lengths

def is_polygon_clockwise(vertices):
    # assume that these are the vertices of a convex polygon in with clockwise or counter-clockwise order
    edge_1 = [vertices[1][0] - vertices[0][0], vertices[1][1] - vertices[0][1]]
    edge_2 = [vertices[2][0] - vertices[1][0], vertices[2][1] - vertices[1][1]]

    sign = np.cross(edge_1, edge_2)

    if sign > 0:
        # counter-clockwise
        return False
    else:
        # clockwise
        return True

if __name__ == "__main__":
    # polygon to clip the line by
    polygon = np.array([[1, 1], [3, 3], [3, 1]])


    # lines to clip (test cases)
    lines = np.array([[[-1, 0], [4, 5]], [[1, 1], [5, 1]], [[1.5, 1.25], [1, 2.5]], [[1, 2], [5, 2]], [[2, 0], [3, 10]],
                      [[1.75, 1.25], [2.75, 2.50]]])

    # on triangle
    # lines = np.array([[[1, 1], [5, 1]]])
    # lines = np.array([[[1, 1], [5, 5]]])
    # lines = np.array([[[3, 0], [3, 5]]])

    # parallel and inside
    # lines = np.array([[[1, 2], [5, 2]]])

    # parallel and outside
    # lines = np.array([[[1, 0], [5, 0]]])
    # lines = np.array([[[-1, 0], [4, 5]]])
    # lines = np.array([[[4, 0], [4, 5]]])

    # going completely through triangle
    # lines = np.array([[[2, 0], [2, 5]]])
    # lines = np.array([[[2, 0], [3, 10]]])

    # going partially through triangle
    # lines = np.array([[[1.5, 1.25], [1, 2.5]]])
    # lines = np.array([[[2, 2.25], [2.75, 1.5]]])

    # completely inside
    # lines = np.array([[[1.75, 1.25], [2.75, 2.50]]])

    # clip lines
    clipped_lines = cyrus_beck_2D(polygon, lines)

    # compute line lengths
    line_lengths = compute_lengths(clipped_lines)

    # visualize the clipped lines
    polygon_vis = polygon.tolist()
    polygon_vis.append(polygon[0])
    x_poly, y_ploy = zip(*polygon_vis)
    plt.plot(x_poly, y_ploy)

    if len(clipped_lines) > 0:
        for line in clipped_lines:
            x_lines = []
            y_lines = []

            x_line, y_line = zip(*line)
            x_lines.append(x_line)
            y_lines.append(y_line)
            plt.plot(x_lines[0], y_lines[0])
    plt.show()