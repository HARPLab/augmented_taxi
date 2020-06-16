import numpy as np
import matplotlib.pyplot as plt

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

def compute_lengths(lines):
    lengths = np.zeros(len(lines))
    n = 0
    for line in lines:
        lengths[n] = np.linalg.norm(line[1] - line[0])
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