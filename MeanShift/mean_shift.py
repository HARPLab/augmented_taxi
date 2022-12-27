import numpy as np
from . import point_grouper as pg
from policy_summarization import computational_geometry as cg
from policy_summarization import probability_utils as p_utils
import matplotlib
import matplotlib.pyplot as plt
from termcolor import colored

MIN_DISTANCE = 0.00001

# from https://github.com/mattnedrich/MeanShift_py

class MeanShift(object):
    def __init__(self, kernel=p_utils.VMF_pdf):
        self.kernel = kernel

    def cluster(self, points, weights=None, kernel_bandwidth=16, iteration_callback=None, downselect_points=None):
        '''
        :param points:
        :param weights: weights corresponding to points (such that points can be unevenly weighted during clustering)
        :param kernel_bandwidth:
        :param iteration_callback:
        :return:
        '''
        points = np.array([[float(v) for v in point] for point in points])
        if(iteration_callback):
            iteration_callback(points, 0)
        shift_points = np.array(points)
        max_min_dist = 1
        iteration_number = 0

        still_shifting = [True] * points.shape[0]
        while max_min_dist > MIN_DISTANCE:
            # print max_min_dist
            max_min_dist = 0
            iteration_number += 1
            for i in range(0, len(shift_points)):
                if not still_shifting[i]:
                    continue
                p_new = shift_points[i]
                p_new_start = p_new

                if downselect_points is not None:
                    # if a method to downselect neighboring points is provided, use it when shifting to the mean
                    downselected_points, downselect_weights = downselect_points(p_new, points, weights)
                    p_new = self._shift_point(p_new, downselected_points, downselect_weights, kernel_bandwidth)
                else:
                    p_new = self._shift_point(p_new, points, weights, kernel_bandwidth)

                dist = cg.geodist(p_new, p_new_start)
                if dist > max_min_dist:
                    max_min_dist = dist
                if dist < MIN_DISTANCE:
                    still_shifting[i] = False
                shift_points[i] = p_new
            if iteration_callback:
                iteration_callback(shift_points, iteration_number)
            print(max_min_dist)
        point_grouper = pg.PointGrouper()
        cluster_centers, group_assignments = point_grouper.group_points(shift_points.tolist())
        return MeanShiftResult(points, cluster_centers, group_assignments, shift_points)

    def _shift_point(self, query_point, neighboring_points, neighboring_point_weights, kernel_bandwidth):
        if len(neighboring_points) > 0:
            neighboring_points = np.array(neighboring_points)

            # weight neighboring points by distance from query point
            point_weights = self.kernel(query_point, kernel_bandwidth, query_point.shape[0], neighboring_points)

            # also account for the original weights assigned to the neighboring points
            if neighboring_point_weights is not None:
                point_weights *= neighboring_point_weights

            # shift the query point to the weighted spherical centroid of its neighboring points
            shifted_point = cg.spherical_centroid(neighboring_points.T, point_weights)
        else:
            # if there is only one other point in the vicinity, simply shift to that point
            print(colored('shifting to one other point'), 'red')
            shifted_point = query_point.squeeze()

        return shifted_point

class MeanShiftResult:
    def __init__(self, original_points, cluster_centers, cluster_assignments, shifted_points):
        self.original_points = original_points
        self.cluster_centers = cluster_centers
        self.cluster_assignments = cluster_assignments
        self.shifted_points = shifted_points

