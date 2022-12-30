import sys
import numpy as np
from policy_summarization import computational_geometry as cg

GROUP_DISTANCE_TOLERANCE = .0001


class PointGrouper(object):
    def group_points(self, shifted_points, orig_points=None, weights=None):
        '''
        :param shifted_points: final resting points after mean-shifting
        :param orig_points: original points before mean-shifting
        :param weights: weights associated with the original points
        :return:
        '''
        group_assignment = []
        shifted_groups = []
        if orig_points is not None:
            orig_groups = []    # for grouping original points
            orig_weights = []   # for grouping weights corresponding to original points
        group_index = 0
        for j, point in enumerate(shifted_points):
            nearest_group_index = self._determine_nearest_group(point, shifted_groups)
            if nearest_group_index is None:
                # create new group
                shifted_groups.append([point])
                group_assignment.append(group_index)
                group_index += 1

                if orig_points is not None:
                    orig_groups.append([orig_points[j]])
                    orig_weights.append([weights[j]])
            else:
                group_assignment.append(nearest_group_index)
                shifted_groups[nearest_group_index].append(point)

                if orig_points is not None:
                    orig_groups[nearest_group_index].append(orig_points[j])
                    orig_weights[nearest_group_index].append(weights[j])

        spherical_centroids = []
        if orig_points is not None:
            for j, group in enumerate(orig_groups):
                if len(group) == 1:
                    spherical_centroids.append(np.array(group[0]).reshape(1, -1))
                else:
                    spherical_centroids.append(cg.spherical_centroid(np.array(group).squeeze().T, orig_weights[j]).reshape(1, -1))
        else:
            for group in shifted_groups:
                spherical_centroids.append(np.array(group[0]).reshape(1, -1))

        return spherical_centroids, np.array(group_assignment)

    def _determine_nearest_group(self, point, groups):
        nearest_group_index = None
        index = 0
        for group in groups:
            distance_to_group = self._distance_to_group(point, group)
            if distance_to_group < GROUP_DISTANCE_TOLERANCE:
                nearest_group_index = index
            index += 1
        return nearest_group_index

    def _distance_to_group(self, point, group):
        min_distance = sys.float_info.max
        for pt in group:
            dist = cg.geodist(np.array(point), np.array(pt))
            if dist < min_distance:
                min_distance = dist
        return min_distance