import sys
import numpy as np
from policy_summarization import computational_geometry as cg

GROUP_DISTANCE_TOLERANCE = .0001


class PointGrouper(object):
    def group_points(self, points):
        group_assignment = []
        groups = []
        group_index = 0
        for point in points:
            nearest_group_index = self._determine_nearest_group(point, groups)
            if nearest_group_index is None:
                # create new group
                groups.append([point])
                group_assignment.append(group_index)
                group_index += 1
            else:
                group_assignment.append(nearest_group_index)
                groups[nearest_group_index].append(point)

        # note the spherical centroid of each cluster
        spherical_centroids = []
        for group in groups:
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