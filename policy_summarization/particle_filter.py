import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import sage.all
import sage.geometry.polyhedron.base as Polyhedron
from termcolor import colored
from spherical_geometry import great_circle_arc as gca
import copy
from numpy.random import uniform
from MeanShift import mean_shift as ms
from scipy.stats import norm

import policy_summarization.BEC_helpers as BEC_helpers
import policy_summarization.BEC_visualization as BEC_viz
import policy_summarization.computational_geometry as cg
from policy_summarization import probability_utils as p_utils

fs = 16

class Particles():
    def __init__(self, positions, eps=1e-5):
        self.positions = np.array(positions)
        self.weights = np.ones(len(positions)) / len(positions)
        self.eps = eps

        self.positions_prev = self.positions.copy()
        self.weights_prev = self.weights.copy()

        # Spherical discretization taken from : A new method to subdivide a spherical surface into equal-area cells,
        # Malkin 2019 https://arxiv.org/pdf/1612.03467.pdf
        self.ele_bin_edges = np.array([0, 0.1721331, 0.34555774, 0.52165622, 0.69434434, 0.86288729,
                                  1.04435092, 1.20822512, 1.39251443, 1.57079633, 1.74907822,
                                  1.93336753, 2.09724173, 2.27870536, 2.44724832, 2.61993643,
                                  2.79603491, 2.96945956, 3.1416])
        self.azi_bin_edges = \
            {0: np.array([0., 2.0943951, 4.1887902, 6.28318531]),
             1: np.array([0., 0.6981317, 1.3962634, 2.0943951, 2.7925268,
                          3.4906585, 4.1887902, 4.88692191, 5.58505361, 6.28318531]),
             2: np.array([0., 0.41887902, 0.83775804, 1.25663706, 1.67551608,
                          2.0943951, 2.51327412, 2.93215314, 3.35103216, 3.76991118,
                          4.1887902, 4.60766923, 5.02654825, 5.44542727, 5.86430629,
                          6.28318531]),
             3: np.array([0., 0.31415927, 0.62831853, 0.9424778, 1.25663706,
                          1.57079633, 1.88495559, 2.19911486, 2.51327412, 2.82743339,
                          3.14159265, 3.45575192, 3.76991118, 4.08407045, 4.39822972,
                          4.71238898, 5.02654825, 5.34070751, 5.65486678, 5.96902604,
                          6.28318531]),
             4: np.array([0., 0.26179939, 0.52359878, 0.78539816, 1.04719755,
                          1.30899694, 1.57079633, 1.83259571, 2.0943951, 2.35619449,
                          2.61799388, 2.87979327, 3.14159265, 3.40339204, 3.66519143,
                          3.92699082, 4.1887902, 4.45058959, 4.71238898, 4.97418837,
                          5.23598776, 5.49778714, 5.75958653, 6.02138592, 6.28318531]),
             5: np.array([0., 0.20943951, 0.41887902, 0.62831853, 0.83775804,
                          1.04719755, 1.25663706, 1.46607657, 1.67551608, 1.88495559,
                          2.0943951, 2.30383461, 2.51327412, 2.72271363, 2.93215314,
                          3.14159265, 3.35103216, 3.56047167, 3.76991118, 3.97935069,
                          4.1887902, 4.39822972, 4.60766923, 4.81710874, 5.02654825,
                          5.23598776, 5.44542727, 5.65486678, 5.86430629, 6.0737458,
                          6.28318531]),
             6: np.array([0., 0.20943951, 0.41887902, 0.62831853, 0.83775804,
                          1.04719755, 1.25663706, 1.46607657, 1.67551608, 1.88495559,
                          2.0943951, 2.30383461, 2.51327412, 2.72271363, 2.93215314,
                          3.14159265, 3.35103216, 3.56047167, 3.76991118, 3.97935069,
                          4.1887902, 4.39822972, 4.60766923, 4.81710874, 5.02654825,
                          5.23598776, 5.44542727, 5.65486678, 5.86430629, 6.0737458,
                          6.28318531]),
             7: np.array([0., 0.17453293, 0.34906585, 0.52359878, 0.6981317,
                          0.87266463, 1.04719755, 1.22173048, 1.3962634, 1.57079633,
                          1.74532925, 1.91986218, 2.0943951, 2.26892803, 2.44346095,
                          2.61799388, 2.7925268, 2.96705973, 3.14159265, 3.31612558,
                          3.4906585, 3.66519143, 3.83972435, 4.01425728, 4.1887902,
                          4.36332313, 4.53785606, 4.71238898, 4.88692191, 5.06145483,
                          5.23598776, 5.41052068, 5.58505361, 5.75958653, 5.93411946,
                          6.10865238, 6.28318531]),
             8: np.array([0., 0.17453293, 0.34906585, 0.52359878, 0.6981317,
                          0.87266463, 1.04719755, 1.22173048, 1.3962634, 1.57079633,
                          1.74532925, 1.91986218, 2.0943951, 2.26892803, 2.44346095,
                          2.61799388, 2.7925268, 2.96705973, 3.14159265, 3.31612558,
                          3.4906585, 3.66519143, 3.83972435, 4.01425728, 4.1887902,
                          4.36332313, 4.53785606, 4.71238898, 4.88692191, 5.06145483,
                          5.23598776, 5.41052068, 5.58505361, 5.75958653, 5.93411946,
                          6.10865238, 6.28318531]),
             9: np.array([0., 0.17453293, 0.34906585, 0.52359878, 0.6981317,
                          0.87266463, 1.04719755, 1.22173048, 1.3962634, 1.57079633,
                          1.74532925, 1.91986218, 2.0943951, 2.26892803, 2.44346095,
                          2.61799388, 2.7925268, 2.96705973, 3.14159265, 3.31612558,
                          3.4906585, 3.66519143, 3.83972435, 4.01425728, 4.1887902,
                          4.36332313, 4.53785606, 4.71238898, 4.88692191, 5.06145483,
                          5.23598776, 5.41052068, 5.58505361, 5.75958653, 5.93411946,
                          6.10865238, 6.28318531]),
             10: np.array([0., 0.17453293, 0.34906585, 0.52359878, 0.6981317,
                           0.87266463, 1.04719755, 1.22173048, 1.3962634, 1.57079633,
                           1.74532925, 1.91986218, 2.0943951, 2.26892803, 2.44346095,
                           2.61799388, 2.7925268, 2.96705973, 3.14159265, 3.31612558,
                           3.4906585, 3.66519143, 3.83972435, 4.01425728, 4.1887902,
                           4.36332313, 4.53785606, 4.71238898, 4.88692191, 5.06145483,
                           5.23598776, 5.41052068, 5.58505361, 5.75958653, 5.93411946,
                           6.10865238, 6.28318531]),
             11: np.array([0., 0.20943951, 0.41887902, 0.62831853, 0.83775804,
                           1.04719755, 1.25663706, 1.46607657, 1.67551608, 1.88495559,
                           2.0943951, 2.30383461, 2.51327412, 2.72271363, 2.93215314,
                           3.14159265, 3.35103216, 3.56047167, 3.76991118, 3.97935069,
                           4.1887902, 4.39822972, 4.60766923, 4.81710874, 5.02654825,
                           5.23598776, 5.44542727, 5.65486678, 5.86430629, 6.0737458,
                           6.28318531]),
             12: np.array([0., 0.20943951, 0.41887902, 0.62831853, 0.83775804,
                           1.04719755, 1.25663706, 1.46607657, 1.67551608, 1.88495559,
                           2.0943951, 2.30383461, 2.51327412, 2.72271363, 2.93215314,
                           3.14159265, 3.35103216, 3.56047167, 3.76991118, 3.97935069,
                           4.1887902, 4.39822972, 4.60766923, 4.81710874, 5.02654825,
                           5.23598776, 5.44542727, 5.65486678, 5.86430629, 6.0737458,
                           6.28318531]),
             13: np.array([0., 0.26179939, 0.52359878, 0.78539816, 1.04719755,
                           1.30899694, 1.57079633, 1.83259571, 2.0943951, 2.35619449,
                           2.61799388, 2.87979327, 3.14159265, 3.40339204, 3.66519143,
                           3.92699082, 4.1887902, 4.45058959, 4.71238898, 4.97418837,
                           5.23598776, 5.49778714, 5.75958653, 6.02138592, 6.28318531]),
             14: np.array([0., 0.31415927, 0.62831853, 0.9424778, 1.25663706,
                           1.57079633, 1.88495559, 2.19911486, 2.51327412, 2.82743339,
                           3.14159265, 3.45575192, 3.76991118, 4.08407045, 4.39822972,
                           4.71238898, 5.02654825, 5.34070751, 5.65486678, 5.96902604,
                           6.28318531]),
             15: np.array([0., 0.41887902, 0.83775804, 1.25663706, 1.67551608,
                           2.0943951, 2.51327412, 2.93215314, 3.35103216, 3.76991118,
                           4.1887902, 4.60766923, 5.02654825, 5.44542727, 5.86430629,
                           6.28318531]),
             16: np.array([0., 0.6981317, 1.3962634, 2.0943951, 2.7925268,
                           3.4906585, 4.1887902, 4.88692191, 5.58505361, 6.28318531]),
             17: np.array([0., 2.0943951, 4.1887902, 6.28318531])}

        self.ele_bin_edges_20 = np.array([0, np.pi/4, np.pi/2, 3 * np.pi/4, np.pi + eps])

        self.azi_bin_edges_20 = \
            {0: np.array([0, 2 * np.pi/3, 4 * np.pi/3, 2 * np.pi + eps]),
             1: np.array([0, 2 * np.pi/7, 4 * np.pi/7, 6 * np.pi/7, 8 * np.pi/7, 10 * np.pi/7, 12 * np.pi/7, 2 * np.pi + eps]),
             2: np.array([0, 2 * np.pi/7, 4 * np.pi/7, 6 * np.pi/7, 8 * np.pi/7, 10 * np.pi/7, 12 * np.pi/7, 2 * np.pi + eps]),
             3: np.array([0, 2 * np.pi/3, 4 * np.pi/3, 2 * np.pi + eps])}

        self.binned = False

        self.cluster_centers = None
        self.cluster_weights = None
        self.cluster_assignments = None

        self.bin_neighbor_mapping = self.initialize_bin_neighbor_mapping()
        self.bin_neighbor_mapping_20 = self.initialize_bin_neighbor_mapping_20()
        self.bin_particle_mapping = None
        self.bin_weight_mapping = None

        self.integral_prob_uniform = 0.8029412189847138   # the total probability on the uniform half of the custom uniform + VMF distribution
        self.integral_prob_VMF = 0.19705878101528612      # the total probability on the VMF half of the custom uniform + VMF distribution
        self.VMF_kappa = 4                                # the concentration parameter of the VMF distribution

    def reinitialize(self, positions):
        self.positions = np.array(positions)
        self.weights = np.ones(len(positions)) / len(positions)

    def initialize_bin_neighbor_mapping(self):
        '''
        Calculate and store the neighboring bins in a 406-cell discretization of the 2-sphere
        Discretization provided by A new method to subdivide a spherical surface into equal-area cells
        https://arxiv.org/pdf/1612.03467.pdf
        '''
        # elements are (elevation_bin, azimuth_bin) pairs
        bin_neighbor_mapping = {0: [[] for _ in range(3)], 1: [[] for _ in range(9)], 2: [[] for _ in range(15)], 3: [[] for _ in range(20)], 4: [[] for _ in range(24)],
                       5: [[] for _ in range(30)],
                       6: [[] for _ in range(30)], 7: [[] for _ in range(36)], 8: [[] for _ in range(36)], 9: [[] for _ in range(36)], 10: [[] for _ in range(36)],
                       11: [[] for _ in range(30)], 12: [[] for _ in range(30)], 13: [[] for _ in range(24)], 14: [[] for _ in range(20)], 15: [[] for _ in range(15)],
                       16: [[] for _ in range(9)], 17: [[] for _ in range(3)]}

        for elevation_bin_idx in bin_neighbor_mapping.keys():
            if elevation_bin_idx == 0 or elevation_bin_idx == 17:
                continue

            n_azimuth_bins = len(bin_neighbor_mapping[elevation_bin_idx])
            for azimuth_bin_idx, _ in enumerate(bin_neighbor_mapping[elevation_bin_idx]):
                if azimuth_bin_idx == 0:
                    # left & right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx, n_azimuth_bins - 1), (elevation_bin_idx, azimuth_bin_idx + 1)])
                    # top left & bottom left
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 1), (elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 1)])

                    # considering top & top right
                    if self.azi_bin_edges[elevation_bin_idx][azimuth_bin_idx + 1] + self.eps < self.azi_bin_edges[elevation_bin_idx - 1][azimuth_bin_idx + 1]:
                        # simply add the top
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx - 1, 0)])
                    else:
                        # else add the top and top right
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx - 1, 0), (elevation_bin_idx - 1, 1)])

                    # considering bottom & bottom right
                    if self.azi_bin_edges[elevation_bin_idx][azimuth_bin_idx + 1] + self.eps < self.azi_bin_edges[elevation_bin_idx + 1][azimuth_bin_idx + 1]:
                        # simply add the bottom
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx + 1, 0)])
                    else:
                        # else add the bottom and bottom right
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx + 1, 0), (elevation_bin_idx + 1, 1)])
                elif azimuth_bin_idx == n_azimuth_bins - 1:
                    # left & right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx, azimuth_bin_idx - 1), (elevation_bin_idx, 0)])

                    # top right & bottom right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx - 1, 0), (elevation_bin_idx + 1, 0)])

                    # considering top left & top left
                    if self.azi_bin_edges[elevation_bin_idx][azimuth_bin_idx] - self.eps > self.azi_bin_edges[elevation_bin_idx - 1][-2]:
                        # simply add the top
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 1)])
                    else:
                        # else add the top and top left
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 1), (elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 2)])

                    # considering bottom & bottom left
                    if self.azi_bin_edges[elevation_bin_idx][azimuth_bin_idx] - self.eps > self.azi_bin_edges[elevation_bin_idx + 1][-2]:
                        # simply add the bottom
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 1)])
                    else:
                        # else add the bottom and bottom left
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 1), (elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 2)])
                else:
                    # left and right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx, azimuth_bin_idx - 1), (elevation_bin_idx, azimuth_bin_idx + 1)])

                    left_edge = self.azi_bin_edges[elevation_bin_idx][azimuth_bin_idx]
                    right_edge = self.azi_bin_edges[elevation_bin_idx][azimuth_bin_idx + 1]
                    # top
                    top_bin_left_edge = np.digitize(left_edge - self.eps, self.azi_bin_edges[elevation_bin_idx - 1])
                    top_bin_right_edge = np.digitize(right_edge + self.eps, self.azi_bin_edges[elevation_bin_idx - 1])
                    if top_bin_left_edge == top_bin_right_edge:
                        # the edge is right skewed, so substract 1
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx - 1, top_bin_right_edge - 1)])
                    else:
                        for x in range(top_bin_left_edge, top_bin_right_edge + 1):
                            bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                                [(elevation_bin_idx - 1, x - 1)])
                    # bottom
                    bottom_bin_left_edge = np.digitize(left_edge - self.eps,
                                self.azi_bin_edges[elevation_bin_idx + 1])
                    bottom_bin_right_edge = np.digitize(right_edge + self.eps,
                                      self.azi_bin_edges[elevation_bin_idx + 1])
                    if bottom_bin_left_edge == bottom_bin_right_edge:
                        # the edge is right skewed, so substract 1
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx + 1, bottom_bin_right_edge - 1)])
                    else:
                        for x in range(bottom_bin_left_edge, bottom_bin_right_edge + 1):
                            bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                                [(elevation_bin_idx + 1, x - 1)])

        # handle corner cases
        bin_neighbor_mapping[0][0].extend([(1, 0), (1, 1), (1, 2), (0, 1), (0, 2), (1, 3), (1, 8)])
        bin_neighbor_mapping[0][1].extend([(1, 3), (1, 4), (1, 5), (0, 0), (0, 2), (1, 2), (1, 6)])
        bin_neighbor_mapping[0][2].extend([(1, 6), (1, 7), (1, 8), (0, 0), (0, 1), (1, 5), (1, 0)])

        bin_neighbor_mapping[17][0].extend([(16, 0), (16, 1), (16, 2), (17, 1), (17, 2), (16, 8), (16, 3)])
        bin_neighbor_mapping[17][1].extend([(16, 3), (16, 4), (16, 5), (17, 0), (17, 2), (16, 2), (16, 6)])
        bin_neighbor_mapping[17][2].extend([(16, 6), (16, 7), (16, 8), (17, 0), (17, 1), (16, 5), (16, 0)])

        return bin_neighbor_mapping

    def initialize_bin_neighbor_mapping_20(self):
        '''
        Calculate and store the neighboring bins in a 20-cell discretization of the 2-sphere
        Discretization provided by A New Equal-area Isolatitudinal Grid on a Spherical Surface
        https://iopscience.iop.org/article/10.3847/1538-3881/ab3a44/pdf
        '''

        # elements are (elevation_bin, azimuth_bin) pairs
        bin_neighbor_mapping = {0: [[] for _ in range(3)], 1: [[] for _ in range(7)], 2: [[] for _ in range(7)],
                                3: [[] for _ in range(3)]}

        for elevation_bin_idx in bin_neighbor_mapping.keys():
            if elevation_bin_idx == 0 or elevation_bin_idx == 3:
                continue

            n_azimuth_bins = len(bin_neighbor_mapping[elevation_bin_idx])
            for azimuth_bin_idx, _ in enumerate(bin_neighbor_mapping[elevation_bin_idx]):
                if azimuth_bin_idx == 0:
                    # left & right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx, n_azimuth_bins - 1), (elevation_bin_idx, azimuth_bin_idx + 1)])
                    # top left & bottom left
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 1),
                         (elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 1)])

                    # considering top & top right
                    if self.azi_bin_edges_20[elevation_bin_idx][azimuth_bin_idx + 1] + self.eps < \
                            self.azi_bin_edges_20[elevation_bin_idx - 1][azimuth_bin_idx + 1]:
                        # simply add the top
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx - 1, 0)])
                    else:
                        # else add the top and top right
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx - 1, 0), (elevation_bin_idx - 1, 1)])

                    # considering bottom & bottom right
                    if self.azi_bin_edges_20[elevation_bin_idx][azimuth_bin_idx + 1] + self.eps < \
                            self.azi_bin_edges_20[elevation_bin_idx + 1][azimuth_bin_idx + 1]:
                        # simply add the bottom
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend([(elevation_bin_idx + 1, 0)])
                    else:
                        # else add the bottom and bottom right
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx + 1, 0), (elevation_bin_idx + 1, 1)])
                elif azimuth_bin_idx == n_azimuth_bins - 1:
                    # left & right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx, azimuth_bin_idx - 1), (elevation_bin_idx, 0)])

                    # top right & bottom right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx - 1, 0), (elevation_bin_idx + 1, 0)])

                    # considering top left & top left
                    if self.azi_bin_edges_20[elevation_bin_idx][azimuth_bin_idx] - self.eps > \
                            self.azi_bin_edges_20[elevation_bin_idx - 1][-2]:
                        # simply add the top
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 1)])
                    else:
                        # else add the top and top left
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 1),
                             (elevation_bin_idx - 1, len(bin_neighbor_mapping[elevation_bin_idx - 1]) - 2)])

                    # considering bottom & bottom left
                    if self.azi_bin_edges_20[elevation_bin_idx][azimuth_bin_idx] - self.eps > \
                            self.azi_bin_edges_20[elevation_bin_idx + 1][-2]:
                        # simply add the bottom
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 1)])
                    else:
                        # else add the bottom and bottom left
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 1),
                             (elevation_bin_idx + 1, len(bin_neighbor_mapping[elevation_bin_idx + 1]) - 2)])
                else:
                    # left and right
                    bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                        [(elevation_bin_idx, azimuth_bin_idx - 1), (elevation_bin_idx, azimuth_bin_idx + 1)])

                    left_edge = self.azi_bin_edges_20[elevation_bin_idx][azimuth_bin_idx]
                    right_edge = self.azi_bin_edges_20[elevation_bin_idx][azimuth_bin_idx + 1]
                    # top
                    top_bin_left_edge = np.digitize(left_edge - self.eps, self.azi_bin_edges_20[elevation_bin_idx - 1])
                    top_bin_right_edge = np.digitize(right_edge + self.eps, self.azi_bin_edges_20[elevation_bin_idx - 1])
                    if top_bin_left_edge == top_bin_right_edge:
                        # the edge is right skewed, so substract 1
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx - 1, top_bin_right_edge - 1)])
                    else:
                        for x in range(top_bin_left_edge, top_bin_right_edge + 1):
                            bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                                [(elevation_bin_idx - 1, x - 1)])
                    # bottom
                    bottom_bin_left_edge = np.digitize(left_edge - self.eps,
                                                       self.azi_bin_edges_20[elevation_bin_idx + 1])
                    bottom_bin_right_edge = np.digitize(right_edge + self.eps,
                                                        self.azi_bin_edges_20[elevation_bin_idx + 1])
                    if bottom_bin_left_edge == bottom_bin_right_edge:
                        # the edge is right skewed, so substract 1
                        bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                            [(elevation_bin_idx + 1, bottom_bin_right_edge - 1)])
                    else:
                        for x in range(bottom_bin_left_edge, bottom_bin_right_edge + 1):
                            bin_neighbor_mapping[elevation_bin_idx][azimuth_bin_idx].extend(
                                [(elevation_bin_idx + 1, x - 1)])

        # handle corner cases
        bin_neighbor_mapping[0][0].extend([(1, 0), (1, 1), (1, 2), (0, 1), (0, 2), (1, 6)])
        bin_neighbor_mapping[0][1].extend([(1, 2), (1, 3), (1, 4), (0, 0), (0, 2)])
        bin_neighbor_mapping[0][2].extend([(1, 4), (1, 5), (1, 6), (1, 0), (0, 1), (0, 0)])

        bin_neighbor_mapping[3][0].extend([(2, 0), (2, 1), (2, 2), (3, 1), (3, 2), (2, 6)])
        bin_neighbor_mapping[3][1].extend([(2, 2), (2, 3), (2, 4), (3, 0), (3, 2)])
        bin_neighbor_mapping[3][2].extend([(2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (2, 0)])

        return bin_neighbor_mapping

    def bin_particles(self):
        '''
        Bin a particle into one of the 406-cell discretization of the 2-sphere
        Discretization provided by A new method to subdivide a spherical surface into equal-area cells
        https://arxiv.org/pdf/1612.03467.pdf
        '''

        # sort particles into bins
        positions_spherical = cg.cart2sph(self.positions.squeeze())

        # dictionary with keys based on elevation, and values based on associated number of azimuth bins
        bin_weight_mapping = {0: np.zeros(3), 1: np.zeros(9), 2: np.zeros(15), 3: np.zeros(20), 4: np.zeros(24),
                       5: np.zeros(30),
                       6: np.zeros(30), 7: np.zeros(36), 8: np.zeros(36), 9: np.zeros(36), 10: np.zeros(36),
                       11: np.zeros(30), 12: np.zeros(30), 13: np.zeros(24), 14: np.zeros(20), 15: np.zeros(15),
                       16: np.zeros(9), 17: np.zeros(3)}

        # contains the indices of points contained within each bin
        bin_particle_mapping = {0: [[] for _ in range(3)], 1: [[] for _ in range(9)], 2: [[] for _ in range(15)], 3: [[] for _ in range(20)], 4: [[] for _ in range(24)],
                       5: [[] for _ in range(30)],
                       6: [[] for _ in range(30)], 7: [[] for _ in range(36)], 8: [[] for _ in range(36)], 9: [[] for _ in range(36)], 10: [[] for _ in range(36)],
                       11: [[] for _ in range(30)], 12: [[] for _ in range(30)], 13: [[] for _ in range(24)], 14: [[] for _ in range(20)], 15: [[] for _ in range(15)],
                       16: [[] for _ in range(9)], 17: [[] for _ in range(3)]}

        elevations, azimuths = zip(*positions_spherical)

        # sort the points into elevation bins
        elevation_bins = np.digitize(elevations, self.ele_bin_edges)

        # for each point, sort into the correct azimuth bin
        for j, elevation_bin in enumerate(elevation_bins):
            azimuth_bin = np.digitize(azimuths[j], self.azi_bin_edges[elevation_bin - 1])
            bin_weight_mapping[elevation_bin - 1][azimuth_bin - 1] += self.weights[j]

            bin_particle_mapping[elevation_bin - 1][azimuth_bin - 1].append(j)

        return bin_particle_mapping, bin_weight_mapping

    def bin_particles_20(self):
        '''
        Bin a particle into one of the 20-cell discretization of the 2-sphere
        Discretization provided by A New Equal-area Isolatitudinal Grid on a Spherical Surface
        https://iopscience.iop.org/article/10.3847/1538-3881/ab3a44/pdf
        '''

        # sort particles into bins
        positions_spherical = cg.cart2sph(self.positions.squeeze())

        # dictionary with keys based on elevation, and values based on associated number of azimuth bins
        bin_weight_mapping = {0: np.zeros(3), 1: np.zeros(7), 2: np.zeros(7), 3: np.zeros(3)}

        # contains the indices of points contained within each bin
        bin_particle_mapping = {0: [[] for _ in range(3)], 1: [[] for _ in range(7)], 2: [[] for _ in range(7)], 3: [[] for _ in range(3)]}

        elevations, azimuths = zip(*positions_spherical)

        # sort the points into elevation bins
        elevation_bins = np.digitize(elevations, self.ele_bin_edges_20)

        # for each point, sort into the correct azimuth bin
        for j, elevation_bin in enumerate(elevation_bins):
            azimuth_bin = np.digitize(azimuths[j], self.azi_bin_edges_20[elevation_bin - 1])
            bin_weight_mapping[elevation_bin - 1][azimuth_bin - 1] += self.weights[j]

            # record indices of points belonging to each bin
            bin_particle_mapping[elevation_bin - 1][azimuth_bin - 1].append(j)

        return bin_particle_mapping, bin_weight_mapping

    def meanshift_plusplus_neighbors(self, query_point, points, weights):
        '''
        Implement "MeanShift++: Extremely Fast Mode-Seeking With Applications to Segmentation and Object Tracking" by
        Jang et al. from CVPR 2021 on 2-sphere
        '''
        # bin this point
        query_point_sph = cg.cart2sph(query_point)
        query_elevation_bin = np.digitize(query_point_sph[0][0], self.ele_bin_edges)
        query_azimuth_bin = np.digitize(query_point_sph[0][1], self.azi_bin_edges[query_elevation_bin - 1])

        # find neighbors
        neighbor_bins = self.bin_neighbor_mapping[query_elevation_bin - 1][query_azimuth_bin - 1]

        # obtain points of neighbors
        neighboring_particles = []
        neighboring_weights = []
        for neighbor_bin in neighbor_bins:
            elevation_bin, azimuth_bin = neighbor_bin
            particle_idxs = self.bin_particle_mapping[elevation_bin][azimuth_bin]
            for idx in particle_idxs:
                neighboring_particles.append(points[idx])
                neighboring_weights.append(weights[idx])

        # obtain points in this bin as well
        particle_idxs = self.bin_particle_mapping[query_elevation_bin - 1][query_azimuth_bin - 1]
        for idx in particle_idxs:
            neighboring_particles.append(points[idx])
            neighboring_weights.append(weights[idx])

        return np.array(neighboring_particles), np.array(neighboring_weights)

    def cluster(self):
        if self.binned == False:
            bin_particle_mapping, bin_weight_mapping = self.bin_particles()
            self.bin_particle_mapping = bin_particle_mapping
            self.bin_weight_mapping = bin_weight_mapping
            self.binned = True

        # cluster particles using mean-shift and store the cluster centers
        mean_shifter = ms.MeanShift()
        # only use a subset of neighboring points to perform meanshift clustering
        # mean_shift_result = mean_shifter.cluster(self.positions.squeeze(), self.weights, downselect_points=self.meanshift_plusplus_neighbors)
        # use all points to perform meanshift clustering
        mean_shift_result = mean_shifter.cluster(self.positions.squeeze(), self.weights)
        self.cluster_centers = mean_shift_result.cluster_centers
        self.cluster_assignments = mean_shift_result.cluster_assignments

        # assign weights to cluster centers by summing up the weights of constituent particles
        cluster_weights = []
        for j, cluster_center in enumerate(self.cluster_centers):
            cluster_weights.append(sum(self.weights[np.where(mean_shift_result.cluster_assignments == j)[0]]))
        self.cluster_weights = cluster_weights

    def reweight(self, constraints):
        '''
        :param constraints: normal of constraints / mean direction of VMF
        :param k: concentration parameter of VMF
        :return: probability of x under this composite distribution (uniform + VMF)
        '''
        for j, x in enumerate(self.positions):
            self.weights[j] = self.weights[j] * self.observation_probability(x, constraints)

    def plot(self, centroid=None, fig=None, ax=None, cluster_centers=None, cluster_weights=None,
                       cluster_assignments=None, plot_prev=False):
        if plot_prev:
            particle_positions = self.positions_prev
            particle_weights = self.weights_prev
        else:
            particle_positions = self.positions
            particle_weights = self.weights

        vis_scale_factor = 10 * len(particle_positions)
        if fig == None:
            fig = plt.figure()
        if ax == None:
            ax = fig.add_subplot(projection='3d')

        if cluster_centers is not None:
            for cluster_id, cluster_center in enumerate(cluster_centers):
                ax.scatter(cluster_center[0][0], cluster_center[0][1], cluster_center[0][2],
                           # s=500 * cluster_weights[cluster_id], c='red', marker='+')
                           s=1000, c='red', marker='+')

            print("# of clusters: {}".format(len(np.unique(cluster_assignments))))

        if cluster_assignments is not None:
            # color the particles according to their cluster assignments if provided
            plt.set_cmap("gist_rainbow")
            ax.scatter(particle_positions[:, 0, 0], particle_positions[:, 0, 1], particle_positions[:, 0, 2],
                       s=particle_weights * vis_scale_factor, c=cluster_assignments)
        else:

            ax.scatter(particle_positions[:, 0, 0], particle_positions[:, 0, 1], particle_positions[:, 0, 2],
                       s=particle_weights * vis_scale_factor, color='tab:blue')

        # if centroid == None:
        #     centroid = cg.spherical_centroid(particle_positions.squeeze().T, particle_weights)
        # ax.scatter(centroid[0], centroid[1], centroid[2], marker='o', c='r', s=100)

        if matplotlib.get_backend() == 'TkAgg':
            ax.set_xlabel('$\mathregular{w_0}$: Mud')
            ax.set_ylabel('$\mathregular{w_1}$: Recharge')
            ax.set_zlabel('$\mathregular{w_2}$: Action')

    def resample_from_index(self, indexes, K=1):
        # resample
        self.positions = self.positions[indexes]

        # sort particles into bins
        positions_spherical = cg.cart2sph(self.positions.squeeze())
        elevations = positions_spherical[:, 0]
        azimuths = positions_spherical[:, 1]

        max_ele_dist = max(elevations) - min(elevations)

        azimuths_sorted = np.sort(azimuths)
        azi_dists = np.empty(len(azimuths))
        azi_dists[0:-1] = np.diff(azimuths_sorted)
        azi_dists[-1] = min(2 * np.pi - (max(azimuths_sorted) - min(azimuths_sorted)), max(azimuths_sorted) - min(azimuths_sorted))

        if np.std(azi_dists[azi_dists > self.eps]) < 0.01 and np.std(azimuths_sorted) > 1:
            # the particles are relatively evenly spaced out across the full range of azimuth
            max_azi_dist = 2 * np.pi
        else:
            # take the largest gap/azimuth distance between two consecutive particles
            max_azi_dist = max(azi_dists)

        # noise suggested by "Novel approach to nonlinear/non-Gaussian Bayesian state estimation" by Gordon et al.
        noise = np.array([np.random.normal(scale=max_ele_dist, size=len(positions_spherical)),
                          np.random.normal(scale=max_azi_dist, size=len(positions_spherical))]).T
        noise *= K * positions_spherical.shape[0] ** (-1/positions_spherical.shape[1])

        positions_spherical += noise

        for j in range(0, len(positions_spherical)):
            self.positions[j, :] = np.array(cg.sph2cart(positions_spherical[j, :])).reshape(1, -1)

        # reset the weights
        self.weights = np.ones(len(self.positions)) / len(self.positions)

    # http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
    def normalized_weighted_variance(self, spherical_centroid=None):
        sum_weights = np.sum(self.weights)

        eff_dof = (sum_weights ** 2) / (np.sum(self.weights ** 2))

        if spherical_centroid is None:
            spherical_centroid = cg.spherical_centroid(self.positions.squeeze().T)

        # take the trace of the covariance matrix as the variance measure
        weighted_var = 0
        for j, particle_position in enumerate(self.positions):
            # geodesic distance
            weighted_var += self.weights[j] * gca.length(spherical_centroid, particle_position,
                                                              degrees=False) ** 2 / sum_weights * (
                                        eff_dof / (eff_dof - 1))

        # normalize
        weighted_var = weighted_var / len(self.positions)

        return weighted_var[0]

    def calc_entropy(self):
        '''
        Implement entropy measure for particle filter as described in Learning Human Ergonomic Preferences for Handovers (Bestick et al. 2018)

        Spherical discretization taken from : A new method to subdivide a spherical surface into equal-area cells,
        Malkin 2019 https://arxiv.org/pdf/1612.03467.pdf
        '''

        if self.binned == False:
            bin_particle_mapping, bin_weight_mapping = self.bin_particles()
            self.bin_particle_mapping = bin_particle_mapping
            self.bin_weight_mapping = bin_weight_mapping
            self.binned = True
        else:
            bin_weight_mapping = self.bin_weight_mapping

        entropy = 0
        for ele_bin in bin_weight_mapping.keys():
            for azi_prob in bin_weight_mapping[ele_bin]:
                if azi_prob > 0:
                    entropy += azi_prob * np.log(azi_prob)
        entropy = entropy * -1

        # perform some basic checks (e.g. that the weights of the particles sum to one)
        sum = 0
        for item in list(bin_weight_mapping.values()):
            sum += np.sum(np.array(item))
        assert np.isclose(sum, 1)

        entropy = entropy.round(4) # for numerical stability

        assert entropy >= 0, "Entropy shouldn't be negative!"

        return entropy

    def calc_info_gain(self, new_constraints):
        new_particles = copy.deepcopy(self)
        new_particles.update(new_constraints)

        return self.calc_entropy() - new_particles.calc_entropy()

    def KLD_resampling(self, k=0, epsilon=.15, N_min=20, N_max=1000, delta=0.01):
        '''
        An implementation of 'Adapting sample size in particle filters through KLD-resampling' (2013) by Li et al.
        :return: A list of particle indexes to resample
        '''

        z = norm.ppf(1 - delta)
        candidate_indexes = []
        resample_indexes = []
        N = N_min

        # todo: I should probably move this somewhere else and get rid of the _20 versions
        # dictionary with keys based on elevation, and values based on associated number of azimuth bins
        bin_occupancy = {0: np.zeros(3), 1: np.zeros(9), 2: np.zeros(15), 3: np.zeros(20), 4: np.zeros(24),
                         5: np.zeros(30),
                         6: np.zeros(30), 7: np.zeros(36), 8: np.zeros(36), 9: np.zeros(36), 10: np.zeros(36),
                         11: np.zeros(30), 12: np.zeros(30), 13: np.zeros(24), 14: np.zeros(20),
                         15: np.zeros(15),
                         16: np.zeros(9), 17: np.zeros(3)}

        while len(resample_indexes) <= N and len(resample_indexes) <= N_max or len(resample_indexes) < N_min:
            if len(candidate_indexes) > 1:
                index = candidate_indexes.pop()
            else:
                # get another set of candidate indexes using systematic resampling
                candidate_indexes = p_utils.systematic_resample(self.weights)
                np.random.shuffle(candidate_indexes)
                candidate_indexes = list(candidate_indexes)
                index = candidate_indexes.pop()

            resample_indexes.append(index)

            position_spherical = cg.cart2sph(self.positions[index])
            elevation = position_spherical[0][0]
            azimuth = position_spherical[0][1]

            elevation_bin = np.digitize(elevation, self.ele_bin_edges)
            azimuth_bin = np.digitize(azimuth, self.azi_bin_edges[elevation_bin - 1])

            if bin_occupancy[elevation_bin - 1][azimuth_bin - 1] == 0:
                k += 1
                bin_occupancy[elevation_bin - 1][azimuth_bin - 1] = 1
                if k > 1:
                    N = (k - 1) / (2 * epsilon) * (1 - 2 / (9 * (k - 1)) + np.sqrt(2 / (9 * (k - 1))) * z) ** 3

        return np.array(resample_indexes)

    def update(self, constraints, c=0.5, reset_threshold_prob=0.001):
        self.weights_prev = self.weights.copy()
        self.positions_prev = self.positions.copy()

        for constraint in constraints:
            self.reweight(constraint)

            if sum(self.weights) < reset_threshold_prob:
                # perform a reset as the particles do not fit the most recently observation well
                # inspired by Sensor Resetting Localization for Poorly Modelled Mobile Robots, Lenser et al. 2000)
                solid_angle = BEC_helpers.calc_solid_angles(constraints)[0]

                # obtain the optimal number of samples to cover the new space using KLD resampling
                n_desired_reset_particles_heuristic = int(np.ceil(solid_angle / (4 * np.pi) * 100))
                print('heuristic number: {}'.format(n_desired_reset_particles_heuristic))
                new_particle_positions = BEC_helpers.sample_human_models_uniform(constraints, n_desired_reset_particles_heuristic)
                joint_particle_positions = np.vstack((np.array(new_particle_positions), self.positions))
                self.reinitialize(joint_particle_positions)
                resample_indexes = self.KLD_resampling()
                n_desired_reset_particles_informed = len(resample_indexes)
                print('informed number: {}'.format(n_desired_reset_particles_informed))

                if len(constraints) == 1:
                    # if there is only one constraint, sample from the VMF + uniform distribution
                    new_particle_positions_uniform = BEC_helpers.sample_human_models_random(constraints, int(np.ceil(
                        n_desired_reset_particles_informed * self.integral_prob_uniform)))

                    mu_constraint = constraints[0][0] / np.linalg.norm(constraints[0][0])
                    new_particle_positions_VMF = p_utils.rand_von_mises_fisher(mu_constraint, kappa=self.VMF_kappa, N=int(np.ceil(n_desired_reset_particles_informed * self.integral_prob_VMF)),
                                                                    halfspace=True)
                    new_particle_positions = np.vstack((np.array(new_particle_positions_uniform), np.expand_dims(new_particle_positions_VMF, 1)))
                else:
                    # otherwise, fall back on simply sampling uniformly from the space obeys all constraints
                    new_particle_positions = BEC_helpers.sample_human_models_uniform(constraints, n_desired_reset_particles_informed)

                joint_particle_positions = np.vstack((np.array(new_particle_positions), self.positions_prev))
                self.reinitialize(joint_particle_positions)

                print(colored('Performed a reset', 'red'))
            else:
                # normalize weights and update particles
                self.weights /= sum(self.weights)

                n_eff = self.calc_n_eff(self.weights)
                # print('n_eff: {}'.format(n_eff))
                if n_eff < c * len(self.weights):
                    # a) use systematic resampling
                    # resample_indexes = p_utils.systematic_resample(self.weights)
                    # self.resample_from_index(resample_indexes)

                    # b) use KLD resampling
                    resample_indexes = self.KLD_resampling()
                    self.resample_from_index(np.array(resample_indexes))

        self.binned = False

    @staticmethod
    def observation_probability(x, constraints, k=4):
        prob = 1
        p = x.shape[1]

        for constraint in constraints:
            dot = constraint.dot(x.T)

            if dot >= 0:
                # use the uniform dist
                prob *= 0.12779
            else:
                # use the VMF dist
                prob *= p_utils.VMF_pdf(constraint, k, p, x, dot=dot)

        return prob

    @staticmethod
    def calc_n_eff(weights):
        return 1. / np.sum(np.square(weights))


def IROS_demonstrations():
    w = np.array([[-3, 3.5, -1]])  # toll, hotswap station, step cost
    w_normalized = w / (np.linalg.norm(w[0, :], ord=2) - np.linalg.norm(w[0, :], ord=2) * 0.05)
    # w_normalized = w / (np.linalg.norm(w[0, :], ord=2))

    prior = [np.array([[0, 0, -1]])]
    n_particles = 200

    constraints_list = [prior]
    constraints_list.extend([[np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]])], [np.array([[ 1,  0, -4]])], [np.array([[0, 1, 2]])],
                   [np.array([[  0,  -1, -10]]), np.array([[ 0, -1, -4]])], [np.array([[1, 1, 0]])]])


    particle_positions = BEC_helpers.sample_human_models_uniform([], n_particles)
    particles = Particles(particle_positions)

    # for visualizing the initial set of particles
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_facecolor('white')
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    #
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    particles.plot(particles)
    plt.show()

    constraints_list = [prior]

    # for testing whether to support incremental updates using constraints or not
    # incremental (i.e. each constraint is provided one at a time)
    # constraints_list.extend([[np.array([[1,  0,  -4]]), np.array([[-1,  0,  2]])], [np.array([[0, -1, -4]]), np.array([[0, 1, 2]])],
    #                [np.array([[1, 1, 0]])]])

    # medium
    # constraints_list.extend([[np.array([[1,  0,  -4]]), np.array([[-1,  0,  2]]), np.array([[0, -1, -4]]), np.array([[0, 1, 2]])]])

    # high (i.e. all constraints are provided all at once)
    # constraints_list.extend([[np.array([[1,  0,  -4]]), np.array([[-1,  0,  2]]), np.array([[0, -1, -4]]), np.array([[0, 1, 2]]),
    #                np.array([[1, 1, 0]])]])

    # for testing PF resetting
    # learned about mud upper and lowerbounds, then got 2-mud wrong (looks good - minor shift)
    # constraints_list.extend([[np.array([[1,  0,  -4]]), np.array([[-1,  0,  2]])], [-np.array([[1,  0,  -4]])]])

    # learned about mud upper and lowerbounds, then got 2-mud wrong, then got 2-mud right (looks good - minor shift)
    # constraints_list.extend([[np.array([[1, 0, -4]]), np.array([[-1, 0, 2]])], [-np.array([[1, 0, -4]])], [np.array([[1, 0, -4]])]])

    # learned about everything, then got 2-mud wrong (looks good - minor shift)
    # constraints_list.extend([[np.array([[1,  0,  -4]]), np.array([[-1,  0,  2]]), np.array([[0, -1, -4]]), np.array([[0, 1, 2]]),
    #                np.array([[1, 1, 0]])], [-np.array([[1, 0, -4]])]])

    # learned about everything, then doesn't pick up the battery even when it's close (looks good - minor shift)
    # constraints_list.extend([[np.array([[1,  0,  -4]]), np.array([[-1,  0,  2]]), np.array([[0, -1, -4]]), np.array([[0, 1, 2]]),
    #                np.array([[1, 1, 0]])], [-np.array([[0, 1, 2]])]])

    # learned about everything, then goes deviates into the mud when you shouldn't (requires a reset)
    constraints_list.extend([[np.array([[1,  0,  -4]]), np.array([[-1,  0,  2]]), np.array([[0, -1, -4]]), np.array([[0, 1, 2]]),
                   np.array([[1, 1, 0]])], [-np.array([[-1, 0, -2]])]])

    # for ICML paper (the constraint that conveys that mud is at least twice as costly as an action)
    # constraints_list.extend([[np.array([[-1,  0,  2]])]])

    # todo: for playing around with what happens if you get a constraint wrong repeatedly
    # constraints_list.extend([[np.array([[0,  1,  2]]), np.array([[-1,  0,  2]])], [np.array([[0, -1, -2]])], [np.array([[0, -1, -2]])], [np.array([[0, -1, -2]])]
    #                          , [np.array([[0, -1, -2]])], [np.array([[0, -1, -2]])], [np.array([[0, -1, -2]])]])

    constraints_running = []
    for j, constraints in enumerate(constraints_list):
        print(j)

        # todo: for playing around with what happens if you get a constraint wrong repeatedly
        # if j == 2:
        #     del constraints_running[2]
        constraints_running.extend(constraints)
        constraints_running = BEC_helpers.remove_redundant_constraints(constraints_running, None, False)

        # print(calc_info_gain(particles, constraints))

        particles.update(constraints)

        seen_dict = {}
        for pos in particles.positions:
            x = tuple(*pos)
            if x not in seen_dict.keys():
                seen_dict[x] = 1
            else:
                seen_dict[x] += 1

        # print("Clustering particles ... ")
        # particles.cluster()

        # print("Avg weight: {}".format(np.average(particles.weights)))
        # print("Total weight: {}".format(np.sum(particles.weights)))
        print("Number of particles: {}".format(len(particles.weights)))

        BEC_viz.visualize_pf_transition(constraints, particles, 'augmented_taxi2', weights=w_normalized)

        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.set_facecolor('white')
        # ax.xaxis.pane.fill = False
        # ax.yaxis.pane.fill = False
        # ax.zaxis.pane.fill = False
        #
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # particles.plot(particles, fig=fig, ax=ax)
        # BEC_viz.visualize_planes(constraints_running, fig=fig, ax=ax)
        # particles.plot(particles, fig=fig, ax=ax, cluster_centers=particles.cluster_centers, cluster_weights=particles.cluster_weights, cluster_assignments=particles.cluster_assignments)

        # visualize spherical polygon
        # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_running)
        # poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
        # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)

        # # visualize the ground truth reward weight
        # w = np.array([[-3, 3.5, -1]])  # toll, hotswap station, step cost
        # w_normalized = w / np.linalg.norm(w[0, :], ord=2)
        # ax.scatter(w_normalized[0, 0], w_normalized[0, 1], w_normalized[0, 2], marker='o', c='r', s=100)

        # for test visualization of sample human models
        # models, model_weights = BEC_helpers.sample_human_models_pf(particles, 8)
        # models = np.array(models)
        # ax.scatter(models[:, 0, 0], models[:, 0, 1], models[:, 0, 2],
        #            s=100, color='black')
        #
        print('Entropy: {}'.format(particles.calc_entropy()))
        plt.show()

if __name__ == "__main__":
    from numpy.random import seed
    seed(2)

    IROS_demonstrations()