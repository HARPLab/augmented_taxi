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
from policy_summarization import particle_filter as pf


class Particles_team(pf.Particles):
    def __init__(self, positions):
        super().__init__(positions)

    def update_jk(self, joint_constraints, c=0.5, reset_threshold_prob=0.001):
        self.weights_prev = self.weights.copy()
        self.positions_prev = self.positions.copy()

        self.reweight_jk(joint_constraints) # particles reweighted even if one of the original constraints is satisfied.

        if sum(self.weights) < reset_threshold_prob:
            self.reset(joint_constraints[0]) # TODO: Check if the same reset procedure applies to the joint knowledge case. Reset particles based on the first constraint for now.
            print('Resetting weights JK...')
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
                print('Resampling...')
                self.resample_from_index(np.array(resample_indexes))

        self.binned = False

    

    def reweight_jk(self, constraints):
        '''
        :param constraints: normal of constraints / mean direction of VMF
        :param k: concentration parameter of VMF
        :return: probability of x under this composite distribution (uniform + VMF)
        '''
        for j, x in enumerate(self.positions):
            self.weights[j] = self.weights[j] * self.observation_probability_jk(x, constraints)

    

    @staticmethod
    def observation_probability_jk(x, joint_constraints, k=4):
        prob = 1
        p = x.shape[1]

        # check if atleast one of the constraints is satisfied, i.e. none of the inverse constraints are satisfied
        constraint_satisfied_flag = False
        for joint_constraint in joint_constraints:
            dot = joint_constraint.dot(x.T)
            if dot >= 0:
                constraint_satisfied_flag = True
                

        # TODO: Update the pdf; this is from Mike's work which is applicable for half-spaces. Here the distribution should be for the union of constraints and thus pdf would vary.
        if constraint_satisfied_flag:
            # use the uniform dist
            prob *= 0.12779
        else:
            # use the VMF dist
            prob *= p_utils.VMF_pdf(joint_constraints[0], k, p, x, dot=dot) # TODO: Update the function. Currently only the first constraint is used.

        return prob