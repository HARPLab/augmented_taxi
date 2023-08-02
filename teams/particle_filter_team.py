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
import teams.utils_teams as utils_teams


class Particles_team(pf.Particles):
    def __init__(self, positions):
        super().__init__(positions)

    def update_jk(self, joint_constraints, c=0.5, reset_threshold_prob=0.001):
        self.weights_prev = self.weights.copy()
        self.positions_prev = self.positions.copy()

        self.reweight_jk(joint_constraints) # particles reweighted even if one of the original constraints is satisfied.

        if sum(self.weights) < reset_threshold_prob:
            self.reset(joint_constraints[0][0]) # TODO: Check if the same reset procedure applies to the joint knowledge case. Reset particles based on the first constraint for now.
            print(colored('Resetting weights JK. The reset process is currently incorrect', 'red'))
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

    
    def reset_jk(self, constraint):
        
        # TODO: Currently, just copied ober the reset function from the single constraint case. Need to modify this to work for the joint knowledge case.
        
        '''
        Reset the particle filter to conservative include a set of new particles that are consistent with the specified constraint
        '''
        new_particle_positions = BEC_helpers.sample_human_models_uniform(constraint, 50)  # 50 is a heuristic number of particles to seed KLD resampling for calculating optimal number of resampled particles
        joint_particle_positions = np.vstack((np.array(new_particle_positions), self.positions))
        self.reinitialize(joint_particle_positions)
        resample_indexes = self.KLD_resampling()
        n_desired_reset_particles_informed = len(resample_indexes)
        print('informed number: {}'.format(n_desired_reset_particles_informed))

        # sample from the VMF + uniform distribution
        new_particle_positions_uniform = BEC_helpers.sample_human_models_random(constraint, int(np.ceil(
            n_desired_reset_particles_informed * self.integral_prob_uniform)))

        mu_constraint = constraint[0] / np.linalg.norm(constraint[0])
        new_particle_positions_VMF = p_utils.rand_von_mises_fisher(mu_constraint, kappa=self.VMF_kappa, N=int(
            np.ceil(n_desired_reset_particles_informed * self.integral_prob_VMF)),
                                                                   halfspace=True)
        new_particle_positions = np.vstack(
            (np.array(new_particle_positions_uniform), np.expand_dims(new_particle_positions_VMF, 1)))

        joint_particle_positions = np.vstack((np.array(new_particle_positions), self.positions_prev))
        self.reinitialize(joint_particle_positions)

        print(colored('Performed a reset', 'red'))







    def reweight_jk(self, constraints, plot_particles_flag = False):
        '''
        :param constraints: normal of constraints / mean direction of VMF
        :param k: concentration parameter of VMF
        :return: probability of x under this composite distribution (uniform + VMF)
        '''

        # this function is primarily for debugging particle update
        def plot_particles():
        
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')

            ax1.title.set_text('Particles before update')
            self.plot(fig=fig, ax=ax1, plot_prev=True)
            ax1.scatter(x[0, 0], x[0, 1], x[0, 2], marker='o', c='r', s=100/2)
            utils_teams.visualize_planes_team(constraints[0], fig=fig, ax=ax1)
            
            ax2.title.set_text('Particles after update')
            self.plot(fig=fig, ax=ax2)
            ax2.scatter(x[0, 0], x[0, 1], x[0, 2], marker='o', c='r', s=100/2)
            utils_teams.visualize_planes_team(constraints[0], fig=fig, ax=ax2)

            print('Particle ', j, 'with weights ', x)
            print('Current weight: ', self.weights_prev[j])
            print('Probability: ', prob)
            print('Updated weight: ', self.weights[j])

            plt.show()



        plot_particles_input = input('Plot particles? (y/n)')
        if plot_particles_input == 'y':
            plot_particles_flag = True
        
        for j, x in enumerate(self.positions):
            
            prob = self.observation_probability_jk(x, constraints, plot_particles_flag)
            self.weights[j] = self.weights[j] * prob
            
            # debug (only possible for high von-mises probabilities. should not normllay happen)            
            if prob > 0.13 or plot_particles_flag:
                plot_particles()
                
            
            

    

    @staticmethod
    def observation_probability_jk(x, joint_constraints, plot_particles_flag, k=4):
        prob = 1
        p = x.shape[1]

        # check if atleast one of the constraints is satisfied, i.e. none of the inverse constraints are satisfied
        constraint_satisfied_flag = []
        
        
        for individual_constraint in joint_constraints:
            # print('Individual constraints: ', individual_constraint)
            constraint_satisfied_flag.append(True)
            
            # check if the constraints for each individual member are satisfied
            for constraint in individual_constraint:
                dot = constraint.dot(x.T)

                if dot < 0:
                    constraint_satisfied_flag[-1] = False


        

        # Update the pdf even if constraints are satisfied for one of the members; this is from Mike's work which is applicable for half-spaces. Here the distribution should be for the union of constraints and thus pdf for uniform distribution would vary.
        if sum(np.array(constraint_satisfied_flag)) > 0  :
            # use the uniform dist
            prob *= 0.12779
        else:
            # use the VMF dist for each of the constraints for all individuals (Design choice).
            
            # vm = []
            for individual_constraint in joint_constraints:
                for i in range(len(individual_constraint)):
                    dot = individual_constraint[i].dot(x.T)
                    # vm.append(p_utils.VMF_pdf(individual_constraint[i], k, p, x, dot=dot))
                    prob *= p_utils.VMF_pdf(individual_constraint[i], k, p, x, dot=dot)
            
            # print('Von mises probabilities for each constraint: ', vm)
            # debug
            if prob > 0.13 or plot_particles_flag:
                print('All constraints: ', joint_constraints)
                print('Von mises joint prob: ', prob)
                print('final constraint_satisfied_flag: ', constraint_satisfied_flag)


        return prob