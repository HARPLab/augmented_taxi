import dill as pickle
import sys, os
import itertools
import copy
from collections import Counter
import operator
import random
import pygame


import sage.all
import sage.geometry.polyhedron.base as Polyhedron
from sage.symbolic.integration.integral import definite_integral
from sage.symbolic.integration.integral import indefinite_integral
import numpy as np
from numpy import linalg as LA
from sklearn import metrics
from scipy.stats import vonmises_fisher
from scipy import integrate
from scipy.optimize import fsolve
from scipy.optimize import minimize
import sympy as sym
from scipy.optimize import least_squares as ls 
from scipy.optimize import root_scalar
from scipy.special import ive

import params_team as params
import policy_summarization.BEC_helpers as BEC_helpers
from policy_summarization import BEC
import policy_summarization.BEC_visualization as BEC_viz
from policy_summarization import particle_filter as pf
from policy_summarization import computational_geometry as cg
from policy_summarization import policy_summarization_helpers as ps_helpers
from policy_summarization import probability_utils as p_utils
from simple_rl.agents import FixedPolicyAgent
from simple_rl.utils import mdp_helpers
from simple_rl.agents import FixedPolicyAgent
from simple_rl.planning import ValueIteration
import teams.teams_helpers as team_helpers
from teams import particle_filter_team as pf_team
import policy_summarization.particle_filter as pf
import teams.utils_teams as utils_teams
from simulation import human_learner_model as hlm
from simulation.sim_helpers import get_human_response



from termcolor import colored
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = '1.0'
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'large'


# # what does base_constraints.pickel contain?

# with open('models/augmented_taxi2/base_constraints.pickle', 'rb') as f:
#             policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, consistent_state_count = pickle.load(f)

# print('policy_constraints...')
# print(policy_constraints)
# print('min_subset_constraints_record...')
# print(min_subset_constraints_record)
# print('env_record...')
# print(env_record)
# print('traj_record....')
# print(traj_record)
# print('traj_features_record....')
# print(traj_features_record)
# print('reward_record......')
# print(reward_record)
# print('consistent_state_count.....')
# print(consistent_state_count)


# # what does BEC constraints.pickle contain?

# with open('models/augmented_taxi2/BEC_constraints.pickle', 'rb') as f:
#             min_BEC_constraints, BEC_lengths_record = pickle.load(f)

# print('min_BEC_constraints...')
# print(min_BEC_constraints)
# print('BEC_lengths_record...')
# print(BEC_lengths_record)


# what does BEC_summary.pickle contain?

# with open('models/augmented_taxi2/BEC_summary.pickle', 'rb') as f:
#             BEC_summary, visited_env_traj_idxs, particles = pickle.load(f)

# print('BEC_summary...')


# for unit in BEC_summary:
#     for summary in unit:
#         particles.update(summary[3])
#         print('summary[3]...')
#         print(summary[3])
#         print('particles...')
#         print(particles)

# print('visited_env_traj_idxs...')
# print(visited_env_traj_idxs)
# print('particles...')
# print(particles)



# what does test_environment.pickle contain?

# with open('models/augmented_taxi2/test_environments.pickle', 'rb') as f:
#             test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers = pickle.load(f)

# print('test_wt_vi_traj_tuples...')
# print(test_wt_vi_traj_tuples)
# print('test_BEC_lengths...')
# print(test_BEC_lengths)
# print('test_BEC_constraints...')
# print(test_BEC_constraints)
# print('selected_env_traj_tracers...')
# print(selected_env_traj_tracers)


#############
# with open('models/augmented_taxi2/gt_policies/wt_vi_traj_params_env00000.pickle', 'rb') as f:
#             wt_vi_traj_env = pickle.load(f)

# print(len(wt_vi_traj_env[0][1].mdp.states))


#################
# with open('models/augmented_taxi2/counterfactual_data_0/model0/cf_data_env00001.pickle', 'rb') as f:
#     best_human_trajs_record_env, constraints_env, human_rewards_env = pickle.load(f)

# with open('models/augmented_taxi2/base_constraints.pickle', 'rb') as f:
#         policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, consistent_state_count = pickle.load(f)


# print(min_subset_constraints_record)

# # min_env_constraints = BEC_helpers.remove_redundant_constraints(min_subset_constraints_record, params.weights, step_cost_flag=True)

# # print(min_env_constraints)

# print(type(min_subset_constraints_record[0]))

# A,b = BEC_helpers.constraints_to_halfspace_matrix(min_subset_constraints_record[0], params.weights, step_cost_flag=True)

# print(A)
# print(b)



############################
# all_env_constraints = []
# for model_idx in range(12):
#     with open('models/augmented_taxi2/counterfactual_data_' + str(0) + '/model' + str(
#             model_idx) + '/cf_data_env' + str(1).zfill(5) + '.pickle', 'rb') as f:
#         best_human_trajs_record_env, constraints_env, human_rewards_env = pickle.load(f)
#     all_env_constraints.append(constraints_env)
#     print(len(constraints_env))
#     print(len(all_env_constraints))

# all_env_constraints_joint = [list(itertools.chain.from_iterable(i)) for i in zip(*all_env_constraints)]
# print('....')
# print(len(all_env_constraints_joint))

##########################################

# constraints

# def visualize_uncertain_constraints(constraints):

#     fig = plt.figure()

#     # plot the constraints
#     ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints)
#     poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
#     # for constraints in [constraints]:
#     #     BEC_viz.visualize_planes(constraints, fig=fig)
#     BEC_viz.visualize_spherical_polygon(poly, fig=fig, plot_ref_sphere=False, alpha=0.75)

#     plt.show()


# with open('models/augmented_taxi2/BEC_constraints.pickle', 'rb') as f:
#     min_BEC_constraints, BEC_lengths_record = pickle.load(f)

# with open('models/augmented_taxi2/BEC_summary.pickle', 'rb') as f:
#     BEC_summary, visited_env_traj_idxs, particles = pickle.load(f)

# particle_positions = BEC_helpers.sample_human_models_uniform([], 10)
# particles = pf.Particles(particle_positions)
# particles.update(params.prior)

# # run through the pre-selected units
# for unit_idx, unit in enumerate(BEC_summary):
#     print("Here are the demonstrations for the unit ", unit_idx)
#     unit_constraints = []
#     running_variable_filter = unit[0][4]

#     # show each demonstration that is part of this unit
#     for subunit in unit:
#         c = subunit[3][0]
#         print(c)
#         c2 = np.array([[c[0][0], c[0][1], c[0][2]]])
#         print(c2)
#         print([c, c2])
#         BEC_viz.visualize_pf_transition([c], particles, particles, params.mdp_class, params.weights['val'])
#         BEC_viz.visualize_pf_transition([c2], particles, particles, params.mdp_class, 0.7*params.weights['val'])
#         plt.show()
#         sys.exit(0)


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# x = np.linspace(-1,1,10)
# z = np.linspace(-1,1,10)

# xx, zz, = np.meshgrid(x, z)

# yy = 0.1*xx + 0.4*zz 
# yy2 = 0.1*xx + 0.4*zz + 0.2
# yy3 = 0.1*xx + 0.4*zz - 0.2
# ax.plot_surface(xx, yy, zz)
# ax.plot_surface(xx, yy2, zz)
# ax.plot_surface(xx, yy3, zz)
# # Set an equal aspect ratio
# ax.set_aspect('equal')

# ax.set_xlabel('$\mathregular{w_0}$: Mud')
# ax.set_ylabel('$\mathregular{w_1}$: Recharge')
# ax.set_zlabel('$\mathregular{w_2}$: Action')

#################################


def plot_sample_learning_rate():

    team_knowledge_common_demo = {'p1': [0, 0.3, 0.4, 0.6, 0.7, 0.75, 0.78, 0.8, 0.81, 0.82, 0.82],
                                'p2': [0, 0.2, 0.35, 0.5, 0.6, 0.65, 0.7, 0.70, 0.72, 0.72, 0.75],
                                'p3': [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.81, 0.81, 0.83, 0.83],
                                'c': [0, 0.1, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.68, 0.7],
                                'j': [0, 0.3, 0.5, 0.6, 0.75, 0.8, 0.82, 0.82, 0.85, 0.85, 0.85]}


    team_knowledge_joint_demo = {'p1': [0, 0.1, 0.25, 0.35, 0.55, 0.7, 0.8, 0.9, 0.95, 0.95],
                                'p2': [0, 0.1, 0.2, 0.3, 0.5, 0.65, 0.75, 0.8, 0.85, 0.9],
                                'p3': [0, 0.1, 0.3, 0.4, 0.6, 0.7, 0.85, 0.90, 0.95, 0.97],
                                'c': [0, 0.1, 0.2, 0.25, 0.3, 0.45, 0.5, 0.6, 0.7, 0.8],
                                'j': [0, 0.2, 0.35, 0.45, 0.6, 0.75, 0.85, 0.9, 0.95, 1.0]}


    team_knowledge_ind_demo =   {'p1': [0, 0.5, 0.7, 0.9, 0.95, 0.95, 0.8, 0.9, 0.95, 0.95, 0.95, 0.95, 0.95],
                                'p2': [0, 0.1, 0.15, 0.2, 0.25, 0.5, 0.7, 0.9, 0.95, 0.95, 0.95, 0.95, 0.95],
                                'p3': [0, 0.1, 0.15, 0.2, 0.2, 0.25, 0.3, 0.35, 0.4, 0.6, 0.8, 0.9, 0.95],
                                'c': [0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.8],
                                'j': [0, 0.2, 0.35, 0.45, 0.6, 0.75, 0.85, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0]}


    fig, axs =plt.subplots(3)
    fig.tight_layout(pad=20.0)

    colors = ["r", "g", "b", "k", "y"]
    labels = ['p1', 'p2', 'p3', 'common', 'joint']

    j = 0
    for i in ['p1', 'p2', 'p3', 'c', 'j']:
        print(np.linspace(1, len(team_knowledge_ind_demo[i])))
        axs[0].plot(np.linspace(1, len(team_knowledge_ind_demo[i]), len(team_knowledge_ind_demo[i])), team_knowledge_ind_demo[i], color = colors[j], label = labels[j] )
        axs[1].plot(np.linspace(1, len(team_knowledge_common_demo[i]), len(team_knowledge_common_demo[i])), team_knowledge_common_demo[i], color = colors[j], label = labels[j])
        axs[2].plot(np.linspace(1, len(team_knowledge_joint_demo[i]), len(team_knowledge_joint_demo[i])), team_knowledge_joint_demo[i], color = colors[j], label = labels[j])
        j += 1

    axs[0].legend(loc="upper left")
    axs[1].legend(loc="upper left")
    axs[2].legend(loc="upper left")

    axs[0].set_title('Learning for individual knowledge (low to high) condition')
    axs[1].set_title('Learning for common knowledge condition')
    axs[2].set_title('Learning for joint knowledge condition')



    plt.show()


def visualize_constraints(constraints):

    def label_axes(ax, mdp_class, weights=None):
        fs = 12
        ax.set_facecolor('white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        if weights is not None:
            ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='r', s=100/2)
        if mdp_class == 'augmented_taxi2':
            ax.set_xlabel('$\mathregular{w_0}$: Mud', fontsize = fs)
            ax.set_ylabel('$\mathregular{w_1}$: Recharge', fontsize = fs)
        elif mdp_class == 'colored_tiles':
            ax.set_xlabel('X: Tile A (brown)')
            ax.set_ylabel('Y: Tile B (green)')
        else:
            ax.set_xlabel('X: Goal')
            ax.set_ylabel('Y: Skateboard')
        ax.set_zlabel('$\mathregular{w_2}$: Action', fontsize = fs)

        ax.view_init(elev=16, azim=-160)

    

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')

    # plot the constraints
    ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints)
    poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    for constraints in [constraints]:
        BEC_viz.visualize_planes(constraints, fig=fig, ax=ax1)
    BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax1, plot_ref_sphere=False, alpha=0.75)

    label_axes(ax1, params.mdp_class, params.weights['val'])

    plt.show()



def sample_valid_region_union(constraints, min_azi=0, max_azi=2 * np.pi, min_ele=0, max_ele=np.pi, n_azi=1000, n_ele=1000):


    # sample along the sphere
    u = np.linspace(min_azi, max_azi, n_azi, endpoint=True)
    v = np.linspace(min_ele, max_ele, n_ele, endpoint=True)

    x = np.outer(np.cos(u), np.sin(v)).reshape(1, -1)
    y = np.outer(np.sin(u), np.sin(v)).reshape(1, -1)
    z = np.outer(np.ones(np.size(u)), np.cos(v)).reshape(1, -1)
    sph_points = np.vstack((x, y, z))

    # see which points on the sphere do not obey all constraints
    dist_to_plane = constraints.dot(sph_points)
    n_constraints_satisfied = np.sum(dist_to_plane < 0, axis=0)
    n_min_constraints = constraints.shape[0]

    idx_valid_sph_points = np.where(n_constraints_satisfied == n_min_constraints)[0]
    valid_sph_x = np.take(x, idx_valid_sph_points)
    valid_sph_y = np.take(y, idx_valid_sph_points)
    valid_sph_z = np.take(z, idx_valid_sph_points)

    return valid_sph_x, valid_sph_y, valid_sph_z



def sample_human_models_uniform_common(constraints, n_models):
    '''
    Summary: sample representative weights that the human could currently attribute to the agent, by greedily selecting
    points that minimize the maximize distance to any other point (k-centers problem)
    '''
    # invert the constraints (to sample points outside the intersection region of constrainst for common knowledge condition)
    constraints = [-x for x in constraints]

    sample_human_models = []

    if len(constraints) > 0:
        constraints_matrix = np.vstack(constraints)

        # obtain x, y, z coordinates on the sphere that obey the constraints
        valid_sph_x, valid_sph_y, valid_sph_z = sample_valid_region_union(constraints_matrix, 0, 2 * np.pi, 0, np.pi, 1000, 1000)

        if len(valid_sph_x) == 0:
            print(colored("Was unable to sample valid human models within the BEC (which is likely too small).",
                        'red'))
            return sample_human_models

        # resample coordinates on the sphere within the valid region (for higher density)
        sph_polygon = cg.cart2sph(np.array([valid_sph_x, valid_sph_y, valid_sph_z]).T)
        sph_polygon_ele = sph_polygon[:, 0]
        sph_polygon_azi = sph_polygon[:, 1]

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

            # reject points that fall inside the desired area

            # see which points on the sphere obey all constraints
            theta_grid, phi_grid = np.meshgrid(theta, phi)
            sph_points = np.array(cg.sph2cart(np.array([phi_grid.flatten(), theta_grid.flatten()]).T))
            dist_to_plane = constraints_matrix.dot(sph_points.T)
            
            # Note: For the common knowledge condition, we check for points that do not satisify the inverted constraints (of the original constraints)
            n_constraints_satisfied = np.sum(dist_to_plane < 0, axis=0)  
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
                select_idxs = BEC_helpers.selectKcities(pairwise.shape[0], pairwise, n_models)
                select_sph_points = valid_sph_points[select_idxs]
                # reshape so that each element is a valid weight vector
                select_sph_points = select_sph_points.reshape(select_sph_points.shape[0], 1, select_sph_points.shape[1])
                sample_human_models.extend(select_sph_points)
    else:
        theta = 2 * np.pi * np.linspace(0, 1, int(np.ceil(np.sqrt(n_models / 2) * 2))) # the range of theta is twice the range of phi, so account for that

        # remove the first and last phi's later to prevent clumps at poles
        phi = np.arccos(1 - 2 * np.linspace(0, 1, int(np.ceil(np.sqrt(n_models / 2))) + 2))
        theta_grid, phi_grid = np.meshgrid(theta, phi[1:-1])

        valid_sph_points = np.array(cg.sph2cart(np.array([phi_grid.flatten(), theta_grid.flatten()]).T))
        # reshape so that each element is a valid weight vector
        valid_sph_points = valid_sph_points.reshape(valid_sph_points.shape[0], 1, valid_sph_points.shape[1])

        # downsize the number of points to the number requested by pulling evenly from the full list
        skip_idxs = np.linspace(0, len(valid_sph_points) - 1, len(valid_sph_points) - n_models).astype(int)
        mask = np.ones(len(valid_sph_points), dtype=bool)
        mask[skip_idxs] = False
        valid_sph_points = valid_sph_points[mask]

        # add in the points at the poles (that were removed earlier)
        valid_sph_points[0] = np.array([[0, 0, 1]])
        valid_sph_points[-1] = np.array([[0, 0, -1]])
        sample_human_models.extend(valid_sph_points)

    return sample_human_models


def label_axes(ax, weights=None):
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    if weights is not None:
        ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='r', s=100/2)
    
    ax.set_xlabel('$\mathregular{w_0}$: Mud')
    ax.set_ylabel('$\mathregular{w_1}$: Recharge')
    ax.set_zlabel('$\mathregular{w_2}$: Action')

    ax.view_init(elev=16, azim=-160)

##############################################



if __name__ == "__main__":

    # visualize expected learning rate
    # plot_sample_learning_rate()

    # #########################################
    # ## plot constraints
    # with open('models/augmented_taxi2/BEC_summary.pickle', 'rb') as f:
    #         BEC_summary, visited_env_traj_idxs, particles = pickle.load(f)


    # # initialize a particle filter model of human
    # particle_positions = BEC_helpers.sample_human_models_uniform([], params.BEC['n_particles'])
    # particles = pf.Particles(particle_positions)
    # particles.update(params.prior)
    # particles_prev = copy.deepcopy(particles)


    # unit_no = 0
    # for unit_idx, unit in enumerate(BEC_summary):
    #     print('Constraints for unit', unit_no)

    #     subunit_no = 0

    #     for subunit in unit:
    #         if unit_no == 1 and subunit_no==1:     # unit and subunit that has more than one constraint
    #             print('Constraints for unit no', unit_no, ', subunit no', subunit_no, ',', subunit[3])
    #             constraints = [ -x for x in subunit[3]]
    #             # visualize_constraints(constraints)

    #             particles.update(constraints)
    #             BEC_viz.visualize_pf_transition(constraints, particles_prev, particles, params.mdp_class, params.weights['val'])
    #             particles_prev = copy.deepcopy(particles)



    #         subunit_no += 1
    #         # break
    #     unit_no += 1
    #     # break
    
    # #########################################
    # # plot sampled human models for common knowledge condition
    
    # with open('models/augmented_taxi2/BEC_summary.pickle', 'rb') as f:
    #         BEC_summary, visited_env_traj_idxs, particles = pickle.load(f)

    # unit_no = 0
    # for unit_idx, unit in enumerate(BEC_summary):
    #     subunit_no = 0
    #     for subunit in unit:
    #         if unit_no == 1 and subunit_no==1:     # unit and subunit that has more than one constraint
    #             print('Constraints for unit no', unit_no, ', subunit no', subunit_no, ',', subunit[3])
    #             constraints = [x for x in subunit[3]]
                
    #             human_models = BEC_helpers.sample_human_models_uniform(constraints, 100)
    #             human_models_common = sample_human_models_uniform_common(constraints, 100)

    #             particles_prev = pf.Particles(human_models)
    #             particles = pf.Particles(human_models_common)

    #             BEC_viz.visualize_pf_transition(constraints, particles_prev, particles, params.mdp_class, params.weights['val'])
    #         subunit_no += 1
    #     unit_no += 1
    

    # #########################################

    
    # plot BEC area and check (TODO)


    # #########################################


    # # check team particle filter
    # particle_positions = BEC_helpers.sample_human_models_uniform([], 100)
    # print(len(particle_positions))
    # particles = pf_team.Particles_team(particle_positions)
    # # particles.plot()
    # # plt.show()

    # mdp_class = 'augmented_taxi2'

    # constraints = [np.array([-1, 0, 2]),np.array([1, 0, -4]), np.array([0, 1, 0]), np.array([0, 1, 2])]


    # for constraint in constraints:

    #     team_helpers.visualize_transition(constraint, particles, mdp_class, weights=None, fig=None)

    # with open('models/augmented_taxi2/BEC_summary.pickle', 'rb') as f:
    #    BEC_summary, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = pickle.load(f)

    # with open('models/augmented_taxi2/BEC_summary.pickle', 'rb') as f:
    #    BEC_summary, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = pickle.load(f)

    # print(len(BEC_summary))

    # unit_no = 0
    # for unit_idx, unit in enumerate(BEC_summary):
    #     subunit_no = 0
    #     for subunit in unit:
    #         # print('Subunit for unit no: ', unit_no, 'and subunit no: ', subunit_no, subunit)
    #         print('Constraints for unit no', unit_no, ', subunit no', subunit_no, ',', subunit[3])                
    #         subunit_no += 1
    #     unit_no += 1


    # #########################################

    ## Check how the team knowledges are being aggregated

    # demo_constraints = [[np.array([[-1, 0, 0]]), np.array([[-1, 0, 2]])]]
    # test_response = {'p1': np.array([[-1, 0, 2]]), 'p2': np.array([[3, 0, -2]])}

    # team_prior, particles_team = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_prior = params.team_prior)
    
    # print('Team prior: ', team_prior)
    # print('Team particles: ', particles_team)
    # print('Demo constraints:', demo_constraints)

    # team_knowledge = copy.deepcopy(team_prior)



    # for member_id in particles_team:
    #     if 'p' in member_id:
    #         team_knowledge = team_helpers.update_team_knowledge(team_knowledge, [test_response[member_id]], params.team_size,  params.weights['val'], params.step_cost_flag, knowledge_to_update = [member_id])
    #         particles_team[member_id].update([test_response[member_id]])
    #     elif member_id == 'common_knowledge':
    #         test_common_constraints = []
    #         for idx, cnt in enumerate(test_response):
    #             test_common_constraints.extend([test_response[cnt]])
    #         print('test_common_constraints: ', test_common_constraints)
    #         team_knowledge = team_helpers.update_team_knowledge(team_knowledge, [], params.team_size,  params.weights['val'], params.step_cost_flag, knowledge_to_update = [member_id])      
    #         particles_team[member_id].update(test_common_constraints)
    #     elif member_id == 'joint_knowledge':
    #         test_joint_constraints = []
    #         for idx, cnt in enumerate(test_response):
    #             test_joint_constraints.append([test_response[cnt]])
    #         print('test_joint_constraints: ', test_joint_constraints)
    #         particles_team[member_id].update_jk(test_joint_constraints)

    # team_helpers.visualize_team_knowledge(particles_team, test_response, params.mdp_class, weights=params.weights['val'], text='Updated team knowledge after test')
    
    
    # particles_team['joint_knowledge'].update_jk(demo_constraints)
    # team_helpers.visualize_transition(demo_constraints, particles_team['joint_knowledge'], params.mdp_class, params.weights['val'], text='Demo 1 for JK Type 1')
    # particles_team['joint_knowledge_2'].update_jk(demo_constraints)
    # team_helpers.visualize_transition(demo_constraints, particles_team['joint_knowledge_2'], params.mdp_class, params.weights['val'], text = 'Demo 1 for JK Type 2')


    ######################################
    # demo_constraints = [[np.array([[-1, 0, 0]]), np.array([[-1, 0, 6]])]]

    # BEC_area = BEC_helpers.calc_solid_angles(demo_constraints)
    # print('BEC area: ', BEC_area)

    # for constraint_set in demo_constraints:
    #     print(len(constraint_set))

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    # for constraints in demo_constraints:
    #     BEC_viz.visualize_planes(constraints, fig=fig, ax=ax1)
    
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax1, plot_ref_sphere=False, alpha=0.75)

    # plt.show()

    ######################################

    # get optimal policies
    # ps_helpers.obtain_env_policies(params.mdp_class, params.data_loc['BEC'], np.expand_dims(params.weights['val'], axis=0), params.mdp_parameters, pool)

    # with open('models/' + params.data_loc['BEC'] + '/team_base_constraints.pickle', 'rb') as f:
    #         policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(f)

    # with open('models/' + params.data_loc['BEC'] + '/team_BEC_constraints.pickle', 'rb') as f:
    #         min_BEC_constraints, BEC_lengths_record = pickle.load(f)
    
    # demo_constraints = [np.array([[-1, 0, 0]]), np.array([[1, 1, 0]])]

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    # # for constraints in min_BEC_constraints:
    # #     BEC_viz.visualize_planes(min_BEC_constraints, fig=fig, ax=ax1)
    
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(min_BEC_constraints)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax1, plot_ref_sphere=False, alpha=0.75)

    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(demo_constraints)
    # poly2 = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    # BEC_viz.visualize_spherical_polygon(poly2, fig=fig, ax=ax1, plot_ref_sphere=False, alpha=0.5, color='r')

    # # plt.show()

    # print(min_BEC_constraints)
    # min_BEC_area = BEC_helpers.calc_solid_angles([min_BEC_constraints])
    # print('min BEC area: ', min_BEC_area)

    # print(demo_constraints)
    # knowledge_area = BEC_helpers.calc_solid_angles([demo_constraints])
    # print('knowledge area: ', knowledge_area)

    # knowledge_spread = np.array(knowledge_area)/np.array(min_BEC_area)

    # # sample particles from knowledge area
    # n_particles = 4500
    # knowledge_particles = pf_team.Particles_team(BEC_helpers.sample_human_models_uniform(demo_constraints, n_particles))

    # # par_pos = knowledge_particles.positions.squeeze()

    # const_id = []
    # x_all = []
    # for j, x in enumerate(knowledge_particles.positions):

    #     all_constraints_satisfied = True
    #     for constraint in min_BEC_constraints:
    #         dot = constraint.dot(x.T)

    #         if dot < 0:
    #             all_constraints_satisfied = False
        
    #     if all_constraints_satisfied:
    #         const_id.append(j)
    #         x_all.append(x)

    # BEC_overlap_ratio = min(min_BEC_area, len(const_id)/n_particles * np.array(knowledge_area))/np.array(min_BEC_area)

    # knowledge_level = 0.5*BEC_overlap_ratio + 0.5/knowledge_spread
    # BEC_overlap_particles = pf_team.Particles_team(np.array(x_all))

    # print('No of particles: ', n_particles)
    # print('Particles overlap: ', len(const_id)/n_particles * np.array(knowledge_area))
    # print('BEC overlap: ', BEC_overlap_ratio)
    # print('knowledge_spread: ', knowledge_spread)
    # print('knowledge_level: ', knowledge_level)

    # BEC_overlap_particles.plot(fig=fig, ax=ax1)
    # ax1.scatter(params.weights['val'][0, 0], params.weights['val'][0, 1], params.weights['val'][0, 2], marker='o', c='r', s=100/2)

    # plt.show()


############

    # w = np.array([[-3, 3.5, -1]]) # toll, hotswap station, step cost
    # w_normalized = w / np.linalg.norm(w[0, :], ord=2)

    # print(w_normalized)

###################



    # # 1) 
    # # team_knowledge = params.team_prior.copy()
    # # team_knowledge['common_knowledge'] = team_helpers.calc_common_knowledge(team_knowledge, 2, params.weights['val'], params.step_cost_flag)
    # # team_knowledge['joint_knowledge'] = team_helpers.calc_joint_knowledge(team_knowledge, 2, params.weights['val'], params.step_cost_flag)

    # # min_unit_constraints = params.prior

    # # min_BEC_constraints =  [np.array([[1, 1, 0]]), np.array([[ 0, -1, -4]]), np.array([[-1,  0,  2]])]


    # # # 2)
    # min_BEC_constraints =  [np.array([[1, 1, 0]]), np.array([[ 0, -1, -4]]), np.array([[-1,  0,  2]])]

    # min_unit_constraints = [np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]])]
    # # # min_unit_constraints = [np.array([[-1,  0,  0]])]

    # # Correct response by both members
    # # team_knowledge = {'p1': [np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])], 
    # #                 'p2': [np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])], 
    # #                 'common_knowledge': [np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])], 
    # #                 'joint_knowledge': [[np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])]]}

    # # Incorrect response by one team member
    # team_knowledge_prior = {'p1': [np.array([[ 0,  0, -1]])], 
    #                         'p2': [np.array([[ 0,  0, -1]])], 
    #                         'common_knowledge': [np.array([[ 0,  0, -1]])], 
    #                         'joint_knowledge': [[np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])]]}



    # team_knowledge_new = {'p1': [np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]]),], 
    #                     'p2': [np.array([[1,  0,  0]]), np.array([[3,  0,  -2]])], 
    #                     'common_knowledge': [np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]]), np.array([[1,  0,  0]]), np.array([[3,  0,  -2]])], 
    #                     'joint_knowledge': [[np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]])], [np.array([[1,  0,  0]]), np.array([[3,  0,  -2]])]]}
    
    # unit_knowledge_level = team_helpers.calc_knowledge_level(team_knowledge_new, min_unit_constraints)
    
    # print('Unit knowledge level: ', unit_knowledge_level)

    # # team_knowledge_level = team_helpers.calc_knowledge_level(team_knowledge, min_BEC_constraints)

    # # print('Team knowledge level: ', team_knowledge_level)

    # plt.show()

    # # inv_constraints = []
    # # for k_id, k_type in enumerate(team_knowledge):
    # #     if 'p' in k_type:
    # #         inv_constraints.extend([-x for x in team_knowledge[k_type]])

    # # inv_joint_constraints = BEC_helpers.remove_redundant_constraints(inv_constraints, params.weights['val'], params.step_cost_flag)

    # joint_constraints = []
    # for k_id, k_type in enumerate(team_knowledge):
    #     if 'p' in k_type:
    #         joint_constraints.extend(team_knowledge[k_type])


    # min_unit_intersection_constraints = min_unit_constraints.copy()
    # min_unit_intersection_constraints.extend(joint_constraints)

    # print('min_unit_intersection_constraints: ', min_unit_intersection_constraints)

    # opposing_constraints = False
    # for cnst in min_unit_intersection_constraints:
    #     for cnst2 in min_unit_intersection_constraints:
    #         print(np.array_equal(-cnst, cnst2))
    #         print(-cnst, cnst2)
    #         if (np.array_equal(-cnst, cnst2)):
    #             opposing_constraints = True                
    
    # if opposing_constraints:
    #     min_unit_BEC_knowledge_intersection = 0
    # else:
    #     min_unit_intersection_constraints = BEC_helpers.remove_redundant_constraints(min_unit_intersection_constraints, params.weights['val'], params.step_cost_flag)
    #     min_unit_BEC_knowledge_intersection = np.array(BEC_helpers.calc_solid_angles([min_unit_intersection_constraints]))
    
    
    # print('opposing_constraints: ', opposing_constraints)

    # min_BEC_area = np.array(BEC_helpers.calc_solid_angles([min_BEC_constraints]))
    # min_unit_area = np.array(BEC_helpers.calc_solid_angles([min_unit_constraints]))

    # ind_knowledge = []
    # ind_intersection_constraints = min_unit_constraints.copy()
    # for ind_constraints in team_knowledge['joint_knowledge']:
    #     ind_knowledge.append(BEC_helpers.calc_solid_angles([ind_constraints]))
    #     ind_intersection_constraints.extend(ind_constraints)
        
    # ind_intersection_constraints = BEC_helpers.remove_redundant_constraints(ind_intersection_constraints, params.weights['val'], params.step_cost_flag)
    # print('min_ind_intersection_constraints: ', ind_intersection_constraints)
    # knowledge_area = sum(np.array(ind_knowledge)) - np.array(BEC_helpers.calc_solid_angles([ind_intersection_constraints]))


    # min_unit_BEC_knowledge_union = min_unit_area + knowledge_area - min_unit_BEC_knowledge_intersection
    
    # # check if the knowledge area is a subset of the BEC area
    # if min_unit_BEC_knowledge_intersection == knowledge_area:
    #     knowledge_level_unit = 1
    # else:
    #     knowledge_level_unit = min_unit_BEC_knowledge_intersection/min_unit_BEC_knowledge_union

    # # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(min_intersection_constraints)
    # # poly = Polyhedron.Polyhedron(ieqs=ieqs)
    # # hrep = np.array(poly.Hrepresentation())

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # utils_teams.visualize_planes_team(min_intersection_constraints, fig=fig, ax=ax1)
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax1, plot_ref_sphere=False, alpha=0.75)



    # label_axes(ax1, params.weights['val'])
    # plt.show()

    # # # remove boundary constraints/facets from consideration
    # # boundary_facet_idxs = np.where(hrep[:, 0] != 0)
    # # hrep_constraints = np.delete(hrep, boundary_facet_idxs, axis=0)
    # # # remove the first column since these constraints goes through the origin
    # # nonredundant_constraints = hrep_constraints[:, 1:]
    # # # reshape so that each element is a valid weight vector
    # # nonredundant_constraints = nonredundant_constraints.reshape(nonredundant_constraints.shape[0], 1, nonredundant_constraints.shape[1])


    # # print('min_BEC_constraints: ', min_BEC_constraints)
    # # print('min_unit_constraints: ', min_unit_constraints)
    # # print('team_knowledge: ', team_knowledge)
    # # print('inv_joint_constraints: ', inv_joint_constraints)
    # # print('min_intersection_constraints: ', min_intersection_constraints)
    # # print('ieqs: ', ieqs)
    # # print('hrep: ', hrep)
    # # print('nonredundant_constraints: ', nonredundant_constraints)

    # print('min_unit_intersection_constraints: ', min_unit_intersection_constraints)
    # print('min BEC area: ', min_BEC_area)
    # print('min unit area: ', min_unit_area)
    # print('knowledge_area: ', knowledge_area)
    # print('min_unit_BEC_knowledge_intersection: ', min_unit_BEC_knowledge_intersection)
    # print('min_unit_BEC_knowledge_union: ', min_unit_BEC_knowledge_union)
    # print('knowledge_level_unit: ', knowledge_level_unit)

    # # plot
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    # ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    # ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    # ax4 = fig.add_subplot(1, 4, 4, projection='3d')

    
    # ax1.title.set_text('min_unit_constraints')    
    # utils_teams.visualize_planes_team(min_unit_constraints, fig=fig, ax=ax1)
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(min_unit_constraints)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax1, plot_ref_sphere=False, alpha=0.75)
    
    # ax2.title.set_text('knowledge_constraints')
    # for ind_constraints in team_knowledge['joint_knowledge']:
    #     utils_teams.visualize_planes_team(ind_constraints, fig=fig, ax=ax2)
    #     ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(ind_constraints)
    #     poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    #     BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax3, plot_ref_sphere=False, alpha=0.75)

    # ax3.title.set_text('min_intersection_constraints')    
    # utils_teams.visualize_planes_team(min_unit_intersection_constraints, fig=fig, ax=ax3)
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(min_unit_intersection_constraints)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax2, plot_ref_sphere=False, alpha=0.75)


    # label_axes(ax1, params.weights['val'])
    # label_axes(ax2, params.weights['val'])
    # label_axes(ax3, params.weights['val'])

    # plt.show()

    # ############################

    # Remedial demo

    # team_knowledge = {'p1': [np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])], 
    #                 'p2': [np.array([[3,  0,  -2]]), np.array([[ 0,  0, -1]])], 
    #                 'common_knowledge': [np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]]), np.array([[3,  0,  -2]])], 
    #                 'joint_knowledge': [[np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])]]}

    # team_knowledge = {'p1' :[np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]]), np.array([[1,  0,  0]])]}

    # min_const = BEC_helpers.remove_redundant_constraints(team_knowledge['p1'],  params.weights['val'], params.step_cost_flag)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    # # utils_teams.visualize_planes_team(team_knowledge['common_knowledge'], fig=fig, ax=ax1)
    # utils_teams.visualize_planes_team(min_const, fig=fig, ax=ax1)

    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(min_const)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax1, plot_ref_sphere=False, alpha=0.75)
    # label_axes(ax1, params.weights['val'])
    # plt.show()

    #############################

    # ## Simulate human response


    # cnst_limit = 1 # limit how many similar constraints to add
    # counter = 0
    # state_count = 0

    # BEC_depth_list = [1]
    # # BEC_depth_list = [1, 2, 3]  # more higher than this does not make sense


    # constraints_env_list = []
    # human_trajs_env_list = []
    # env_list = []

    # for env_idx in range(64):
    # # for env_idx in range(1):
        
    #     filename = 'models/augmented_taxi2/gt_policies/wt_vi_traj_params_env' + str(env_idx).zfill(5) + '.pickle'

    #     print(filename)

    
    #     with open(filename, 'rb') as f:
    #         wt_vi_traj_env = pickle.load(f)

    #     mdp = wt_vi_traj_env[0][1].mdp
    #     agent = FixedPolicyAgent(wt_vi_traj_env[0][1].policy)
    #     weights = mdp.weights

    #     constraints_list = []
    #     human_trajs_list = []

    #     for BEC_depth in BEC_depth_list:
    #         # print('BEC_depth: ', BEC_depth)
    #         action_seq_list = list(itertools.product(mdp.actions, repeat=BEC_depth))

    #         # print('action_seq_list: ', action_seq_list)

    #         for state in mdp.states:
    #             # print('State: ', state)
    #             state_count += 1
    #             # constraints = []
    #             # human_trajs = []
    #             traj_opt = mdp_helpers.rollout_policy(mdp, agent, cur_state=state)

    #             for sas_idx in range(len(traj_opt)):
    #                 # reward features of optimal action
    #                 mu_sa = mdp.accumulate_reward_features(traj_opt[sas_idx:], discount=True)

    #                 sas = traj_opt[sas_idx]
    #                 cur_state = sas[0]

    #                 # currently assumes that all actions are executable from all states
    #                 for action_seq in action_seq_list:
    #                     traj_hyp = mdp_helpers.rollout_policy(mdp, agent, cur_state=cur_state, action_seq=action_seq)
    #                     mu_sb = mdp.accumulate_reward_features(traj_hyp, discount=True)

    #                     new_constraint = mu_sa - mu_sb
    #                     # new_constraint = mu_sb

    #                     count = sum(np.array_equal(new_constraint, arr) for arr in constraints_list)

    #                     if count < cnst_limit:
    #                         constraints_list.append(new_constraint)
    #                         human_trajs_list.append(traj_hyp)
    #                         counter += 1


    #     constraints_env_list.append(constraints_list)
    #     human_trajs_env_list.append(human_trajs_list)
    #     env_list.append(env_idx)


    #     print('state_count: ', state_count)

    # # print('constraints_env_list: ', constraints_env_list)

    # # print('Set of all constraints: ', list(set(constraints_env_list)))

    # set_env_constraints_traj = list(zip(env_list, constraints_env_list, human_trajs_env_list))

    # # print('set_env_constraints_traj: ', set_env_constraints_traj)

    # with open('models/augmented_taxi2/human_trajs_env_constraints_2.pickle', 'wb') as f:
    #     pickle.dump(set_env_constraints_traj, f)


    # # visualize trajectories
    # traj_id = 1
    # for traj_hyp in traj_hyp_list:
    #     print('Visualizing trajecory no ', traj_id, ' ....')
    #     print('Trajectory constraints: ', constraints[traj_id-1])
    #     mdp.visualize_trajectory(traj_hyp)
    #     traj_id += 1


## Load the saved files and visualize trajectories

    # with open('models/augmented_taxi2/human_trajs_env_constraints_2.pickle', 'rb') as f:
    #     set_env_constraints_traj = pickle.load(f)

    
    # for env_constraints_traj in set_env_constraints_traj:
    #     env_id = env_constraints_traj[0]
    #     env_constraints = env_constraints_traj[1]
    #     env_trajectories = env_constraints_traj[2]
    #     traj_id = 1
    

    #     filename = 'models/augmented_taxi2/gt_policies/wt_vi_traj_params_env' + str(env_id).zfill(5) + '.pickle'

    #     # print(len(env_trajectories))
    #     # print(len(env_constraints))

    #     for traj in env_trajectories:
    #         print('For environment ', env_id, 'visualize trajectory ', traj_id, 'with constraints: ', env_constraints[traj_id-1])

    #         with open(filename, 'rb') as f:
    #             wt_vi_traj_env = pickle.load(f)

    #         mdp = wt_vi_traj_env[0][1].mdp
    #         mdp.visualize_trajectory(traj)

    #         traj_id += 1


#############################################

    # # new_constraint = [1, 0, 2]
    # new_constraint = [0, 0, 0]

    # if (new_constraint == np.array([0, 0, 0])).all():
    #     print('Constraint is zero')
    # else:
    #     print('Constraint is not zero')
            

#############################################################3

# # Check human response simulation integration

#     env_idx = 3
#     response_distribution = 'mixed'

#     human_history = []

#     team_size = 2
    
#     with open('models/augmented_taxi2/team_base_constraints.pickle', 'rb') as f:
#             policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(f)

    
#     human_traj = []
#     cnst = []

#     # a) find the sub_optimal responses
#     BEC_depth_list = [1]

#     filename = 'models/augmented_taxi2/gt_policies/wt_vi_traj_params_env' + str(env_idx).zfill(5) + '.pickle'
    
#     with open(filename, 'rb') as f:
#         wt_vi_traj_env = pickle.load(f)

#     mdp = wt_vi_traj_env[0][1].mdp
#     agent = FixedPolicyAgent(wt_vi_traj_env[0][1].policy)


#     opt_traj = traj_record[env_idx][0]



#     mdp.set_init_state(opt_traj[0][0])
    
#     weights = mdp.weights

#     constraints_list_correct = []
#     human_trajs_list_correct = []
#     constraints_list_incorrect = []
#     human_trajs_list_incorrect = []



#     for BEC_depth in BEC_depth_list:
#         # print('BEC_depth: ', BEC_depth)
#         action_seq_list = list(itertools.product(mdp.actions, repeat=BEC_depth))

#         traj_opt = mdp_helpers.rollout_policy(mdp, agent)
#         print('Optimal Trajectory length: ', len(traj_opt))
#         traj_hyp = []

#         for sas_idx in range(len(traj_opt)):
        
#             # reward features of optimal action
#             mu_sa = mdp.accumulate_reward_features(traj_opt[sas_idx:], discount=True)

#             sas = traj_opt[sas_idx]
#             cur_state = sas[0]
#             # if sas_idx > 0:
#             #     traj_hyp = traj_opt[:sas_idx-1]

#             # currently assumes that all actions are executable from all states
#             for action_seq in action_seq_list:
#                 traj_hyp = []
#                 if sas_idx > 0:
#                     traj_hyp = traj_opt[:sas_idx-1]

#                 traj_hyp_human = mdp_helpers.rollout_policy(mdp, agent, cur_state=cur_state, action_seq=action_seq)
#                 traj_hyp.extend(traj_hyp_human)
                
#                 mu_sb = mdp.accumulate_reward_features(traj_hyp, discount=True)

#                 new_constraint = mu_sa - mu_sb

#                 count = sum(np.array_equal(new_constraint, arr) for arr in constraints_list_correct) + sum(np.array_equal(new_constraint, arr) for arr in constraints_list_incorrect)

#                 if count < team_size: # one sample trajectory for each constriant is sufficient; but just for a variety gather one trajectory for each person for each constraint, if possible
#                     print('Hyp traj len: ', len(traj_hyp))
#                     print('new_constraint: ', new_constraint)
#                     if (new_constraint == np.array([0, 0, 0])).all():
#                         constraints_list_correct.append(new_constraint)
#                         human_trajs_list_correct.append(traj_hyp) 
#                     else:
#                         constraints_list_incorrect.append(new_constraint)
#                         human_trajs_list_incorrect.append(traj_hyp)

           
    
#     print('Constraints list correct: ', constraints_list_correct)
#     print('Constraints list incorrect: ', constraints_list_incorrect)
    
#     # b) find the counterfactual human responses
#     sample_human_models = BEC_helpers.sample_human_models_uniform([], 8)

#     for model_idx, human_model in enumerate(sample_human_models):

#         mdp.weights = human_model
#         vi_human = ValueIteration(mdp, sample_rate=1)
#         vi_human.run_vi()

#         if not vi_human.stabilized:
#             skip_human_model = True
        
#         if not skip_human_model:
#             human_opt_trajs = mdp_helpers.rollout_policy_recursive(vi_human.mdp, vi_human, cur_state, [])
#             for human_opt_traj in human_opt_trajs:
#                 human_traj_rewards = mdp.accumulate_reward_features(human_opt_traj, discount=True)
#                 mu_sa = mdp.accumulate_reward_features(traj_opt, discount=True)
#                 new_constraint = mu_sa - human_traj_rewards

#                 count = sum(np.array_equal(new_constraint, arr) for arr in constraints_list_correct) + sum(np.array_equal(new_constraint, arr) for arr in constraints_list_incorrect)

#                 if count < team_size:
#                     print('Hyp traj len: ', len(traj_hyp))
#                     print('new_constraint: ', new_constraint)
#                     if (new_constraint == np.array([0, 0, 0])).all():
#                         constraints_list_correct.append(new_constraint)
#                         human_trajs_list_correct.append(human_opt_traj) 
#                     else:
#                         constraints_list_incorrect.append(new_constraint)
#                         human_trajs_list_incorrect.append(human_opt_traj)

#     print('Constraints list correct after human models: ', constraints_list_correct)
#     print('Constraints list incorrect after human models: ', constraints_list_incorrect)

#     # Currently coded for a team size of 2
#     if response_distribution == 'correct':

#         for i in range(team_size):
#             random_index = random.randint(0, len(constraints_list_correct)-1)
#             human_traj.append(human_trajs_list_correct[random_index])
#             cnst.append(constraints_list_correct[random_index])

#             constraints_list_correct.pop(random_index)
#             human_trajs_list_correct.pop(random_index)
        
#     elif response_distribution == 'incorrect':
#         for i in range(team_size):
#             random_index = random.randint(0, len(constraints_list_incorrect)-1)
#             human_traj.append(human_trajs_list_incorrect[random_index])
#             cnst.append(constraints_list_incorrect[random_index])

#             constraints_list_incorrect.pop(random_index)
#             human_trajs_list_incorrect.pop(random_index)

#     elif response_distribution == 'mixed':
#         for i in range(team_size):
#             if i%2 == 0:
#                 random_index = random.randint(0, len(constraints_list_correct)-1)
#                 human_traj.append(human_trajs_list_correct[random_index])
#                 cnst.append(constraints_list_correct[random_index])

#                 constraints_list_correct.pop(random_index)
#                 human_trajs_list_correct.pop(random_index)
#             else:
#                 random_index = random.randint(0, len(constraints_list_incorrect)-1)
#                 human_traj.append(human_trajs_list_incorrect[random_index])
#                 cnst.append(constraints_list_incorrect[random_index])

#                 constraints_list_incorrect.pop(random_index)
#                 human_trajs_list_incorrect.pop(random_index)

#     human_history.append((env_idx, human_traj))

#     print('N of human_traj: ', len(human_traj))

#     print('Visualizing human trajectory ....')
#     for ht in human_traj:
#         print('human_traj len: ', len(ht))
#         print('constraint: ', cnst[human_traj.index(ht)])
#         mdp.visualize_trajectory(ht)
#     pygame.quit()

##########################
    # min_intersection_constraints = [np.zeros((params.team_size), dtype=int)]
    # # min_BEC_constraints =  [np.array([[1, 1, 0]]), np.array([[ 0, -1, -4]]), np.array([[-1,  0,  2]])]
    # print(min_intersection_constraints)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    # utils_teams.visualize_planes_team(min_intersection_constraints, fig=fig, ax=ax1)
    # plt.show()

#####################
    # ck = np.zeros((3, 1), dtype=int)
    # print(ck.shape)
    # team_knowledge = [ck.reshape(ck.shape[1], ck.shape[0])]
    # print(team_knowledge)

    # if (team_knowledge[0] == 0).all():
    #     print('All zeros!')

    # knowledge_particles = pf_team.Particles_team(BEC_helpers.sample_human_models_uniform([], 500))


    # # team constraints
    # min_unit_constraints = [[np.array([[-1,  0,  2]])], [np.array([[1,  0,  -2]])]]
    # print(min_unit_constraints)

    # # knowledge_particles_prev = copy.deepcopy(knowledge_particles)

    # knowledge_particles.update([np.array([[0,  0,  -1]])])

    # knowledge_particles.update_jk(min_unit_constraints)

    # team_helpers.visualize_transition(min_unit_constraints, knowledge_particles, params.mdp_class, weights=params.weights['val'], demo_strategy = 'joint_knowledge')

#####################
# # Debug knowledge calculation

#     # team_knowledge = {'p1': [np.array([[0,  0,  -1]])], 'p2': [np.array([[ 0,  0, -1]])], 'common_knowledge': [np.array([[ 0,  0, -1]])], 'joint_knowledge': [[np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])]]}

#     team_knowledge, particles_team = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_prior = params.team_prior)
        
#     min_BEC_constraints:  [np.array([[1, 1, 0]]), np.array([[ 0, -1, -4]]), np.array([[-1,  0,  2]])]
    
#     min_unit_constraints = [ [np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]])], 
#                              [np.array([[0, 1, 2]])] ]
    

#     actual_team_constraints = [ [  [[np.array([[-1,  0,  2]])], [np.array([[ 0,  0, -1]])]],   [[np.array([[-1,  0,  0]])], [np.array([[ 0,  0, -1]])]]  ], 
#                                 [  [[np.array([[0, 1, 2]])], [np.array([[0, 0, 1]])]]  ]  ] 
    

#     for actual_cnst_idx, actual_cnst in enumerate(actual_team_constraints):
#         print('actual_cnst: ', actual_cnst)
#         team_cnsts = []
#         for cnst in actual_cnst:
#             for m_id in range(1, 3):
#                 member_id = 'p' + str(m_id)
#                 team_knowledge = team_helpers.update_team_knowledge(team_knowledge, cnst[m_id-1], 2, params.weights['val'], params.step_cost_flag, knowledge_to_update=[member_id])
#                 particles_team[member_id].update(cnst[m_id-1])
#                 team_helpers.visualize_transition(cnst[m_id-1], particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Knowledge change for set for ' + member_id)
#                 team_cnsts.extend(cnst[m_id-1])
#             team_knowledge = team_helpers.update_team_knowledge(team_knowledge, [], 2, params.weights['val'], params.step_cost_flag, knowledge_to_update=['common_knowledge', 'joint_knowledge'])

#             particles_team['common_knowledge'].update([team_cnsts])
#             team_helpers.visualize_transition(team_cnsts, particles_team['common_knowledge'], params.mdp_class, params.weights['val'], text = 'Knowledge change for set for common knowledge')

#             particles_team['joint_knowledge'].update_jk(cnst)
#             team_helpers.visualize_transition([cnst], particles_team['joint_knowledge'], params.mdp_class, params.weights['val'], text = 'Knowledge change for set for joint knowledge')



#             print('constraints: ', cnst)
#             print('team_knowledge: ', team_knowledge)
#             print('unit knowledge_level: ', team_helpers.calc_knowledge_level(team_knowledge, min_unit_constraints[actual_cnst_idx]))

    ###########################

    # ## Majority Rules
    # # opp_constraints = [np.array([[-1, 0, 2]]), np.array([[ 1, 0, -2]])]
    # # opp_idx_unique = [1, 2]
    # # resp_cat = ['incorrect', 'correct']
    # test_constraints_team_expanded =   [np.array([[ -1,  0, 2]]), np.array([[-1,  0,  2]]), np.array([[ 1,  0, -2]])]
    # opposing_idx =  [[0, 2], [2, 0], [1, 2], [2, 1]]
    # response_category_team = ['correct', 'correct', 'incorrect']

    # opp_idx_unique = []
    # for i in range(len(opposing_idx)):
    #     opp_idx_unique.extend(x for x in opposing_idx[i] if x not in opp_idx_unique)

    # print('opp_idx_unique: ', opp_idx_unique)
    # opp_constraints = [test_constraints_team_expanded[x] for x in opp_idx_unique]
    # print('opp_constraints: ', opp_constraints)
    # resp_cat = [response_category_team[x] for x in opp_idx_unique]
    # print('resp_cat: ', resp_cat)
    # opp_set = []
    # count_opp_set = []
    # resp_cat_set = []
    # for j in range(len(opp_constraints)):
    #     opp_c = opp_constraints[j]
    #     if len(opp_set) > 0:
    #         in_minimal_set = False
    #         for i in range(len(opp_set)):
    #             opp_c_set = opp_set[i]
    #             if (opp_c == opp_c_set).all():
    #                 in_minimal_set = True
    #                 count_opp_set[i] += 1

    #         if not in_minimal_set:
    #             opp_set.append(opp_c)
    #             count_opp_set.append(1)
    #             resp_cat_set.append(resp_cat[j])
    #     else:
    #         opp_set.append(opp_c)
    #         count_opp_set.append(1)
    #         resp_cat_set.append(resp_cat[j])
            
    # # print('minimal opposing constraints: ', opp_set)
    # # print('count_opp_set: ', count_opp_set)
    # # print('response set: ', resp_cat_set)

    # max_count_idx = [i for i in range(len(count_opp_set)) if count_opp_set[i] == max(count_opp_set)]

    # # print('max_count_idx: ', max_count_idx)
    # if len(max_count_idx) == 1:
    #     # print('Majority response: ', resp_cat_set[max_count_idx[0]])
    #     maj_cnst = opp_set[max_count_idx[0]]
    # else:        
    #     # print('Checking when there are multiple max counts')
    #     # print('Majority response: ', 'incorrect')
    #     maj_cnst = [opp_set[x] for x in max_count_idx if resp_cat_set[x] == 'incorrect']
        
    # # print('Majority constraint: ', maj_cnst)


    # # # Option 3: Assign common knowledge as the knowledge of the person(s) who got it incorrect, even if majority got it correct
    # alternate_team_constraints = [opp_set[i] for i in range(len(resp_cat_set)) if resp_cat_set[i] == 'incorrect'] #incorrect constraint. Assume that no one learned even if one person did not get it correct. Logical for the common knowledge where everyone is expected to know something about the robot.
    # print('Incorrect_team_constraints: ', alternate_team_constraints)



#########################################################

    # # constraints = [np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]]), np.array([[ 0,  0, -1]]), np.array([[ 3,  0, -2]]), np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])]
    # constraints = [np.array([[-1,  0,  2]]), np.array([[3,  0,  -2]]), np.array([[ 0,  0, -1]])]

    # # print(BEC_helpers.remove_redundant_constraints(constraints, params.weights['val'], params.step_cost_flag))
    # # common_constraints = BEC_helpers.remove_redundant_constraints(constraints, params.weights['val'], params.step_cost_flag)

    # # unit_constraint_flag = True
    # # for constraint in common_constraints:
    # #     print('Constraint: ', constraint, 'norm: ', LA.norm(constraint,1))
    # #     if LA.norm(constraint,1) !=1:
    # #         unit_constraint_flag = False
    # # if unit_constraint_flag:
    # #     utils_teams.visualize_planes_team(common_constraints)
    # #     plt.show()
    
    # unit_constraint_flag = True
    # unit_common_constraint_flag = True
    # constraints_copy = constraints.copy()
    
    # while not unit_constraint_flag and unit_common_constraint_flag:
        
    #     common_constraints = BEC_helpers.remove_redundant_constraints(constraints_copy, params.weights['val'], params.step_cost_flag)
    #     print('Origninal common knowledge: ', constraints_copy)
    #     print('Remove redundant common knowledge: ', common_constraints)

    #     for constraint in common_constraints:
    #         if LA.norm(constraint) !=1:
    #             unit_constraint_flag = False

    #     for constraint in common_constraints:
    #         if LA.norm(constraint) !=1:
    #             unit_common_constraint_flag = False

    #     if not unit_constraint_flag and unit_common_constraint_flag:
    #         constraints_copy.pop(0)  # remove the first constraint


#########################################################
    # x = [np.array([[1,  0,  0]])]
    
    # visualize_constraints(x)

    # particles_sample = pf_team.Particles_team(BEC_helpers.sample_human_models_uniform([], 50))

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1, projection='3d')

    # particles_sample.plot(fig=fig, ax=ax1)
    # plt.show()

####################################

    # mu_rep = np.array([1,  0,  0])
    # x_edge_rep = np.array([0,  0,  1])  # a point on the edge of the constraint mu
    # # x_edge = [1/np.sqrt(2), 1/np.sqrt(2), 0] 
    # kappa = 1.23
    # vmf = vonmises_fisher(mu_rep, kappa)
    # pdf_vmf = vmf.pdf(x_edge_rep)
    # # print(pdf_vmf)

    # phi_lim = [0, np.pi]
    # theta_lim = [np.pi/2, 3*np.pi/2]    # these limits are specific to the constraint mu = [1, 0, 0]
    # f1 = lambda phi, theta: kappa*np.exp(kappa*np.array([np.cos(theta)* np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).dot(mu_rep))*np.sin(phi)/(2*np.pi*(np.exp(kappa)-np.exp(-kappa)))
    # int_probability_vmf = integrate.dblquad(f1, theta_lim[0], theta_lim[1], phi_lim[0], phi_lim[1])
    # x1 = 1 / (4 * np.pi * pdf_vmf)
    # x2 = 1 / (int_probability_vmf * x1 + 0.5)

    # print(int_probability_vmf)


############################



    # mu = np.array([1,  0,  0])
    # kappa_list = np.linspace(0.0001, 10, 100)
    # u_cdf = np.linspace(0.0, 1, 100)
    # correct_hs_likelihood = np.zeros(len(kappa_list))
    # p = 3


    # for k in range(len(kappa_list)):
    #     kappa = kappa_list[k]
    #     human_model = hlm.cust_pdf(mu, kappa, p)
    #     # create samples
    #     n = 1000
    #     dot_n = np.zeros([n, 1], int)

    #     cust_samps = human_model.rvs(size=n)
    #     # print(cust_samps)

    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(1, 1, 1, projection='3d')

    #     # particles_sample = pf_team.Particles_team(cust_samps)
    #     # particles_sample.plot(fig=fig, ax=ax1)
    #     # plt.show()

    #     for i in range(n):
    #         dot = cust_samps[i].dot(mu)
    #         if dot >= 0:
    #             dot_n[i] = 1
        
    #     correct_hs_likelihood[k] = sum(dot_n)/n


    # print(correct_hs_likelihood)

    # sns.lineplot(x=kappa_list, y=correct_hs_likelihood, title='Correct human response likelihood')
    # plt.show()


    # correct_hs_likelihood = np. array([0.628, 0.607, 0.646, 0.665, 0.68,  0.683, 0.681, 0.679, 0.68,  0.722, 0.7,   0.711,
    #                                     0.739, 0.739, 0.744, 0.739, 0.745, 0.76,  0.773, 0.76,  0.789, 0.784, 0.788, 0.782,
    #                                     0.791, 0.836, 0.822, 0.807, 0.826, 0.831, 0.837, 0.824, 0.859, 0.847, 0.848, 0.854,
    #                                     0.844, 0.869, 0.862, 0.852, 0.875, 0.854, 0.853, 0.872, 0.88,  0.871, 0.874, 0.891,
    #                                     0.873, 0.882, 0.897, 0.892, 0.881, 0.897, 0.913, 0.891, 0.904, 0.887, 0.896, 0.9,
    #                                     0.917, 0.905, 0.91,  0.911, 0.913, 0.914, 0.914, 0.905, 0.904, 0.906, 0.905, 0.921,
    #                                     0.909, 0.904, 0.909, 0.912, 0.922, 0.922, 0.925, 0.919, 0.921, 0.923, 0.916, 0.922,
    #                                     0.919, 0.926, 0.927, 0.924, 0.922, 0.943, 0.927, 0.937, 0.941, 0.934, 0.925, 0.937,
    #                                     0.924, 0.932, 0.93,  0.935])

    # u_cdf_ls = np.zeros(len(u_cdf))
    # for i in range(len(u_cdf)):

    #     # u_cdf_ls[i] = min(max(u_cdf[i] + random.uniform(-0.1, 0.1), 0), 1)
    #     u_cdf_ls[i] = random.uniform(13, 3.2)

    # sns.lineplot(x=u_cdf, y=u_cdf_ls).set(title='Correct human response likelihood')
    # plt.xlabel("uniform cdf")
    # plt.ylabel("Probability of sampling correct response")
    # plt.show()

    ######################################

    
    
    #####
    # rng = np.random.default_rng()
    # mu, kappa = np.array([1, 0, 0]), 2
    # samples = vonmises_fisher(mu, kappa).rvs(10, random_state=rng)

    # print(samples)

    # mu_fit, kappa_fit = vonmises_fisher.fit(samples)
    # print(mu_fit, kappa_fit)

    # print(np.linalg.norm(mu_fit, 2))



    #### from scipy vmf.fit  ####

    # mu = np.array([1,  0,  0])

    # x_list = BEC_helpers.sample_human_models_uniform([], 1000)

    # print(x_list)

    # x_list_correct = np.empty([3])
    # x_list_incorrect = np.empty([3])
    

    # for i in range(len(x_list)):
    #     if mu.dot(x_list[i].T) >= 0:
    #         x_list_correct = np.vstack((x_list_correct, x_list[i]))
    #     else:
    #         x_list_incorrect = np.vstack((x_list_incorrect, x_list[i]))

    
    # kappa = 1
    # vmf = vonmises_fisher(mu, kappa)

    # pdf_incorrect = np.zeros(len(x_list_incorrect))
    # pdf_correct = np.zeros(len(x_list_correct))

    # print(x_list_incorrect)
    # print(x_list_correct)

    # for i in range(len(x_list_incorrect)):
    #     pdf_incorrect[i] = vmf.pdf(x_list_incorrect[i, :])

    # for i in range(len(x_list_correct)):
    #     pdf_correct[i] = vmf.pdf(x_list_correct[i, :])

    # print(np.max(pdf_incorrect))
    # print(np.max(pdf_correct))


######################

    # solve for kappa given the pdf at the intersection of uniform and VMF

    # p = 3
    # x = np.array([0,  0,  1])
    # f = 0.25/np.pi

    # func = lambda kappa : np.exp(kappa) - np.exp(-kappa) - kappa/(2*np.pi*f)
    # func_prime = lambda kappa : np.exp(kappa) + np.exp(-kappa) - 1/(2*np.pi*f)  # derivative
    # func_prime2 = lambda kappa : np.exp(kappa) - np.exp(-kappa)  # second derivative

    # root_res = root_scalar(func, x0 = 1e-6, method="halley", fprime = func_prime, fprime2 = func_prime2,
    #                            bracket=(1e-8, 1e9))

    # root_res = root_scalar(func, x0 = 1e-6, method="secant", fprime = func_prime,
    #                            bracket=(1e-8, 1e9))

    # root_res = minimize(func, x0 = 1e-6, method = 'L-BFGS-B', bounds = [(0, None)])

    # root_res = ls(func, x0 = 1e-6, bounds = (0, np.inf))
    
    # print(root_res)

    # print(p_utils.VMF_pdf(mu, root_res.root, p, np.array([0, 0, 1])))

    # vmf = vonmises_fisher(mu, 0.00001)

    # print(f)
    # print(vmf.pdf(np.array([0, 0, 1])))


####################
    ## Solve for kappa given uniform cdf

    # LHS = 0.2

    # def integrand(phi, theta, kappa):
    #     return ( kappa*np.sin(phi)*np.exp(kappa*np.cos(theta)*np.sin(phi)) ) / ( 2*np.pi*(np.exp(kappa)-np.exp(-kappa)) )


    # def func(kappa):
    #     y, err = integrate.dblquad(integrand, np.pi/2, 3*np.pi/2, 0, np.pi, args = (kappa, ))

    #     return LHS-y
    

    # sol = fsolve(func, 0.001)

    # # sol = ls(func, x0 = 1, bounds = (0, np.inf))

    # # sol = minimize(func, x0 = 1, method = 'L-BFGS-B', bounds = [(0, None)])


    # # Brute force solve

    # x_range = np.array([0.001, 10])
    # sol_found = False
    # # while not sol_found:
    # #     x = np.mean(x_range)
    # #     if func(x) > 0:
    # #         x_range[1] = x
    # #     else:
    # #         x_range[0] = x
        
    # #     if np.abs(func(x)) < 0.001:
    # #         sol_found = True
        
    # #     print(x, x_range, func(x))

    # print(sol)


    #############

    # mu = np.array([1,  0,  0])
    # u_cdf_list = np.linspace(0.5, 0.99, 100)
    # kappa_list = np.zeros(len(u_cdf_list))
    # correct_hs_likelihood = np.zeros(len(u_cdf_list))
    # p = 3


    # for k in range(len(u_cdf_list)):
    #     u_cdf = u_cdf_list[k]
    #     human_model = hlm.cust_pdf_uniform(mu, u_cdf)
    #     kappa_list[k] = human_model.kappa
    #     # create samples
    #     n = 1000
    #     dot_n = np.zeros([n, 1], int)

    #     cust_samps = human_model.rvs(size=n)
    #     # print(cust_samps)

    #     # fig = plt.figure()
    #     # ax1 = fig.add_subplot(1, 1, 1, projection='3d')

    #     # particles_sample = pf_team.Particles_team(cust_samps)
    #     # particles_sample.plot(fig=fig, ax=ax1)
    #     # plt.show()

    #     for i in range(n):
    #         dot = cust_samps[i].dot(mu)
    #         if dot >= 0:
    #             dot_n[i] = 1
        
    #     correct_hs_likelihood[k] = sum(dot_n)/n


    # print(correct_hs_likelihood)


    # fig, ax = plt.subplots(2)

    # ax[0].plot(u_cdf_list, correct_hs_likelihood)
    # ax[0].set_xlabel('Uniform distribution mass')
    # ax[0].set_ylabel('Probability of sampling a correct response')


    # ax[1].plot(kappa_list, correct_hs_likelihood)
    # ax[1].set_xlabel('Kappa of VMF')
    # ax[1].set_ylabel('Probability of sampling a correct response')
    # plt.show()


    ##############################

    # # check if sampled response is matching expected response and if likelihood matches the final proportion of correct responses sampled

    # with open('models/augmented_taxi2/human_resp_debug.pickle', 'rb') as f:
    #     env_idx, test_constraints, opt_traj, team_likelihood_correct_response = pickle.load(f)

    # print('test_constraints: ', test_constraints)
    # print('team likelihood correct response: ', team_likelihood_correct_response)

    # human_traj_team = []
    # response_type_team = []
    # initial_likely_response_type_team = []
    # likely_correct_response_count = 0
    # likely_incorrect_response_count = 0

    # for i in range(100):
    #     print('Sampling human response for iteration ', i+1, ' ...')
    #     human_traj, response_type, lcr, lir, initial_likely_response_type = get_human_response(env_idx, test_constraints[0], opt_traj, 0.55)
    #     likely_correct_response_count += lcr
    #     likely_incorrect_response_count += lir
    #     human_traj_team.append(human_traj)
    #     response_type_team.append(response_type)
    #     initial_likely_response_type_team.append(initial_likely_response_type)
    #     # print('Simulated  response length for player ', i+1, ' : ', len(human_traj_team[i]))
    
    # print('Likelihood of initially sampled response to be correct :', len([x for x in initial_likely_response_type_team if x == 'correct'])/len(initial_likely_response_type_team))
    # print('Likelihood of plausible correct response: ', likely_correct_response_count/(likely_correct_response_count + likely_incorrect_response_count))
    # print('Likelihood of actual correct response: ', len([x for x in response_type_team if x == 'correct'])/len(response_type_team))

    ###############################

    ## check if scaled uniform distribution matches the expected likelihood
    # mu = np.array([1,  0,  0])
    # u_cdf_list = np.linspace(0.5, 1, 50)


    # N_loops = 5

    # correct_hs_likelihood = np.zeros([N_loops, len(u_cdf_list)])

    # for j in range(N_loops):
    #     print('Loop no: ', j)
    #     for u in range(len(u_cdf_list)):
    #         # u_cdf = random.sample(u_cdf_list)
    #         u_cdf = u_cdf_list[u]
    #         human_model = hlm.cust_pdf_uniform(mu, u_cdf)
            
    #         # create samples
    #         n = 1000
    #         dot_n = np.zeros([n, 1], int)

    #         cust_samps = human_model.rvs(size=n)
    #         # print(cust_samps)

    #         # fig = plt.figure()
    #         # ax1 = fig.add_subplot(1, 1, 1, projection='3d')

    #         # particles_sample = pf_team.Particles_team(cust_samps)
    #         # particles_sample.plot(fig=fig, ax=ax1)
    #         # plt.show()

    #         for i in range(n):
    #             dot = cust_samps[i].dot(mu)  # note that here we are not checking if the human model are stable for Value Iteration or not
    #             if dot >= 0:
    #                 dot_n[i] = 1
            
    #         correct_hs_likelihood[j, u] = sum(dot_n)/n

    # correct_hs_likelihood_expand = correct_hs_likelihood.flatten()
    # u_cdf_list_expand = np.repeat(u_cdf_list, N_loops, axis=0)

    # correct_hs_likelihood_mean = np.mean(correct_hs_likelihood, axis=0)
    # correct_hs_likelihood_std = np.std(correct_hs_likelihood, axis=0)

    # # print('u_cdf_list: ', u_cdf_list)
    # # print('correct_hs_likelihood: ', correct_hs_likelihood)
    # # print('correct_hs_likelihood_expand: ', correct_hs_likelihood_expand)
    # # print('u_cdf_list_expand: ', u_cdf_list_expand)



    # plt.plot(u_cdf_list, correct_hs_likelihood_mean)
    # plt.fill_between(u_cdf_list, correct_hs_likelihood_mean - correct_hs_likelihood_std, correct_hs_likelihood_mean + correct_hs_likelihood_std, color='b', alpha=0.2)
    # plt.xlabel("uniform cdf")
    # plt.ylabel("Probability of sampling correct response")
    # plt.show()

    ##################################################################



    # check constraint reset

    # team_knowledge_new = {'p1': [np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]]),], 
    #                 'p2': [np.array([[1,  0,  0]]), np.array([[3,  0,  -2]])], 
    #                 'common_knowledge': [np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]]), np.array([[1,  0,  0]]), np.array([[3,  0,  -2]])], 
    #                 'joint_knowledge': [[np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]])], [np.array([[1,  0,  0]]), np.array([[3,  0,  -2]])]]}
    

    # reset_constraint = np.array([[-1,  0,  2]])

    # reset_index = [i for i in range(len(team_knowledge_new['p1'])) if (team_knowledge_new['p1'][i] == reset_constraint).all()]

    # print(reset_index)

    # check joint knowledge non-intersection 

    
    # team_knowledge = {'p1': [np.array([[0,  0,  -1]])], 
    #                 'p2': [np.array([[0,  0,  -1]])], 
    #                 'common_knowledge': [np.array([[0,  0,  -1]]), np.array([[0,  0,  -1]])], 
    #                 'joint_knowledge': [[np.array([[0,  0,  -1]])], [np.array([[0,  0,  -1]])]]}

    # ind_knowledge = []
    # ind_intersection_constraints = []
    # for ind_constraints in team_knowledge['joint_knowledge']:
    #     ind_knowledge.append(BEC_helpers.calc_solid_angles([ind_constraints]))
    #     ind_intersection_constraints.extend(ind_constraints)

    # min_ind_intersection_constraints = BEC_helpers.remove_redundant_constraints_team(ind_intersection_constraints, params.weights['val'], params.step_cost_flag)

    # print('ind_intersection_constraints :', ind_intersection_constraints)
    # print('min_ind_intersection_constraints :', min_ind_intersection_constraints)
    # print('ind_knowledge :', ind_knowledge)
    # print('Flag - Non intersecting individual constraints', team_helpers.check_for_non_intersecting_constraints(min_ind_intersection_constraints))

    #################

    # # check how redundant constraints are removed for some cases

    # min_unit_constraints =  [np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]])]
    # knowledge_constraints =  [np.array([[-1,  0,  0]]), np.array([[ 1,  0, -2]])]
    
    # min_unit_area = np.array(BEC_helpers.calc_solid_angles([min_unit_constraints]))
    # knowledge_area = np.array(BEC_helpers.calc_solid_angles([knowledge_constraints]))

    # min_unit_intersection_constraints = min_unit_constraints.copy()
    # # print('unit_intersection_constraints before update: ', min_unit_intersection_constraints)
    # min_unit_intersection_constraints.extend(knowledge_constraints)
    # min_unit_intersection_constraints_old = copy.deepcopy(min_unit_intersection_constraints)
    # print('min_unit_intersection_constraints before removing redundant constraints: ', min_unit_intersection_constraints)
    # print('opposing constraints: ', team_helpers.check_opposing_constraints(min_unit_intersection_constraints))
    # min_unit_intersection_constraints = BEC_helpers.remove_redundant_constraints(min_unit_intersection_constraints, params.weights['val'], params.step_cost_flag)
    # print('min_unit_intersection_constraints after removing redundant constraints: ', min_unit_intersection_constraints)

    
            
    # if not team_helpers.check_for_non_intersecting_constraints(min_unit_intersection_constraints):
    #     min_unit_BEC_knowledge_intersection = np.array(BEC_helpers.calc_solid_angles([min_unit_intersection_constraints]))
    #     print('min_unit_BEC_knowledge_intersection: ', min_unit_BEC_knowledge_intersection)
    #     min_unit_BEC_knowledge_union = min_unit_area + knowledge_area - min_unit_BEC_knowledge_intersection
    #     print('min_unit_BEC_knowledge_union: ', min_unit_BEC_knowledge_union)
    #     # check if the knowledge area is a subset of the BEC area
    #     if min_unit_BEC_knowledge_intersection == knowledge_area:
    #         knowledge_level = 1
    #     else:
    #         kl = min_unit_BEC_knowledge_intersection/min_unit_BEC_knowledge_union
    #         knowledge_level = kl
    #         # knowledge_level[knowledge_type] = min(1, max(0, kl))  
    # else:
    #     knowledge_level = 0 

    # print('knowledge_level: ', knowledge_level)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    # ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    # ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    # utils_teams.visualize_planes_team(min_unit_constraints, fig=fig, ax=ax1, alpha=0.5)
    # utils_teams.visualize_planes_team(min_unit_intersection_constraints_old, fig=fig, ax=ax2, alpha=0.5)
    # utils_teams.visualize_planes_team(min_unit_intersection_constraints, fig=fig, ax=ax3, alpha=0.5)

    # plt.show()


    ############################
    # check how non-intersecting constraints can be recursively checked and removed

    # knowledge_constraints =  [np.array([[0,  0,  -1]]), np.array([[ -1,  0, 2]]), np.array([[ 1,  0, -1]])]

    # int_cnsts = team_helpers.majority_rules_non_intersecting_team_constraints(knowledge_constraints, params.weights['val'], params.step_cost_flag)

    # print('max_int_cnsts: ', int_cnsts)

    ##########
    # x = [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -2]]), np.array([[ 2, -1, -2]])]]
    # reset_constraint = [np.array([ 2, -1, -2])]
    # kc_id = 1

    # # for i in range(len(x)):
    # #     print('reset_constraint[0]: ', reset_constraint[0])
    # #     print('x[i][0]: ', x[i][0])
    # #     if (reset_constraint[0] == x[i][0]).all():
    # #         reset_index = i

    # reset_index = [i for i in range(len(x[kc_id])) if (x[kc_id][i] == reset_constraint[0]).all()]

    # print(reset_index)

    ######
    # check majority rules output form (list or array)
    # opp_constraints = [np.array([[0, 1, 2]]), np.array([[ 0, -1, -2]]), np.array([[ 0, -1, -2]])]

    # opposing_constraints_flag, opposing_constraints_count, opposing_idx = team_helpers.check_opposing_constraints(opp_constraints, 0)

    # test_constraints_team_expanded = team_helpers.majority_rules_opposing_team_constraints(opposing_idx, opp_constraints, ['incorrect', 'correct', 'correct'])

    # print('test_constraints_team_expanded: ', test_constraints_team_expanded)

    ############
    # x =  [np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]]), np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])]
    # x =  [np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]])]
    # x =  [np.array([[ 0, 0, -1]]), np.array([[ -1,  0, 2]]), np.array([[1, 0, 0]])]
    # x =  [np.array([[ 0,  0, -1]]), np.array([[1, 0, 0]])]

    # x = [np.array([[1, 1, 0]]), np.array([[ 0, -1, -4]]), np.array([[-1,  0,  2]]), np.array([[0, 0, 1]]), np.array([[0, 1, 0]]), np.array([[1, 0, 0]])]


    # x_min = BEC_helpers.remove_redundant_constraints(x, params.weights['val'], params.step_cost_flag)
    # print('x: ', x)
    # print('x_min: ', x_min)
    # print('Area x: ', BEC_helpers.calc_solid_angles([x]))
    # print('Area x_min: ', BEC_helpers.calc_solid_angles([x_min]))

    # N_min_cnst_in_cnst_list = 0
    # for cnst in x_min:
    #     if any((i==cnst).all() for i in x):
    #         non_intersecting_constraints_flag = False
    #         N_min_cnst_in_cnst_list += 1
    
    # print('N_min_cnst_in_cnst_list: ', N_min_cnst_in_cnst_list)

    # if N_min_cnst_in_cnst_list == len(x_min):
    #     print('Intersecting constraints...')

    # y= [i for i in x_min if i in x]

    # if len([i for i in x_min if i in x]) == len(x_min):
    #     print('Intersecting constraints')
    # else:
    #     print('Non-intersecting constraints')


    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # utils_teams.visualize_planes_team(x, fig=fig, ax=ax1, alpha=0.5)
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(x)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax1, plot_ref_sphere=False, color = 'b')

    # utils_teams.visualize_planes_team(x_min, fig=fig, ax=ax2, alpha=0.5)
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(x_min)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax2, plot_ref_sphere=False, color = 'b')

    # plt.show()

    ###############################

    # team_knowledge = {'p1': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])], [np.array([[0, 1, 2]])], [np.array([[ 1, -1, -4]]), np.array([[1, 1, 0]]), np.array([[-2, -1,  0]])]], 
    #                   'p2': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])], [np.array([[0, 1, 2]])], [np.array([[-1,  0,  2]]), np.array([[1, 1, 0]]), np.array([[-2, -1,  0]])]], 
    #                   'p3': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])], [np.array([[0, 1, 2]])], [np.array([[-1,  0,  2]]), np.array([[1, 1, 0]]), np.array([[-2, -1,  0]])]], 
    #                   'common_knowledge': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])], [np.array([[0, 1, 2]])], [np.array([[-1,  0,  2]]), np.array([[ 1, -1, -4]]), np.array([[1, 1, 0]]), np.array([[-2, -1,  0]])]], 
    #                   'joint_knowledge': [[[np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])]], [[np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])], [np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])], [np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])]], 
    #                                       [[np.array([[0, 1, 2]])], [np.array([[0, 1, 2]])], [np.array([[0, 1, 2]])]], [[np.array([[ 1, -1, -4]]), np.array([[1, 1, 0]]), np.array([[-2, -1,  0]])], [np.array([[-1,  0,  2]]), np.array([[1, 1, 0]]), np.array([[-2, -1,  0]])], [np.array([[-1,  0,  2]]), np.array([[1, 1, 0]]), np.array([[-2, -1,  0]])]]] }
    # BEC_constraints =  [np.array([[1, 1, 0]]), np.array([[ 0, -1, -4]]), np.array([[-1,  0,  2]])]
    # unit_constraints = [np.array([[-1,  0,  0]]), np.array([[0, 1, 2]]), np.array([[-2, -1,  0]]), np.array([[-1,  0,  2]]), np.array([[1, 1, 0]])]
    # kc_id = 3
    # knowledge_constraints = copy.deepcopy(team_knowledge['p1'][kc_id])

    # x = copy.deepcopy(unit_constraints)
    # x.extend(knowledge_constraints)
    # x_min= BEC_helpers.remove_redundant_constraints(x, params.weights['val'], params.step_cost_flag)
    # min_unit_constraints = BEC_helpers.remove_redundant_constraints(unit_constraints, params.weights['val'], params.step_cost_flag)    
    
    # intersection_area = np.array(BEC_helpers.calc_solid_angles([x]))
    # min_unit_area = np.array(BEC_helpers.calc_solid_angles([min_unit_constraints]))
    # knowledge_area = np.array(BEC_helpers.calc_solid_angles([knowledge_constraints]))

    # union_area =  knowledge_area + min_unit_area - intersection_area

    # print('unit_knowledge_intersection_constraints: ', x)
    # print('min_unit_knowledge_intersection_constraints: ', x_min)
    # print('min_unit_constraints: ', min_unit_constraints)
    # print('knowledge_constraints: ', knowledge_constraints)

    # print('intersection area: ', intersection_area)
    # print('min unit area: ', min_unit_area)
    # print('knowledge area: ', knowledge_area)
    # print('union area: ', union_area)
    # print('knowledge_level: ', intersection_area/union_area)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    # ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    # ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    # ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    
    # utils_teams.visualize_planes_team(x, fig=fig, ax=ax1, alpha=0.5)
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(x)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax1, plot_ref_sphere=False, color = 'b')

    # utils_teams.visualize_planes_team(x_min, fig=fig, ax=ax2, alpha=0.5)
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(x_min)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax2, plot_ref_sphere=False, color = 'b')

    # utils_teams.visualize_planes_team(min_unit_constraints, fig=fig, ax=ax3, alpha=0.5)
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(min_unit_constraints)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax3, plot_ref_sphere=False, color = 'b')

    # utils_teams.visualize_planes_team(knowledge_constraints, fig=fig, ax=ax4, alpha=0.5)
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(knowledge_constraints)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax4, plot_ref_sphere=False, color = 'b')

    # label_axes(ax1, params.weights['val'])
    # label_axes(ax2, params.weights['val'])
    # label_axes(ax3, params.weights['val'])
    # label_axes(ax4, params.weights['val'])

    # plt.show()


    ############
    # # team_knowledge = {'p1': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])], [np.array([[0, 1, 2]])], [np.array([[3, 0, 0]])]], 
    # #                   'p2': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])], [np.array([[0, 1, 2]])], [np.array([[ 1,  0, -2]]), np.array([[1, 1, 0]]), np.array([[-2, -1,  0]])]], 
    # #                   'p3': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])], [np.array([[0, 1, 2]])], [np.array([[ 1,  0, -2]]), np.array([[1, 1, 0]]), np.array([[-2, -1,  0]])]], 
    # #                   'common_knowledge': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])], [np.array([[0, 1, 2]])], [np.array([[ 1,  0, -2]]), np.array([[1, 1, 0]]), np.array([[-2, -1,  0]]), np.array([[1, 0, 0]])]], 
    # #                   'joint_knowledge': [[[np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])]], [[np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])], [np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])], [np.array([[-1,  0,  2]]), np.array([[-1,  0,  0]])]], [[np.array([[0, 1, 2]])], [np.array([[0, 1, 2]])], [np.array([[0, 1, 2]])]], [[np.array([[3, 0, 0]])], [np.array([[ 1,  0, -2]]), np.array([[1, 1, 0]]), np.array([[-2, -1,  0]])], [np.array([[ 1,  0, -2]]), np.array([[1, 1, 0]]), np.array([[-2, -1,  0]])]]]}
    # BEC_constraints =  [np.array([[1, 1, 0]]), np.array([[ 0, -1, -4]]), np.array([[-1,  0,  2]])]
    # unit_constraints = [np.array([[-1,  0,  0]]), np.array([[0, 1, 2]]), np.array([[-2, -1,  0]]), np.array([[-1,  0,  2]]), np.array([[1, 1, 0]])]
    
    
    # team_knowledge =  {'p1': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -2]])]], 
    #                    'p2': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -2]])]], 
    #                    'p3': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]], 
    #                    'common_knowledge': [[np.array([[ 0,  0, -1]])], [np.array([[1,  0,  -2]]), np.array([[ 1,  0, -4]])]], 
    #                    'joint_knowledge': [[[np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])]], 
    #                                        [[np.array([[ 1,  0, -2]])], 
    #                                         [np.array([[ 1,  0, -2]])], 
    #                                         [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]]]}
    


    # # p3 =  [np.array([[ 0,  0, -1]]), np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]]), np.array([[ 0, -1, -2]]), np.array([[0, 1, 4]])]
    # # p3_min = BEC_helpers.remove_redundant_constraints(p3, params.weights['val'], params.step_cost_flag)
    # # print(p3_min)

    
    # kc_id_list = range(len(team_knowledge['p1']))

    # # min_KC_constraints =  [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]

    # # unit_learning_goal_reached = team_helpers.check_unit_learning_goal_reached(team_knowledge, min_KC_constraints, 1)

    # # print(unit_learning_goal_reached)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # # # for cnsts in team_knowledge['joint_knowledge']:
    # for kc_index in kc_id_list:
    #     for i in range(len(team_knowledge['joint_knowledge'][kc_index])):
    #         cnst = team_knowledge['joint_knowledge'][kc_index][i]
    #         print('Plotting joint constraints: ', cnst)
    #         ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(cnst)
    #         poly = Polyhedron.Polyhedron(ieqs=ieqs)
    #         BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax1, plot_ref_sphere=False, alpha=0.75)

    # # plt.show()

    # knowledge_constraints = []
    # knowledge_id = 'common_knowledge'
    # kc_id_list = range(len(team_knowledge[knowledge_id]))


    # # team_knowledge = team_helpers.update_team_knowledge(team_knowledge, kc_id, kc_reset_flag, test_constraints, params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = ['common_knowledge'])


    
    # for kc_index in kc_id_list:
    #     if knowledge_id == 'joint_knowledge':
    #         # # print('team_knowledge[knowledge_type][kc_index]: ', team_knowledge[knowledge_type][kc_index])
    #         for i in range(len(team_knowledge[knowledge_id][kc_index])):
    #             if first_index:
    #                 knowledge_constraints.append(utils_teams.flatten_list(copy.deepcopy(team_knowledge[knowledge_id][kc_index][i])))
    #                 # # print(colored('team_knowledge_constraints flattened: ' + str(team_knowledge_constraints), 'blue' ))
    #             else:
    #                 knowledge_constraints[i].extend(team_knowledge[knowledge_id][kc_index][i])
    #         first_index = False
    #     else:
    #         # print('team_knowledge[knowledge_type][kc_index]: ', team_knowledge[knowledge_type][kc_index])
    #         if len(knowledge_constraints) == 0:
    #             knowledge_constraints = copy.deepcopy(team_knowledge[knowledge_id][kc_index])
    #         else:
    #             knowledge_constraints.extend(team_knowledge[knowledge_id][kc_index])



    # print('knowledge_constraints:', knowledge_constraints)
    # min_const = BEC_helpers.remove_redundant_constraints(knowledge_constraints,  params.weights['val'], params.step_cost_flag)


    # utils_teams.visualize_planes_team(min_const, fig=fig, ax=ax2)
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(min_const)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax2, plot_ref_sphere=False, alpha=0.75)

    # utils_teams.visualize_planes_team(knowledge_constraints, fig=fig, ax=ax1)

    # plt.show()






    
    # # kc_id = 3


    # # knowledge_constraints = copy.deepcopy(team_knowledge['joint_knowledge'][kc_id])



    # ########################

    # # test_constraints_team =  [[np.array([[-1,  0,  0]])], [np.array([[-1,  0,  0]])], [np.array([[2, 0, 0]])]]

    # # response_category_team = ['correct', 'correct', 'incorrect']
    # # opposing_constraints_count = 0
    # # non_intersecting_constraints_count = 0

    # # test_constraints_team_expanded = []
    # # for test_constraints in test_constraints_team:
    # #     test_constraints_team_expanded.extend(test_constraints)

    # # print(colored('test_constraints_team_expanded: ' + str(test_constraints_team_expanded), 'red'))
    
    # # opposing_constraints_flag, opposing_constraints_count, opposing_idx = team_helpers.check_opposing_constraints(test_constraints_team_expanded, opposing_constraints_count)
    # # print('Opposing constraints normal loop? ', opposing_constraints_flag)
    # # # Assign majority rules and update common knowledge and joint knowledge accordingly
    # # if opposing_constraints_flag:
    # #     test_constraints_team_expanded = team_helpers.majority_rules_opposing_team_constraints(opposing_idx, test_constraints_team_expanded, response_category_team)

    # #     print(colored('test_constraints_team_expanded after majority rules opposinng constraints: ' + str(test_constraints_team_expanded), 'red' ))

    # # non_intersecting_constraints_flag, non_intersecting_constraints_count = team_helpers.check_for_non_intersecting_constraints(test_constraints_team_expanded, params.weights['val'], params.step_cost_flag, non_intersecting_constraints_count)
    # # print('Non-intersecting constraints normal loop? ', non_intersecting_constraints_flag)
    # # # Assign majority rules and update common knowledge and joint knowledge accordingly
    # # if non_intersecting_constraints_flag:
    # #     test_constraints_team_expanded, intersecting_constraints = team_helpers.majority_rules_non_intersecting_team_constraints(test_constraints_team_expanded, params.weights['val'], params.step_cost_flag)

    # #     print(colored('test_constraints_team_expanded after majority rules non intersecting: ' + str(test_constraints_team_expanded), 'red'))

    # # # double check again
    # # print('Double checking agan...')
    # # opposing_constraints_flag, opposing_constraints_count, opposing_idx = team_helpers.check_opposing_constraints(test_constraints_team_expanded, opposing_constraints_count)
    # # non_intersecting_constraints_flag, non_intersecting_constraints_count = team_helpers.check_for_non_intersecting_constraints(test_constraints_team_expanded, params.weights['val'], params.step_cost_flag, non_intersecting_constraints_count)

    # ##################

    # # all_unit_constraints = [np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]]), np.array([[0, 1, 2]]), np.array([[-1,  0,  0]]), np.array([[0, 1, 2]]), 
    # #                         np.array([[-2, -1,  0]]), np.array([[-1,  0,  2]]), np.array([[1, 1, 0]])]
    
    # # min_BEC_constraints = [np.array([[1, 1, 0]]), np.array([[ 0, -1, -4]]), np.array([[-1,  0,  2]])]

    # # min_all_unit_constraints = BEC_helpers.remove_redundant_constraints(all_unit_constraints, params.weights['val'], params.step_cost_flag)
    

    # # print('unit constraints: ', all_unit_constraints)
    # # print('min unit constraints: ', min_all_unit_constraints)
    # # print('BEC constraints: ', min_BEC_constraints)


    # # team_knowledge = {'p1': [min_all_unit_constraints]}

    # # all_unit_constraints_area = BEC_helpers.calc_solid_angles([all_unit_constraints])
    # # min_unit_constraints_area = BEC_helpers.calc_solid_angles([min_all_unit_constraints])
    # # BEC_area = BEC_helpers.calc_solid_angles([min_BEC_constraints])

    # # print('All Units Area: ', all_unit_constraints_area)
    # # print('Min Units Area: ', min_unit_constraints_area)
    # # print('BEC Area: ', BEC_area)

    # # knowledge_level = team_helpers.calc_knowledge_level(team_knowledge, min_BEC_constraints)

    # # print(knowledge_level)

    # # fig = plt.figure()
    # # ax1 = fig.add_subplot(1, 1, 1, projection='3d')

    # # utils_teams.visualize_planes_team(min_all_unit_constraints, fig=fig, ax=ax1, alpha=0.5)
    # # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(min_all_unit_constraints)
    # # poly = Polyhedron.Polyhedron(ieqs=ieqs)
    # # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax1, plot_ref_sphere=False, color = 'y')

    # # utils_teams.visualize_planes_team(min_BEC_constraints, fig=fig, ax=ax1, alpha=0.5)
    # # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(min_BEC_constraints)
    # # poly = Polyhedron.Polyhedron(ieqs=ieqs)
    # # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax1, plot_ref_sphere=False, color = 'b')

    # # plt.show()

    # ################################

    # # with open('models/' + params.data_loc['BEC'] + '/team_base_constraints.pickle', 'rb') as f:
    # #     policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(f)

    # # with open('models/' + params.data_loc['BEC'] + '/team_BEC_constraints.pickle', 'rb') as f:
    # #     min_BEC_constraints, BEC_lengths_record = pickle.load(f)


    # # print('Policy constraints: ', policy_constraints)

    # # print('min_subset_constraints_record: ', min_subset_constraints_record)

    # # print('min_BEC_constraints: ', min_BEC_constraints)

    # # print('BEC_lengths_record: ', BEC_lengths_record)

    # ##############################

    # # x = np.array([ 0.87365874, -0.44951806, -0.18615563])

    # # min_BEC_constraints_running = [np.array([[0, 0, -1]])]

    # # particles_team = pf_team.Particles_team(BEC_helpers.sample_human_models_uniform([], 500))
    # # # particles = pf.Particles(BEC_helpers.sample_human_models_uniform([], 500))
    # # particles_team.update(min_BEC_constraints_running)

    # # sample_human_models = BEC_helpers.sample_human_models_uniform(min_BEC_constraints_running, 20)
    # # sample_human_models_pf = BEC_helpers.sample_human_models_pf(particles_team, 20)

    # # team_helpers.plot_sampled_models(particles_team, sample_human_models, weights=params.weights['val'], fig=None, text='Sampled human models')


###############################

    # min_KC_constraints = [np.array([[  0,  -1, -10]]), np.array([[ 0, -1, -6]])]

    # with open('models/' + params.data_loc['BEC'] + '/team_base_constraints.pickle', 'rb') as f:
    #     policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(f)
    
    # with open('models/' + params.data_loc['BEC'] + '/team_BEC_constraints.pickle', 'rb') as f:
    #         min_BEC_constraints, BEC_lengths_record = pickle.load(f)

    # with open('models/augmented_taxi2/BEC_summary.pickle', 'rb') as f:
    #         summary, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = pickle.load(f)

    # running_variable_filter_unit =  np.array([[1, 0, 0]])

    # preliminary_tests, visited_env_traj_idxs = team_helpers.obtain_diagnostic_tests(params.data_loc['BEC'], summary[-1], visited_env_traj_idxs, min_KC_constraints, min_subset_constraints_record, traj_record, traj_features_record, running_variable_filter_unit, mdp_features_record)


    # print(preliminary_tests)

    # x = [np.array([[ 0,  0, -1]]), np.array([[ -2,  0, -2]]), np.array([[ 1,  0, -4]])]

    # x =  [np.array([[2, 0, 0]]), np.array([[-1,  0,  0]]), np.array([[-1,  0,  0]])]

    # x_min = BEC_helpers.remove_redundant_constraints(x, params.weights['val'], params.step_cost_flag)
    # print(x_min)

    # non_intersecting_constraints_flag, _ = team_helpers.check_for_non_intersecting_constraints(x, params.weights['val'], params.step_cost_flag)
    # print('non_intersecting_constraints_flag: ', non_intersecting_constraints_flag)
    # if non_intersecting_constraints_flag:
    #     x, _ = team_helpers.majority_rules_non_intersecting_team_constraints(x, params.weights['val'], params.step_cost_flag)

    # print('intersecting constraints: ', x)

    # fig = plt.figure()
    # ax0 = fig.add_subplot(1, 1, 1, projection='3d')
    # utils_teams.visualize_planes_team(x, fig=fig, ax=ax0)
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(x)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax0, plot_ref_sphere=False, alpha=0.75)
    # plt.show()

    #######################

    # team_knowledge =  {'p1': [[np.array([[ 0,  0, -1]])], [ np.array([[ -1,  0, 4]]), np.array([[ -1,  0, 2]]) ] ], 
    #                    'p2': [[np.array([[ 0,  0, -1]])], [ np.array([[ 1,  0, -4]]), np.array([[ 1,  0, -2]]) ] ], 
    #                    'p3': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]], 
    #                    'common_knowledge': [], 
    #                    'joint_knowledge': []}



    # test_constraints_team = [[ np.array([[ -1,  0, 4]]),  np.array([[ 1,  0, -4]]),  np.array([[ 1,  0, -4]])], 
    #                          [ np.array([[ -1,  0, 2]]), np.array([[ 1,  0, -2]]), np.array([[ -1,  0, 2]])] ]

    # # constraints = [np.array([[ 0,  0, -1]]),  np.array([[ 1,  0, -4]]), np.array([[ 1,  0, -2]]) ] 

    # # x_min = BEC_helpers.remove_redundant_constraints(constraints, params.weights['val'], params.step_cost_flag)
    # # print(x_min)

    # x, int_index = team_helpers.majority_rules_non_intersecting_team_constraints(test_constraints_team, params.weights['val'], params.step_cost_flag, test_flag = True) 

    # print(x)
    # print(int_index)




    #################################

    # team_knowledge = {'p1': [[np.array([[ 0,  0, -1]])], [np.array([[1, 0, 0]])]], 
    #                   'p2': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  0]])]], 
    #                   'p3': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  0]])]], 
    #                   'common_knowledge': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  0]])]], 
    #                   'joint_knowledge': [[[np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])]], [[np.array([[1, 0, 0]])], [np.array([[-1,  0,  0]])], [np.array([[-1,  0,  0]])]]]}
    
    # min_KC_constraints =  [np.array([[-1,  0,  0]])]  



    # team_knowledge =  {'p1': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -2]])]], 
    #                   'p2': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]], 
    #                   'p3': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]], 
    #                   'common_knowledge': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]])]], 
    #                   'joint_knowledge': [[[np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])]], 
    #                                       [[np.array([[ 1,  0, -2]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]]]}
    
    # KC_constraints =  [np.array([[ 0,  0, -1]]), np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]

    # min_KC_constraints = BEC_helpers.remove_redundant_constraints(KC_constraints, params.weights['val'], params.step_cost_flag)

    # min_BEC_constraints = [np.array([[1, 1, 0]]), np.array([[ 0, -1, -4]]), np.array([[-1,  0,  2]])]

    # all_unit_constraints = [np.array([[-2, -1,  0]]), np.array([[1, 1, 0]]), np.array([[-1,  0,  2]])]
                                                   

    # print('min_KC_constraints: ', min_KC_constraints)
    
    # knowledge = team_helpers.calc_knowledge_level(team_knowledge, all_unit_constraints)

    # print(knowledge)

    #######################

    # x =   [np.array([[ 0,  0, -1]]), np.array([[ -1,  0, 4]]), np.array([[1, 1, 0]]), np.array([[ 0, -1, -4]]), np.array([[-1,  0,  2]])]
    # # x = [np.array([[ 1,  0, -2]]),  np.array([[-1,  0,  2]])]

    # # x=  [np.array([[ 0,  0, -1]]), np.array([[ 1,  0, -2]]), np.array([[ 0,  0, -1]]), np.array([[-1,  0,  4]])]

    # print(team_helpers.check_for_non_intersecting_constraints(x, params.weights['val'], params.step_cost_flag))

    # print(BEC_helpers.calc_solid_angles([x]))


    # fig = plt.figure()
    # ax0 = fig.add_subplot(1, 1, 1, projection='3d')
    # utils_teams.visualize_planes_team(x, fig=fig, ax=ax0)
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(x)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax0, plot_ref_sphere=False, alpha=0.75)
    # plt.show()

    #################################

    # team_knowledge =  {'p1': [[np.array([[ 0,  0, -1]])], [np.array([[1,  0,  -4]]), np.array([[-1,  0,  2]])]], 
    #                   'p2': [[np.array([[ 0,  0, -1]])], [np.array([[ 2,  0, -2]])]], 
    #                   'p3': [[np.array([[ 0,  0, -1]])], [np.array([[ -1,  0, 4]])]], 
    #                   'common_knowledge': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]])]], 
    #                   'joint_knowledge': [[[np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])]], 
    #                                       [[np.array([[1,  0,  -4]]), np.array([[ -1,  0, 2]])], [np.array([[1,  0,  -1]])], [np.array([[ -1,  0, 4]])]]]}
    
    # min_KC_constraints =  [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]


    # # knowledge = team_helpers.calc_knowledge_level(team_knowledge, min_KC_constraints)

    # # fig = team_helpers.visualize_team_knowledge_constraints(team_knowledge, params.weights['val'], params.step_cost_flag, particles_team = None, kc_id = None, fig=None, text=None, plot_min_constraints_flag = False, plot_text_flag = False, min_unit_constraints = [], plot_filename = 'team_knowledge_constraints', fig_title = None)

    # # plot_constraints = [np.array([[ 0,  0, -1]]), np.array([[-1,  0,  2]]), np.array([[1,  0,  -1]])]

    # plot_constraints = [[np.array([[1,  0,  -4]]), np.array([[-1,  0,  2]])], [np.array([[ 2,  0, -2]])], [np.array([[ -1,  0, 4]])] ]
    

    # plot_constraints, _ = team_helpers.majority_rules_non_intersecting_team_constraints(plot_constraints, params.weights['val'], params.step_cost_flag, test_flag=True)

    # print('plot_constraints: ', plot_constraints)
    
    # # print(knowledge)

    #############################

    # test_constraints_team =  [[np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[-1,  0,  4]])]]

    # # majority_rules_non_intersecting_team_constraints(test_constraints_team, weights, step_cost_flag, test_flag = False)

    # intersections_to_check =  [(0, 1), (0, 2), (1, 2)]

    # max_constraints = [test_constraints_team[cnst_id] for cnst_id in intersections_to_check[0] ]

    # print(max_constraints)


    #################################

    # team_knowledge:  {'p1': [[array([[ 0,  0, -1]])], [array([[-1,  0,  2]]), array([[-1,  0,  4]])], [array([[ 0, -1, -2]]), array([[0, 1, 4]])]], 'p2': [[array([[ 0,  0, -1]])], [array([[ 1,  0, -2]]), array([[ 1,  0, -4]])], [array([[ 0, -1, -2]]), array([[0, 1, 4]])]], 'p3': [[array([[ 0,  0, -1]])], [array([[-1,  0,  2]]), array([[ 1,  0, -4]])], [array([[0, 1, 2]]), array([[ 0, -1, -4]])]], 'common_knowledge': [[array([[ 0,  0, -1]])], [array([[-1,  0,  2]]), array([[-1,  0,  4]])], [array([[ 0, -1, -2]]), array([[0, 1, 4]])]], 'joint_knowledge': [[[array([[ 0,  0, -1]])], [array([[ 0,  0, -1]])], [array([[ 0,  0, -1]])]], [[array([[-1,  0,  2]]), array([[-1,  0,  4]])], [array([[ 1,  0, -2]]), array([[ 1,  0, -4]])], [array([[-1,  0,  2]]), array([[ 1,  0, -4]])]], [[array([[ 0, -1, -2]]), array([[0, 1, 4]])], [array([[ 0, -1, -2]]), array([[0, 1, 4]])], [array([[0, 1, 2]]), array([[ 0, -1, -4]])]]]}
    
    
    # min_KC_constraints:  [array([[ 0, -1, -4]]), array([[0, 1, 2]])]


    # team_knowledge =  {'p1': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])]], 
    #                   'p2': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])]], 
    #                   'p3': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[ 0, -1, -2]]), np.array([[0, 1, 4]])]], 
    #                   'common_knowledge': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]])], [np.array([[ 0, -1, -4]]), np.array([[0, 1, 2]])]], 
    #                   'joint_knowledge': [[[np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])]], [[np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]], [[np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], [np.array([[ 0, -1, -2]]), np.array([[0, 1, 4]])]]]}
   
    # min_KC_constraints =  [np.array([[ 0, -1, -4]]), np.array([[0, 1, 2]])]

    # min_BEC_constraints = [np.array([[1, 1, 0]]), np.array([[ 0, -1, -4]]), np.array([[-1,  0,  2]])]

    # unit_knowledge = team_helpers.calc_knowledge_level(team_knowledge, min_KC_constraints, kc_id_list = [2], plot_flag = False, fig_title = 'Unit knowledge')
    # print(colored('unit_knowledge: ', 'blue'), unit_knowledge)
    # BEC_knowledge = team_helpers.calc_knowledge_level(team_knowledge, min_BEC_constraints, plot_flag = True, fig_title = 'BEC knowledge')
    # print(colored('BEC_knowledge: ', 'blue'), BEC_knowledge)


    #####################

    # team_knowledge =  {'p1': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]], 
    #                   'p2': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  4]])]], 
    #                   'p3': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]], 
    #                   'common_knowledge': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]])]], 
    #                   'joint_knowledge': [[[np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])]], 
    #                                       [[np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[-1,  0,  4]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]]]}


    # team_knowledge =  {'p1': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -2]]), np.array([[-1,  0,  4]])]], 
    #                    'p2': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]])]], 
    #                    'p3': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]])]], 
    #                    'common_knowledge': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]])]], 
    #                    'joint_knowledge': [[[np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])]], 
    #                                        [[np.array([[ 1,  0, -2]]), np.array([[-1,  0,  4]])], [np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]])], [np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]])]]]}


    # team_knowledge = {'p1': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], [np.array([[ 2, -1, -2]]), np.array([[-1,  1,  0]]), np.array([[1, 1, 0]])]], 
    #                 'p2': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], [np.array([[ 1, -1, -2]]), np.array([[-1,  1,  0]]), np.array([[ 0, -1, -2]])]], 
    #                 'p3': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  1,  0]]), np.array([[1, 1, 0]])]], 
    #                 'common_knowledge': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]])], [np.array([[ 0, -1, -4]]), np.array([[0, 1, 2]])], [np.array([[ 2, -1, -2]]), np.array([[-1,  1,  0]]), np.array([[ 0, -1, -2]]), np.array([[1, 1, 0]])]], 
    #                 'joint_knowledge': [[[np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])]], [[np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]], [[np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])]], [[np.array([[ 2, -1, -2]]), np.array([[-1,  1,  0]]), np.array([[1, 1, 0]])], [np.array([[ 1, -1, -2]]), np.array([[-1,  1,  0]]), np.array([[ 0, -1, -2]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  1,  0]]), np.array([[1, 1, 0]])]]]}
   

    team_knowledge = {'p1': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], [np.array([[ 1, -1,  0]])]], 
                      'p2': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], [np.array([[1, 1, 0]])]], 
                      'p3': [[np.array([[ 0,  0, -1]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], [np.array([[1, 1, 0]])]], 
                      'common_knowledge': [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]])], [np.array([[ 0, -1, -4]]), np.array([[0, 1, 2]])], [np.array([[ 1, -1,  0]])]], 
                      'joint_knowledge': [[[np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])], [np.array([[ 0,  0, -1]])]], 
                                          [[np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]], 
                                          [[np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])]], 
                                          [[np.array([[ 1, -1,  0]])], [np.array([[1, 1, 0]])], [np.array([[1, 1, 0]])]]]}


    # min_KC_constraints =  [np.array([[ 1, 0, -4]]), np.array([[-1, 0, 2]])]

    min_KC_constraints = [np.array([[ 1, 1, 0]])]

    min_BEC_constraints = [np.array([[1, 1, 0]]), np.array([[ 0, -1, -4]]), np.array([[-1,  0,  2]])]

    # unit_knowledge = team_helpers.calc_knowledge_level(team_knowledge, min_KC_constraints, kc_id_list = [1], plot_flag = False, fig_title = 'Unit knowledge')
    # print(colored('unit_knowledge: ', 'blue'), unit_knowledge)
    # BEC_knowledge = team_helpers.calc_knowledge_level(team_knowledge, min_BEC_constraints, plot_flag = False, fig_title = 'BEC knowledge')
    # print(colored('BEC_knowledge: ', 'blue'), BEC_knowledge)

    # team_helpers.visualize_team_knowledge_constraints(team_knowledge, params.weights['val'], params.step_cost_flag, particles_team = None, kc_id = None, fig=None, text=None, plot_min_constraints_flag = False, plot_text_flag = False, min_unit_constraints = [], plot_filename = 'team_knowledge_constraints', fig_title = None)
    # plt.show()

    ###################################


    # x = [np.array([[ 0,  0, -1]]), np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]
    # x = [ np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]
    # x = [np.array([[ 0,  0, -1]]), np.array([[ 3,  0, -2]]), np.array([[ 1,  0, -4]])]
    # x = [np.array([[ 0,  0, -1]]), np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]]), np.array([[ 0, -1, -2]]), np.array([[0, 1, 4]])]
    # x = [np.array([[ 0,  0, -1]]), np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]]), np.array([[0, 1, 4]])]
    
    
    # print(team_helpers.check_for_non_intersecting_constraints(x, params.weights['val'], params.step_cost_flag))

    # x = BEC_helpers.remove_redundant_constraints(x, params.weights['val'], params.step_cost_flag)
    # print(x)
    # print(BEC_helpers.calc_solid_angles([x]))

    ##########################

    # x = [ np.array([[ 0,  0, -1]]), np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]]), np.array([[ 0, -1, -2]]), np.array([[ 0,  1,  4]]) ]

    # print(team_helpers.check_for_non_intersecting_constraints(x, params.weights['val'], params.step_cost_flag))

    #############

    test_constraints_team = [[np.array([[1, -1,  0]])], [np.array([[1, 1, 0]])], [np.array([[1, 1, 0]])]]

    test_constraints_team_expanded = [np.array([[1, -1,  0]]), np.array([[1, 1, 0]]), np.array([[1, 1, 0]])]

    print(team_helpers.check_for_non_intersecting_constraints(test_constraints_team_expanded, params.weights['val'], params.step_cost_flag))

    alternate_team_constraints, intersecting_cnst = team_helpers.majority_rules_non_intersecting_team_constraints(test_constraints_team, params.weights['val'], params.step_cost_flag, test_flag=True)

    print(alternate_team_constraints)

    ck = [[np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[ 1,  0, -4]])], [np.array([[ 0, -1, -4]]), np.array([[0, 1, 2]])], [np.array([[ 1, -1,  0]])]]

    print(BEC_helpers.remove_redundant_constraints(test_constraints_team_expanded, params.weights['val'], params.step_cost_flag))
