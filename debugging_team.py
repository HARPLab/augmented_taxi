import dill as pickle
import sys, os
import policy_summarization.BEC_helpers as BEC_helpers
import params_team as params
import itertools
import copy
from policy_summarization import BEC
import policy_summarization.BEC_visualization as BEC_viz

import sage.all
import sage.geometry.polyhedron.base as Polyhedron
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from policy_summarization import particle_filter as pf
import numpy as np
from policy_summarization import computational_geometry as cg
from termcolor import colored
from sklearn import metrics
from policy_summarization import policy_summarization_helpers as ps_helpers

import teams.teams_helpers as team_helpers
from teams import particle_filter_team as pf_team
import teams.utils_teams as utils_teams
from simple_rl.agents import FixedPolicyAgent
from simple_rl.utils import mdp_helpers

import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = '1.0'
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'large'

from collections import Counter
import operator
from simple_rl.agents import FixedPolicyAgent
from simple_rl.planning import ValueIteration
import random
import pygame
from numpy import linalg as LA

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

    fig = plt.figure()

    # plot the constraints
    ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints)
    poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    for constraints in [constraints]:
        BEC_viz.visualize_planes(constraints, fig=fig)
    BEC_viz.visualize_spherical_polygon(poly, fig=fig, plot_ref_sphere=False, alpha=0.75)

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


    # # 2)
    # min_BEC_constraints =  [np.array([[1, 1, 0]]), np.array([[ 0, -1, -4]]), np.array([[-1,  0,  2]])]

    # min_unit_constraints = [np.array([[-1,  0,  0]]), np.array([[-1,  0,  2]])]
    # # min_unit_constraints = [np.array([[-1,  0,  0]])]

    # team_knowledge = {'p1': [np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])], 
    #                 'p2': [np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])], 
    #                 'common_knowledge': [np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])], 
    #                 'joint_knowledge': [[np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])], [np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])]]}
    
    # # knowledge_level = team_helpers.calc_knowledge_level(team_knowledge, min_unit_constraints)


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

    # # fig = plt.figure()
    # # ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    # # utils_teams.visualize_planes_team(min_intersection_constraints, fig=fig, ax=ax1)
    # # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax1, plot_ref_sphere=False, alpha=0.75)
    # # label_axes(ax1, params.weights['val'])
    # # plt.show()

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

    # constraints = [np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]]), np.array([[ 0,  0, -1]]), np.array([[ 3,  0, -2]]), np.array([[-1,  0,  2]]), np.array([[ 0,  0, -1]])]
    constraints = [np.array([[-1,  0,  2]]), np.array([[3,  0,  -2]]), np.array([[ 0,  0, -1]])]

    # print(BEC_helpers.remove_redundant_constraints(constraints, params.weights['val'], params.step_cost_flag))
    # common_constraints = BEC_helpers.remove_redundant_constraints(constraints, params.weights['val'], params.step_cost_flag)

    # unit_constraint_flag = True
    # for constraint in common_constraints:
    #     print('Constraint: ', constraint, 'norm: ', LA.norm(constraint,1))
    #     if LA.norm(constraint,1) !=1:
    #         unit_constraint_flag = False
    # if unit_constraint_flag:
    #     utils_teams.visualize_planes_team(common_constraints)
    #     plt.show()
    
    unit_constraint_flag = True
    unit_common_constraint_flag = True
    constraints_copy = constraints.copy()
    
    while not unit_constraint_flag and unit_common_constraint_flag:
        
        common_constraints = BEC_helpers.remove_redundant_constraints(constraints_copy, params.weights['val'], params.step_cost_flag)
        print('Origninal common knowledge: ', constraints_copy)
        print('Remove redundant common knowledge: ', common_constraints)

        for constraint in common_constraints:
            if LA.norm(constraint,1) !=1:
                unit_constraint_flag = False

        for constraint in common_constraints:
            if LA.norm(constraint,1) !=1:
                unit_common_constraint_flag = False

        if not unit_constraint_flag and unit_common_constraint_flag:
            constraints_copy.pop(0)  # remove the first constraint




