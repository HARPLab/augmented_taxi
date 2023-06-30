import dill as pickle
import sys, os
import policy_summarization.BEC_helpers as BEC_helpers
import params
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

import teams.teams_helpers as team_helpers

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

    
    # plot BEC area and check

    







