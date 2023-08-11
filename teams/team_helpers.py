## This contains all new codes related to the robot teaching to a team (to be modularized and structured later)

# Python imports.
import sys
import dill as pickle
import numpy as np
import copy
from termcolor import colored
from pathos.multiprocessing import ProcessPool as Pool
import sage.all
import sage.geometry.polyhedron.base as Polyhedron
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import itertools
from sklearn import metrics

# Other imports.
sys.path.append("simple_rl")
import params
from simple_rl.agents import FixedPolicyAgent
from simple_rl.planning import ValueIteration
from simple_rl.utils import make_mdp
from policy_summarization import bayesian_IRL
from policy_summarization import policy_summarization_helpers as ps_helpers
from policy_summarization import BEC
import policy_summarization.multiprocessing_helpers as mp_helpers
from simple_rl.utils import mdp_helpers
import policy_summarization.BEC_helpers as BEC_helpers
import policy_summarization.BEC_visualization as BEC_viz
from teams import particle_filter_team as pf_team
import teams.utils_teams as utils_teams
from policy_summarization import probability_utils as p_utils
from policy_summarization import computational_geometry as cg
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = '1.0'
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'large'

import random
from scipy.spatial import geometric_slerp
import matplotlib.tri as mtri




##########################################################




def calc_common_knowledge(team_knowledge, team_size, weights, step_cost_flag):

    
    ''''for i in range(team_size):
        member_id = 'p' + str(i+1)
        if i==0:
            constraints = team_knowledge[member_id].copy()
        else:
            constraints.extend(team_knowledge[member_id])''''

    constraints = team_knowledge["members_list"].copy()
    common_constraints = BEC_helpers.remove_redundant_constraints(constraints, weights, step_cost_flag)

    return common_constraints



''''def calc_joint_knowledge(team_knowledge, team_size, weights=None, step_cost_flag=None):

    # TODO: Calculate joint constraints
    # Joint constraints is normally expressed as the union of constraints of individuals. 

    # ## Method 1: Joint constraints expressed as inverse of the inverted and minimal constraints of individual members
    # for i in range(team_size):
    #     member_id = 'p' + str(i+1)
    #     if i==0:
    #         inv_constraints = [-x for x in team_knowledge[member_id]]
    #     else:
    #         inv_constraints.extend([-x for x in team_knowledge[member_id]])

    # print('Team knowledge:', team_knowledge)
    # print('Inverted constraints:', inv_constraints)
    # inv_joint_constraints = BEC_helpers.remove_redundant_constraints(inv_constraints, weights, step_cost_flag)

    # joint_constraints = [-x for x in inv_joint_constraints]


    ## Method 2: Here we just represent the constraints of each individual as a separate list.

    for i in range(team_size):
        member_id = 'p' + str(i+1)
        if i==0:
            joint_constraints = [team_knowledge[member_id]]
        else:
            joint_constraints.append(team_knowledge[member_id])


    # print('Team knowledge:', team_knowledge)
    # print('Joint constraints:', joint_constraints)

    return team_knowledge["members_list"]''''



def update_team_knowledge(team_knowledge, new_constraints, team_size, weights, step_cost_flag, knowledge_to_update = 'all'):

    team_knowledge_updated = team_knowledge.copy()

    # checking what needs to be updated: common_knowledge, joint_knowledge, or both
    if knowledge_to_update == 'all':
        knowledge_types = list(team_knowledge.keys())
    else:
        knowledge_types = knowledge_to_update.copy()

    # iterating throught the knowledge types
    for knowledge_type in knowledge_types:
        # if the knowledge type is not one of the base knowledge types will simply add the new_constraints to that knowledge id
        if knowledge_type != 'joint_knowledge' and  knowledge_type != 'common_knowledge':
            knowledge = team_knowledge[knowledge_type].copy()
            print('Old_knowledge: ', knowledge)
            knowledge.extend(new_constraints)
            print('New_knowledge: ', knowledge)
            # print('New_knowledge: ', new_knowledge)
            # print('New constraints: ', new_constraints)
            new_knowledge = BEC_helpers.remove_redundant_constraints(knowledge, weights, step_cost_flag)

            team_knowledge_updated[knowledge_type] = new_knowledge.copy()
            '''
            elif  knowledge_type == 'joint_knowledge':
                team_knowledge_updated[knowledge_type] = calc_joint_knowledge(team_knowledge_updated, team_size, weights=weights, step_cost_flag=step_cost_flag)'''
        elif  knowledge_type == 'common_knowledge':
            team_knowledge_updated[knowledge_type] = calc_common_knowledge(team_knowledge_updated, team_size, weights=weights, step_cost_flag=step_cost_flag)
        else:
            # would be reached in the case of joint_knowledge since this is now automatically updated
            print(colored('Unknow knowledge type to update or joint_knowledge is being updated even though it is automatically updated', 'red'))

    return team_knowledge_updated



# seems like this fincution was from original framework
def check_info_gain(info_gains_record, consistent_state_count):
    '''
    Do a quick check of whether there's any information to be gained from any of the trajectories.
    '''
    no_info_flag = False
    max_info_gain = 1

    if consistent_state_count:
        info_gains = np.array(info_gains_record)
        if np.sum(info_gains > 1) == 0:
            no_info_flag = True
    else:
        info_gains_flattened_across_models = list(itertools.chain.from_iterable(info_gains_record))
        info_gains_flattened_across_envs = list(itertools.chain.from_iterable(info_gains_flattened_across_models))
        if sum(np.array(info_gains_flattened_across_envs) > 1) == 0:
            no_info_flag = True
    
    return no_info_flag

                
# seems like this function was from original framework
def check_and_update_variable_filter(min_subset_constraints_record = None, variable_filter = None, nonzero_counter = None, initialize_filter_flag = False, no_info_flag = False):
    
    teaching_complete_flag = True
    
    if initialize_filter_flag:
        # true constraints
        min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record for item in sublist]
        min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record_flattened for item in sublist]
        min_subset_constraints_record_array = np.array(min_subset_constraints_record_flattened)


        # for variable scaffolding:
        # count how many nonzero constraints are present for each reward weight (i.e. variable) in the minimum BEC constraints
        # (which are obtained using one-step deviations). mask variables in order of fewest nonzero constraints for variable scaffolding
        # rationale: the variable with the most nonzero constraints, often the step cost, serves as a good reference/ratio variable
        nonzero_counter = (min_subset_constraints_record_array != 0).astype(float)
        nonzero_counter = np.sum(nonzero_counter, axis=0)
        nonzero_counter = nonzero_counter.flatten()

        # initialize variable filter
        variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
        print('variable filter: {}'.format(variable_filter))

        teaching_complete_flag = False

    else:
        # no need to continue search for demonstrations if none of them will improve the human's understanding
        if no_info_flag:
            # if no variables had been filtered out, then there are no more informative demonstrations to be found
            if not np.any(variable_filter):
                teaching_complete_flag = True
            else:
                # no more informative demonstrations with this variable filter, so update it
                variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
                print(colored('Did not find any more informative demonstrations.', 'red'))
                print('Updated variable filter: {}'.format(variable_filter))
                teaching_complete_flag = False
                
    
    return variable_filter, nonzero_counter, teaching_complete_flag


# use the member id list and just return the sorted variable
#going to try to not use this anymore
'''def find_ascending_individual_knowledge(team_knowledge, min_BEC_constraints):

    # Sorts based on ascending order of knowledge

    team_knowledge_level = calc_knowledge_level(team_knowledge, min_BEC_constraints)
    # sorted_kl = sorted(team_knowledge_level)
    sorted_kl = dict(sorted(team_knowledge_level.items(), key=lambda item: item[1])) # why is this a dictionary when it is being changed to a list later
    # I think the above line can be rewritten to:
    # sorted_kl = list(sorted(team_knowledge_level.values))
    # only issue I see with this is it doesnt remove the common or joint knowledge items 
    # shouldn't it only sort the members?

    #is this for debugging?
    print('sorted_kl: ', sorted_kl)
    ascending_order_of_knowledge = []
    for i, kl in enumerate(sorted_kl):
        print(kl)
        if 'p' in kl:
            ascending_order_of_knowledge.append(kl)
    ascending_order_of_knowledge = sorted(team_knowledge_level['members_list'])

    return ascending_order_of_knowledge'''



def calc_knowledge_level(team_knowledge, min_unit_constraints, weights = None, step_cost_flag = False):
    # 
    knowledge_level = {}
    

    print('Calculating knowledge level...')


    def calc_inverse_constraints():
        inv_constraints = []
        for k_id, k_type in enumerate(team_knowledge):
            if 'p' in k_type:
                inv_constraints.extend([-x for x in team_knowledge[k_type]])

        inv_joint_constraints = BEC_helpers.remove_redundant_constraints(inv_constraints, weights, step_cost_flag)

        return inv_constraints, inv_joint_constraints

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
    

    for knowledge_id, knowledge_type in enumerate(team_knowledge):

        print('Calculating knowledge level for: ', knowledge_type)

        if knowledge_type == 'members_list':
            # TODO: Knowledge metric for joint knowledge
            
            ## Method 1: Joint constraints expressed as inverse of the inverted and minimal constraints of individual members
            #basically doing calc_join_knowledge here would be much simplier with the members already being stored in a list
            # can be optimized further with members being stored in list
            '''joint_constraints = []
            for k_id, k_type in enumerate(team_knowledge):
                if 'p' in k_type:
                    joint_constraints.extend(team_knowledge[k_type])'''
            joint_constraints = team_knowledge["members_list"]

            min_unit_intersection_constraints = min_unit_constraints.copy()
            min_unit_intersection_constraints.extend(joint_constraints)


            opposing_constraints = False
            for cnst in min_unit_intersection_constraints:
                for cnst2 in min_unit_intersection_constraints:
                    # print(np.array_equal(-cnst, cnst2))
                    # print(-cnst, cnst2)
                    if (np.array_equal(-cnst, cnst2)):
                        opposing_constraints = True                
            
            if opposing_constraints:
                min_unit_BEC_knowledge_intersection = 0
            else:
                min_unit_intersection_constraints = BEC_helpers.remove_redundant_constraints(min_unit_intersection_constraints, params.weights['val'], params.step_cost_flag)
                min_unit_BEC_knowledge_intersection = np.array(BEC_helpers.calc_solid_angles([min_unit_intersection_constraints]))
            
            print('min_unit_intersection_constraints: ', min_unit_intersection_constraints)
            print('opposing_constraints: ', opposing_constraints)

            min_unit_area = np.array(BEC_helpers.calc_solid_angles([min_unit_constraints]))

            knowledge_area = 0
            for ind_constraints in team_knowledge['members_list']:
                ind_intersection_constraints = min_unit_constraints.copy()
                ind_intersection_constraints.extend(ind_constraints)
                ind_intersection_constraints = BEC_helpers.remove_redundant_constraints(ind_intersection_constraints, params.weights['val'], params.step_cost_flag)
                print('ind_intersection_constraints: ', ind_intersection_constraints)
                knowledge_area += np.array(BEC_helpers.calc_solid_angles([ind_intersection_constraints]))

            min_unit_BEC_knowledge_union = min_unit_area + knowledge_area - min_unit_BEC_knowledge_intersection
            
            # check if the knowledge area is a subset of the BEC area
            # TODO: Check this joint knowledge metric
            if min_unit_BEC_knowledge_intersection == knowledge_area/2:   #TODO: Update this checker!
                knowledge_level[knowledge_type] = 1
            else:
                knowledge_level[knowledge_type] = min_unit_BEC_knowledge_intersection/min_unit_BEC_knowledge_union

            

        else:
        #     # print('min_BEC_constraints: ', [min_BEC_constraints])
        #     min_BEC_area = BEC_helpers.calc_solid_angles([min_BEC_constraints])
        #     # print('knowledge: ', team_knowledge[knowledge_type])
        #     knowledge_area = BEC_helpers.calc_solid_angles([team_knowledge[knowledge_type]])
            
        #     # n_particles = 5000
        #     n_particles = 500
        #     knowledge_particles = pf_team.Particles_team(BEC_helpers.sample_human_models_uniform(team_knowledge[knowledge_type], n_particles))

        #     const_id = []
        #     for j, x in enumerate(knowledge_particles.positions):

        #         all_constraints_satisfied = True
        #         for constraint in min_BEC_constraints:
        #             dot = constraint.dot(x.T)

        #             if dot < 0:
        #                 all_constraints_satisfied = False
                
        #         if all_constraints_satisfied:
        #             const_id.append(j)
            
        #     BEC_overlap_area = min(min_BEC_area, len(const_id)/n_particles * np.array(knowledge_area))
            
        #     # # Method 1: Calculates the knowledge level (0 to 1) based on two factors: ratio of BEC area to knowledge area and % of overlap of BEC area with knowledge area
        #     # knowledge_spread = np.array(knowledge_area)/np.array(min_BEC_area)
        #     # BEC_overlap_ratio = BEC_overlap_area/np.array(min_BEC_area)
        #     # knowledge_level[knowledge_type] = 0.5*BEC_overlap_ratio + 0.5/knowledge_spread

        #     # Method 2: Calculate knowledge level based on the ratio of BEC overlap area with the total knowledge area
        #     knowledge_level[knowledge_type] = BEC_overlap_area/np.array(knowledge_area)

            ##############################################

            # Method 3: Use Jaccard's index for set similarity (Intersection over Union)

            # print('min_BEC_constraints: ', [min_BEC_constraints])
            min_unit_area = np.array(BEC_helpers.calc_solid_angles([min_unit_constraints]))
            
            knowledge_area = np.array(BEC_helpers.calc_solid_angles([team_knowledge[knowledge_type]]))

            min_unit_intersection_constraints = min_unit_constraints.copy()
            min_unit_intersection_constraints.extend(team_knowledge[knowledge_type])
            print('min_unit_intersection_constraints: ', min_unit_intersection_constraints)
            min_unit_intersection_constraints = BEC_helpers.remove_redundant_constraints(min_unit_intersection_constraints, weights, step_cost_flag)

            min_unit_BEC_knowledge_intersection = np.array(BEC_helpers.calc_solid_angles([min_unit_intersection_constraints]))

            print('min_BEC_knowledge_intersection: ', min_unit_BEC_knowledge_intersection)


            if min_unit_BEC_knowledge_intersection > 0:
                min_unit_BEC_knowledge_union = min_unit_area + knowledge_area - min_unit_BEC_knowledge_intersection
            else:
                min_unit_BEC_knowledge_union = 1
            
            # check if the knowledge area is a subset of the BEC area
            if min_unit_BEC_knowledge_intersection == knowledge_area:
                knowledge_level[knowledge_type] = 1
            else:
                knowledge_level[knowledge_type] = min_unit_BEC_knowledge_intersection/min_unit_BEC_knowledge_union


            # Method 4: An improved method for disjoint sets to see how close the disjoint is Generalized Intersection over Union (see https://giou.stanford.edu/)
            # TODO: Later




    return knowledge_level



def optimize_visuals_team(data_loc, best_env_idxs, best_traj_idxs, chunked_traj_record, summary, type='training'):
    visual_dissimilarities = np.zeros(len(best_env_idxs))
    complexities = np.zeros(len(best_env_idxs))

    prev_env_idx = None
    for j, best_env_idx in enumerate(best_env_idxs):

        # assuming that environments are provided in order of monotonically increasing indexes
        if prev_env_idx != best_env_idx:
            # reset the visual dissimilarity dictionary for a new MDP
            average_dissimilarity_dict = {}

            filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
            with open(filename, 'rb') as f:
                wt_vi_traj_env = pickle.load(f)
            best_mdp = wt_vi_traj_env[0][1].mdp

        if len(summary) >= 1:
            
            # debug
            for i in range(len(summary)):
                print('Summary element :', i, ',', summary[i])
            
            if type == 'training':

                # only consider the most recent demo
                visual_dissimilarities[j] = best_mdp.measure_visual_dissimilarity(
                        chunked_traj_record[best_env_idx][best_traj_idxs[j]][0][0], summary[0][2], summary[0][3][0][0])
            elif type == 'testing':
                # consider all previous demos
                for demo in summary:
                    visual_dissimilarities[j] += best_mdp.measure_visual_dissimilarity(
                    chunked_traj_record[best_env_idx][best_traj_idxs[j]][0][0], demo[0], demo[1][0][0])

                visual_dissimilarities /= len(summary)
            else:
                raise AssertionError("Unsupported type for visual optimization")
        else:
            first_state = chunked_traj_record[best_env_idx][best_traj_idxs[j]][0][0]

            # compare visual dissimilarity of this state to other states in this MDP, trying to minimize dissimilarity.
            # the rationale behind this is that you want to have a starting demonstration that can be easily followed
            # up by visually similar demonstrations
            if first_state in average_dissimilarity_dict:
                visual_dissimilarities[j] = average_dissimilarity_dict[first_state]
            else:
                average_dissimilarity = 0
                for other_state_idx, other_state in enumerate(best_mdp.states):
                    if first_state != other_state:
                        average_dissimilarity += best_mdp.measure_visual_dissimilarity(first_state, best_mdp, other_state)

                average_dissimilarity = average_dissimilarity / (len(best_mdp.states) - 1)
                average_dissimilarity_dict[first_state] = average_dissimilarity

                visual_dissimilarities[j] = round(average_dissimilarity)

        # get demos of low visual complexity
        complexities[j] = best_mdp.measure_env_complexity(chunked_traj_record[best_env_idx][best_traj_idxs[j]][0][0])

        prev_env_idx = best_env_idx

    tie_breaker = np.arange(len(best_env_idxs))
    np.random.shuffle(tie_breaker)

    if type == 'testing':
        # if obtaining tests, opt for greatest complexity and dissimilarity to previous demonstrations
        complexities *= -1
        visual_dissimilarities *= -1

    # sort first for visual simplicity, then visual similarity  (sorts from small to large values)
    sorted_zipped = sorted(zip(complexities, visual_dissimilarities, tie_breaker, best_env_idxs, best_traj_idxs))
    complexities_sorted, visual_dissimilarities_sorted, _, best_env_idxs_sorted, best_traj_idxs_sorted = list(
        zip(*sorted_zipped))

    best_env_idx = best_env_idxs_sorted[0]
    best_traj_idx = best_traj_idxs_sorted[0]

    return best_env_idx, best_traj_idx



def particles_for_demo_strategy(demo_strategy, team_knowledge, team_particles, team_size, weights, step_cost_flag, n_particles, min_BEC_constraints, teammate_idx=0):

    # particles = []
    # # prior knowledge
    # if demo_strategy != 'joint_knowledge':
    #     if demo_strategy =='individual_knowledge_low':
    #         ind_knowledge_ascending = find_ascending_individual_knowledge(team_knowledge, min_BEC_constraints)
    #         prior = team_knowledge[ind_knowledge_ascending[teammate_idx]].copy()
    #     elif demo_strategy == 'individual_knowledge_high':
    #         ind_knowledge_ascending = find_ascending_individual_knowledge(team_knowledge, min_BEC_constraints)
    #         prior = team_knowledge[ind_knowledge_ascending[len(ind_knowledge_ascending) - teammate_idx - 1]].copy()
    #     elif demo_strategy == 'common_knowledge':
    #         prior = calc_common_knowledge(team_knowledge, team_size, weights, step_cost_flag)
        
    #     # particles for human models
    #     particle_positions = BEC_helpers.sample_human_models_uniform([], n_particles)
    #     particles = pf_team.Particles_team(particle_positions)
    #     particles.update(prior)
    
    # elif demo_strategy == 'joint_knowledge':
    #     prior = calc_joint_knowledge(team_knowledge, team_size, weights, step_cost_flag)
    #     particle_positions = sample_human_models_uniform_joint_knowledge(prior, n_particles)
    #     particles = pf_team.Particles_team(particle_positions)


    # else:
    #     print('Unsupported demo strategy for sampling particles!')
    # return prior, particles

    ###########################################
    team_knowledge_level = calc_knowledge_level(team_knowledge, min_BEC_constraints)['members_list']
    # particles to consider while generating demos
    if demo_strategy =='individual_knowledge_low':
        knowledge_id = np.argmin(team_knowledge_level)
    
    elif demo_strategy == 'individual_knowledge_high':
        knowledge_id = np.argmax(team_knowledge_level)
    
    elif demo_strategy == 'common_knowledge':
        #particles = team_particles[demo_strategy]
        knowledge_id = demo_strategy
    elif demo_strategy == 'joint_knowledge':
        knowledge_id = 'members_list'
    
    else:
        print('Unsupported demo strategy for sampling particles!')

    particles = team_particles['members_list'][knowledge_id]

    
    return knowledge_id, particles




    
def sample_valid_region_jk(joint_constraints, min_azi, max_azi, min_ele, max_ele, n_azi, n_ele):
    # sample along the sphere
    u = np.linspace(min_azi, max_azi, n_azi, endpoint=True)
    v = np.linspace(min_ele, max_ele, n_ele, endpoint=True)

    x = np.outer(np.cos(u), np.sin(v)).reshape(1, -1)
    y = np.outer(np.sin(u), np.sin(v)).reshape(1, -1)
    z = np.outer(np.ones(np.size(u)), np.cos(v)).reshape(1, -1)
    sph_points = np.vstack((x, y, z))

    # see which points on the sphere obey atleast one of the original constraints
    dist_to_plane = joint_constraints.dot(sph_points)
    n_joint_constraints_satisfied = np.sum(dist_to_plane >= 0, axis=0)
    # n_min_constraints = constraints.shape[0]

    idx_valid_sph_points = np.where(n_joint_constraints_satisfied > 0)[0]
    valid_sph_x = np.take(x, idx_valid_sph_points)
    valid_sph_y = np.take(y, idx_valid_sph_points)
    valid_sph_z = np.take(z, idx_valid_sph_points)

    return valid_sph_x, valid_sph_y, valid_sph_z    
    

def sample_human_models_uniform_joint_knowledge(joint_constraints, n_models):
    '''
    Summary: sample representative weights that the human could currently attribute to the agent, by greedily selecting
    points that minimize the maximize distance to any other point (k-centers problem)
    '''

    sample_human_models = []

    if len(joint_constraints) > 0:
        joint_constraints_matrix = np.vstack(joint_constraints)

        # obtain x, y, z coordinates on the sphere that obey the constraints
        valid_sph_x, valid_sph_y, valid_sph_z = sample_valid_region_jk(joint_constraints_matrix, 0, 2 * np.pi, 0, np.pi, 1000, 1000)

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

            # reject points that fall outside of the desired area

            # see which points on the sphere obey atleast one of constraints. 
            # see which points on the sphere obey atleast one of the original constraints
            theta_grid, phi_grid = np.meshgrid(theta, phi)
            sph_points = np.array(cg.sph2cart(np.array([phi_grid.flatten(), theta_grid.flatten()]).T))
            dist_to_plane = joint_constraints_matrix.dot(sph_points.T)
            n_joint_constraints_satisfied = np.sum(dist_to_plane >= 0, axis=0)
            # n_min_constraints = constraints_matrix.shape[0]

            idx_valid_sph_points = np.where(n_joint_constraints_satisfied > 0)[0]
            valid_sph_points = sph_points[idx_valid_sph_points, :]

            # greedily select k 'centers' such that the maximum distance from any point to a center is minimized
            # solution is never worse than twice the optimal solution (2-approximation greedy algorithm)
            # https://www.geeksforgeeks.org/k-centers-problem-set-1-greedy-approximate-algorithm/
            if len(valid_sph_points) == n_models:
                sample_human_models.extend(valid_sph_points)
            else:
                valid_sph_points_latllong = cg.cart2latlong(valid_sph_points)
                pairwise = metrics.pairwise.haversine_distances(valid_sph_points_latllong)
                select_idxs = BEC_helpers.selectKcities(pairwise.shape[0], pairwise, n_models)
                select_sph_points = valid_sph_points[select_idxs]
                # reshape so that each element is a valid weight vector
                select_sph_points = select_sph_points.reshape(select_sph_points.shape[0], 1, select_sph_points.shape[1])
                sample_human_models.extend(select_sph_points)
    else:
        points = cg.generate_equidistributed_points_on_sphere(n_models)
        points = np.expand_dims(points, 1)
        sample_human_models.extend(points)

    return sample_human_models







    return prior, particles




def obtain_team_summary(data_loc, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count,
                        n_train_demos, particles_demo, variable_filter, nonzero_counter, BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs):


    summary_variant = 'counterfactual'
    # summary_variant = 'particle-filter'
    

    # # obtain demo summary        
    # # particle filters take a lot of time to converge. Avoid using this!
    # if summary_variant == 'particle-filter':
    #     BEC_summary, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = obtain_summary_particle_filter_team(data_loc, particles_demo, BEC_summary, variable_filter, nonzero_counter, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count,
    #                    min_BEC_constraints_running, n_train_demos=np.inf, downsample_threshold=float("inf"), consider_human_models_jointly=True, c=0.001, obj_func_proportion=1, min_info_gain=0.01, visited_env_traj_idxs = visited_env_traj_idxs)


    if summary_variant == 'counterfactual':
        BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = obtain_summary_counterfactual_team(data_loc, particles_demo, variable_filter, nonzero_counter, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count,
                       BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs = visited_env_traj_idxs, n_train_demos=np.inf, downsample_threshold=float("inf"), consider_human_models_jointly=True, c=0.001, obj_func_proportion=1)

        

    return BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo




def sample_team_pf(team_size, n_particles, weights, step_cost_flag, team_prior=None):

    particles_team = {}
    particles_team['members_list'] = []
    # particles for individual team members
    for i in range(team_size):
        particles_team['members_list'][i] = pf_team.Particles_team(BEC_helpers.sample_human_models_uniform([], n_particles))

        if team_prior is not None:
            particles_team['members_list'][i].update(team_prior['members_list'][i])

    
    # particles for aggregated team knowledge - common knowledge
    particles_team['common_knowledge'] = pf_team.Particles_team(BEC_helpers.sample_human_models_uniform([], n_particles))
    if team_prior is not None:
        team_prior['common_knowledge'] = calc_common_knowledge(team_prior, team_size, weights, step_cost_flag)
        particles_team['common_knowledge'].update(team_prior['common_knowledge'])
    

    # particles for aggregated team knowledge - joint knowledge (both methods should produce similar particles; check and if they are similar method 1 is more streamlined)
    # method 1
    '''particles_team['joint_knowledge'] = pf_team.Particles_team(BEC_helpers.sample_human_models_uniform([], n_particles))
    if team_prior is not None:
        particles_team['joint_knowledge'].update_jk(team_prior['members_list'])'''
    
    # # method 2
    # if team_prior is not None:
    #     team_prior['joint_knowledge'] = calc_joint_knowledge(team_prior, team_size, weights, step_cost_flag)
    #     particle_positions = sample_human_models_uniform_joint_knowledge(team_prior['joint_knowledge'], n_particles)
    #     particles_team['joint_knowledge_2'] = pf_team.Particles_team(particle_positions)



    return team_prior, particles_team



# def downsample_team_models_pf(particles_team, n_models):

    
#     sampled_team_models = {}
#     sampled_team_model_weights = {}

#     for member_id in particles_team:
#         sampled_human_model_idxs = []
#         particles_team[member_id].cluster()
#         if len(particles_team[member_id].cluster_centers) > n_models:
#             # if there are more clusters than number of sought human models, return the spherical centroids of the top n
#             # most frequently counted cluster indexes selected by systematic resampling
#             while len(sampled_human_model_idxs) < n_models:
#                 indexes = p_utils.systematic_resample(particles_team[member_id].cluster_weights)
#                 unique_idxs, counts = np.unique(indexes, return_counts=True)
#                 # order the unique indexes via their frequency
#                 unique_idxs_sorted = [x for _, x in sorted(zip(counts, unique_idxs), reverse=True)]

#                 for idx in unique_idxs_sorted:
#                     # add new unique indexes to the human models that will be considered for counterfactual reasoning
#                     if idx not in sampled_human_model_idxs:
#                         sampled_human_model_idxs.append(idx)

#                     if len(sampled_human_model_idxs) == n_models:
#                         break

#             sampled_human_models = [particles_team[member_id].cluster_centers[i] for i in sampled_human_model_idxs]
#             sampled_human_model_weights = np.array([particles_team[member_id].cluster_weights[i] for i in sampled_human_model_idxs])
#             sampled_human_model_weights /= np.sum(sampled_human_model_weights)  # normalize
#         elif len(particles_team[member_id].cluster_centers) == n_models:
#             sampled_human_models = particles_team[member_id].cluster_centers
#             sampled_human_model_weights = np.array(particles_team[member_id].cluster_weights) # should already be normalized
#         else:
#             # if there are fewer clusters than number of sought human models, use systematic sampling to determine how many
#             # particles from each cluster to return (using the k-cities algorithm to ensure that they are diverse)
#             indexes = p_utils.systematic_resample(particles_team[member_id].cluster_weights, N=n_models)
#             unique_idxs, counts = np.unique(indexes, return_counts=True)

#             sampled_human_models = []
#             sampled_human_model_weights = []
#             for j, unique_idx in enumerate(unique_idxs):
#                 # particles of this cluster
#                 clustered_particles = particles_team[member_id].positions[np.where(particles_team[member_id].cluster_assignments == unique_idx)]
#                 clustered_particles_weights = particles_team[member_id].weights[np.where(particles_team[member_id].cluster_assignments == unique_idx)]

#                 # use the k-cities algorithm to obtain a diverse sample of weights from this cluster
#                 pairwise = metrics.pairwise.euclidean_distances(clustered_particles.reshape(-1, 3))
#                 select_idxs = BEC_helpers.selectKcities(pairwise.shape[0], pairwise, counts[j])
#                 sampled_human_models.extend(clustered_particles[select_idxs])
#                 sampled_human_model_weights.extend(clustered_particles_weights[select_idxs])

#             sampled_human_model_weights = np.array(sampled_human_model_weights)
#             sampled_human_model_weights /= np.sum(sampled_human_model_weights)

#         # update member models
#         sampled_team_models[member_id] = sampled_human_models
#         sampled_team_model_weights[member_id] = sampled_human_model_weights
    

#     return sampled_team_models, sampled_team_model_weights



def show_demonstrations(unit, particles_demo, mdp_class, weights, visualize_pf_transition=False, unit_count=0):
    # TODO: WIP
    
    # print('Summary:', summary)
    
    print("Here are the demonstrations for this unit")
    unit_constraints = []
    running_variable_filter_unit = unit[0][4]

    # print('Unit:', unit)
    n = 1
    # show each demonstration that is part of this unit       
    for subunit in unit:
        subunit[0].visualize_trajectory(subunit[1])
        unit_constraints.extend(subunit[3])
        
        # debug
        # print('Constraint ', n, 'for this unit: ', subunit[3])
        

        # update particle filter with demonstration's constraint
        particles_demo.update(subunit[3])
        
        # visualize the updated particle filter
        if visualize_pf_transition:
            visualize_transition(subunit[3], particles_demo, mdp_class, weights, text='Demo ' + str(n) + 'for Summary/Unit ' + str(unit_count+1))

        n += 1

    return unit_constraints, running_variable_filter_unit



def visualize_spherical_polygon_jk(poly, fig=None, ax=None, alpha=1.0, color='y', plot_ref_sphere=True):
    '''
    Visualize the spherical polygon created by the intersection between the constraint polyhedron and a unit sphere
    '''
    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.add_subplot(projection='3d')

    vertices = np.array(poly.vertices())
    vertex_adj = poly.vertex_adjacency_matrix()
    vertex_adj_triu = np.triu(vertex_adj)

    # pull out vertices of the constraint polyhedron that lie on a constraint plane (as opposed to vertices that are
    # formed due to boundary constraints, where the latter simply clutter the visualization)
    spherical_polygon_vertices = {}
    for vertex_idx, vertex in enumerate(vertices):
        if (vertex != np.array([0, 0, 0])).any() and BEC_helpers.lies_on_constraint_plane(poly, vertex):
            vertex_normed = vertex / np.linalg.norm(vertex)
            spherical_polygon_vertices[vertex_idx] = vertex_normed

            # ax.scatter(vertex_normed[0], vertex_normed[1], vertex_normed[2], marker='o', c='g', s=50)

    t_vals = np.linspace(0, 1, 50)

    # plot the spherical BEC polygon
    for spherical_polygon_vertex_idx in spherical_polygon_vertices.keys():
        vertex_normed = spherical_polygon_vertices[spherical_polygon_vertex_idx]

        # plot the spherically interpolated lines between adjacent vertices that lie on a constraint plane
        adj_vertex_idxs = vertex_adj_triu[spherical_polygon_vertex_idx]
        for adj_vertex_idx, is_adj in enumerate(adj_vertex_idxs):
            # if this vertex is adjacent, not at the origin, and lies on a constraint plane ...
            if is_adj == 1 and (vertices[adj_vertex_idx] != np.array([0, 0, 0])).any() and (adj_vertex_idx in spherical_polygon_vertices.keys()):
                adj_vertex_normed = spherical_polygon_vertices[adj_vertex_idx]
                result = geometric_slerp(vertex_normed, adj_vertex_normed, t_vals)
                # ax.plot(result[:, 0], result[:, 1], result[:, 2], c='k')

    if plot_ref_sphere:
        # plot full unit sphere for reference
        u = np.linspace(0, 2 * np.pi, 30, endpoint=True)
        v = np.linspace(0, np.pi, 30, endpoint=True)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='y', alpha=0.2)

    # plot only the potion of the sphere corresponding to the valid solid angle
    hrep = np.array(poly.Hrepresentation())
    boundary_facet_idxs = np.where(hrep[:, 0] != 0)
    # first remove boundary constraint planes (those with a non-zero offset)
    min_constraints = np.delete(hrep, boundary_facet_idxs, axis=0)
    min_constraints = min_constraints[:, 1:]

    # obtain x, y, z coordinates on the sphere that obey the constraints
    valid_sph_x, valid_sph_y, valid_sph_z = sample_valid_region_jk(min_constraints, 0, 2 * np.pi, 0, np.pi, 1000, 1000)

    if len(valid_sph_x) == 0:
        print(colored("Was unable to sample valid points for visualizing the BEC (which is likely too small).",
                      'red'))
        return

    # resample coordinates on the sphere within the valid region (for higher density)
    sph_polygon = cg.cart2sph(np.array([valid_sph_x, valid_sph_y, valid_sph_z]).T)
    sph_polygon_ele = sph_polygon[:, 0]
    sph_polygon_azi = sph_polygon[:, 1]

    # obtain (the higher density of) x, y, z coordinates on the sphere that obey the constraints
    valid_sph_x, valid_sph_y, valid_sph_z = \
        sample_valid_region_jk(min_constraints, min(sph_polygon_azi), max(sph_polygon_azi), min(sph_polygon_ele), max(sph_polygon_ele), 50, 50)

    # create a triangulation mesh on which to interpolate using spherical coordinates
    sph_polygon = cg.cart2sph(np.array([valid_sph_x, valid_sph_y, valid_sph_z]).T)
    valid_ele = sph_polygon[:, 0]
    valid_azi = sph_polygon[:, 1]

    tri = mtri.Triangulation(valid_azi, valid_ele)

    # reject triangles that are too large (which often result from connecting non-neighboring vertices) w/ a corresp mask
    dev_x = np.ptp(valid_sph_x[tri.triangles], axis=1) > 5 * np.mean(np.ptp(valid_sph_x[tri.triangles], axis=1))
    dev_y = np.ptp(valid_sph_y[tri.triangles], axis=1) > 5 * np.mean(np.ptp(valid_sph_y[tri.triangles], axis=1))
    dev_z = np.ptp(valid_sph_z[tri.triangles], axis=1) > 5 * np.mean(np.ptp(valid_sph_z[tri.triangles], axis=1))
    first_or = np.logical_or(dev_x, dev_y)
    second_or = np.logical_or(first_or, dev_z)
    tri.set_mask(second_or)

    # plot valid x, y, z coordinates on sphere as a mesh of valid triangles
    ax.plot_trisurf(valid_sph_x, valid_sph_y, valid_sph_z, triangles=tri.triangles, mask=second_or, color=color, alpha=alpha)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)



def visualize_transition(constraints, particles, mdp_class, weights=None, fig=None, text=None, demo_strategy = 'common_knowledge'):

    # From BEC_viz.visualize_pf_transition function
   
    '''
    Visualize the change in particle filter due to constraints
    '''
    if fig == None:
        fig = plt.figure()
    def label_axes(ax, mdp_class, weights=None):
        ax.set_facecolor('white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        if weights is not None:
            ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='r', s=100/2)
        if mdp_class == 'augmented_taxi2':
            ax.set_xlabel('$\mathregular{w_0}$: Mud')
            ax.set_ylabel('$\mathregular{w_1}$: Recharge')
        elif mdp_class == 'colored_tiles':
            ax.set_xlabel('X: Tile A (brown)')
            ax.set_ylabel('Y: Tile B (green)')
        else:
            ax.set_xlabel('X: Goal')
            ax.set_ylabel('Y: Skateboard')
        ax.set_zlabel('$\mathregular{w_2}$: Action')

        ax.view_init(elev=16, azim=-160)

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d', sharex=ax1, sharey=ax1, sharez=ax1)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d', sharex=ax1, sharey=ax1, sharez=ax1)
    ax1.title.set_text('Particles before Demonstration')
    ax2.title.set_text('Constraint corresponding to demonstration')
    ax3.title.set_text('Particles after demonstration')

    fig.suptitle(text, fontsize=30)

    # plot particles before and after the constraints
    particles.plot(fig=fig, ax=ax1, plot_prev=True)
    particles.plot(fig=fig, ax=ax3)

    # plot the constraints
    print('Constraints.... :', constraints)
    ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints)
    poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    for constraints in [constraints]:
        BEC_viz.visualize_planes(constraints, fig=fig, ax=ax2)
    for constraints in [constraints]:
        BEC_viz.visualize_planes(constraints, fig=fig, ax=ax1)
    for constraints in [constraints]:
        BEC_viz.visualize_planes(constraints, fig=fig, ax=ax3)
    
    if demo_strategy == 'joint_knowledge':
        for ind_constraints in constraints:
            ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(ind_constraints)
            poly = Polyhedron.Polyhedron(ieqs=ieqs)
            BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax2, plot_ref_sphere=False, alpha=0.75)
    else:
        ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints)
        poly = Polyhedron.Polyhedron(ieqs=ieqs)
        BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax2, plot_ref_sphere=False, alpha=0.75)

    label_axes(ax1, mdp_class, weights)
    label_axes(ax2, mdp_class, weights)
    label_axes(ax3, mdp_class, weights)

    # New: Add what constraints are being shown in the demo to the plot
    x_loc = 0.5
    y_loc = 0.1
    fig.text(0.2, y_loc, 'Constraints in this demo: ', fontsize=20)
    for cnst in constraints:
        fig.text(x_loc, y_loc, str(cnst), fontsize=20)
        y_loc -= 0.05


    # https://stackoverflow.com/questions/41167196/using-matplotlib-3d-axes-how-to-drag-two-axes-at-once
    # link the pan of the three axes together
    def on_move(event):
        if event.inaxes == ax1:
            ax2.view_init(elev=ax1.elev, azim=ax1.azim)
            ax3.view_init(elev=ax1.elev, azim=ax1.azim)
        elif event.inaxes == ax2:
            ax1.view_init(elev=ax2.elev, azim=ax2.azim)
            ax3.view_init(elev=ax2.elev, azim=ax2.azim)
        elif event.inaxes == ax3:
            ax1.view_init(elev=ax3.elev, azim=ax3.azim)
            ax2.view_init(elev=ax3.elev, azim=ax3.azim)
            return
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    plt.show()



def visualize_team_knowledge(particles_team, mdp_class, fig=None, weights=None, text=None):


    if fig == None:
       fig = plt.figure()
    
    def label_axes(ax, mdp_class, weights=None):
        ax.set_facecolor('white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        if weights is not None:
            ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='r', s=100/2)
        if mdp_class == 'augmented_taxi2':
            ax.set_xlabel('$\mathregular{w_0}$: Mud')
            ax.set_ylabel('$\mathregular{w_1}$: Recharge')

    #may need to add a case to plot each individual member (if needed)
    n_subplots = len(particles_team)
    i = 1
    for knowledge_id, knowledge_type  in enumerate(particles_team):
        if knowledge_type == 'members_list':
            ax = fig.add_subplot(1, n_subplots, i, projection='3d')
            ax.title.set_text('Particles for knowledge: \n ' + "joint_knowledge")
            particles_team[knowledge_type].plot(fig=fig, ax=ax)
            label_axes(ax, mdp_class, weights)
            i += 1
        else: 
            ax = fig.add_subplot(1, n_subplots, i, projection='3d')
            ax.title.set_text('Particles for knowledge: \n ' + str(knowledge_type))
            particles_team[knowledge_type].plot(fig=fig, ax=ax)
            label_axes(ax, mdp_class, weights)
            i += 1

    fig.suptitle(text, fontsize=30)

    plt.show()






def visualize_BEC_area(constraints):

    
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    for constraints in [constraints]:
        BEC_viz.visualize_planes(constraints, fig=fig, ax=ax1)

    plt.show()



# def visualize_planes_team(constraints, fig=None, ax=None, alpha=0.5, color=None):
#     '''
#     Plot the planes associated with the normal vectors contained within 'constraints'
#     '''
#     if fig == None:
#         fig = plt.figure()
#     if ax == None:
#         ax = fig.add_subplot(projection='3d')

#     x = np.linspace(-1, 1, 10)
#     y = np.linspace(-1, 1, 10)
#     z = np.linspace(-1, 1, 10)

#     X_xy, Y_xy, = np.meshgrid(x, y)

#     for constraint in constraints:
#         if constraint[0, 2] != 0:
#             Z = (-constraint[0, 0] * X_xy - constraint[0, 1] * Y_xy) / constraint[0, 2]
#             if color is not None:
#                 ax.plot_surface(X_xy, Y_xy, Z, alpha=alpha, color=color)
#             else:
#                 ax.plot_surface(X_xy, Y_xy, Z, alpha=alpha)
#         elif constraint[0, 1] != 0:
#             X_xz, Z_xz, = np.meshgrid(x, z)
#             Y = (-constraint[0, 0] * X_xz - constraint[0, 2] * Z_xz) / constraint[0, 1]
#             if color is not None:
#                 ax.plot_surface(X_xz, Y, Z_xz, alpha=alpha, color=color)
#             else:
#                 ax.plot_surface(X_xz, Y, Z_xz, alpha=alpha)
#         else:
#             Y_yz, Z_yz, = np.meshgrid(y, z)
#             X = (-constraint[0, 1] * Y_yz - constraint[0, 2] * Z_yz) / constraint[0, 0]
#             if color is not None:
#                 ax.plot_surface(X, Y_yz, Z_yz, alpha=alpha, color=color)
#             else:
#                 ax.plot_surface(X, Y_yz, Z_yz, alpha=alpha)



def obtain_summary_counterfactual_team(data_loc, particles_demo, variable_filter, nonzero_counter, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count,
                       summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs=[], n_train_demos=3, prior=[], downsample_threshold=float("inf"), consider_human_models_jointly=True, c=0.001, obj_func_proportion=1):


    # TODO: WIP
    
    unit = []

    # impose prior
    min_BEC_constraints_running = prior.copy()

    # count how many nonzero constraints are present for each reward weight (i.e. variable) in the minimum BEC constraints
    # (which are obtained using one-step deviations). mask variables in order of fewest nonzero constraints for variable scaffolding
    # rationale: the variable with the most nonzero constraints, often the step cost, serves as a good reference/ratio variable
    min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record for item in sublist]
    min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record_flattened for item in sublist]
    min_subset_constraints_record_array = np.array(min_subset_constraints_record_flattened)

    # for variable scaffolding
    nonzero_counter = (min_subset_constraints_record_array != 0).astype(float)
    nonzero_counter = np.sum(nonzero_counter, axis=0)
    nonzero_counter = nonzero_counter.flatten()

    variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
    running_variable_filter = variable_filter.copy()
    print('variable filter: {}'.format(variable_filter))

    # clear the demonstration generation log
    open('models/' + data_loc + '/demo_gen_log.txt', 'w').close()

    while summary_count < n_train_demos:
        
        # visualize_constraints(min_BEC_constraints_running, weights, step_cost_flag, fig_name=str(summary_count) + '.png', just_save=True)

        # (approximately) uniformly divide up the valid BEC area along 2-sphere
        sample_human_models = BEC_helpers.sample_human_models_uniform(min_BEC_constraints_running, n_human_models)

        if len(sample_human_models) == 0:
            print(colored("Likely cannot reduce the BEC further through additional demonstrations. Returning.", 'red'))
            return summary, summary_count, visited_env_traj_idxs, min_BEC_constraints_running, particles_demo

        info_gains_record = []
        overlap_in_opt_and_counterfactual_traj_record = []

        print("Length of summary: {}".format(summary_count))
        with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
            myfile.write('Length of summary: {}\n'.format(summary_count))

        for model_idx, human_model in enumerate(sample_human_models):
            print(colored('Model #: {}'.format(model_idx), 'red'))
            print(colored('Model val: {}'.format(human_model), 'red'))

            with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
                myfile.write('Model #: {}\n'.format(model_idx))
                myfile.write('Model val: {}\n'.format(human_model))

            # based on the human's current model, obtain the information gain generated when comparing to the agent's
            # optimal trajectories in each environment (human's corresponding optimal trajectories and constraints
            # are saved for reference later)
            print("Obtaining counterfactual information gains:")

            cf_data_dir = 'models/' + data_loc + '/counterfactual_data_' + str(summary_count) + '/model' + str(model_idx)
            os.makedirs(cf_data_dir, exist_ok=True)
            if consider_human_models_jointly:
                args = [(data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]), traj_record[i], None, min_BEC_constraints_running, step_cost_flag, summary_count, variable_filter, mdp_features_record[i], consider_human_models_jointly) for i in range(len(traj_record))]
                info_gain_envs = list(tqdm(pool.imap(BEC.compute_counterfactuals, args), total=len(args)))

                info_gains_record.append(info_gain_envs)
            else:
                args = [(data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]), traj_record[i], None, min_BEC_constraints_running, step_cost_flag, summary_count, variable_filter, mdp_features_record[i], consider_human_models_jointly) for i in range(len(traj_record))]
                info_gain_envs, overlap_in_opt_and_counterfactual_traj_env = zip(*pool.imap(BEC.compute_counterfactuals, tqdm(args), total=len(args)))

                info_gains_record.append(info_gain_envs)
                overlap_in_opt_and_counterfactual_traj_record.append(overlap_in_opt_and_counterfactual_traj_env)

        with open('models/' + data_loc + '/info_gains_' + str(summary_count) + '.pickle', 'wb') as f:
            pickle.dump(info_gains_record, f)


        # do a quick check of whether there's any information to be gained from any of the trajectories
        no_info_flag = False
        max_info_gain = 1
        if consistent_state_count:
            info_gains = np.array(info_gains_record)
            if np.sum(info_gains > 1) == 0:
                no_info_flag = True
        else:
            info_gains_flattened_across_models = list(itertools.chain.from_iterable(info_gains_record))
            info_gains_flattened_across_envs = list(itertools.chain.from_iterable(info_gains_flattened_across_models))
            if sum(np.array(info_gains_flattened_across_envs) > 1) == 0:
                no_info_flag = True

        # no need to continue search for demonstrations if none of them will improve the human's understanding
        # if no_info_flag:
        #     # if no variables had been filtered out, then there are no more informative demonstrations to be found
        #     if not np.any(variable_filter):
        #         break
        #     else:
        #         # no more informative demonstrations with this variable filter, so update it
        #         variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
        #         print(colored('Did not find any informative demonstrations.', 'red'))
        #         print('variable filter: {}'.format(variable_filter))

        #         continue
        
        if no_info_flag:
            print(colored('Did not find any informative demonstrations. Moving onto next unit, if applicable.', 'red'))
            summary.append(unit)
            return summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo

        if consider_human_models_jointly:
            print("Combining the most limiting constraints across human models:")
            args = [(i, range(len(sample_human_models)), data_loc, summary_count, weights, step_cost_flag, variable_filter, mdp_features_record[i],
                     traj_record[i], min_BEC_constraints_running, None, True, True) for
                    i in range(len(traj_record))]
            info_gains_record, min_env_constraints_record, n_diff_constraints_record, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = zip(
                *pool.imap(BEC.combine_limiting_constraints_IG, tqdm(args)))

            with open('models/' + data_loc + '/info_gains_joint' + str(summary_count) + '.pickle', 'wb') as f:
                pickle.dump(info_gains_record, f)

            differing_constraint_count = 1          # number of constraints in the running human model that would differ after showing a particular demonstration
            max_differing_constraint_count = max(list(itertools.chain(*n_diff_constraints_record)))
            print("max_differing_constraint_count: {}".format(max_differing_constraint_count))
            no_info_flag = True

            # try to find a demonstration that will yield the fewest changes in the constraints defining the running human model while maximizing the information gain
            while no_info_flag and differing_constraint_count <= max_differing_constraint_count:
                # the possibility that no demonstration provides information gain must be checked for again,
                # in case all limiting constraints involve a masked variable and shouldn't be considered for demonstration yet
                if consistent_state_count:
                    info_gains = np.array(info_gains_record)
                    n_diff_constraints = np.array(n_diff_constraints_record)
                    traj_overlap_pcts = np.array(overlap_in_opt_and_counterfactual_traj_avg)

                    # obj_function = info_gains * (traj_overlap_pcts + c)  # objective 2: scaled
                    obj_function = info_gains

                    # not considering demos where there is no info gain helps ensure that the final demonstration
                    # provides the maximum info gain (in conjuction with previously shown demonstrations)
                    obj_function[info_gains == 1] = 0
                    obj_function[n_diff_constraints != differing_constraint_count] = 0

                    max_info_gain = np.max(info_gains)
                    if max_info_gain == 1:
                        no_info_flag = True
                        differing_constraint_count += 1
                    else:
                        # if visuals aren't considered, then you can simply return one of the demos that maximizes the obj function
                        # best_env_idx, best_traj_idx = np.unravel_index(np.argmax(obj_function), info_gains.shape)

                        if obj_func_proportion == 1:
                            # a) select the trajectory with the maximal information gain
                            best_env_idxs, best_traj_idxs = np.where(obj_function == max(obj_function.flatten()))
                        else:
                            # b) select the trajectory closest to the desired partial information gain (to obtain more demonstrations0
                            obj_function_flat = obj_function.flatten()
                            obj_function_flat.sort()

                            best_obj = obj_function_flat[-1]
                            target_obj = obj_func_proportion * best_obj
                            target_idx = np.argmin(abs(obj_function_flat - target_obj))
                            closest_obj = obj_function_flat[target_idx]
                            best_env_idxs, best_traj_idxs = np.where(obj_function == obj_function_flat[closest_obj])

                        # filter best_env_idxs and best_traj_idxs to only include those that haven't been visited yet
                        candidate_envs_trajs = list(zip(best_env_idxs, best_traj_idxs))
                        filtered_candidate_envs_trajs = [env_traj for env_traj in candidate_envs_trajs if env_traj not in visited_env_traj_idxs]
                        if len(filtered_candidate_envs_trajs) > 0:
                            best_env_idxs, best_traj_idxs = zip(*filtered_candidate_envs_trajs)
                            if (running_variable_filter == variable_filter).all():
                                # we're still in the same unit so try and optimize visuals wrt other demonstrations in this unit
                                best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, traj_record, unit)
                            else:
                                # we've moved on to a different unit so no need to consider previous demonstrations when optimizing visuals
                                best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, traj_record, [])
                            no_info_flag = False
                        else:
                            print("Ran into a conflict with a previously shown demonstration")
                            no_info_flag = True
                            differing_constraint_count += 1
                else:
                    best_obj = float('-inf')
                    best_env_idxs = []
                    best_traj_idxs = []

                    if obj_func_proportion == 1:
                        # a) select the trajectory with the maximal information gain
                        for env_idx, info_gains_per_env in enumerate(info_gains_record):
                            for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
                                if info_gain_per_traj > 1 and n_diff_constraints_record[env_idx][traj_idx] == differing_constraint_count:

                                    # obj = info_gain_per_traj * (
                                    #             overlap_in_opt_and_counterfactual_traj_avg[env_idx][traj_idx] + c)  # objective 2: scaled
                                    obj = info_gain_per_traj

                                    if np.isclose(obj, best_obj):
                                        best_env_idxs.append(env_idx)
                                        best_traj_idxs.append(traj_idx)
                                    elif obj > best_obj:
                                        best_obj = obj

                                        best_env_idxs = [env_idx]
                                        best_traj_idxs = [traj_idx]
                                    if info_gain_per_traj > max_info_gain:
                                        max_info_gain = info_gain_per_traj
                                        print("new max info: {}".format(max_info_gain))
                    else:
                        # b) select the trajectory closest to the desired partial information gain (to obtain more demonstrations)
                        # first find the max information gain
                        for env_idx, info_gains_per_env in enumerate(info_gains_record):
                            for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
                                if info_gain_per_traj > 1 and n_diff_constraints_record[env_idx][traj_idx] == differing_constraint_count:
                                    obj = info_gain_per_traj

                                    if np.isclose(obj, best_obj):
                                        pass
                                    elif obj > best_obj:
                                        best_obj = obj

                                    if info_gain_per_traj > max_info_gain:
                                        max_info_gain = info_gain_per_traj
                                        print("new max info: {}".format(max_info_gain))

                        target_obj = obj_func_proportion * best_obj
                        closest_obj_dist = float('inf')

                        for env_idx, info_gains_per_env in enumerate(info_gains_record):
                            for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
                                if info_gain_per_traj > 1 and n_diff_constraints_record[env_idx][traj_idx] == differing_constraint_count:

                                    obj = info_gain_per_traj

                                    if np.isclose(abs(target_obj - obj), closest_obj_dist):
                                        best_env_idxs.append(env_idx)
                                        best_traj_idxs.append(traj_idx)
                                    elif abs(target_obj - obj) < closest_obj_dist:
                                        closest_obj_dist = abs(obj - target_obj)

                                        best_env_idxs = [env_idx]
                                        best_traj_idxs = [traj_idx]

                    if max_info_gain == 1:
                        no_info_flag = True
                        differing_constraint_count += 1
                    else:
                        # filter best_env_idxs and best_traj_idxs to only include those that haven't been visited yet
                        candidate_envs_trajs = list(zip(best_env_idxs, best_traj_idxs))
                        filtered_candidate_envs_trajs = [env_traj for env_traj in candidate_envs_trajs if env_traj not in visited_env_traj_idxs]
                        if len(filtered_candidate_envs_trajs) > 0:
                            best_env_idxs, best_traj_idxs = zip(*filtered_candidate_envs_trajs)
                            if (running_variable_filter == variable_filter).all():
                                # we're still in the same unit so try and optimize visuals wrt other demonstrations in this unit
                                best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, traj_record, unit)
                            else:
                                # we've moved on to a different unit so no need to consider previous demonstrations when optimizing visuals
                                best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, traj_record, [])
                            no_info_flag = False
                        else:
                            print("Ran into a conflict with a previously shown demonstration")
                            no_info_flag = True
                            differing_constraint_count += 1

            with open('models/' + data_loc + '/best_env_idxs' + str(summary_count) + '.pickle', 'wb') as f:
                pickle.dump((best_env_idx, best_traj_idx, best_env_idxs, best_traj_idxs), f)

            print("current max info: {}".format(max_info_gain))
            # no need to continue search for demonstrations if none of them will improve the human's understanding
            # if no_info_flag:
            #     # if no variables had been filtered out, then there are no more informative demonstrations to be found
            #     if not np.any(variable_filter):
            #         break
            #     else:
            #         # no more informative demonstrations with this variable filter, so update it
            #         variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
            #         print(colored('Did not find any informative demonstrations.', 'red'))
            #         print('variable filter: {}'.format(variable_filter))
            #         continue

            if no_info_flag:
                print(colored('Did not find any informative demonstrations. Moving onto next unit, if applicable.', 'red'))
                summary.append(unit)
                return summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo
            

            best_traj = traj_record[best_env_idx][best_traj_idx]

            filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
            with open(filename, 'rb') as f:
                wt_vi_traj_env = pickle.load(f)
            best_mdp = wt_vi_traj_env[0][1].mdp
            best_mdp.set_init_state(best_traj[0][0]) # for completeness
            min_BEC_constraints_running.extend(min_env_constraints_record[best_env_idx][best_traj_idx])
            min_BEC_constraints_running = BEC_helpers.remove_redundant_constraints(min_BEC_constraints_running, weights, step_cost_flag)
            
            visited_env_traj_idxs.append((best_env_idx, best_traj_idx))
            
            # This is just an additionla check. Variable filter should not be changing within this function
            if (running_variable_filter == variable_filter).all():
                unit.append([best_mdp, best_traj, (best_env_idx, best_traj_idx), min_env_constraints_record[best_env_idx][best_traj_idx], variable_filter, sample_human_models])
                summary_count += 1
            else:
                
                print(colored('This should not happen! Variable filter is getting updated within obtain_counterfactual_demo', 'red'))
                
                # summary.append(unit)

                # unit = [[best_mdp, best_traj, (best_env_idx, best_traj_idx), min_env_constraints_record[best_env_idx][best_traj_idx], variable_filter, sample_human_models]]
                # running_variable_filter = variable_filter.copy()
                # summary_count += 1
            
            
        # else:
        #     # todo: this section of code is no longer supported (e.g. doesn't try to provide demonstrations that convey
        #     #  one new constraint at a time if possible) and should be deprecated
            
        #     print(colored('Running into a supposedly unsupported section', 'red'))
            
        #     # b) consider each human model separately
        #     if consistent_state_count:
        #         info_gains = np.array(info_gains_record)
        #         traj_overlap_pcts = np.array(overlap_in_opt_and_counterfactual_traj_record)

        #         # obj_function = info_gains * (traj_overlap_pcts + c)                  # objective 2: scaled
        #         obj_function = info_gains

        #         obj_function[info_gains == 1] = 0

        #         select_model, best_env_idx, best_traj_idx = np.unravel_index(np.argmax(obj_function), info_gains.shape)

        #         if max(info_gains.flatten()) == 1:
        #             no_info_flag = True
        #         else:
        #             best_env_idx, best_traj_idx = np.unravel_index(np.argmax(obj_function), info_gains.shape)
        #             best_env_idxs, best_traj_idxs = np.where(obj_function == max(obj_function.flatten()))
        #             max_info_gain = np.max(info_gains)

        #             best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, traj_record, summary)
        #     else:
        #         best_obj = float('-inf')
        #         select_model = best_env_idx = best_traj_idx = 0
        #         for model_idx, info_gains_per_model in enumerate(info_gains_record):
        #             for env_idx, info_gains_per_env in enumerate(info_gains_per_model):
        #                 for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
        #                     if info_gain_per_traj > 1:
        #                         # obj = info_gain_per_traj * (overlap_in_opt_and_counterfactual_traj_record[model_idx][env_idx][traj_idx] + c)
        #                         obj = info_gain_per_traj

        #                         if np.isclose(obj, best_obj):
        #                             best_env_idxs.append(env_idx)
        #                             best_traj_idxs.append(traj_idx)
        #                         elif obj > best_obj:
        #                             best_obj = obj
        #                             max_info_gain = info_gain_per_traj

        #                             best_env_idxs = [env_idx]
        #                             best_traj_idxs = [traj_idx]
        #                             select_model = model_idx

        #                         if info_gain_per_traj > max_info_gain:
        #                             max_info_gain = info_gain_per_traj

        #         if max_info_gain == 1:
        #             no_info_flag = True
        #         else:
        #             best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, traj_record, summary)

        #     if no_info_flag:
        #         if variable_filter is None:
        #             break
        #         else:
        #             # no more informative demonstrations with this variable filter, so update it
        #             variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
        #             print(colored('Did not find any informative demonstrations.', 'red'))
        #             print('variable filter: {}'.format(variable_filter))
        #             continue

        #     with open('models/' + data_loc + '/counterfactual_data_' + str(summary_count) + '/model' + str(
        #             select_model) + '/cf_data_env' + str(
        #             best_env_idx).zfill(5) + '.pickle', 'rb') as f:
        #         constraints_env = pickle.load(f)

        #     # best_human_trajs = best_human_trajs_record_env[best_traj_idx]
        #     best_traj = traj_record[best_env_idx][best_traj_idx]

        #     filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
        #     with open(filename, 'rb') as f:
        #         wt_vi_traj_env = pickle.load(f)
        #     best_mdp = wt_vi_traj_env[0][1].mdp
        #     best_mdp.set_init_state(best_traj[0][0]) # for completeness
        #     min_BEC_constraints_running.extend(constraints_env[best_traj_idx])

        #     particles_demo.update(constraints_env[best_traj_idx])

        #     min_BEC_constraints_running = BEC_helpers.remove_redundant_constraints(min_BEC_constraints_running, weights, step_cost_flag)
        #     if (running_variable_filter == variable_filter).all():
        #         unit.append([best_mdp, best_traj, (best_env_idx, best_traj_idx), constraints_env[best_traj_idx], variable_filter,
        #                     sample_human_models, select_model])
        #         summary_count += 1
        #     else:
        #         summary.append(unit)

        #         unit = [[best_mdp, best_traj, (best_env_idx, best_traj_idx), constraints_env[best_traj_idx], variable_filter,
        #                     sample_human_models, select_model]]
        #         running_variable_filter = variable_filter.copy()
        #         summary_count += 1

        #     visited_env_traj_idxs.append((best_env_idx, best_traj_idx))

        print(colored('Max infogain: {}'.format(max_info_gain), 'blue'))
        with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
            myfile.write('Max infogain: {}\n'.format(max_info_gain))
            myfile.write('\n')

        # save the summary along the way (for each completed unit)
        with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
            pickle.dump((summary, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo), f)

    # add any remaining demonstrations
    summary.append(unit)

    # this method doesn't always finish, so save the summary along the way (for each completed unit)
    with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
        pickle.dump((summary, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo), f)

    return summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo



def obtain_remedial_demos_tests(data_loc, previous_demos, visited_env_traj_idxs, min_BEC_constraints, min_subset_constraints_record, traj_record, traj_features_record, variable_filter, mdp_features_record, downsample_threshold=float("inf"), opt_simplicity=True, opt_similarity=True, type = 'training'):

    preliminary_test_info = []

    # if you're looking for demonstrations that will convey the most constraining BEC region or will be employing scaffolding,
    # obtain the demos needed to convey the most constraining BEC region
    BEC_constraints = min_BEC_constraints.copy()
    print('Remedial constraints:', BEC_constraints)
    BEC_constraint_bookkeeping = BEC_helpers.perform_BEC_constraint_bookkeeping(BEC_constraints,
                                                                                min_subset_constraints_record, visited_env_traj_idxs, traj_record, traj_features_record, mdp_features_record, variable_filter=variable_filter)
    
    
    
    if len(BEC_constraint_bookkeeping[0]) <= 0:
        nn_BEC_constraint_bookkeeping, minimal_distances = BEC_helpers.perform_nn_BEC_constraint_bookkeeping(BEC_constraints,
                                                                                            min_subset_constraints_record, visited_env_traj_idxs, traj_record, traj_features_record, mdp_features_record, variable_filter=variable_filter)
    
        BEC_bookkeeping = nn_BEC_constraint_bookkeeping.copy()
        print('BEC_constraint_bookkeeping length:', len(nn_BEC_constraint_bookkeeping))
        print('BEC_constraint_bookkeeping :', nn_BEC_constraint_bookkeeping)

    else:
        BEC_bookkeeping = BEC_constraint_bookkeeping.copy()
    
    while len(BEC_constraints) > 0:
        # downsampling strategy 1: randomly cull sets with too many members for computational feasibility
        # for j, set in enumerate(sets):
        #     if len(set) > downsample_threshold:
        #         sets[j] = random.sample(set, downsample_threshold)

        # downsampling strategy 2: if there are any env_traj pairs that cover more than one constraint, use it and remove all
        # env_traj pairs that would've conveyed the same constraints
        # initialize all env_traj tuples with covering the first min BEC constraint
        env_constraint_mapping = {}
        for key in BEC_bookkeeping[0]:
            env_constraint_mapping[key] = [0]
        max_constraint_count = 1  # what is the max number of desired constraints that one env / demo can convey
        max_env_traj_tuples = [key]

        # for all other env_traj tuples,
        for constraint_idx, env_traj_tuples in enumerate(BEC_bookkeeping[1:]):
            for env_traj_tuple in env_traj_tuples:
                # if this env_traj tuple has already been seen previously
                if env_traj_tuple in env_constraint_mapping.keys():
                    env_constraint_mapping[env_traj_tuple].append(constraint_idx + 1)
                    # and adding another constraint to this tuple increases the highest constraint coverage by a single tuple
                    if len(env_constraint_mapping[env_traj_tuple]) > max_constraint_count:
                        # update the max values, replacing the max tuple
                        max_constraint_count = len(env_constraint_mapping[env_traj_tuple])
                        max_env_traj_tuples = [env_traj_tuple]
                    # otherwise, if it simply equals the highest constraint coverage, add this tuple to the contending list
                    elif len(env_constraint_mapping[env_traj_tuple]) == max_constraint_count:
                        max_env_traj_tuples.append(env_traj_tuple)
                else:
                    env_constraint_mapping[env_traj_tuple] = [constraint_idx + 1]

        if max_constraint_count == 1:
            # no one demo covers multiple constraints. so greedily select demos from base list that is mot visually complex
            # filter for the most visually complex environment
            for env_traj_tuples in BEC_bookkeeping[::-1]:
                best_env_idxs, best_traj_idxs = zip(*env_traj_tuples)
                best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs,
                                                                          traj_record, previous_demos, type='training')
                max_complexity_env_traj_tuple = (best_env_idx, best_traj_idx)
                preliminary_test_info.append((max_complexity_env_traj_tuple, [BEC_constraints.pop()]))

        else:
            # filter for the most visually complex environment that can cover multiple constraints
            best_env_idxs, best_traj_idxs = zip(*max_env_traj_tuples)
            best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs,
                                                                      traj_record, previous_demos, type='training')
            max_complexity_env_traj_tuple = (best_env_idx, best_traj_idx)

            constituent_constraints = []
            for idx in env_constraint_mapping[max_complexity_env_traj_tuple]:
                constituent_constraints.append(BEC_constraints[idx])

            preliminary_test_info.append((max_complexity_env_traj_tuple, constituent_constraints))

            for constraint_idx in sorted(env_constraint_mapping[max_complexity_env_traj_tuple], reverse=True):
                del BEC_constraints[constraint_idx]

    preliminary_tests = []
    for info in preliminary_test_info:
        env_idx, traj_idx = info[0]
        traj = traj_record[env_idx][traj_idx]
        constraints = info[1]
        filename = mp_helpers.lookup_env_filename(data_loc, env_idx)
        with open(filename, 'rb') as f:
            wt_vi_traj_env = pickle.load(f)
        best_mdp = wt_vi_traj_env[0][1].mdp
        best_mdp.set_init_state(traj[0][0])  # for completeness
        preliminary_tests.append([best_mdp, traj, (env_idx, traj_idx), constraints, variable_filter])
        visited_env_traj_idxs.append((env_idx, traj_idx))

    return preliminary_tests, visited_env_traj_idxs



def check_unit_learning_goal_reached(team_knowledge, min_unit_constraints):

    
    check_unit_learning_goal_reached = False
    
    knowledge_level = calc_knowledge_level(team_knowledge, min_unit_constraints)
    if knowledge_level['common_knowledge'] > 0.4:
        check_unit_learning_goal_reached = True
    

    return check_unit_learning_goal_reached



# def obtain_summary_particle_filter_team(data_loc, particles, summary, variable_filter, nonzero_counter, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count,
#                        min_BEC_constraints_running, n_train_demos=np.inf, downsample_threshold=float("inf"), consider_human_models_jointly=True, c=0.001, obj_func_proportion=1, min_info_gain=0.01, visited_env_traj_idxs=[]):


#     # count how many nonzero constraints are present for each reward weight (i.e. variable) in the minimum BEC constraints
#     # (which are obtained using one-step deviations). mask variables in order of fewest nonzero constraints for variable scaffolding
#     # rationale: the variable with the most nonzero constraints, often the step cost, serves as a good reference/ratio variable
#     min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record for item in sublist]
#     min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record_flattened for item in sublist]
#     min_subset_constraints_record_array = np.array(min_subset_constraints_record_flattened)

#     print('variable filter: {}'.format(variable_filter))

#     # running_variable_filter = variable_filter.copy()

#     # clear the demonstration generation log
#     open('models/' + data_loc + '/demo_gen_log.txt', 'w').close()

#     while len(summary) < n_train_demos:
#         # visualize_constraints(min_BEC_constraints_running, weights, step_cost_flag, fig_name=str(len(summary)) + '.png', just_save=True)
#         sample_human_models = BEC_helpers.sample_human_models_pf(particles, n_human_models)

#         if len(sample_human_models) == 0:
#             print(colored("Likely cannot reduce the BEC further through additional demonstrations. Returning.", 'red'))
#             return summary

#         info_gains_record = []
#         overlap_in_opt_and_counterfactual_traj_record = []

#         print("Length of summary: {}".format(len(summary)))
#         with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
#             myfile.write('Length of summary: {}\n'.format(len(summary)))
            

#         for model_idx, human_model in enumerate(sample_human_models[0]):
#             print(colored('Model #: {}'.format(model_idx), 'red'))
#             print(colored('Model val: {}'.format(human_model), 'red'))

#             with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
#                 myfile.write('Model #: {}\n'.format(model_idx))
#                 myfile.write('Model val: {}\n'.format(human_model))
                

#             # based on the human's current model, obtain the information gain generated when comparing to the agent's
#             # optimal trajectories in each environment (human's corresponding optimal trajectories and constraints
#             # are saved for reference later)
#             print("Obtaining counterfactual information gains:")

#             cf_data_dir = 'models/' + data_loc + '/counterfactual_data_' + str(len(summary)) + '/model' + str(model_idx)
            
#             # cf_data_dir = 'models/' + data_loc + '/counterfactual_data_' + str(summary_id) + '/model' + str(model_idx)
#             os.makedirs(cf_data_dir, exist_ok=True)
#             if consider_human_models_jointly:
#                 args = [(data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]), traj_record[i], particles, min_BEC_constraints_running, step_cost_flag, len(summary), variable_filter, mdp_features_record[i], consider_human_models_jointly) for i in range(len(traj_record))]

#                 info_gain_envs = list(tqdm(pool.imap(compute_counterfactuals_team, args), total=len(args)))

#                 info_gains_record.append(info_gain_envs)
#             else:
#                 args = [(data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]), traj_record[i], particles, min_BEC_constraints_running, step_cost_flag, len(summary), variable_filter, mdp_features_record[i], consider_human_models_jointly) for i in range(len(traj_record))]
#                 info_gain_envs, overlap_in_opt_and_counterfactual_traj_env = zip(*pool.imap(compute_counterfactuals_team, tqdm(args), total=len(args)))

#                 info_gains_record.append(info_gain_envs)
#                 overlap_in_opt_and_counterfactual_traj_record.append(overlap_in_opt_and_counterfactual_traj_env)


#         with open('models/' + data_loc + '/info_gains_' + str(len(summary)) + '.pickle', 'wb') as f:
#             pickle.dump(info_gains_record, f)

#         # do a quick check of whether there's any information to be gained from any of the trajectories
#         no_info_flag = False
#         max_info_gain = -np.inf
#         if consistent_state_count:
#             info_gains = np.array(info_gains_record)
#             if np.sum(info_gains > 0) == 0:
#                 no_info_flag = True
#         else:
#             info_gains_flattened_across_models = list(itertools.chain.from_iterable(info_gains_record))
#             info_gains_flattened_across_envs = list(itertools.chain.from_iterable(info_gains_flattened_across_models))
#             if sum(np.array(info_gains_flattened_across_envs) > 0) == 0:
#                 no_info_flag = True

#         # no need to continue search for demonstrations if none of them will improve the human's understanding
#         if no_info_flag:
#             print('No information gained...')
#             break


#         # todo: now that we're utilizing a probabilistic human model, we should account for the probability of having
#         #  selected each human model in the information gain calculation (e.g. by taking an expectation over the information
#         #  gain), rather than simply combine all of the constraints that could've been generated (which is currently done)

#         # todo: the code below isn't updated to provide demonstrations that only differ by a single constraint at a time,
#         #  nor does it ensure that the selected demonstrations don't conflict with prior selected demonstrations that are
#         #  specified through visited_env_traj_idxs (e.g. ones that will be used for assessment tests after teaching).
#         #  see obtain_summary_counterfactual() for a more updated version
#         print("Combining the most limiting constraints across human models:")
#         args = [(i, range(len(sample_human_models)), data_loc, len(summary), weights, step_cost_flag, variable_filter, mdp_features_record[i],
#                  traj_record[i], min_BEC_constraints_running, particles, True, False) for
#                 i in range(len(traj_record))]
#         info_gains_record, min_env_constraints_record, n_diff_constraints_record, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = zip(
#             *pool.imap(BEC.combine_limiting_constraints_IG, tqdm(args)))

#         # the possibility that no demonstration provides information gain must be checked for again,
#         # in case all limiting constraints involve a masked variable and shouldn't be considered for demonstration yet
#         if consistent_state_count:
#             info_gains = np.array(info_gains_record)
#             traj_overlap_pcts = np.array(overlap_in_opt_and_counterfactual_traj_avg)

#             # obj_function = info_gains * (traj_overlap_pcts + c)  # objective 2: scaled
#             obj_function = info_gains

#             # not considering demos where there is no info gain helps ensure that the final demonstration
#             # provides the maximum info gain (in conjuction with previously shown demonstrations)
#             obj_function[info_gains <= 0] = 0

#             max_info_gain = np.max(info_gains)

#             if max_info_gain <= min_info_gain:
#                 no_info_flag = True
#             else:
#                 # if visuals aren't considered, then you can simply return one of the demos that maximizes the obj function
#                 # best_env_idx, best_traj_idx = np.unravel_index(np.argmax(obj_function), info_gains.shape)

#                 if obj_func_proportion == 1:
#                     # a) select the trajectory with the maximal information gain
#                     best_env_idxs, best_traj_idxs = np.where(obj_function == max(obj_function.flatten()))
#                 else:
#                     # b) select the trajectory closest to the desired partial information gain (to obtain more demonstrations0
#                     obj_function_flat = obj_function.flatten()
#                     obj_function_flat.sort()

#                     best_obj = obj_function_flat[-1]
#                     target_obj = obj_func_proportion * best_obj
#                     target_idx = np.argmin(abs(obj_function_flat - target_obj))
#                     closest_obj = obj_function_flat[target_idx]
#                     best_env_idxs, best_traj_idxs = np.where(obj_function == obj_function_flat[closest_obj])

#                 best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, traj_record, summary)
#         else:
#             best_obj = float('-inf')
#             best_env_idxs = []
#             best_traj_idxs = []

#             if obj_func_proportion == 1:
#                 # a) select the trajectory with the maximal information gain
#                 for env_idx, info_gains_per_env in enumerate(info_gains_record):
#                     for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
#                         if info_gain_per_traj > 0:

#                             # obj = info_gain_per_traj * (
#                             #             overlap_in_opt_and_counterfactual_traj_avg[env_idx][traj_idx] + c)  # objective 2: scaled
#                             obj = info_gain_per_traj

#                             if np.isclose(obj, best_obj):
#                                 best_env_idxs.append(env_idx)
#                                 best_traj_idxs.append(traj_idx)
#                             elif obj > best_obj:
#                                 best_obj = obj

#                                 best_env_idxs = [env_idx]
#                                 best_traj_idxs = [traj_idx]
#                             if info_gain_per_traj > max_info_gain:
#                                 max_info_gain = info_gain_per_traj
#                                 print("new max info: {}".format(max_info_gain))
#             else:
#                 # b) select the trajectory closest to the desired partial information gain (to obtain more demonstrations)
#                 # first find the max information gain
#                 for env_idx, info_gains_per_env in enumerate(info_gains_record):
#                     for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
#                         if info_gain_per_traj > 0:
#                             obj = info_gain_per_traj

#                             if np.isclose(obj, best_obj):
#                                 pass
#                             elif obj > best_obj:
#                                 best_obj = obj

#                             if info_gain_per_traj > max_info_gain:
#                                 max_info_gain = info_gain_per_traj
#                                 print("new max info: {}".format(max_info_gain))

#                 target_obj = obj_func_proportion * best_obj
#                 closest_obj_dist = float('inf')

#                 for env_idx, info_gains_per_env in enumerate(info_gains_record):
#                     for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
#                         if info_gain_per_traj > 0:

#                             obj = info_gain_per_traj

#                             if np.isclose(abs(target_obj - obj), closest_obj_dist):
#                                 best_env_idxs.append(env_idx)
#                                 best_traj_idxs.append(traj_idx)
#                             elif abs(target_obj - obj) < closest_obj_dist:
#                                 closest_obj_dist = abs(obj - target_obj)

#                                 best_env_idxs = [env_idx]
#                                 best_traj_idxs = [traj_idx]

#             if max_info_gain < min_info_gain:
#                 no_info_flag = True
#             elif max_info_gain == min_info_gain:
#                 no_info_flag = True
#                 print(colored('Max info gain is equal to min info gain....!', 'red'))
#             else:
#                 best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, traj_record, summary)

#         print("current max info: {}".format(max_info_gain))
#         # no need to continue search for demonstrations if none of them will improve the human's understanding
#         if no_info_flag:
#             # if no variables had been filtered out, then there are no more informative demonstrations to be found
#             if not np.any(variable_filter):
#                 break
#             else:
#                 # no more informative demonstrations with this variable filter, so update it
#                 variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
#                 print(colored('Did not find any informative demonstrations.', 'red'))
#                 print('variable filter: {}'.format(variable_filter))
#                 continue

#         best_traj = traj_record[best_env_idx][best_traj_idx]

#         filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
#         with open(filename, 'rb') as f:
#             wt_vi_traj_env = pickle.load(f)
#         best_mdp = wt_vi_traj_env[0][1].mdp
#         best_mdp.set_init_state(best_traj[0][0]) # for completeness
#         min_BEC_constraints_running.extend(min_env_constraints_record[best_env_idx][best_traj_idx])
#         min_BEC_constraints_running = BEC_helpers.remove_redundant_constraints(min_BEC_constraints_running, weights, step_cost_flag)
        
#         # # check and update unit
#         # if (running_variable_filter == variable_filter).all():
#         #     unit.append([best_mdp, best_traj, (best_env_idx, best_traj_idx), min_env_constraints_record[best_env_idx][best_traj_idx], variable_filter, sample_human_models])
#         # else:
#         #     summary.append(unit)
#         #     unit = [[best_mdp, best_traj, (best_env_idx, best_traj_idx), min_env_constraints_record[best_env_idx][best_traj_idx], variable_filter, sample_human_models]]
#         #     running_variable_filter = variable_filter.copy()

#         summary.append([best_mdp, best_traj, (best_env_idx, best_traj_idx), min_env_constraints_record[best_env_idx][best_traj_idx], variable_filter, sample_human_models])

#         visited_env_traj_idxs.append((best_env_idx, best_traj_idx))

#         particles.update(min_env_constraints_record[best_env_idx][best_traj_idx])

#         print(colored('Max infogain: {}'.format(max_info_gain), 'blue'))
#         with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
#             myfile.write('Max infogain: {}\n'.format(max_info_gain))
#             myfile.write('\n')

#         print(colored('entropy: {}'.format(particles.calc_entropy()), 'blue'))


#         # this method doesn't always finish, so save the summary along the way (for each completed unit)
#         with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
#             pickle.dump((summary, min_BEC_constraints_running, visited_env_traj_idxs, particles), f)

#     return summary, min_BEC_constraints_running, visited_env_traj_idxs, particles



# def compute_counterfactuals_team(args):
#     data_loc, model_idx, env_idx, w_human_normalized, env_filename, trajs_opt, particles, min_BEC_constraints_running, step_cost_flag, summary_len, variable_filter, mdp_features, consider_human_models_jointly = args

#     skip_env = False

#     # if the mdp has a feature that is meant to be filtered out, then skip this environment
#     if variable_filter.dot(mdp_features.T) > 0:
#         skip_env = True

#     if not skip_env:
#         with open(env_filename, 'rb') as f:
#             wt_vi_traj_env = pickle.load(f)

#         agent = wt_vi_traj_env[0][1]
#         weights = agent.mdp.weights

#         human = copy.deepcopy(agent)
#         mdp = human.mdp
#         mdp.weights = w_human_normalized
#         vi_human = ValueIteration(mdp, sample_rate=1)
#         vi_human.run_vi()

#         if not vi_human.stabilized:
#             skip_env = True
#             print(colored('Skipping environment' + str(env_idx) + '...', 'red'))

#     if not skip_env:
#         # only consider counterfactual trajectories from human models whose value iteration have converged and whose
#         # mdp does not have a feature that is meant to be filtered out
#         # best_human_trajs_record_env = []
#         constraints_env = []
#         info_gain_env = []
#         # human_rewards_env = []
#         overlap_in_opt_and_counterfactual_traj_env = []

#         for traj_idx, traj_opt in enumerate(trajs_opt):
#             constraints = []

#             # # a) accumulate the reward features and generate a single constraint
#             # mu_sa = mdp.accumulate_reward_features(traj_opt, discount=True)
#             # traj_hyp = mdp_helpers.rollout_policy(vi_human.mdp, vi_human)
#             # mu_sb = vi_human.mdp.accumulate_reward_features(traj_hyp, discount=True)
#             # constraints.append(mu_sa - mu_sb)
#             # best_human_trajs_record = [] # for symmetry with below
#             # best_human_reward = weights.dot(mu_sb.T)  # for symmetry with below

#             # b) contrast differing expected feature counts for each state-action pair along the agent's optimal trajectory
#             # best_human_trajs_record = []
#             # best_human_reward = 0
#             for sas_idx in range(len(traj_opt)):
#                 # reward features of optimal action
#                 mu_sa = mdp.accumulate_reward_features(traj_opt[sas_idx:], discount=True)

#                 sas = traj_opt[sas_idx]
#                 cur_state = sas[0]

#                 # obtain all optimal trajectory rollouts according to the human's model (assuming that it's a reasonable policy that has converged)
#                 human_opt_trajs = mdp_helpers.rollout_policy_recursive(vi_human.mdp, vi_human, cur_state, [])

#                 cur_best_reward = float('-inf')
#                 best_reward_features = []
#                 best_human_traj = []
#                 # select the human's possible trajectory that has the highest true reward (i.e. give the human's policy the benefit of the doubt)
#                 for traj in human_opt_trajs:
#                     mu_sb = mdp.accumulate_reward_features(traj, discount=True)  # the human and agent should be working with identical mdps
#                     reward_hyp = weights.dot(mu_sb.T)
#                     if reward_hyp > cur_best_reward:
#                         cur_best_reward = reward_hyp
#                         best_reward_features = mu_sb
#                         best_human_traj = traj

#                 # # todo: for testing how much computation and time I save by not doing a recursive rollout
#                 # best_human_traj = mdp_helpers.rollout_policy(vi_human.mdp, vi_human, cur_state, [])
#                 # best_reward_features = mdp.accumulate_reward_features(best_human_traj,
#                 #                                        discount=True)  # the human and agent should be working with identical mdps
#                 # cur_best_reward = weights.dot(best_reward_features.T)

#                 # only store the reward of the full trajectory
#                 # if sas_idx == 0:
#                 #     best_human_reward = cur_best_reward
#                 #     traj_opt_feature_count = mu_sa
#                 constraints.append(mu_sa - best_reward_features)
#                 # best_human_trajs_record.append(best_human_traj)

#             if len(constraints) > 0:
#                 constraints = BEC_helpers.remove_redundant_constraints(constraints, weights, step_cost_flag)

#             if particles is not None:
#                 info_gain = particles.calc_info_gain(constraints)
#             else:
#                 info_gain = BEC_helpers.calculate_information_gain(min_BEC_constraints_running, constraints,
#                                                                    weights, step_cost_flag)

#             # human_rewards_env.append(best_human_reward)
#             # best_human_trajs_record_env.append(best_human_trajs_record)
#             constraints_env.append(constraints)
#             info_gain_env.append(info_gain)

#             # if not consider_human_models_jointly:
#             #     # you should only consider the overlap for the first counterfactual human trajectory (as opposed to
#             #     # counterfactual trajectories that could've arisen from states after the first state)
#             #     overlap_pct = BEC_helpers.calculate_counterfactual_overlap_pct(best_human_trajs_record[0], traj_opt)
#             #
#             #     overlap_in_opt_and_counterfactual_traj_env.append(overlap_pct)

#     # else just populate with dummy variables
#     else:
#         # best_human_trajs_record_env = [[[]] for i in range(len(trajs_opt))]
#         constraints_env = [[] for i in range(len(trajs_opt))]
#         info_gain_env = [0 for i in range(len(trajs_opt))]
#         if not consider_human_models_jointly:
#             overlap_in_opt_and_counterfactual_traj_env = [float('inf') for i in range(len(trajs_opt))]
#         # human_rewards_env = [np.array([[0]]) for i in range(len(trajs_opt))]

#     if summary_len is not None:
#         with open('models/' + data_loc + '/counterfactual_data_' + str(summary_len) + '/model' + str(model_idx) +
#                   '/cf_data_env' + str(env_idx).zfill(5) + '.pickle', 'wb') as f:
#             pickle.dump(constraints_env, f)
#         # print('Info gain for Summary length ' + str(summary_len) + ' of model ' + str(model_idx) + 'of env' + str(env_idx) + 'is' + str(info_gain_env) + '...')

#     if consider_human_models_jointly:
#         return info_gain_env
#     else:
#         return info_gain_env, overlap_in_opt_and_counterfactual_traj_env



# def obtain_team_summary_counterfactuals(data_loc, demo_strategy, team_models, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag, pool, team_size, n_human_models, consistent_state_count,
#                                 BEC_summary = None, n_train_demos=3, team_prior={}, downsample_threshold=float("inf"), consider_human_models_jointly=True, c=0.001, obj_func_proportion=1):

#     # TODO: This is still a work in progress




#     # prior knowledge
#     if demo_strategy =='individual_knowledge_low':
#         ind_knowledge_ascending = find_ascending_individual_knowledge(team_prior)
#         prior = team_prior[ind_knowledge_ascending[0]].copy()
#     elif demo_strategy == 'individual_knowledge_high':
#         ind_knowledge_ascending = find_ascending_individual_knowledge(team_prior)
#         prior = team_prior[ind_knowledge_ascending[len(ind_knowledge_ascending) - 1]].copy()
#     elif demo_strategy == 'joint_knowledge':
#         prior = calc_joint_knowledge(team_prior)
#     elif demo_strategy == 'common_knowledge':
#         prior = calc_common_knowledge(team_prior)

#     min_BEC_constraints_running = prior.copy() 


#     ## demo generation loop
#     while summary_count < n_train_demos:
        
#         if summary_count == 0:
#             if demo_strategy == 'individual_knowledge_low':
#                 sample_human_models = team_models[ind_knowledge_ascending[0]].copy()
#             elif demo_strategy == 'individual_knowledge_high':
#                 sample_human_models = team_models[ind_knowledge_ascending[len(ind_knowledge_ascending) - 1]].copy()
#             else:
#                 sample_human_models = BEC_helpers.sample_human_models_uniform(min_BEC_constraints_running, n_human_models)
#         else:
#             sample_human_models = BEC_helpers.sample_human_models_uniform(min_BEC_constraints_running, n_human_models)

#         # print('Initial sample human models for person {} : {}'.format(p_id, sample_human_models))
        
#         if len(sample_human_models) == 0:
#             print(colored("Likely cannot reduce the BEC further through additional demonstrations. Returning.", 'red'))
#             return summary, visited_env_traj_idxs

#         info_gains_record = []

#         print("Length of summary: {}".format(summary_count))
#         with open('models/' + data_loc + '/teams_demo_gen_log.txt', 'a') as myfile:
#             myfile.write('Length of summary: {}\n'.format(summary_count))

#         for model_idx, human_model in enumerate(sample_human_models):
#             print(colored('Model #: {}'.format(model_idx), 'red'))
#             print(colored('Model val: {}'.format(human_model), 'red'))

#             with open('models/' + data_loc + '/teams_demo_gen_log.txt', 'a') as myfile:
#                 myfile.write('Model #: {}\n'.format(model_idx))
#                 myfile.write('Model val: {}\n'.format(human_model))

#             # based on the human's current model, obtain the information gain generated when comparing to the agent's
#             # optimal trajectories in each environment (human's corresponding optimal trajectories and constraints
#             # are saved for reference later)
#             print("Obtaining counterfactual information gains:")

#             cf_data_dir = 'models/' + data_loc + '/teams_counterfactual_data_' + str(summary_count) + '/model' + str(model_idx)
#             os.makedirs(cf_data_dir, exist_ok=True)

#             pool.restart()
#             args = [(data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]), traj_record[i], None, min_BEC_constraints_running, step_cost_flag, summary_count, variable_filter, consider_human_models_jointly) for i in range(len(traj_record))]
#             info_gain_envs = list(tqdm(pool.imap(compute_counterfactuals_team, args), total=len(args)))
#             pool.close()
#             pool.join()
#             pool.terminate()

#             info_gains_record.append(info_gain_envs)

#         with open('models/' + data_loc + '/teams_info_gains_' + str(summary_count) + '.pickle', 'wb') as f:
#             pickle.dump(info_gains_record, f)


#         # do a quick check of whether there's any information to be gained from any of the trajectories
#         no_info_flag = check_info_gain(info_gains_record, consistent_state_count)

#         # no need to continue search for demonstrations if none of them will improve the human's understanding
#         if no_info_flag:
#             break


#         print("Combining the most limiting constraints across human models:")
#         pool.restart()
#         args = [(i, len(sample_human_models), data_loc, summary_count, weights, step_cost_flag, variable_filter,
#                 traj_record[i], min_BEC_constraints_running, None) for
#                 i in range(len(traj_record))]
#         info_gains_record, min_env_constraints_record, n_diff_constraints_record, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = zip(
#             *pool.imap(combine_limiting_constraints_IG, tqdm(args)))
#         pool.close()
#         pool.join()
#         pool.terminate()

#         with open('models/' + data_loc + '/teams_info_gains_joint' + str(summary_count) + '.pickle', 'wb') as f:
#             pickle.dump(info_gains_record, f)

#         differing_constraint_count = 1          # number of constraints in the running human model that would differ after showing a particular demonstration
#         max_differing_constraint_count = max(list(itertools.chain(*n_diff_constraints_record)))
#         print("max_differing_constraint_count: {}".format(max_differing_constraint_count))
#         no_info_flag = True
#         max_info_gain = 1

#         # try to find a demonstration that will yield the fewest changes in the constraints defining the running human model while maximizing the information gain
#         while no_info_flag and differing_constraint_count <= max_differing_constraint_count:
#             # the possibility that no demonstration provides information gain must be checked for again,
#             # in case all limiting constraints involve a masked variable and shouldn't be considered for demonstration yet
#             if consistent_state_count:
#                 info_gains = np.array(info_gains_record)
#                 n_diff_constraints = np.array(n_diff_constraints_record)
#                 traj_overlap_pcts = np.array(overlap_in_opt_and_counterfactual_traj_avg)

#                 # obj_function = info_gains * (traj_overlap_pcts + c)  # objective 2: scaled
#                 obj_function = info_gains

#                 # not considering demos where there is no info gain helps ensure that the final demonstration
#                 # provides the maximum info gain (in conjuction with previously shown demonstrations)
#                 obj_function[info_gains == 1] = 0
#                 obj_function[n_diff_constraints != differing_constraint_count] = 0

#                 max_info_gain = np.max(info_gains)
#                 if max_info_gain == 1:
#                     no_info_flag = True
#                     differing_constraint_count += 1
#                 else:
#                     # if visuals aren't considered, then you can simply return one of the demos that maximizes the obj function
#                     # best_env_idx, best_traj_idx = np.unravel_index(np.argmax(obj_function), info_gains.shape)

#                     if obj_func_proportion == 1:
#                         # a) select the trajectory with the maximal information gain
#                         best_env_idxs, best_traj_idxs = np.where(obj_function == max(obj_function.flatten()))
#                     else:
#                         # b) select the trajectory closest to the desired partial information gain (to obtain more demonstrations0
#                         obj_function_flat = obj_function.flatten()
#                         obj_function_flat.sort()

#                         best_obj = obj_function_flat[-1]
#                         target_obj = obj_func_proportion * best_obj
#                         target_idx = np.argmin(abs(obj_function_flat - target_obj))
#                         closest_obj = obj_function_flat[target_idx]
#                         best_env_idxs, best_traj_idxs = np.where(obj_function == obj_function_flat[closest_obj])

#                     # we're still in the same unit so try and optimize visuals wrt other demonstrations in this unit
#                     best_env_idx, best_traj_idx = optimize_visuals_team(data_loc, best_env_idxs, best_traj_idxs, traj_record, unit)
#                     no_info_flag = False
#             else:
#                 best_obj = float('-inf')
#                 best_env_idxs = []
#                 best_traj_idxs = []

#                 if obj_func_proportion == 1:
#                     # a) select the trajectory with the maximal information gain
#                     for env_idx, info_gains_per_env in enumerate(info_gains_record):
#                         for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
#                             if info_gain_per_traj > 1 and n_diff_constraints_record[env_idx][traj_idx] == differing_constraint_count:

#                                 # obj = info_gain_per_traj * (
#                                 #             overlap_in_opt_and_counterfactual_traj_avg[env_idx][traj_idx] + c)  # objective 2: scaled
#                                 obj = info_gain_per_traj

#                                 if np.isclose(obj, best_obj):
#                                     best_env_idxs.append(env_idx)
#                                     best_traj_idxs.append(traj_idx)
#                                 elif obj > best_obj:
#                                     best_obj = obj

#                                     best_env_idxs = [env_idx]
#                                     best_traj_idxs = [traj_idx]
#                                 if info_gain_per_traj > max_info_gain:
#                                     max_info_gain = info_gain_per_traj
#                                     print("new max info: {}".format(max_info_gain))
#                 else:
#                     # b) select the trajectory closest to the desired partial information gain (to obtain more demonstrations)
#                     # first find the max information gain
#                     for env_idx, info_gains_per_env in enumerate(info_gains_record):
#                         for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
#                             if info_gain_per_traj > 1 and n_diff_constraints_record[env_idx][traj_idx] == differing_constraint_count:
#                                 obj = info_gain_per_traj

#                                 if np.isclose(obj, best_obj):
#                                     pass
#                                 elif obj > best_obj:
#                                     best_obj = obj

#                                 if info_gain_per_traj > max_info_gain:
#                                     max_info_gain = info_gain_per_traj
#                                     print("new max info: {}".format(max_info_gain))

#                     target_obj = obj_func_proportion * best_obj
#                     closest_obj_dist = float('inf')

#                     for env_idx, info_gains_per_env in enumerate(info_gains_record):
#                         for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
#                             if info_gain_per_traj > 1 and n_diff_constraints_record[env_idx][traj_idx] == differing_constraint_count:

#                                 obj = info_gain_per_traj

#                                 if np.isclose(abs(target_obj - obj), closest_obj_dist):
#                                     best_env_idxs.append(env_idx)
#                                     best_traj_idxs.append(traj_idx)
#                                 elif abs(target_obj - obj) < closest_obj_dist:
#                                     closest_obj_dist = abs(obj - target_obj)

#                                     best_env_idxs = [env_idx]
#                                     best_traj_idxs = [traj_idx]

#                 if max_info_gain == 1:
#                     no_info_flag = True
#                     differing_constraint_count += 1
#                 else:
#                     # we're still in the same unit so try and optimize visuals wrt other demonstrations in this unit
#                     best_env_idx, best_traj_idx = optimize_visuals_team(data_loc, best_env_idxs, best_traj_idxs, traj_record, unit)
#                     no_info_flag = False

#         with open('models/' + data_loc + '/teams_best_env_idxs' + str(summary_count) + '.pickle', 'wb') as f:
#             pickle.dump((best_env_idx, best_traj_idx, best_env_idxs, best_traj_idxs), f)

#         print("current max info: {}".format(max_info_gain))
        

#         if no_info_flag:
#             break

#         best_traj = traj_record[best_env_idx][best_traj_idx]

#         filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
#         with open(filename, 'rb') as f:
#             wt_vi_traj_env = pickle.load(f)
#         best_mdp = wt_vi_traj_env[0][1].mdp
#         best_mdp.set_init_state(best_traj[0][0]) # for completeness
#         min_BEC_constraints_running.extend(min_env_constraints_record[best_env_idx][best_traj_idx])
#         min_BEC_constraints_running = BEC_helpers.remove_redundant_constraints(min_BEC_constraints_running, weights, step_cost_flag)
#         if (running_variable_filter == variable_filter).all():
#             unit.append([demo_strategy, p_id, best_mdp, best_traj, (best_env_idx, best_traj_idx), min_env_constraints_record[best_env_idx][best_traj_idx], variable_filter, sample_human_models])
#             summary_count += 1
#         else:
#             summary.append(unit)

#             unit = [[demo_strategy, p_id, best_mdp, best_traj, (best_env_idx, best_traj_idx), min_env_constraints_record[best_env_idx][best_traj_idx], variable_filter, sample_human_models]]
#             running_variable_filter = variable_filter.copy()
#             summary_count += 1
        
#         visited_env_traj_idxs.append((best_env_idx, best_traj_idx))


#         print(colored('Max infogain: {}'.format(max_info_gain), 'blue'))
#         with open('models/' + data_loc + '/teams_demo_gen_log.txt', 'a') as myfile:
#             myfile.write('Max infogain: {}\n'.format(max_info_gain))
#             myfile.write('\n')

#     # add any remaining demonstrations
#     summary.append(unit)


#     return summary, visited_env_traj_idxs