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
import random
import csv


# Other imports.
sys.path.append("simple_rl")
import params_team as params
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
from sklearn.metrics.pairwise import haversine_distances
from numpy import linalg as LA
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##########################################################




def calc_common_knowledge(team_knowledge, team_size, weights, step_cost_flag, kc_id=None):

    print('Team knowledge: ', team_knowledge)

    for i in range(team_size):
        member_id = 'p' + str(i+1)
        
        if kc_id is None:
            kc_id = len(team_knowledge[member_id])-1  # constraints of last KC

        if i==0:
            constraints = copy.deepcopy(team_knowledge[member_id][kc_id])
        else:
            constraints.extend(team_knowledge[member_id][kc_id])

        # # print('Constraints so far:', constraints)

    # # check for opposing constraints (since remove redundant constraints gives the perpendicular axes for opposing constraints)

    # opposing_constraints_flag, _, _ = check_opposing_constraints(constraints)
    # if opposing_constraints_flag:
    #     raise RuntimeError('Opposing constraints found in common knowledge!')

    non_intersecting_cnsts_flag, _ = check_for_non_intersecting_constraints(constraints, weights, step_cost_flag)
    if non_intersecting_cnsts_flag:
        raise RuntimeError('Non-intersecting constraints found in common knowledge!')

    # # print('Opposing constraints flag: ', opposing_constraints_flag)

    # # if not opposing_constraints_flag:
    # unit_constraint_flag = True
    # unit_common_constraint_flag = True
    constraints_copy = copy.deepcopy(constraints)

    common_constraints = BEC_helpers.remove_redundant_constraints(constraints_copy, weights, step_cost_flag)
    # print('Common knowledge after removing redundant constraints: ', common_constraints)


    return common_constraints



def check_opposing_constraints(test_constraints_team_expanded, opposing_constraints_count=0):
    
    if params.debug_kl_calc:
        print('Checking for opposing constraints in list: ', test_constraints_team_expanded)
    
    opposing_constraints_flag = False
    opposing_idx = []
    new_opposing_constraints_count = 0
    for cnst_idx, cnst in enumerate(test_constraints_team_expanded):
        for cnst2_idx, cnst2 in enumerate(test_constraints_team_expanded):
            # # print('-cnst[0]: ', -cnst[0])
            # # print('cnst2[0]): ', cnst2[0])
            if (BEC_helpers.equal_constraints(-cnst[0], cnst2[0])):
                opposing_constraints_flag = True  
                opposing_idx.append([cnst_idx, cnst2_idx])
                new_opposing_constraints_count += 1 # gets double-counted
                
    # print('opposing_constraints_flag: ', opposing_constraints_flag)
    
    return opposing_constraints_flag, opposing_constraints_count+new_opposing_constraints_count/2, opposing_idx


def calc_joint_knowledge(team_knowledge, team_size, kc_id = None):

    # Joint constraints is normally expressed as the union of constraints of individuals. 

    # ## Method 1: Joint constraints expressed as inverse of the inverted and minimal constraints of individual members
    # for i in range(team_size):
    #     member_id = 'p' + str(i+1)
    #     if i==0:
    #         inv_constraints = [-x for x in team_knowledge[member_id]]
    #     else:
    #         inv_constraints.extend([-x for x in team_knowledge[member_id]])

    # # print('Team knowledge:', team_knowledge)
    # # print('Inverted constraints:', inv_constraints)
    # inv_joint_constraints = BEC_helpers.remove_redundant_constraints(inv_constraints, weights, step_cost_flag)

    # joint_constraints = [-x for x in inv_joint_constraints]


    ## Method 2: Here we just represent the constraints of each individual as a separate list.
    for i in range(team_size):
        member_id = 'p' + str(i+1)
        if kc_id is None:
            kc_id = len(team_knowledge[member_id])-1  # constraints of last KC

        if i==0:
            joint_constraints = [team_knowledge[member_id][kc_id]]
        else:
            joint_constraints.append(team_knowledge[member_id][kc_id])

    # # print('Team knowledge:', team_knowledge)
    # # print('Joint constraints:', joint_constraints)


    return joint_constraints




def update_team_knowledge(team_knowledge, kc_id, kc_reset_flag, new_constraints, team_size, weights, step_cost_flag, knowledge_to_update = 'all'):

    # print(colored('Updating team knowledge...', 'blue'))
    team_knowledge_updated = copy.deepcopy(team_knowledge)

    if knowledge_to_update == 'all':
        knowledge_types = list(team_knowledge.keys())
    else:
        knowledge_types = copy.deepcopy(knowledge_to_update)


    for knowledge_type in knowledge_types:

        # print('Knowledge type: ', knowledge_type)
        # print('Knowledge constraints: ', team_knowledge[knowledge_type])
        # print('KC index: ', kc_id)
        # print('Knowledge len: ', len(team_knowledge[knowledge_type]))
        # print('New constraints: ', new_constraints)

        if knowledge_type != 'joint_knowledge' and  knowledge_type != 'common_knowledge':
            new_knowledge = copy.deepcopy(new_constraints)
        
        elif  knowledge_type == 'joint_knowledge':
            new_knowledge = calc_joint_knowledge(team_knowledge_updated, team_size)
        
        elif  knowledge_type == 'common_knowledge':
            new_knowledge = calc_common_knowledge(team_knowledge_updated, team_size, weights=weights, step_cost_flag=step_cost_flag)
        
        # else:
            # print(colored('Unknown knowledge type to update.', 'red'))

        if kc_id > len(team_knowledge[knowledge_type]):
            print(colored('kc_id is greater than the number of knowledge constraints in the team knowledge.', 'red'))
            # raise RuntimeError('kc_id is greater than the number of knowledge constraints in the team knowledge.')

        # print('New knowledge: ', new_knowledge)

        # if knowledge_type != 'joint_knowledge' and kc_id == len(team_knowledge[knowledge_type]):
        #     # print('Appendin knowledge...')
        #     team_knowledge_updated[knowledge_type].append(new_knowledge)
        # elif knowledge_type == 'joint_knowledge' and kc_id == len(team_knowledge[knowledge_type])/params.team_size:
        #     # print('Appendin joint knowledge...')
        #     team_knowledge_updated[knowledge_type].append(new_knowledge)
        if kc_id == len(team_knowledge[knowledge_type]):
            # print(colored('Appending knowledge...', 'red'))
            team_knowledge_updated[knowledge_type].append(new_knowledge)
        elif knowledge_type != 'joint_knowledge' and kc_reset_flag:
            # print(colored('Reset knowledge...', 'red'))
            team_knowledge_updated[knowledge_type][kc_id] = copy.deepcopy(new_knowledge)
        elif  knowledge_type == 'joint_knowledge':
            # print(colored('Update joint knowledge. Always reset..', 'red'))
            team_knowledge_updated[knowledge_type][kc_id] = copy.deepcopy(new_knowledge)
        else:
            # print(colored('Extend knowledge..', 'red'))
            team_knowledge_updated[knowledge_type][kc_id].extend(new_knowledge)

    # print('team_knowledge_updated: ', team_knowledge_updated)

    return team_knowledge_updated




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

                

def check_and_update_variable_filter(min_subset_constraints_record = None, variable_filter = None, nonzero_counter = None, initialize_filter_flag = False):
    
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
        # print('variable filter: {}'.format(variable_filter))

        teaching_complete_flag = False

    else:
        # if no variables had been filtered out, then there are no more informative demonstrations to be found
        # print('Current variable filter: ', variable_filter)
        if not np.any(variable_filter):
            teaching_complete_flag = True
            print('teaching_complete_flag: ', teaching_complete_flag)
        else:
            # no more informative demonstrations with this variable filter, so update it
            variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
            # print(colored('Did not find any more informative demonstrations.', 'red'))
            # print('Updated variable filter: {}'.format(variable_filter))
            teaching_complete_flag = False
            # print('Updated variable filter: ', variable_filter)
                
    
    return variable_filter, nonzero_counter, teaching_complete_flag



def find_ascending_individual_knowledge(team_knowledge, min_BEC_constraints, particles_team_teacher = None, kc_id=None):

    # Sorts based on ascending order of knowledge
    if kc_id is None:
        team_knowledge_level = calc_knowledge_level(team_knowledge, min_BEC_constraints, particles_team_teacher = None)
    else:
        team_knowledge_level = calc_knowledge_level(team_knowledge, min_BEC_constraints, particles_team_teacher = None, kc_id_list=[kc_id])

    sorted_kl = dict(sorted(team_knowledge_level.items(), key=lambda item: item[1]))
    print('sorted_kl: ', sorted_kl)
    ascending_order_of_knowledge = []
    for i, kl in enumerate(sorted_kl):
        # print(kl)
        if 'p' in kl:
            ascending_order_of_knowledge.append(kl)

    return ascending_order_of_knowledge



def check_for_non_intersecting_constraints(constraints, weights, step_cost_flag, non_intersecting_constraints_count=0):

    if params.debug_kl_calc:
        print('Checking for non-intersecting constraints in list:', constraints)


    if len(constraints) == 1:
        non_intersecting_constraints_flag = False
        return non_intersecting_constraints_flag, non_intersecting_constraints_count

    elif len(constraints) == 0:
        raise RuntimeError('Empty constraint list!')
    
    else:

        # # Method 1: crude check after redundancy removal. Assuming it would always result in unit constraints
        # # check for non-intersecting constraints
        # all_unit_norm_flag = True
        # for cnst in constraints:
        #     # print('cnst: ', cnst)
        #     if LA.norm(cnst) != 1:
        #         all_unit_norm_flag = False
        # # print('step_cost_flag: ', step_cost_flag)
        # # print('weights: ', weights)
        # min_constraints = BEC_helpers.remove_redundant_constraints(constraints, weights, step_cost_flag)

        # # print('min constraints: ', min_constraints)
        # # print('all_unit_norm_flag: ', all_unit_norm_flag)

        # if not all_unit_norm_flag:
        #     non_intersecting_constraints_flag = True
        #     for cnst in min_constraints:
        #         if LA.norm(cnst) != 1:
        #             non_intersecting_constraints_flag = False
        # else:
        #     # all reduced constraints are in the original constraints (happens when there are intersecting unit norm constraints)
        #     N_min_cnst_in_cnst_list = 0
        #     for cnst in min_constraints:
        #         if any((i==cnst).all() for i in constraints):
        #             non_intersecting_constraints_flag = False
        #             N_min_cnst_in_cnst_list += 1

        #     if N_min_cnst_in_cnst_list == len(min_constraints):
        #         # # print('Intersecting constraints...')
        #         non_intersecting_constraints_flag = False

        
        
        ## check for opposing constraints
        opposing_constraints_flag, _, opposing_idx = check_opposing_constraints(constraints)

        if opposing_constraints_flag:
            non_intersecting_constraints_flag = True

        else:
            ## Method 2: Check for reduced unit norm constrainst not in original constraint list
            min_constraints = BEC_helpers.remove_redundant_constraints(constraints, weights, step_cost_flag)
            # constraints_not_in_set = [cnst for cnst in min_constraints if cnst not in constraints]
            constraints_not_in_set = []
            for cnst in min_constraints:
                cnst_present_flag = False
                for cnst2 in constraints:
                    if BEC_helpers.equal_constraints(cnst, cnst2):
                        cnst_present_flag = True
                        break
                
                if not cnst_present_flag:
                    if len(constraints_not_in_set) == 0:
                        constraints_not_in_set = [cnst]
                    else:
                        constraints_not_in_set.append(cnst)
            
            if params.debug_kl_calc:
                print('min_constraints: ', min_constraints)
                print('constraints_not_in_set: ', constraints_not_in_set)

            # double check if these are unit norm - commented it because noticed that for some constraints the non-intersecting minimal constraints are not necessarily unit norm!
            # Example case for above: [np.array([[ 0,  0, -1]]), np.array([[ -1,  0, 4]]), np.array([[1, 1, 0]]), np.array([[ 0, -1, -4]]), np.array([[-1,  0,  2]])]
            if len(constraints_not_in_set) > 0:
                non_intersecting_constraints_flag = True
                # for c in constraints_not_in_set:
                #     if LA.norm(c) != 1:
                #         non_intersecting_constraints_flag = False
            else:
                non_intersecting_constraints_flag = False


    return non_intersecting_constraints_flag, non_intersecting_constraints_count+1



def calc_expected_learning(team_knowledge_expected, kc_id, particles_team_teacher, min_BEC_constraints, new_constraints, params, loop_count, kc_reset_flag = False, viz_flag = False):

    # print(colored('Calculating expected learning for KC ' + str(kc_id) + 'in loop ' + str(loop_count) + '...', 'blue'))
    # # print('Current expected team knowledge: ', team_knowledge_expected)
    # # print('New constraints: ', new_constraints)
    team_knowledge_expected = update_team_knowledge(team_knowledge_expected, kc_id, kc_reset_flag, new_constraints, params.team_size,  params.weights['val'], params.step_cost_flag)

    # # print('Updated expected team knowledge: ', team_knowledge_expected)

    p = 1
    while p <= params.team_size:
        member_id = 'p' + str(p)
        # print('PF update of ', member_id)
        plot_title = 'Teacher belief (Expected knowledge change) for loop ' + str(loop_count+1) + ' for player ' + member_id
        particles_team_teacher[member_id].update(new_constraints, plot_title = plot_title, viz_flag = viz_flag)
        
        ########## debugging - calculate proportion of particles that are within the BEC
        particles_team_teacher[member_id].calc_particles_probability(new_constraints)
        ##########################
        
        if viz_flag:
            visualize_transition(new_constraints[0], particles_team_teacher[member_id], params.mdp_class, params.weights['val'], text = 'Expected knowledge change for set ' + str(loop_count+1) + ' for player ' + member_id, plot_filename ='ek_p' + str(p) + '_loop_' + str(loop_count+1))
        p += 1
            
    # Update common knowledge model
    plot_title = 'Teacher belief (Expected knowledge change) for loop ' + str(loop_count+1) + ' for common knowledge'
    particles_team_teacher['common_knowledge'].update(new_constraints, plot_title = plot_title, viz_flag = viz_flag) # PF update of common_knowledge
    # if viz_flag:
    #     visualize_transition(new_constraints, particles_team_teacher_expected['common_knowledge'], params.mdp_class, params.weights['val'], text = 'Expected knowledge change for set' + str(loop_count+1) + ' for common knowledge', plot_filename ='ek_ck_loop_' + str(loop_count+1))
    
    # Method: Use new joint knowledge of team
    new_constraints_team = []
    for p in range(params.team_size):
        new_constraints_team.append(new_constraints)
    particles_team_teacher['joint_knowledge'].update_jk(new_constraints_team)

    # if viz_flag:
    #     visualize_transition(new_constraints_team, particles_team_teacher_expected['joint_knowledge'], params.mdp_class, params.weights['val'], text = 'Expected knowledge change for set ' + str(loop_count+1) + ' for joint knowledge',  knowledge_type = 'joint_knowledge', plot_filename ='ek_jk_loop_' + str(loop_count+1))

    return team_knowledge_expected, particles_team_teacher





def calc_knowledge_level(team_knowledge, min_unit_constraints, particles_team_teacher = None, kc_id_list = None, weights = None, step_cost_flag = False, plot_flag = False, debug_flag=True, fig_title=''):
    
    
    def plot_text(constraints, knowledge, fig, i):
        x_loc = [0.1, 0.3, 0.5, 0.7, 0.9]
        y_loc = 0.2
        fig.text(x_loc[i], y_loc, 'Knowledge level: ' + str(knowledge), fontsize=12)
        y_loc -= 0.03
        fig.text(x_loc[i], y_loc, 'Knowledge constraints: ', fontsize=12)
        y_loc -= 0.02
        for cnst in constraints:
            fig.text(x_loc[i], y_loc, s=str(cnst), fontsize=12)
            y_loc -= 0.02


    def calc_recursive_intersection_area(constraint_list, N_intersecting_sets, weights, step_cost_flag, fixed_constraint = []):
        
        N_sets = len(constraint_list)
        intersections_to_check = list(itertools.combinations(range(len(constraint_list)), N_intersecting_sets))
        intersection_area = 0

        # if params.debug_kl_calc:
        #     print('Constraint list: ', constraint_list)
            # print('Intersections to check: ', intersections_to_check)
            # print('Fixed constraint: ', fixed_constraint)  # will always be part of the intersection


        for i in range(len(intersections_to_check)):
            constraints = []
            for j in intersections_to_check[i]:
                constraints.extend(constraint_list[j])

            constraints.extend(fixed_constraint) # in case there is a fixed constraint
            
            # if params.debug_kl_calc:
            #     print('Constraint combinations: ', intersections_to_check[i])
                # print('Constraints: ', constraints)

            if len(constraints) > 0 and not check_for_non_intersecting_constraints(constraints, weights, step_cost_flag)[0]:
                
                # remove redundant constraints before calculating area
                constraints = BEC_helpers.remove_redundant_constraints(constraints, weights, step_cost_flag)
                # if params.debug_kl_calc:
                #     print('Min constraints to calculate area: ', constraints)
# 
                intersection_area += (-1)**(N_intersecting_sets+1) * np.array(BEC_helpers.calc_solid_angles([constraints]))

            else:
                if params.debug_kl_calc:
                    print(colored('Non intersecting constraint set!', 'red'))
        
        if params.debug_kl_calc:
            print('N_intersecting_sets: ', N_intersecting_sets, 'intersection_area: ', intersection_area)
            


        return 0 if N_intersecting_sets > N_sets else intersection_area + calc_recursive_intersection_area(constraint_list, N_intersecting_sets+1, weights, step_cost_flag, fixed_constraint)

    

    def calc_joint_knowledge_separately(ind_intersection_constraints,  min_unit_constraints, weights, step_cost_flag):

        if params.debug_kl_calc:
            print('ind_intersection_constraints: ', ind_intersection_constraints)
            print('min_unit_constraints: ', min_unit_constraints)

        ind_union_constraints = copy.deepcopy(ind_intersection_constraints)
        ind_union_constraints.extend([min_unit_constraints])

        if params.debug_kl_calc:
            print('ind_union_constraints: ', ind_union_constraints)

        min_unit_BEC_knowledge_union = calc_recursive_intersection_area(ind_union_constraints, 1, weights, step_cost_flag)
        min_unit_BEC_knowledge_intersection = calc_recursive_intersection_area(ind_intersection_constraints, 1, weights, step_cost_flag, fixed_constraint = min_unit_constraints)
        
        if params.debug_kl_calc:
            print('min_unit_BEC_knowledge_intersection :', min_unit_BEC_knowledge_intersection)
            print('min_unit_BEC_knowledge_union :', min_unit_BEC_knowledge_union)


        # check for non-intersecting member knowledge
        m_id = 0
        for member_knowledge in ind_intersection_constraints:
            print('member_knowledge: ', member_knowledge)
            if check_for_non_intersecting_constraints(member_knowledge, weights, step_cost_flag)[0]:
                member_id = 'p' + str(m_id+1)
                min_unit_BEC_knowledge_union += BEC_helpers.calc_solid_angles([params.team_prior[member_id]])
            m_id += 1

        if params.debug_kl_calc:
            print('min_unit_BEC_knowledge_union after correction for null member knowledge:', min_unit_BEC_knowledge_union)
        

        return min_unit_BEC_knowledge_intersection/min_unit_BEC_knowledge_union

    
    # # print('team_knowledge: ', team_knowledge)
    # # print('min_unit_constraints: ', min_unit_constraints)
    
    ######
    
    knowledge_level = {}
    min_unit_BEC_knowledge_intersection = 0
    min_unit_BEC_knowledge_union = 0

    # specific knowledge component or overall knowledge
    if kc_id_list is None:
        kc_id_list = range(len(team_knowledge['p1']))


    if plot_flag:
        fig = visualize_team_knowledge_constraints(team_knowledge, weights, step_cost_flag, particles_team_teacher = particles_team_teacher, plot_min_constraints_flag = False, plot_text_flag = False, min_unit_constraints = min_unit_constraints, plot_filename = 'team_knowledge_constraints', fig_title = fig_title)
        

    plot_id = 0
    for knowledge_id, knowledge_type in enumerate(team_knowledge):


        # print(colored('Calculating knowledge level for: ', 'blue'), knowledge_type)
        # print('knowledge constraints: ', team_knowledge[knowledge_type])
        # combine constraints of all KCs
        team_knowledge_constraints = []
        
        # combine knowledge constraints
        if knowledge_type == 'joint_knowledge':
            # for each player
            for mem_id in range(len(team_knowledge[knowledge_type][kc_id_list[0]])):
                team_knowledge_constraints.append(copy.deepcopy(team_knowledge[knowledge_type][kc_id_list[0]][mem_id]))

                for rem_kc_id in kc_id_list[1:]:
                    team_knowledge_constraints[mem_id].extend(team_knowledge[knowledge_type][rem_kc_id][mem_id])

        else:
            for kc_index in kc_id_list:
                # print('team_knowledge[knowledge_type][kc_index]: ', team_knowledge[knowledge_type][kc_index])
                if len(team_knowledge_constraints) == 0:
                    team_knowledge_constraints = copy.deepcopy(team_knowledge[knowledge_type][kc_index])
                else:
                    team_knowledge_constraints.extend(team_knowledge[knowledge_type][kc_index])

        # print(colored('team_knowledge_constraints: ', 'blue'), team_knowledge_constraints)

        # calculate knowledge level
        if knowledge_type == 'joint_knowledge':
            ## Knowledge metric for joint knowledge
            knowledge_level[knowledge_type] = calc_joint_knowledge_separately(team_knowledge_constraints, min_unit_constraints, weights, step_cost_flag)
        else:
            # Method 3: Use Jaccard's index for set similarity (Intersection over Union)
            if not check_for_non_intersecting_constraints(team_knowledge_constraints, weights, step_cost_flag)[0]:
                team_knowledge_constraints = BEC_helpers.remove_redundant_constraints(team_knowledge_constraints, weights, step_cost_flag)  # before calculating solid angles constraints set has to be a minimum set

                # double check that these are indeed minimum constraints
                min_unit_area = np.array(BEC_helpers.calc_solid_angles([min_unit_constraints]))            
                knowledge_area = np.array(BEC_helpers.calc_solid_angles([team_knowledge_constraints]))

                min_unit_intersection_constraints = copy.deepcopy(min_unit_constraints)
                # print('unit_intersection_constraints before update: ', min_unit_intersection_constraints)
                min_unit_intersection_constraints.extend(team_knowledge_constraints)
                # print('unit_intersection_constraints after update: ', min_unit_intersection_constraints)
                
                # if non-intersecting constraints are found, then calculate knowledge level
                if not check_for_non_intersecting_constraints(min_unit_intersection_constraints, weights, step_cost_flag)[0]:
                    # remove redundant constraints before calculating area
                    min_unit_intersection_constraints = BEC_helpers.remove_redundant_constraints(min_unit_intersection_constraints, weights, step_cost_flag)
                    min_unit_BEC_knowledge_intersection = np.array(BEC_helpers.calc_solid_angles([min_unit_intersection_constraints]))
                    min_unit_BEC_knowledge_union = min_unit_area + knowledge_area - min_unit_BEC_knowledge_intersection

                    # print('min_unit_intersection_constraints: ', min_unit_intersection_constraints)
                    # print('min_unit_BEC_knowledge_intersection: ', min_unit_BEC_knowledge_intersection)
                    # print('min_unit_BEC_knowledge_union: ', min_unit_BEC_knowledge_union)

                    # check if the knowledge area is a subset of the BEC area
                    if np.abs(min_unit_BEC_knowledge_intersection - knowledge_area) < 0.001:
                        print('Knowledge area is a subset of the BEC area!..')
                        knowledge_level[knowledge_type] = [np.array(1.0)]
                    else:
                        kl = min_unit_BEC_knowledge_intersection/min_unit_BEC_knowledge_union
                        # knowledge_level[knowledge_type] = copy.deepcopy(kl)
                        knowledge_level[knowledge_type] = [np.array(min(1, max(0, kl)))]
                else:
                    knowledge_level[knowledge_type] = [np.array(0.0)]  # if constraints need to be relaxed/reset it would have happened before when particles were being reset. Constraints should not be reset unless the associated particles are reset!
            else:
                    knowledge_level[knowledge_type] = [np.array(0.0)] # if constraints need to be relaxed/reset it would have happened before when particles were being reset. Constraints should not be reset unless the associated particles are reset!

            
            # Method 4: An improved method for disjoint sets to see how close the disjoint is Generalized Intersection over Union (see https://giou.stanford.edu/)
            # TODO: Later

            # sanity check
            knowledge_not_zero_flag = False
            if knowledge_type == 'joint_knowledge' or knowledge_type == 'common_knowledge':
                for know_id, know_type in enumerate(team_knowledge):
                    if 'p' in know_type:
                        if knowledge_level[know_type] !=0:
                            knowledge_not_zero_flag = True
                if not knowledge_not_zero_flag:
                    if knowledge_level[knowledge_type] != 0:
                        # raise RuntimeError('Knowledge level should be zero if there is no knowledge for any individual!')
                        # print(colored('Knowledge level should be zero if there is no knowledge for any individual!', 'red'))
                        knowledge_level[knowledge_type] =  [np.array(0.0)]
        
        if plot_flag:
            plot_text(team_knowledge_constraints, knowledge_level[knowledge_type], fig, plot_id)
            plot_id += 1
            plt.show()

        # # print(colored('Calculated knowledge level for: ', 'blue'), knowledge_type)
        # # print('min_unit_constraints: ', min_unit_constraints)
        # # print('knowledge constraints: ', team_knowledge_constraints)
        # # print('min_unit_intersection_constraints: ', min_unit_intersection_constraints)
        # # print('min unit area: ', min_unit_area)
        # # print('knowledge_area: ', knowledge_area)
        # # print('min_unit_BEC_knowledge_intersection: ', min_unit_BEC_knowledge_intersection)
        # # print('min_unit_BEC_knowledge_union: ', min_unit_BEC_knowledge_union)
        # # print('knowledge_level_unit: ', knowledge_level[knowledge_type])


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
            # for i in range(len(summary)):
                # print('Summary element :', i, ',', summary[i])
            
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
    
    # particles to consider while generating demos
    if demo_strategy =='individual_knowledge_low':
        ind_knowledge_ascending = find_ascending_individual_knowledge(team_knowledge, min_BEC_constraints, kc_id=0) # kc_id matters only when team members have a different prior
        knowledge_id = ind_knowledge_ascending[teammate_idx]
    
    elif demo_strategy == 'individual_knowledge_high':
        ind_knowledge_ascending = find_ascending_individual_knowledge(team_knowledge, min_BEC_constraints, kc_id=0)
        knowledge_id = ind_knowledge_ascending[len(ind_knowledge_ascending) - teammate_idx - 1]
    
    elif demo_strategy == 'common_knowledge' or demo_strategy == 'joint_knowledge':
        knowledge_id = demo_strategy
    
    # else:
        # print('Unsupported demo strategy for sampling particles!')

    particles = copy.deepcopy(team_particles[knowledge_id])

    
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
    





def obtain_team_summary(data_loc, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count,
                        n_train_demos, particles_demo, knowledge_id, variable_filter, nonzero_counter, BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, obj_func_proportion=1):


    summary_variant = 'counterfactual' # uses min_BEC_constraints_running to sample human models and could sample models from 
    # summary_variant = 'particle-filter'
    

    # # obtain demo summary        
    # # particle filters take a lot of time to converge. Avoid using this!
    # if summary_variant == 'particle-filter':
    #     BEC_summary, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = obtain_summary_particle_filter_team(data_loc, particles_demo, knowledge_id, BEC_summary, variable_filter, nonzero_counter, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count,
    #                    min_BEC_constraints_running, n_train_demos=np.inf, downsample_threshold=float("inf"), consider_human_models_jointly=True, c=0.001, obj_func_proportion=obj_func_proportion, min_info_gain=0.01, visited_env_traj_idxs = visited_env_traj_idxs)
    #     summary_count = len(BEC_summary)

    if summary_variant == 'counterfactual':
        BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = obtain_summary_counterfactual_team(data_loc, particles_demo, knowledge_id, variable_filter, nonzero_counter, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count,
                       BEC_summary, summary_count, min_BEC_constraints_running, params.BEC['n_human_models_precomputed'], visited_env_traj_idxs = visited_env_traj_idxs, n_train_demos=np.inf, downsample_threshold=float("inf"), consider_human_models_jointly=True, c=0.001, obj_func_proportion=obj_func_proportion)

        

    return BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo




def sample_team_pf(team_size, n_particles, weights, step_cost_flag, team_learning_factor=None, team_prior=None, pf_flag = 'teacher'):

    particles_team = {}
    
    # particles for individual team members
    for i in range(team_size):
        member_id = 'p' + str(i+1)

        if team_learning_factor is not None:
            learning_factor = team_learning_factor[i]
        else:
            learning_factor = params.default_learning_factor_teacher


        if pf_flag == 'learner':
            particles_team[member_id] = pf_team.Particles_team(BEC_helpers.sample_human_models_uniform([], n_particles), learning_factor)
        elif pf_flag == 'teacher':
            particles_team[member_id] = pf_team.Particles_team(BEC_helpers.sample_human_models_uniform([], n_particles), params.default_learning_factor_teacher)
        
        
        #### debug - save particles!
        # pf_particles = pd.DataFrame({'Pos': str(particles_team[member_id].positions)})
        # pf_particles.to_csv()

        # with open('data/simulation/sampling_tests/particles_set_4_' + member_id + '.csv', 'w', newline='') as csv_file:
        #     csv_writer = csv.writer(csv_file)
        #     csv_writer.writerow(['Position'])  # Write header
        #     csv_writer.writerows(particles_team[member_id].positions)  # Write data rows
            # for i in range(len(particles_team[member_id].positions)):
            #     csv_writer.writerow(particles_team[member_id].positions[i])  # Write data rows
        #######################


        if team_prior is not None:
            # for cnst in team_prior[member_id]:
            # print('team_prior[member_id]: ', team_prior[member_id]) 
            plot_title = 'Teacher belief for player ' + str(member_id) + 'for prior'
            particles_team[member_id].update(team_prior[member_id][0], learning_factor = 1 , plot_title = plot_title, viz_flag = False) # team prior is in team knowledge format. Hence use kc_id = 0 to get prior

                # visualize_transition(cnst, particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Simulated knowledge change for ' + str(member_id) )
            
            particles_team[member_id].knowledge_update(team_prior[member_id])


    if pf_flag == 'teacher':
        # particles for aggregated team knowledge - common knowledge
        particles_team['common_knowledge'] = pf_team.Particles_team(BEC_helpers.sample_human_models_uniform([], n_particles), params.default_learning_factor_teacher)
        if team_prior is not None:
            team_prior['common_knowledge'] = [calc_common_knowledge(team_prior, team_size, weights, step_cost_flag)]
            plot_title = 'Teacher belief for common knowledge for prior'
            particles_team['common_knowledge'].update(team_prior['common_knowledge'][0], plot_title=plot_title, viz_flag = False)  # team prior is in team knowledge format. Hence use kc_id = 0 to get prior
            particles_team['common_knowledge'].knowledge_update(team_prior['common_knowledge'])
        

        # particles for aggregated team knowledge - joint knowledge (both methods should produce similar particles; check and if they are similar method 1 is more streamlined)
        # method 1
        particles_team['joint_knowledge'] = pf_team.Particles_team(BEC_helpers.sample_human_models_uniform([], n_particles), params.default_learning_factor_teacher)
        if team_prior is not None:
            team_prior['joint_knowledge'] = [calc_joint_knowledge(team_prior, team_size)]
            particles_team['joint_knowledge'].update_jk(team_prior['joint_knowledge'])
            particles_team['joint_knowledge'].knowledge_update(team_prior['joint_knowledge'])
    

        # # method 2
        # if team_prior is not None:
        #     team_prior['joint_knowledge'] = calc_joint_knowledge(team_prior, team_size, weights, step_cost_flag)
        #     particle_positions = sample_human_models_uniform_joint_knowledge(team_prior['joint_knowledge'], n_particles)
        #     particles_team['joint_knowledge_2'] = pf_team.Particles_team(particle_positions)
        
        return team_prior, particles_team
    
    else:
        return particles_team




def show_demonstrations(unit, particles_demo, mdp_class, weights, loop_count, viz_flag=False):
    # TODO: WIP
    
    # # print('Summary:', summary)
    
    # # print("Here are the demonstrations for this unit")
    # # print('Unit to demonstrate:', unit)

    unit_constraints = []
    demo_ids = []
    running_variable_filter_unit = unit[0][4]

    n = 1
    # show each demonstration that is part of this unit       
    for subunit in unit:
        if viz_flag:
            # print('Visualzing trajectory...')
            subunit[0].visualize_trajectory(subunit[1])
        unit_constraints.extend(subunit[3])
        demo_ids.append(subunit[2])
        
        # debug
        # # print('Constraint ', n, 'for this unit: ', subunit[3])
        
        # update particle filter with demonstration's constraint (Being done while generating the summary, so commented out here)
        # particles_demo.update(subunit[3])
        

        n += 1

    return unit_constraints, demo_ids, running_variable_filter_unit



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
        # print(colored("Was unable to sample valid points for visualizing the BEC (which is likely too small).", 'red'))
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



def visualize_transition(constraints, particles, mdp_class, weights=None, fig=None, text=None, knowledge_type = 'common_knowledge', plot_filename = 'transition'):

    # From BEC_viz.visualize_pf_transition function
   
    '''
    Visualize the change in particle filter due to constraints
    '''
    if fig == None:
        fig = plt.figure()

    def label_axes(ax, mdp_class, weights=None, view_params = None):
        fs = 12
        ax.set_facecolor('white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        if weights is not None:
            ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='r', s=100/2)
        if mdp_class == 'augmented_taxi2':
            ax.set_xlabel(r'$\mathregular{w_0}$: Mud', fontsize = fs)
            ax.set_ylabel(r'$\mathregular{w_1}$: Recharge', fontsize = fs)
        elif mdp_class == 'colored_tiles':
            ax.set_xlabel('X: Tile A (brown)')
            ax.set_ylabel('Y: Tile B (green)')
        else:
            ax.set_xlabel('X: Goal')
            ax.set_ylabel('Y: Skateboard')
        ax.set_zlabel('$\mathregular{w_2}$: Action', fontsize = fs)

        # ax.view_init(elev=16, azim=-160)
        if not view_params is None:
            elev = view_params[0]
            azim = view_params[1]
        else:
            elev=16
            azim = -160
        ax.view_init(elev=elev, azim=azim)
    

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d', sharex=ax1, sharey=ax1, sharez=ax1)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d', sharex=ax1, sharey=ax1, sharez=ax1)
    ax1.title.set_text('Particles before Demonstration')
    ax2.title.set_text('Constraint corresponding to demonstration')
    ax3.title.set_text('Particles after demonstration')

    fig.suptitle(text, fontsize=30)

    # cluster particles
    # particles.cluster()
    # print(colored('Cluster centers: ' + str(particles.cluster_centers) + '. Cluster weights: ' + str(particles.cluster_weights), 'blue'))
    
    # plot particles before and after the constraints
    particles.plot(fig=fig, ax=ax1, plot_prev=True)
    particles.plot(fig=fig, ax=ax3)

    ## plot the constraints
    # for constraints in [constraints]:
    #     BEC_viz.visualize_planes(constraints, fig=fig, ax=ax2)
    # for constraints in [constraints]:
    #     BEC_viz.visualize_planes(constraints, fig=fig, ax=ax1)
    # for constraints in [constraints]:
    #     BEC_viz.visualize_planes(constraints, fig=fig, ax=ax3)

    if len(constraints) > 0:
        for ax_n in [ax1, ax2, ax3]:
            if knowledge_type == 'joint_knowledge':
                for constraint in constraints:
                    BEC_viz.visualize_planes(constraint, fig=fig, ax=ax_n)
            else:
                if len(constraints) > 1:
                    BEC_viz.visualize_planes(constraints, fig=fig, ax=ax_n)
                    if constraints[0][0][0] == 0:
                        view_params = [16, -160]
                    elif constraints[0][0][1] == 0:
                        view_params = [2, -100]
                    elif constraints[0][0][2] == 0:
                        view_params = [2, -60]
                    else:
                        view_params = [16, -160]
                else:
                    BEC_viz.visualize_planes([constraints], fig=fig, ax=ax_n)
                    if constraints[0][0] == 0:
                        view_params = [16, -160]
                    elif constraints[0][1] == 0:
                        view_params = [2, -100]
                    elif constraints[0][2] == 0:
                        view_params = [2, -60]
                    else:
                        view_params = [16, -160]

    
    # plot the spherical polygon corresponding to the constraints
    if len(constraints) > 0:
        if knowledge_type == 'joint_knowledge':
            for ind_constraints in constraints:
                ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(ind_constraints)
                poly = Polyhedron.Polyhedron(ieqs=ieqs)
                BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax2, plot_ref_sphere=False, alpha=0.75)
                # print('Plotting joint knowledge spherical polygon..')
        else:
            ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints)
            poly = Polyhedron.Polyhedron(ieqs=ieqs)
            BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax2, plot_ref_sphere=False, alpha=0.75)

    label_axes(ax1, mdp_class, weights, view_params = view_params)
    label_axes(ax2, mdp_class, weights, view_params = view_params)
    label_axes(ax3, mdp_class, weights, view_params = view_params)

    # New: Add what constraints are being shown in the demo to the plot
    if len(constraints) > 0:
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
    # fig.savefig('plots/' + plot_filename +'.png', dpi=300)
    # plt.pause(10)
    # plt.close()



def visualize_team_knowledge_constraints(team_knowledge, weights, step_cost_flag, particles_team_teacher = None, kc_id = None, fig=None, text=None, plot_min_constraints_flag = False, plot_text_flag = False, min_unit_constraints = [], plot_filename = 'team_knowledge_constraints', fig_title = None):

    ###########  functions ###################
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


    def plot_text(constraints, fig, i):
        x_loc = [0.05, 0.22, 0.40, 0.58, 0.8]
        y_loc = 0.2
        fig.text(x_loc[i], y_loc, 'Knowledge constraints: ', fontsize=12)
        y_loc -= 0.02
        for cnst in constraints:
            fig.text(x_loc[i], y_loc, s=str(cnst), fontsize=12)
            y_loc -= 0.02


    # https://stackoverflow.com/questions/41167196/using-matplotlib-3d-axes-how-to-drag-two-axes-at-once
    # link the pan of the three axes together
    def on_move(event):

        if event.inaxes == ax0:
            ax1.view_init(elev=ax0.elev, azim=ax0.azim)
            ax2.view_init(elev=ax0.elev, azim=ax0.azim)
            # ax3.view_init(elev=ax0.elev, azim=ax0.azim)
            # ax4.view_init(elev=ax0.elev, azim=ax0.azim)
        elif event.inaxes == ax1:
            ax0.view_init(elev=ax1.elev, azim=ax1.azim)
            ax2.view_init(elev=ax1.elev, azim=ax1.azim)
            # ax3.view_init(elev=ax1.elev, azim=ax1.azim)
            # ax4.view_init(elev=ax1.elev, azim=ax1.azim)
        elif event.inaxes == ax2:
            ax0.view_init(elev=ax2.elev, azim=ax2.azim)
            ax1.view_init(elev=ax2.elev, azim=ax2.azim)
            # ax3.view_init(elev=ax2.elev, azim=ax2.azim)
            # ax4.view_init(elev=ax2.elev, azim=ax2.azim)
        elif event.inaxes == ax3:
            ax0.view_init(elev=ax3.elev, azim=ax3.azim)
            ax1.view_init(elev=ax3.elev, azim=ax3.azim)
            ax2.view_init(elev=ax3.elev, azim=ax3.azim)
            ax4.view_init(elev=ax3.elev, azim=ax3.azim)
        elif event.inaxes == ax4:
            ax0.view_init(elev=ax4.elev, azim=ax4.azim)
            ax1.view_init(elev=ax4.elev, azim=ax4.azim)
            ax2.view_init(elev=ax4.elev, azim=ax4.azim)
            ax3.view_init(elev=ax4.elev, azim=ax4.azim)
            return
        
        fig.canvas.draw_idle()
    
    ###########  functions ###################


    # print(colored('Plotting team knowledge constraints..', 'blue'))
    
    if fig == None:
        fig = plt.figure()
        ax0 = fig.add_subplot(1, 5, 1, projection='3d')
        ax1 = fig.add_subplot(1, 5, 2, projection='3d')
        ax2 = fig.add_subplot(1, 5, 3, projection='3d')
        ax3 = fig.add_subplot(1, 5, 4, projection='3d')
        ax4 = fig.add_subplot(1, 5, 5, projection='3d')
        fig.suptitle(fig_title, fontsize=16)

    if kc_id is None:
        kc_id_list = range(len(team_knowledge['p1']))
    else:
        kc_id_list = [kc_id]
    

    plot_id = 0
    for knowledge_id, knowledge_type  in enumerate(team_knowledge):
        
        # # plot constraints - old
        # plot_constraints = []
        # first_index = True

        # for kc_index in kc_id_list:
        #     if knowledge_type == 'joint_knowledge':
        #         for i in range(len(team_knowledge[knowledge_type][kc_index])):
        #             if first_index:
        #                 plot_constraints.append(copy.deepcopy(team_knowledge[knowledge_type][kc_index][i]))
        #             else:
        #                 plot_constraints[i].extend(team_knowledge[knowledge_type][kc_index][i])
        #         first_index = False
        #     else:
        #         if len(plot_constraints) == 0:
        #             plot_constraints = copy.deepcopy(team_knowledge[knowledge_type][kc_index])
        #         else:
        #             plot_constraints.extend(team_knowledge[knowledge_type][kc_index])


        # plot constraints - updated
        plot_constraints = []
        if knowledge_type == 'joint_knowledge':
            # for each player
            for mem_id in range(len(team_knowledge[knowledge_type][kc_id_list[0]])):
                plot_constraints.append(copy.deepcopy(team_knowledge[knowledge_type][kc_id_list[0]][mem_id]))

                for rem_kc_id in kc_id_list[1:]:
                    plot_constraints[mem_id].extend(team_knowledge[knowledge_type][rem_kc_id][mem_id])

        else:
            for kc_index in kc_id_list:
                # print('team_knowledge[knowledge_type][kc_index]: ', team_knowledge[knowledge_type][kc_index])
                if len(plot_constraints) == 0:
                    plot_constraints = copy.deepcopy(team_knowledge[knowledge_type][kc_index])
                else:
                    plot_constraints.extend(team_knowledge[knowledge_type][kc_index])
        

        # choose plot axes
        if knowledge_type == 'common_knowledge':
            plot_ax = ax3
        elif knowledge_type == 'joint_knowledge':
            plot_ax = ax4
        elif knowledge_type == 'p1':
            plot_ax = ax0
        elif knowledge_type == 'p2':
            plot_ax = ax1
        elif knowledge_type == 'p3':
            plot_ax = ax2

        # plot knowledge constraints
        if knowledge_type != 'joint_knowledge':
            # print('plot_constraints: ', plot_constraints)
            # print('weights: ', weights)

            # check for non intersecting constraints
            # opposing_constraints_flag, _, opposing_idx = check_opposing_constraints(plot_constraints)
            # if opposing_constraints_flag:
            #     plot_constraints = majority_rules_opposing_team_constraints(opposing_idx, plot_constraints, [])
            #     print(colored('Majority rules for opposing constraints plotting knowledge constraints of ' + str(knowledge_type), 'red'))

            non_intersecting_constraints_flag, _ = check_for_non_intersecting_constraints(plot_constraints, weights, step_cost_flag)
            if non_intersecting_constraints_flag:
                plot_constraints, _ = majority_rules_non_intersecting_team_constraints(plot_constraints, weights, step_cost_flag)
                print(colored('Majority rules for non intersecting constraints plotting knowledge constraints of ' + str(knowledge_type), 'red'))
                # print('plot_constraints after majority rules: ', plot_constraints)
            
            # plot unit constraints
            utils_teams.visualize_planes_team(plot_constraints, fig=fig, ax=plot_ax)
            ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(plot_constraints)
            poly = Polyhedron.Polyhedron(ieqs=ieqs)
            BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=plot_ax, plot_ref_sphere=False, alpha=0.75)

        else:

            for i in range(len(plot_constraints)):
                cnsts = plot_constraints[i]
                # print('Plotting joint constraints: ', cnsts)

                # check for non opposing and non-intersecting constraints
                # opposing_constraints_flag, _, opposing_idx = check_opposing_constraints(cnsts)
                # if opposing_constraints_flag:
                #     cnsts = majority_rules_opposing_team_constraints(opposing_idx, cnsts, [])
                #     print(colored('Majority rules for opposing constraints plotting knowledge constraints of ' + str(knowledge_type), 'red'))

                non_intersecting_constraints_flag, _ = check_for_non_intersecting_constraints(cnsts, weights, step_cost_flag)
                if non_intersecting_constraints_flag:
                    cnsts, _ = majority_rules_non_intersecting_team_constraints(cnsts, weights, step_cost_flag)
                    print(colored('Majority rules for non intersecting constraints plotting knowledge constraints of ' + str(knowledge_type), 'red'))
                    # print('cnsts after majority rules: ', cnsts)

                cnsts = BEC_helpers.remove_redundant_constraints(cnsts, weights, step_cost_flag)
                utils_teams.visualize_planes_team(cnsts, fig=fig, ax=plot_ax)
                ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(cnsts)
                poly = Polyhedron.Polyhedron(ieqs=ieqs)
                BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=plot_ax, plot_ref_sphere=False, alpha=0.75)

            plot_ax.set_title(knowledge_type)

            if plot_text_flag:
                plot_text(plot_constraints, fig, knowledge_id)

        # plot min constraints
        if plot_min_constraints_flag:
            # print('Plotting knowledge constraints..')
            utils_teams.visualize_planes_team(min_unit_constraints, fig=fig, ax=plot_ax, alpha=0.2)
            ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(min_unit_constraints)
            poly = Polyhedron.Polyhedron(ieqs=ieqs)
            BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=plot_ax, plot_ref_sphere=False, color = 'b')

        # plot particles
        if particles_team_teacher is not None:
            particles_team_teacher[knowledge_type].plot(fig=fig, ax=plot_ax)

        # plot text
        plot_ax.set_title(knowledge_type)
        plot_text(plot_constraints, fig, plot_id)

        plot_id += 1



    fig.canvas.mpl_connect('motion_notify_event', on_move)

    # plt.show()
    # fig.savefig('plots/' + plot_filename +'.png', dpi=300)
    # plt.pause(10)
    # plt.close()

    return fig




def visualize_BEC_area(BEC_constraints, fig, ax1):

    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    
    # # plot BEC constraints
    # for constraints in [BEC_constraints]:
    #     BEC_viz.visualize_planes(constraints, fig=fig, ax=ax1)

    # plot BEC area
    ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(BEC_constraints)
    poly = Polyhedron.Polyhedron(ieqs=ieqs)
    BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax1, plot_ref_sphere=False, alpha=0.75, color='b')

    # plt.show()



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



def obtain_summary_counterfactual_team(data_loc, particles_demo, member_id, variable_filter, nonzero_counter, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count,
                       summary, summary_count, min_BEC_constraints_running, n_human_models_precomputed, visited_env_traj_idxs=[], n_train_demos=3, downsample_threshold=float("inf"), consider_human_models_jointly=True, c=0.001, obj_func_proportion=1):

    
    # unit = []

    # # impose prior
    # min_BEC_constraints_running = copy.deepcopy(prior)

    # # count how many nonzero constraints are present for each reward weight (i.e. variable) in the minimum BEC constraints
    # # (which are obtained using one-step deviations). mask variables in order of fewest nonzero constraints for variable scaffolding
    # # rationale: the variable with the most nonzero constraints, often the step cost, serves as a good reference/ratio variable
    # min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record for item in sublist]
    # min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record_flattened for item in sublist]
    # min_subset_constraints_record_array = np.array(min_subset_constraints_record_flattened)

    # # for variable scaffolding
    # nonzero_counter = (min_subset_constraints_record_array != 0).astype(float)
    # nonzero_counter = np.sum(nonzero_counter, axis=0)
    # nonzero_counter = nonzero_counter.flatten()

    # variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
    # running_variable_filter = copy.deepcopy(variable_filter)
    # # # print('variable filter: {}'.format(variable_filter))

    # # clear the demonstration generation log
    # open('models/' + data_loc + '/demo_gen_log.txt', 'w').close()


    # sample_human_models_ref = BEC_helpers.sample_human_models_uniform([], n_human_models_precomputed)


    # while summary_count < n_train_demos:
        

        # ################ Computing counterfactual constraints in real-time  ###############################

        # # visualize_constraints(min_BEC_constraints_running, weights, step_cost_flag, fig_name=str(summary_count) + '.png', just_save=True)

        # # (approximately) uniformly divide up the valid BEC area along 2-sphere
        # sample_human_models = BEC_helpers.sample_human_models_uniform(min_BEC_constraints_running, n_human_models)

        # if len(sample_human_models) == 0:
        #     # print(colored("Likely cannot reduce the BEC further through additional demonstrations. Returning.", 'red'))
        #     return summary, summary_count, visited_env_traj_idxs, min_BEC_constraints_running, particles_demo

        # info_gains_record = []
        # overlap_in_opt_and_counterfactual_traj_record = []

        # # print("Length of summary: {}".format(summary_count))
        # with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
        #     myfile.write('Length of summary: {}\n'.format(summary_count))
        
        # for model_idx, human_model in enumerate(sample_human_models):
        #     # print(colored('Model #: {}'.format(model_idx), 'red'))
        #     # print(colored('Model val: {}'.format(human_model), 'red'))

        #     with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
        #         myfile.write('Model #: {}\n'.format(model_idx))
        #         myfile.write('Model val: {}\n'.format(human_model))

        #     # based on the human's current model, obtain the information gain generated when comparing to the agent's
        #     # optimal trajectories in each environment (human's corresponding optimal trajectories and constraints
        #     # are saved for reference later)
        #     # print("Obtaining counterfactual information gains:")

        #     cf_data_dir = 'models/' + data_loc + '/counterfactual_data_' + str(summary_count) + '/model' + str(model_idx)
        #     os.makedirs(cf_data_dir, exist_ok=True)
        #     if consider_human_models_jointly:
        #         args = [(data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]), traj_record[i], None, min_BEC_constraints_running, step_cost_flag, summary_count, variable_filter, mdp_features_record[i], consider_human_models_jointly) for i in range(len(traj_record))]
        #         info_gain_envs = list(tqdm(pool.imap(BEC.compute_counterfactuals, args), total=len(args)))

        #         info_gains_record.append(info_gain_envs)
        #     else:
        #         args = [(data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]), traj_record[i], None, min_BEC_constraints_running, step_cost_flag, summary_count, variable_filter, mdp_features_record[i], consider_human_models_jointly) for i in range(len(traj_record))]
        #         info_gain_envs, overlap_in_opt_and_counterfactual_traj_env = zip(*pool.imap(BEC.compute_counterfactuals, tqdm(args), total=len(args)))

        #         info_gains_record.append(info_gain_envs)
        #         overlap_in_opt_and_counterfactual_traj_record.append(overlap_in_opt_and_counterfactual_traj_env)

        # with open('models/' + data_loc + '/info_gains_' + str(summary_count) + '.pickle', 'wb') as f:
        #     pickle.dump(info_gains_record, f)


        # # do a quick check of whether there's any information to be gained from any of the trajectories
        # no_info_flag = False
        # max_info_gain = 1
        # if consistent_state_count:
        #     info_gains = np.array(info_gains_record)
        #     if np.sum(info_gains > 1) == 0:
        #         no_info_flag = True
        # else:
        #     info_gains_flattened_across_models = list(itertools.chain.from_iterable(info_gains_record))
        #     info_gains_flattened_across_envs = list(itertools.chain.from_iterable(info_gains_flattened_across_models))
        #     if sum(np.array(info_gains_flattened_across_envs) > 1) == 0:
        #         no_info_flag = True

        
        # if no_info_flag:
        #     # print(colored('Did not find any more informative demonstrations. Moving onto next unit, if applicable.', 'red'))
        #     if len(unit) > 0:
        #         summary.append(unit)
            
        #     return summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo
        

        # # print("Combining the most limiting constraints across human models:")
        # args = [(i, range(len(sample_human_models)), data_loc, summary_count, weights, step_cost_flag, variable_filter, mdp_features_record[i],
        #             traj_record[i], min_BEC_constraints_running, None, True, True) for
        #         i in range(len(traj_record))]
        # info_gains_record, min_env_constraints_record, n_diff_constraints_record, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = zip(
        #     *pool.imap(BEC.combine_limiting_constraints_IG, tqdm(args)))

        
        ####################################################################################################


    ################ Using precomputed counterfactual constraints  ###############################
    unit = []
    # if len(min_BEC_constraints_running) == 0:
    #     min_BEC_constraints_running = copy.deepcopy(params.prior) # impose prior

    # Sample counterfactual models not with minimum BEC constraints but from the individual knowledge (from their constraints)
    constraint_space_to_sample_human_models = []
    
    kc_id_list = range(len(particles_demo.knowledge_constraints))
    first_index = True
    for kc_index in kc_id_list:
        if member_id == 'joint_knowledge':
            # print('particles_demo.knowledge_constraints[kc_index] ', particles_demo.knowledge_constraints[kc_index])
            for i in range(len(particles_demo.knowledge_constraints[kc_index])):
                if first_index:
                    constraint_space_to_sample_human_models.append(utils_teams.flatten_list(copy.deepcopy(particles_demo.knowledge_constraints[kc_index][i])))
                    # # print(colored('team_knowledge_constraints flattened: ' + str(team_knowledge_constraints), 'blue' ))
                else:
                    constraint_space_to_sample_human_models[i].extend(particles_demo.knowledge_constraints[kc_index][i])
            first_index = False
        else:
            # print('particles_demo.knowledge_constraints[kc_index] ', particles_demo.knowledge_constraints[kc_index])
            if len(constraint_space_to_sample_human_models) == 0:
                constraint_space_to_sample_human_models = copy.deepcopy(particles_demo.knowledge_constraints[kc_index])
            else:
                constraint_space_to_sample_human_models.extend(particles_demo.knowledge_constraints[kc_index])


    # print(colored('constraint_space_to_sample_human_models:', 'red'), constraint_space_to_sample_human_models)

    # choose the maximum subset of constraints in case there is no common region among them
    if member_id == 'joint_knowledge':
        constraint_space_to_sample_human_models_majority = []
        for ind_cnst_space in constraint_space_to_sample_human_models:
            if check_for_non_intersecting_constraints(ind_cnst_space, weights, step_cost_flag)[0]:
                ind_cnst_space = majority_rules_non_intersecting_team_constraints(ind_cnst_space, weights, step_cost_flag)[0]
            constraint_space_to_sample_human_models_majority.append(ind_cnst_space)
        
        constraint_space_to_sample_human_models = copy.deepcopy(constraint_space_to_sample_human_models_majority)
    else:
        if check_for_non_intersecting_constraints(constraint_space_to_sample_human_models, weights, step_cost_flag)[0]:
            constraint_space_to_sample_human_models = majority_rules_non_intersecting_team_constraints(constraint_space_to_sample_human_models, weights, step_cost_flag)[0]
    
    print(colored('Initial knowledge constraints to sample human models from: ', 'blue')) 
    print(constraint_space_to_sample_human_models)
    
    running_variable_filter = copy.deepcopy(variable_filter)
    # print('variable filter: {}'.format(variable_filter))

    # clear the demonstration generation log
    open('models/' + data_loc + '/demo_gen_log.txt', 'w').close()

    sample_human_models_ref = BEC_helpers.sample_human_models_uniform([], n_human_models_precomputed)
    
    while summary_count < n_train_demos:
        
        # print('Knowledge constraints for sampling particles {}'.format(constraint_space_to_sample_human_models))
        # print('n_human_models: ', n_human_models)
        
        # # sample counterfactual human models; sample equally from each team member knowledge for joint knowledge case
        sample_human_models = []
        if member_id != 'joint_knowledge':
            sample_human_models = BEC_helpers.sample_human_models_uniform(constraint_space_to_sample_human_models, n_human_models)
        else:
            for i in range(len(constraint_space_to_sample_human_models)):
                # print('constraint_space_to_sample_human_models[i]: ', constraint_space_to_sample_human_models[i])
                if len(sample_human_models) == 0:
                    sample_human_models.extend(BEC_helpers.sample_human_models_uniform(constraint_space_to_sample_human_models[i], int(n_human_models/len(constraint_space_to_sample_human_models))))
                else:
                    sample_human_models.extend(BEC_helpers.sample_human_models_uniform(constraint_space_to_sample_human_models[i], int(n_human_models/len(constraint_space_to_sample_human_models))))

        
        # sample_human_models, model_weights = BEC_helpers.sample_human_models_pf(particles_demo, n_human_models)

        if len(sample_human_models) == 0:
            print(colored("Likely cannot reduce the BEC further through additional demonstrations. Returning.", 'red'))
            return summary, summary_count, visited_env_traj_idxs, min_BEC_constraints_running, particles_demo
        
        # print("Length of summary: {}".format(summary_count))
        with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
            myfile.write('Length of summary: {}\n'.format(summary_count))

        # print(colored('sample_human_models: ', 'red'), sample_human_models)

        # obtain the indices of the reference human models (that have precomputed constraints) that are closest to the sampled human models
        sample_human_models_ref_latllong = cg.cart2latlong(np.array(sample_human_models_ref).squeeze())
        sample_human_models_latlong = cg.cart2latlong(np.array(sample_human_models).squeeze())
        distances = haversine_distances(sample_human_models_latlong, sample_human_models_ref_latllong)
        min_model_idxs = np.argmin(distances, axis=1)

        # plot sampled human models and nearest particles
        if params.plot_sampled_models_flag:                    
            plot_sampled_models(particles_demo, min_BEC_constraints_running, sample_human_models, weights=weights, fig=None, text='Sampled human models based on ' + member_id)



        # # print('min_model_idxs: ', min_model_idxs)
        # # print('model distances:', distances[:, min_model_idxs])

        # # print("Combining the most limiting constraints across human models:")
        args = [(i, min_model_idxs, data_loc, 'precomputed', weights, step_cost_flag, variable_filter,
                    mdp_features_record[i],
                    traj_record[i], min_BEC_constraints_running, None, True, True) for
                i in range(len(traj_record))]
        info_gains_record, min_env_constraints_record, n_diff_constraints_record, _, _ = zip(
            *pool.imap(BEC.combine_limiting_constraints_IG, tqdm(args)))


        # # print('info_gains_record: ', info_gains_record)
        # # print('min_env_constraints_record: ', min_env_constraints_record)
        # # print('n_diff_constraints_record: ', n_diff_constraints_record)

        # do a quick check of whether there's any information to be gained from any of the trajectories
        no_info_flag = False
        if consistent_state_count:
            info_gains = np.array(info_gains_record)
            if np.sum(info_gains > 1) == 0:
                no_info_flag = True
        else:
            info_gains_flattened_across_envs_models = list(itertools.chain.from_iterable(info_gains_record))
            # # print('info_gains_record 2: ', info_gains_record)
            # # print('info_gains_flattened_across_models 2: ', info_gains_flattened_across_envs_models)
            if sum(np.array(info_gains_flattened_across_envs_models) > 1) == 0:
                no_info_flag = True

        if no_info_flag:
            print(colored('Did not find any more informative demonstrations. Moving onto next unit, if applicable.', 'red'))
            if len(unit) > 0:
                summary.append(unit)
            return summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo

        ####################################################################################################


        with open('models/' + data_loc + '/info_gains_joint' + str(summary_count) + '.pickle', 'wb') as f:
            pickle.dump(info_gains_record, f)

        differing_constraint_count = 1          # number of constraints in the running human model that would differ after showing a particular demonstration
        max_differing_constraint_count = max(list(itertools.chain(*n_diff_constraints_record)))
        # # print("max_differing_constraint_count: {}".format(max_differing_constraint_count))
        no_info_flag = True
        max_info_gain = 1

        # try to find a demonstration that will yield the fewest changes in the constraints defining the running human model while maximizing the information gain
        while no_info_flag and differing_constraint_count <= max_differing_constraint_count:
            # the possibility that no demonstration provides information gain must be checked for again,
            # in case all limiting constraints involve a masked variable and shouldn't be considered for demonstration yet
            if consistent_state_count:
                info_gains = np.array(info_gains_record)
                n_diff_constraints = np.array(n_diff_constraints_record)
                obj_function = info_gains

                # traj_overlap_pcts = np.array(overlap_in_opt_and_counterfactual_traj_avg)
                # obj_function = info_gains * (traj_overlap_pcts + c)  # objective 2: scaled
                

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
                best_env_idx = None
                best_traj_idx = None

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
                                    # # print("new max info: {}".format(max_info_gain))
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
                                    # # print("new max info: {}".format(max_info_gain))

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
                        # print("Ran into a conflict with a previously shown demonstration")
                        no_info_flag = True
                        differing_constraint_count += 1



        # # print("current max info: {}".format(max_info_gain))
        
        if no_info_flag:
            print(colored('Did not find any more informative demonstrations. Moving onto next unit too, if applicable.', 'red'))
            if len(unit) > 0:
                summary.append(unit)

            if params.plot_sampled_models_flag:
                plot_sampled_models(particles_demo, unit[3], sample_human_models, weights=weights, fig=None, text='Sampled human models based on ' + member_id)

            return summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo
        
        
        if best_env_idx is not None:
            with open('models/' + data_loc + '/best_env_idxs' + str(summary_count) + '.pickle', 'wb') as f:
                pickle.dump((best_env_idx, best_traj_idx, best_env_idxs, best_traj_idxs), f)

            best_traj = traj_record[best_env_idx][best_traj_idx]

            filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
            with open(filename, 'rb') as f:
                wt_vi_traj_env = pickle.load(f)
            best_mdp = wt_vi_traj_env[0][1].mdp
            best_mdp.set_init_state(best_traj[0][0]) # for completeness
            new_constraint = min_env_constraints_record[best_env_idx][best_traj_idx]
            min_BEC_constraints_running.extend(new_constraint)
            min_BEC_constraints_running = BEC_helpers.remove_redundant_constraints(min_BEC_constraints_running, weights, step_cost_flag)
            
            # print('Updating particles with constraint: ', new_constraint)
            plot_title = 'Demo particles update after demonstration summary ' + str(summary_count)
            particles_demo.update(new_constraint, plot_title = plot_title)

            
            # added newly (12/1/23) to separate sampling space and demo info space
            # print('constraint_space_to_sample_human_models: ', constraint_space_to_sample_human_models)
            constraint_space_to_sample_human_models.extend(new_constraint)
            constraint_space_to_sample_human_models = BEC_helpers.remove_redundant_constraints(min_BEC_constraints_running, weights, step_cost_flag)
            
            visited_env_traj_idxs.append((best_env_idx, best_traj_idx))
        
            # This is just an additionla check. Variable filter should not be changing within this function
            if (running_variable_filter == variable_filter).all():
                unit.append([best_mdp, best_traj, (best_env_idx, best_traj_idx), new_constraint, variable_filter, sample_human_models])
                summary_count += 1
            
        

        # # print(colored('Max infogain: {}'.format(max_info_gain), 'blue'))
        with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
            myfile.write('Max infogain: {}\n'.format(max_info_gain))
            myfile.write('\n')

        # save the summary along the way (for each completed unit)
        with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
            pickle.dump((summary, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo), f)

    # add any remaining demonstrations
    if len(unit) > 0:
        summary.append(unit)

    # this method doesn't always finish, so save the summary along the way (for each completed unit)
    with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
        pickle.dump((summary, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo), f)

    return summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo






def check_unit_learning_goal_reached(team_knowledge, min_unit_constraints, kc_id):

    print(colored('Checking if unit goal is reached...', 'blue'))

    check_unit_learning_goal_reached = False
    
    knowledge_level = calc_knowledge_level(team_knowledge, min_unit_constraints, kc_id_list=[kc_id])
    
    # if knowledge_level['common_knowledge'] > 0.7:
    #     unit_learning_goal_reached_flag = True
    knowledge_types = list(team_knowledge.keys())

    unit_learning_goal_reached_flag = True

    # print('knowledge_level: ', knowledge_level)

    for ind_knowledge in knowledge_types:
        if 'p' in ind_knowledge:
            # print('ind_knowledge[kc_id][0]: ', type(np.array(knowledge_level[ind_knowledge][0])))
            # print('params.learning_goal: ', type(params.learning_goal))
            if np.array(knowledge_level[ind_knowledge][0]).astype(float) < params.learning_goal:
                unit_learning_goal_reached_flag = False
    

    return unit_learning_goal_reached_flag






def compute_counterfactuals_team(args):
    data_loc, model_idx, env_idx, w_human_normalized, env_filename, trajs_opt, particles, min_BEC_constraints_running, step_cost_flag, summary_len, variable_filter, mdp_features, consider_human_models_jointly = args

    skip_env = False

    # if the mdp has a feature that is meant to be filtered out, then skip this environment
    if variable_filter.dot(mdp_features.T) > 0:
        skip_env = True

    if not skip_env:
        with open(env_filename, 'rb') as f:
            wt_vi_traj_env = pickle.load(f)

        agent = wt_vi_traj_env[0][1]
        weights = agent.mdp.weights

        human = copy.deepcopy(agent)
        mdp = human.mdp
        mdp.weights = w_human_normalized
        vi_human = ValueIteration(mdp, sample_rate=1)
        vi_human.run_vi()

        if not vi_human.stabilized:
            skip_env = True
            # print(colored('Skipping environment' + str(env_idx) + '...', 'red'))

    if not skip_env:
        # only consider counterfactual trajectories from human models whose value iteration have converged and whose
        # mdp does not have a feature that is meant to be filtered out
        # best_human_trajs_record_env = []
        constraints_env = []
        info_gain_env = []
        # human_rewards_env = []
        overlap_in_opt_and_counterfactual_traj_env = []

        for traj_idx, traj_opt in enumerate(trajs_opt):
            constraints = []

            # # a) accumulate the reward features and generate a single constraint
            # mu_sa = mdp.accumulate_reward_features(traj_opt, discount=True)
            # traj_hyp = mdp_helpers.rollout_policy(vi_human.mdp, vi_human)
            # mu_sb = vi_human.mdp.accumulate_reward_features(traj_hyp, discount=True)
            # constraints.append(mu_sa - mu_sb)
            # best_human_trajs_record = [] # for symmetry with below
            # best_human_reward = weights.dot(mu_sb.T)  # for symmetry with below

            # b) contrast differing expected feature counts for each state-action pair along the agent's optimal trajectory
            # best_human_trajs_record = []
            # best_human_reward = 0
            for sas_idx in range(len(traj_opt)):
                # reward features of optimal action
                mu_sa = mdp.accumulate_reward_features(traj_opt[sas_idx:], discount=True)

                sas = traj_opt[sas_idx]
                cur_state = sas[0]

                # obtain all optimal trajectory rollouts according to the human's model (assuming that it's a reasonable policy that has converged)
                human_opt_trajs = mdp_helpers.rollout_policy_recursive(vi_human.mdp, vi_human, cur_state, [])

                cur_best_reward = float('-inf')
                best_reward_features = []
                best_human_traj = []
                # select the human's possible trajectory that has the highest true reward (i.e. give the human's policy the benefit of the doubt)
                for traj in human_opt_trajs:
                    mu_sb = mdp.accumulate_reward_features(traj, discount=True)  # the human and agent should be working with identical mdps
                    reward_hyp = weights.dot(mu_sb.T)
                    if reward_hyp > cur_best_reward:
                        cur_best_reward = reward_hyp
                        best_reward_features = mu_sb
                        best_human_traj = traj

                # # todo: for testing how much computation and time I save by not doing a recursive rollout
                # best_human_traj = mdp_helpers.rollout_policy(vi_human.mdp, vi_human, cur_state, [])
                # best_reward_features = mdp.accumulate_reward_features(best_human_traj,
                #                                        discount=True)  # the human and agent should be working with identical mdps
                # cur_best_reward = weights.dot(best_reward_features.T)

                # only store the reward of the full trajectory
                # if sas_idx == 0:
                #     best_human_reward = cur_best_reward
                #     traj_opt_feature_count = mu_sa
                constraints.append(mu_sa - best_reward_features)
                # best_human_trajs_record.append(best_human_traj)

            if len(constraints) > 0:
                constraints = BEC_helpers.remove_redundant_constraints(constraints, weights, step_cost_flag)

            if particles is not None:
                info_gain = particles.calc_info_gain(constraints)
            else:
                info_gain = BEC_helpers.calculate_information_gain(min_BEC_constraints_running, constraints,
                                                                   weights, step_cost_flag)

            # human_rewards_env.append(best_human_reward)
            # best_human_trajs_record_env.append(best_human_trajs_record)
            constraints_env.append(constraints)
            info_gain_env.append(info_gain)

            # if not consider_human_models_jointly:
            #     # you should only consider the overlap for the first counterfactual human trajectory (as opposed to
            #     # counterfactual trajectories that could've arisen from states after the first state)
            #     overlap_pct = BEC_helpers.calculate_counterfactual_overlap_pct(best_human_trajs_record[0], traj_opt)
            #
            #     overlap_in_opt_and_counterfactual_traj_env.append(overlap_pct)

    # else just populate with dummy variables
    else:
        # best_human_trajs_record_env = [[[]] for i in range(len(trajs_opt))]
        constraints_env = [[] for i in range(len(trajs_opt))]
        info_gain_env = [0 for i in range(len(trajs_opt))]
        if not consider_human_models_jointly:
            overlap_in_opt_and_counterfactual_traj_env = [float('inf') for i in range(len(trajs_opt))]
        # human_rewards_env = [np.array([[0]]) for i in range(len(trajs_opt))]

    if summary_len is not None:
        with open('models/' + data_loc + '/counterfactual_data_' + str(summary_len) + '/model' + str(model_idx) +
                  '/cf_data_env' + str(env_idx).zfill(5) + '.pickle', 'wb') as f:
            pickle.dump(constraints_env, f)
        # # print('Info gain for Summary length ' + str(summary_len) + ' of model ' + str(model_idx) + 'of env' + str(env_idx) + 'is' + str(info_gain_env) + '...')

    if consider_human_models_jointly:
        return info_gain_env
    else:
        return info_gain_env, overlap_in_opt_and_counterfactual_traj_env



def remove_redundant_constraints_team(constraints, weights, step_cost_flag):
    '''
    Summary: Remove redundant constraints
    '''
    if step_cost_flag:
        # Remove redundant constraint that do not change the underlying intersection between the BEC region and the
        # L1 constraints
        try:
            BEC_length_all_constraints, nonredundant_constraint_idxs = BEC_helpers.calculate_BEC_length(constraints, weights,
                                                                                            step_cost_flag)
            # print('')
        except:
            # a subset of these constraints aren't numerically stable (e.g. you can have a constraint that's ever so slightly
            # over the ground truth reward weight and thus fail to yield a proper polygonal convex hull. remove the violating constraints
            A, b = BEC_helpers.constraints_to_halfspace_matrix(constraints, weights, step_cost_flag)
            violating_idxs = BEC_helpers.indicate_violating_constraints(A, b)

            for violating_idx in sorted(violating_idxs[0], reverse=True):
                del constraints[violating_idx]

            BEC_length_all_constraints, nonredundant_constraint_idxs = BEC_helpers.calculate_BEC_length(constraints, weights,
                                                                                            step_cost_flag)
            

        nonredundant_constraints = [constraints[x] for x in nonredundant_constraint_idxs]

        for query_idx, query_constraint in enumerate(constraints):
            if query_idx not in nonredundant_constraint_idxs:
                pass
            else:
                # see if this is truly non-redundant or crosses an L1 constraint exactly where another constraint does
                constraints_other = []
                for constraint_idx, constraint in enumerate(nonredundant_constraints):
                    if not BEC_helpers.equal_constraints(query_constraint, constraint):
                        constraints_other.append(constraint)
                if len(constraints_other) > 0:
                    BEC_length = BEC_helpers.calculate_BEC_length(constraints_other, weights, step_cost_flag)[0]

                    # simply remove the first redundant constraint. can also remove the redundant constraint that's
                    # 1) conveyed by the fewest environments, 2) conveyed by a higher minimum complexity environment,
                    # 3) doesn't work as well with visual similarity of other nonredundant constraints
                    if np.isclose(BEC_length, BEC_length_all_constraints):
                        nonredundant_constraints = constraints_other
    else:
        # remove constraints that don't belong in the minimal H-representation of the corresponding polyhedron (not
        # including the boundary constraints/facets)
        # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints)
        ieqs = constraints_to_halfspace_matrix_sage_team(constraints)
        # print('ieqs: ', ieqs)

        poly = Polyhedron.Polyhedron(ieqs=ieqs)
        hrep = np.array(poly.Hrepresentation())

        # print('hrep: ', hrep)
        
        # remove boundary constraints/facets from consideration
        boundary_facet_idxs = np.where(hrep[:, 0] != 0)
        hrep_constraints = np.delete(hrep, boundary_facet_idxs, axis=0)
        # remove the first column since these constraints goes through the origin
        nonredundant_constraints = hrep_constraints[:, 1:]

        # reshape so that each element is a valid weight vector
        nonredundant_constraints = nonredundant_constraints.reshape(nonredundant_constraints.shape[0], 1, nonredundant_constraints.shape[1])


    return list(nonredundant_constraints)



def constraints_to_halfspace_matrix_sage_team(constraints):
    '''
    Summary: convert list of halfspace constraints into an array of halfspace constraints. Add bounding cube

    Halfspace representation of a convex polygon (Ax < b):
    [-1,7,3,4] represents the inequality 7x_1 + 3x_2 + 4x_3 >= 1
    '''
    constraints_stacked = np.vstack(constraints)
    constraints_stacked = np.insert(constraints_stacked, 0, np.zeros((constraints_stacked.shape[0]), dtype='int'), axis=1)
    # constraints_stacked = np.vstack((constraints_stacked, np.array([1, 1, 0, 0]), np.array([1, -1, 0, 0]), np.array([1, 0, 1, 0]), np.array([1, 0, -1, 0]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, -1])))
    ieqs = constraints_stacked

    return ieqs



def majority_rules_opposing_team_constraints(opposing_idx, test_constraints_team_expanded, response_category_team):

    opp_idx_unique = []
    for i in range(len(opposing_idx)):
        opp_idx_unique.extend(x for x in opposing_idx[i] if x not in opp_idx_unique)

    print('opp_idx_unique: ', opp_idx_unique)
    opp_constraints = [test_constraints_team_expanded[x] for x in opp_idx_unique]
    non_opposing_constraints = [test_constraints_team_expanded[i] for i in range(len(test_constraints_team_expanded)) if i not in opp_idx_unique]

    print('non_opposing_constraints before process: ', non_opposing_constraints)
    # print('opp_constraints: ', opp_constraints)
    # resp_cat = [response_category_team[x] for x in opp_idx_unique]
    # # print('resp_cat: ', resp_cat)
    opp_set = []
    count_opp_set = []
    # resp_cat_set = []
    for j in range(len(opp_constraints)):
        opp_c = opp_constraints[j]
        if len(opp_set) > 0:
            in_minimal_set = False
            for i in range(len(opp_set)):
                opp_c_set = opp_set[i]
                # if (opp_c == opp_c_set).all():
                if BEC_helpers.equal_constraints(opp_c, opp_c_set):
                    in_minimal_set = True
                    count_opp_set[i] += 1

            if not in_minimal_set:
                opp_set.append(opp_c)
                count_opp_set.append(1)
                # resp_cat_set.append(resp_cat[j])
        else:
            opp_set.append(opp_c)
            count_opp_set.append(1)
            # resp_cat_set.append(resp_cat[j])
            
    # # print('minimal opposing constraints: ', opp_set)
    # # print('count_opp_set: ', count_opp_set)
    # # print('response set: ', resp_cat_set)

    max_count_idx = [i for i in range(len(count_opp_set)) if count_opp_set[i] == max(count_opp_set)]

    # # print('max_count_idx: ', max_count_idx)
    if len(max_count_idx) == 1:
        # # print('Majority response: ', resp_cat_set[max_count_idx[0]])
        maj_cnst = [opp_set[max_count_idx[0]]]
    else:        
        # # print('Checking when there are multiple max counts')
        # # print('Majority response: ', 'incorrect')
        # maj_cnst = [opp_set[x] for x in max_count_idx if resp_cat_set[x] == 'incorrect'] # choose the set with incorrect responses
        maj_cnst = [opp_set[random.randint(0, len(opp_set)-1)]] # randomly choose a set (unbiased)
        
    # # print('Majority constraint: ', maj_cnst)

    # alternate_team_constraints = copy.deepcopy(maj_cnst)  # majority rules
    # print('Majority_team_constraints: ', alternate_team_constraints, ' count: ', count_opp_set[max_count_idx[0]])
    # # print('Maj team constraints type: ', type(alternate_team_constraints))
    non_opposing_constraints.extend(maj_cnst)
    print('non_opposing_constraints after process: ', non_opposing_constraints)

    return non_opposing_constraints



def majority_rules_non_intersecting_team_constraints(test_constraints_team, weights, step_cost_flag, test_flag = False):

    
    # non_intersecting_flag_cnst = []
    intersecting_constraint_idx = []
    max_intersecting_cnsts = []
    intersecting_cnst = []
    alternate_team_constraints = []
    N_max_cnst = 0

    # print(colored('Number of loops:' + str(N_loops), 'blue'))
    # print('test_constraints_team: ', test_constraints_team)

    # TODO:
    if test_flag:
        test_constraints_team_formatted = copy.deepcopy(test_constraints_team)
        N_intersecting_sets = len(test_constraints_team)
    else:
        test_constraints_team_formatted = []
        for cnst in test_constraints_team:
            test_constraints_team_formatted.append([cnst])
        N_intersecting_sets = len(test_constraints_team_formatted)



    while N_intersecting_sets > 1:
        intersections_to_check = list(itertools.combinations(range(len(test_constraints_team_formatted)), N_intersecting_sets))

        # print('Constraint list: ', test_constraints_team_formatted)
        # print('N_intersecting_sets: ', N_intersecting_sets)
        # print('Intersections to check: ', intersections_to_check)

        for int_check_id in range(len(intersections_to_check)):
            constraints = []
            for ind_int in intersections_to_check[int_check_id]:
                # if test_flag:
                # print('test_constraints_team_formatted: ', test_constraints_team_formatted[ind_int])
                # constraints.extend(utils_teams.flatten_list(test_constraints_team_expanded[j]))
                constraints.extend(test_constraints_team_formatted[ind_int])
                # else:
                #     constraints.append(test_constraints_team_expanded[j])

            # print('Constraints to check for non-intersection: ', constraints)

            non_intersecting_cnst_flag, _ = check_for_non_intersecting_constraints(constraints, weights, step_cost_flag)
            # non_intersecting_flag_cnst.extend([non_intersecting_cnst_flag, constraints])
            # # print('non_intersecting_cnst_flag: ', non_intersecting_cnst_flag, ' constraints: ', constraints)

            if not non_intersecting_cnst_flag:
                # print('max_intersecting_cnsts: ', max_intersecting_cnsts)
                # print('constraints: ', constraints)

                if len(max_intersecting_cnsts) == 0:
                    max_intersecting_cnsts = [copy.deepcopy(constraints)]
                    # max_intersecting_cnsts = [ test_constraints_team_formatted[cnst_id] for cnst_id in intersections_to_check[int_check_id] ]
                    intersecting_cnst = [intersections_to_check[int_check_id]]
                    N_max_cnst = len(constraints)
                else:
                    # # print('max_intersecting_cnsts[0]): ', max_intersecting_cnsts[0])
                    if len(constraints) > N_max_cnst:   # update the max intersecting constraints
                        max_intersecting_cnsts = [copy.deepcopy(constraints)]
                        # max_intersecting_cnsts = [ test_constraints_team_formatted[cnst_id] for cnst_id in intersections_to_check[int_check_id] ]
                        intersecting_cnst = [intersections_to_check[int_check_id]]
                        N_max_cnst = len(constraints)
                    elif len(constraints) == N_max_cnst:         # joint max intersecting constraints
                        # max_intersecting_cnsts.append([ test_constraints_team_formatted[cnst_id] for cnst_id in intersections_to_check[int_check_id] ])
                        max_intersecting_cnsts.append(copy.deepcopy(constraints))
                        intersecting_cnst.append(intersections_to_check[int_check_id])
                
                # print('Updated max int cnts: ', max_intersecting_cnsts)
                
            # # plot
            # if non_intersecting_cnst_flag:
            #     # print('Plotting..')
            #     # plot actual knowledge constraints for this knowledge type
            #     fig = plt.figure()
            #     ax = fig.add_subplot(projection='3d')
            #     utils_teams.visualize_planes_team(constraints, fig=fig, ax=ax, alpha=0.5)
            #     ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints)
            #     poly = Polyhedron.Polyhedron(ieqs=ieqs)
            #     BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, color = 'b')
            #     plt.show()

        N_intersecting_sets -= 1

        # print('max_intersecting_cnsts:' , max_intersecting_cnsts)

        if len(max_intersecting_cnsts) > 0:
            break


    # ## max intersecting constraints
    # if len(max_intersecting_cnsts) > 0:
    #     for max_int_cnsts in max_intersecting_cnsts:
    #         if len(max_int_cnsts) == 1:
    #             if len(alternate_team_constraints) == 0:
    #                 alternate_team_constraints = max_int_cnsts
    #             else:
    #                 alternate_team_constraints.extend(max_int_cnsts)
    #         elif len(max_int_cnsts) > 1:
    #             rand_index = random.randint(0, len(max_int_cnsts)-1)
    #             if len(alternate_team_constraints) == 0:
    #                 alternate_team_constraints = max_int_cnsts[rand_index]
    #             else:
    #                 alternate_team_constraints.extend(max_int_cnsts[rand_index])

    # else:
    #     # There are no intersections among any of the teammates, so randomly choose the constraints of a teammate
    #     intersecting_cnst = random.randint(0, len(test_constraints_team_expanded)-1)
    #     alternate_team_constraints = test_constraints_team_expanded[intersecting_cnst]

    if len(max_intersecting_cnsts) == 1:
        alternate_team_constraints = max_intersecting_cnsts[0]
    elif len(max_intersecting_cnsts) > 1:
        rand_index = random.randint(0, len(max_intersecting_cnsts)-1)
        alternate_team_constraints = max_intersecting_cnsts[rand_index]
    else:
        # There are no intersections among any of the teammates, so randomly choose the constraints of a teammate
        intersecting_cnst = random.randint(0, len(test_constraints_team_formatted)-1)
        alternate_team_constraints = test_constraints_team_formatted[intersecting_cnst]



    

    return alternate_team_constraints, intersecting_cnst



def plot_sampled_models(particles, constraints, human_models, weights=None, fig=None, text=None):
    

    def label_axes(ax):
        fs = 12
        ax.set_facecolor('white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xlabel(r'$\mathregular{w_0}$: Mud', fontsize = fs)
        ax.set_ylabel(r'$\mathregular{w_1}$: Recharge', fontsize = fs)
        ax.set_zlabel('$\mathregular{w_2}$: Action', fontsize = fs)
        ax.view_init(elev=16, azim=-160)



    # print('human_models: ', human_models)

    if fig == None:
        fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    if weights is not None:
            ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='g', s=50)

    fig.suptitle(text, fontsize=30)

    # plot particles before and after the constraints
    particles.plot(fig=fig, ax=ax)

    ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints)
    poly = Polyhedron.Polyhedron(ieqs=ieqs)
    BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.5, color='y')

    for hm in human_models:
        # print(hm)
        ax.scatter(hm[0][0], hm[0][1], hm[0][2], marker='+', c='r', s=50)

    label_axes(ax)

    plt.show()

#####################

def obtain_diagnostic_tests(data_loc, previous_demos, visited_env_traj_idxs, min_BEC_constraints, min_subset_constraints_record, traj_record, traj_features_record, variable_filter, mdp_features_record, downsample_threshold=float("inf"), opt_simplicity=True, opt_similarity=True):
    
    preliminary_test_info = []

    # if you're looking for demonstrations that will convey the most constraining BEC region or will be employing scaffolding,
    # obtain the demos needed to convey the most constraining BEC region
    BEC_constraints = min_BEC_constraints.copy()
    BEC_constraint_bookkeeping = BEC_helpers.perform_BEC_constraint_bookkeeping(BEC_constraints,
                                                                                min_subset_constraints_record, visited_env_traj_idxs, traj_record, traj_features_record, mdp_features_record, variable_filter=variable_filter)
    
    if params.debug_hm_sampling:
        print('BEC_constraints: {}'.format(BEC_constraints))
        print('BEC_constraint_bookkeeping for generating diagnostic tests: {}'.format(BEC_constraint_bookkeeping))


    # check for any exact matches in BEC_constraints_bookkeeping
    exact_BEC_flag = True
    nn_BEC_constraint_id = []
    for BEC_cnst_id in range(len(BEC_constraint_bookkeeping)):
        if len(BEC_constraint_bookkeeping[BEC_cnst_id]) == 0:
            nn_BEC_constraint_id.append(BEC_cnst_id)
            exact_BEC_flag = False

    if not exact_BEC_flag:
        nn_BEC_constraint_bookkeeping, minimal_distances = BEC_helpers.perform_nn_BEC_constraint_bookkeeping(BEC_constraints,
                                                                                            min_subset_constraints_record, visited_env_traj_idxs, traj_record, traj_features_record, mdp_features_record, variable_filter=variable_filter)
        if params.debug_hm_sampling:
            print('NN BEC_constraint_bookkeeping for generating diagnostic tests: {}'.format(BEC_constraint_bookkeeping))

        #Final BEC bookkeeping
        BEC_constraint_bookkeeping = [nn_BEC_constraint_bookkeeping[x] if x in nn_BEC_constraint_id else BEC_constraint_bookkeeping[x] for x in range(len(BEC_constraint_bookkeeping))]

        if params.debug_hm_sampling:
            print('Updated BEC_constraint_bookkeeping for generating diagnostic tests: {}'.format(BEC_constraint_bookkeeping))
        

    # print(colored('Entering while loop to calculate BEC constraints', 'blue'))
    while len(BEC_constraints) > 0:
        # downsampling strategy 1: randomly cull sets with too many members for computational feasibility
        # for j, set in enumerate(sets):
        #     if len(set) > downsample_threshold:
        #         sets[j] = random.sample(set, downsample_threshold)

        # downsampling strategy 2: if there are any env_traj pairs that cover more than one constraint, use it and remove all
        # env_traj pairs that would've conveyed the same constraints
        # initialize all env_traj tuples with covering the first min BEC constraint
        env_constraint_mapping = {}
        for key in BEC_constraint_bookkeeping[0]:
            env_constraint_mapping[key] = [0]
        max_constraint_count = 1  # what is the max number of desired constraints that one env / demo can convey
        max_env_traj_tuples = [key]

        # print('max_env_traj_tuples that cover 1st BEC constraint: {}'.format(max_env_traj_tuples))

        # for all other env_traj tuples,
        for constraint_idx, env_traj_tuples in enumerate(BEC_constraint_bookkeeping[1:]):
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

        # print('max_env_traj_tuples that cover {} BEC constraints: {}'.format(max_constraint_count, max_env_traj_tuples))


        if max_constraint_count == 1:
            # no one demo covers multiple constraints. so greedily select demos from base list that is mot visually complex
            # filter for the most visually complex environment
            for env_traj_tuples in BEC_constraint_bookkeeping[::-1]:
                best_env_idxs, best_traj_idxs = zip(*env_traj_tuples)
                best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs,
                                                                        traj_record, previous_demos, type='testing')
                max_complexity_env_traj_tuple = (best_env_idx, best_traj_idx)
                preliminary_test_info.append((max_complexity_env_traj_tuple, [BEC_constraints.pop()]))

                if params.debug_hm_sampling:
                    print('max_complexity_env_traj_tuple: ', max_complexity_env_traj_tuple)

        else:
            # filter for the most visually complex environment that can cover multiple constraints
            best_env_idxs, best_traj_idxs = zip(*max_env_traj_tuples)
            best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs,
                                                                    traj_record, previous_demos, type='testing')
            max_complexity_env_traj_tuple = (best_env_idx, best_traj_idx)

            if params.debug_hm_sampling:
                print('env_constraint_mapping: ', env_constraint_mapping)
                print('max_complexity_env_traj_tuple: ', max_complexity_env_traj_tuple)
            
            constituent_constraints = []
            for idx in env_constraint_mapping[max_complexity_env_traj_tuple]:
                # Temp
                # print('idx: ', idx)
                # print('BEC_constraints: ', BEC_constraints)
                constituent_constraints.append(BEC_constraints[idx])

            preliminary_test_info.append((max_complexity_env_traj_tuple, constituent_constraints))

            for constraint_idx in sorted(env_constraint_mapping[max_complexity_env_traj_tuple], reverse=True):
                del BEC_constraints[constraint_idx]
                del BEC_constraint_bookkeeping[constraint_idx] # added by Suresh. BEC_constraints and BEC_constraint_bookkeeping should be in sync.

    
    preliminary_tests = []
    for info in preliminary_test_info:
        env_idx, traj_idx = info[0]
        traj = traj_record[env_idx][traj_idx]
        constraints = info[1]
        filename = mp_helpers.lookup_env_filename(data_loc, env_idx)
        with open(filename, 'rb') as f:
            wt_vi_traj_env = pickle.load(f)
        vi = wt_vi_traj_env[0][1]
        best_mdp = wt_vi_traj_env[0][1].mdp
        best_mdp.set_init_state(traj[0][0])  # for completeness
        preliminary_tests.append([best_mdp, traj, (env_idx, traj_idx), constraints, vi, wt_vi_traj_env[0][3]])
        visited_env_traj_idxs.append((env_idx, traj_idx))

    # else:
        
    #     nn_BEC_constraint_bookkeeping, minimal_distances = BEC_helpers.perform_nn_BEC_constraint_bookkeeping(BEC_constraints,
    #                                                                                         min_env_constraints_record, visited_env_traj_idxs, traj_record, traj_features_record, mdp_features_record, variable_filter=variable_filter)
    #     print('{} approximate candidates for remedial demo/test'.format(len(nn_BEC_constraint_bookkeeping[0])))
    #     if len(nn_BEC_constraint_bookkeeping[0]) > 0:
    #         # the human's incorrect response can be corrected with similar enough counterexample
    #         best_env_idxs, best_traj_idxs = list(zip(*nn_BEC_constraint_bookkeeping[0]))

    #         # simply optimize for the visuals of the direct counterexample
    #         best_env_idxs, best_traj_idxs = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs,
    #                                                                     traj_record,
    #                                                                     previous_demonstrations, type=type, return_all_equiv=True)

    #         print('Failed constraint: {}'.format(minimal_distances[0][2]))
    #         print('Similar-enough constraint: {}'.format(minimal_distances[0][3]))

        
    #     # sample human models again and find most informative nearest demo
    #     sample_human_models, model_weights = BEC_helpers.sample_human_models_uniform(particles, n_human_models)

    #     # obtain the indices of the reference human models (that have precomputed constraints) that are closest to the sampled human models
    #     sample_human_models_ref_latllong = cg.cart2latlong(np.array(sample_human_models_ref).squeeze())
    #     sample_human_models_latlong = cg.cart2latlong(np.array(sample_human_models).squeeze())
    #     distances = haversine_distances(sample_human_models_latlong, sample_human_models_ref_latllong)
    #     min_model_idxs = np.argmin(distances, axis=1)
        
    #     args = [(i, min_model_idxs, data_loc, 'precomputed', weights, step_cost_flag, variable_filter,
    #         mdp_features_record[i],
    #         traj_record[i], [], None, False, False) for
    #     i in range(len(traj_record))]
    #     info_gains_record, min_env_constraints_record, n_diff_constraints_record, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = zip(
    #         *pool.imap(combine_limiting_constraints_IG, tqdm(args)))
    
        
    #     if len(best_env_idxs) > 1 and n_human_models_precomputed != 0:
    #         # optimizing information gain is currently only implemented for the instance where precomputed PF-based counterfactuals constraints are avilable
    #         best_env_idx, best_traj_idx = BEC_helpers.optimize_information_gain(particles, best_env_idxs, best_traj_idxs, min_model_idxs, model_weights, data_loc, weights,
    #                                     step_cost_flag, type)
    #     else:
    #         best_env_idx = best_env_idxs[0]
    #         best_traj_idx = best_traj_idxs[0]

    #     traj = traj_record[best_env_idx][best_traj_idx]
    #     filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
    #     with open(filename, 'rb') as f:
    #         wt_vi_traj_env = pickle.load(f)
    #     best_mdp = wt_vi_traj_env[0][1].mdp
    #     best_mdp.set_init_state(traj[0][0])  # for completeness
    #     remedial_demonstrations.append([best_mdp, traj, (best_env_idx, best_traj_idx), BEC_constraints, variable_filter])
    #     visited_env_traj_idxs.append((best_env_idx, best_traj_idx))

    return preliminary_tests, visited_env_traj_idxs





def simulate_team_learning(kc_id, particles_team_learner, new_constraints, params, loop_count, kc_reset_flag=True, viz_flag=False, learner_update_type = 'noise'):


    print(colored('Simulating team learning for KC ' + str(kc_id) + 'in loop ' + str(loop_count) + '...', 'red'))

    p = 1
    while p <= params.team_size:
        member_id = 'p' + str(p)
        # print('PF update of ', member_id)
        plot_title = 'Learner belief update after demonstration for loop ' + str(loop_count+1) + ' for player ' + member_id
        particles_team_learner[member_id].update(new_constraints, plot_title = plot_title, model_type = learner_update_type, viz_flag = viz_flag)
        particles_team_learner[member_id].calc_particles_probability(new_constraints)
        ##########################
        
        if viz_flag:
            visualize_transition(new_constraints[0], particles_team_learner[member_id], params.mdp_class, params.weights['val'], text = 'Simulated knowledge change for set ' + str(loop_count+1) + ' for player ' + member_id, plot_filename ='ek_p' + str(p) + '_loop_' + str(loop_count+1))
        p += 1
            

    return particles_team_learner



############################################################################























# def obtain_remedial_demos_tests(data_loc, previous_demos, visited_env_traj_idxs, min_BEC_constraints, min_subset_constraints_record, traj_record, traj_features_record, variable_filter, mdp_features_record, downsample_threshold=float("inf"), opt_simplicity=True, opt_similarity=True, type = 'training'):

#     preliminary_test_info = []

#     # if you're looking for demonstrations that will convey the most constraining BEC region or will be employing scaffolding,
#     # obtain the demos needed to convey the most constraining BEC region
#     BEC_constraints = min_BEC_constraints.copy()
#     # print('Remedial constraints:', BEC_constraints)
#     BEC_constraint_bookkeeping = BEC_helpers.perform_BEC_constraint_bookkeeping(BEC_constraints,
#                                                                                 min_subset_constraints_record, visited_env_traj_idxs, traj_record, traj_features_record, mdp_features_record, variable_filter=variable_filter)
    
    
    
#     if len(BEC_constraint_bookkeeping[0]) <= 0:
#         nn_BEC_constraint_bookkeeping, minimal_distances = BEC_helpers.perform_nn_BEC_constraint_bookkeeping(BEC_constraints,
#                                                                                             min_subset_constraints_record, visited_env_traj_idxs, traj_record, traj_features_record, mdp_features_record, variable_filter=variable_filter)
    
#         BEC_bookkeeping = nn_BEC_constraint_bookkeeping.copy()
#         # print('BEC_constraint_bookkeeping length:', len(nn_BEC_constraint_bookkeeping))
#         # print('BEC_constraint_bookkeeping :', nn_BEC_constraint_bookkeeping)

#     else:
#         BEC_bookkeeping = BEC_constraint_bookkeeping.copy()
    
#     while len(BEC_constraints) > 0:
#         # downsampling strategy 1: randomly cull sets with too many members for computational feasibility
#         # for j, set in enumerate(sets):
#         #     if len(set) > downsample_threshold:
#         #         sets[j] = random.sample(set, downsample_threshold)

#         # downsampling strategy 2: if there are any env_traj pairs that cover more than one constraint, use it and remove all
#         # env_traj pairs that would've conveyed the same constraints
#         # initialize all env_traj tuples with covering the first min BEC constraint
#         env_constraint_mapping = {}
#         for key in BEC_bookkeeping[0]:
#             env_constraint_mapping[key] = [0]
#         max_constraint_count = 1  # what is the max number of desired constraints that one env / demo can convey
#         max_env_traj_tuples = [key]

#         # for all other env_traj tuples,
#         for constraint_idx, env_traj_tuples in enumerate(BEC_bookkeeping[1:]):
#             for env_traj_tuple in env_traj_tuples:
#                 # if this env_traj tuple has already been seen previously
#                 if env_traj_tuple in env_constraint_mapping.keys():
#                     env_constraint_mapping[env_traj_tuple].append(constraint_idx + 1)
#                     # and adding another constraint to this tuple increases the highest constraint coverage by a single tuple
#                     if len(env_constraint_mapping[env_traj_tuple]) > max_constraint_count:
#                         # update the max values, replacing the max tuple
#                         max_constraint_count = len(env_constraint_mapping[env_traj_tuple])
#                         max_env_traj_tuples = [env_traj_tuple]
#                     # otherwise, if it simply equals the highest constraint coverage, add this tuple to the contending list
#                     elif len(env_constraint_mapping[env_traj_tuple]) == max_constraint_count:
#                         max_env_traj_tuples.append(env_traj_tuple)
#                 else:
#                     env_constraint_mapping[env_traj_tuple] = [constraint_idx + 1]

#         if max_constraint_count == 1:
#             # no one demo covers multiple constraints. so greedily select demos from base list that is mot visually complex
#             # filter for the most visually complex environment
#             for env_traj_tuples in BEC_bookkeeping[::-1]:
#                 best_env_idxs, best_traj_idxs = zip(*env_traj_tuples)
#                 best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs,
#                                                                           traj_record, previous_demos, type='training')
#                 max_complexity_env_traj_tuple = (best_env_idx, best_traj_idx)
#                 preliminary_test_info.append((max_complexity_env_traj_tuple, [BEC_constraints.pop()]))

#         else:
#             # filter for the most visually complex environment that can cover multiple constraints
#             best_env_idxs, best_traj_idxs = zip(*max_env_traj_tuples)
#             best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs,
#                                                                       traj_record, previous_demos, type='training')
#             max_complexity_env_traj_tuple = (best_env_idx, best_traj_idx)

#             constituent_constraints = []
#             for idx in env_constraint_mapping[max_complexity_env_traj_tuple]:
#                 constituent_constraints.append(BEC_constraints[idx])

#             preliminary_test_info.append((max_complexity_env_traj_tuple, constituent_constraints))

#             for constraint_idx in sorted(env_constraint_mapping[max_complexity_env_traj_tuple], reverse=True):
#                 del BEC_constraints[constraint_idx]

#     preliminary_tests = []
#     for info in preliminary_test_info:
#         env_idx, traj_idx = info[0]
#         traj = traj_record[env_idx][traj_idx]
#         constraints = info[1]
#         filename = mp_helpers.lookup_env_filename(data_loc, env_idx)
#         with open(filename, 'rb') as f:
#             wt_vi_traj_env = pickle.load(f)
#         best_mdp = wt_vi_traj_env[0][1].mdp
#         best_mdp.set_init_state(traj[0][0])  # for completeness
#         preliminary_tests.append([best_mdp, traj, (env_idx, traj_idx), constraints, variable_filter])
#         visited_env_traj_idxs.append((env_idx, traj_idx))

#     return preliminary_tests, visited_env_traj_idxs



# def obtain_summary_particle_filter_team(data_loc, particles, summary, variable_filter, nonzero_counter, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count,
#                        min_BEC_constraints_running, n_train_demos=np.inf, downsample_threshold=float("inf"), consider_human_models_jointly=True, c=0.001, obj_func_proportion=1, min_info_gain=0.01, visited_env_traj_idxs=[]):


    # # count how many nonzero constraints are present for each reward weight (i.e. variable) in the minimum BEC constraints
    # # (which are obtained using one-step deviations). mask variables in order of fewest nonzero constraints for variable scaffolding
    # # rationale: the variable with the most nonzero constraints, often the step cost, serves as a good reference/ratio variable
    # min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record for item in sublist]
    # min_subset_constraints_record_flattened = [item for sublist in min_subset_constraints_record_flattened for item in sublist]
    # min_subset_constraints_record_array = np.array(min_subset_constraints_record_flattened)

    # # print('variable filter: {}'.format(variable_filter))

    # # running_variable_filter = variable_filter.copy()

    # # clear the demonstration generation log
    # open('models/' + data_loc + '/demo_gen_log.txt', 'w').close()

    # while len(summary) < n_train_demos:
    #     # visualize_constraints(min_BEC_constraints_running, weights, step_cost_flag, fig_name=str(len(summary)) + '.png', just_save=True)
    #     sample_human_models = BEC_helpers.sample_human_models_pf(particles, n_human_models)

    #     if len(sample_human_models) == 0:
    #         # print(colored("Likely cannot reduce the BEC further through additional demonstrations. Returning.", 'red'))
    #         return summary

    #     info_gains_record = []
    #     overlap_in_opt_and_counterfactual_traj_record = []

    #     # print("Length of summary: {}".format(len(summary)))
    #     with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
    #         myfile.write('Length of summary: {}\n'.format(len(summary)))
            

    #     for model_idx, human_model in enumerate(sample_human_models[0]):
    #         # print(colored('Model #: {}'.format(model_idx), 'red'))
    #         # print(colored('Model val: {}'.format(human_model), 'red'))

    #         with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
    #             myfile.write('Model #: {}\n'.format(model_idx))
    #             myfile.write('Model val: {}\n'.format(human_model))
                

    #         # based on the human's current model, obtain the information gain generated when comparing to the agent's
    #         # optimal trajectories in each environment (human's corresponding optimal trajectories and constraints
    #         # are saved for reference later)
    #         # print("Obtaining counterfactual information gains:")

    #         cf_data_dir = 'models/' + data_loc + '/counterfactual_data_' + str(len(summary)) + '/model' + str(model_idx)
            
    #         # cf_data_dir = 'models/' + data_loc + '/counterfactual_data_' + str(summary_id) + '/model' + str(model_idx)
    #         os.makedirs(cf_data_dir, exist_ok=True)
    #         if consider_human_models_jointly:
    #             args = [(data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]), traj_record[i], particles, min_BEC_constraints_running, step_cost_flag, len(summary), variable_filter, mdp_features_record[i], consider_human_models_jointly) for i in range(len(traj_record))]

    #             info_gain_envs = list(tqdm(pool.imap(compute_counterfactuals_team, args), total=len(args)))

    #             info_gains_record.append(info_gain_envs)
    #         else:
    #             args = [(data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]), traj_record[i], particles, min_BEC_constraints_running, step_cost_flag, len(summary), variable_filter, mdp_features_record[i], consider_human_models_jointly) for i in range(len(traj_record))]
    #             info_gain_envs, overlap_in_opt_and_counterfactual_traj_env = zip(*pool.imap(compute_counterfactuals_team, tqdm(args), total=len(args)))

    #             info_gains_record.append(info_gain_envs)
    #             overlap_in_opt_and_counterfactual_traj_record.append(overlap_in_opt_and_counterfactual_traj_env)


    #     with open('models/' + data_loc + '/info_gains_' + str(len(summary)) + '.pickle', 'wb') as f:
    #         pickle.dump(info_gains_record, f)

    #     # do a quick check of whether there's any information to be gained from any of the trajectories
    #     no_info_flag = False
    #     max_info_gain = -np.inf
    #     if consistent_state_count:
    #         info_gains = np.array(info_gains_record)
    #         if np.sum(info_gains > 0) == 0:
    #             no_info_flag = True
    #     else:
    #         info_gains_flattened_across_models = list(itertools.chain.from_iterable(info_gains_record))
    #         info_gains_flattened_across_envs = list(itertools.chain.from_iterable(info_gains_flattened_across_models))
    #         if sum(np.array(info_gains_flattened_across_envs) > 0) == 0:
    #             no_info_flag = True

    #     # no need to continue search for demonstrations if none of them will improve the human's understanding
    #     if no_info_flag:
    #         # print('No information gained...')
    #         break


    #     # todo: now that we're utilizing a probabilistic human model, we should account for the probability of having
    #     #  selected each human model in the information gain calculation (e.g. by taking an expectation over the information
    #     #  gain), rather than simply combine all of the constraints that could've been generated (which is currently done)

    #     # todo: the code below isn't updated to provide demonstrations that only differ by a single constraint at a time,
    #     #  nor does it ensure that the selected demonstrations don't conflict with prior selected demonstrations that are
    #     #  specified through visited_env_traj_idxs (e.g. ones that will be used for assessment tests after teaching).
    #     #  see obtain_summary_counterfactual() for a more updated version
    #     # print("Combining the most limiting constraints across human models:")
    #     args = [(i, range(len(sample_human_models)), data_loc, len(summary), weights, step_cost_flag, variable_filter, mdp_features_record[i],
    #              traj_record[i], min_BEC_constraints_running, particles, True, False) for
    #             i in range(len(traj_record))]
    #     info_gains_record, min_env_constraints_record, n_diff_constraints_record, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = zip(
    #         *pool.imap(BEC.combine_limiting_constraints_IG, tqdm(args)))

    #     # the possibility that no demonstration provides information gain must be checked for again,
    #     # in case all limiting constraints involve a masked variable and shouldn't be considered for demonstration yet
    #     if consistent_state_count:
    #         info_gains = np.array(info_gains_record)
    #         traj_overlap_pcts = np.array(overlap_in_opt_and_counterfactual_traj_avg)

    #         # obj_function = info_gains * (traj_overlap_pcts + c)  # objective 2: scaled
    #         obj_function = info_gains

    #         # not considering demos where there is no info gain helps ensure that the final demonstration
    #         # provides the maximum info gain (in conjuction with previously shown demonstrations)
    #         obj_function[info_gains <= 0] = 0

    #         max_info_gain = np.max(info_gains)

    #         if max_info_gain <= min_info_gain:
    #             no_info_flag = True
    #         else:
    #             # if visuals aren't considered, then you can simply return one of the demos that maximizes the obj function
    #             # best_env_idx, best_traj_idx = np.unravel_index(np.argmax(obj_function), info_gains.shape)

    #             if obj_func_proportion == 1:
    #                 # a) select the trajectory with the maximal information gain
    #                 best_env_idxs, best_traj_idxs = np.where(obj_function == max(obj_function.flatten()))
    #             else:
    #                 # b) select the trajectory closest to the desired partial information gain (to obtain more demonstrations0
    #                 obj_function_flat = obj_function.flatten()
    #                 obj_function_flat.sort()

    #                 best_obj = obj_function_flat[-1]
    #                 target_obj = obj_func_proportion * best_obj
    #                 target_idx = np.argmin(abs(obj_function_flat - target_obj))
    #                 closest_obj = obj_function_flat[target_idx]
    #                 best_env_idxs, best_traj_idxs = np.where(obj_function == obj_function_flat[closest_obj])

    #             best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, traj_record, summary)
    #     else:
    #         best_obj = float('-inf')
    #         best_env_idxs = []
    #         best_traj_idxs = []

    #         if obj_func_proportion == 1:
    #             # a) select the trajectory with the maximal information gain
    #             for env_idx, info_gains_per_env in enumerate(info_gains_record):
    #                 for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
    #                     if info_gain_per_traj > 0:

    #                         # obj = info_gain_per_traj * (
    #                         #             overlap_in_opt_and_counterfactual_traj_avg[env_idx][traj_idx] + c)  # objective 2: scaled
    #                         obj = info_gain_per_traj

    #                         if np.isclose(obj, best_obj):
    #                             best_env_idxs.append(env_idx)
    #                             best_traj_idxs.append(traj_idx)
    #                         elif obj > best_obj:
    #                             best_obj = obj

    #                             best_env_idxs = [env_idx]
    #                             best_traj_idxs = [traj_idx]
    #                         if info_gain_per_traj > max_info_gain:
    #                             max_info_gain = info_gain_per_traj
    #                             # print("new max info: {}".format(max_info_gain))
    #         else:
    #             # b) select the trajectory closest to the desired partial information gain (to obtain more demonstrations)
    #             # first find the max information gain
    #             for env_idx, info_gains_per_env in enumerate(info_gains_record):
    #                 for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
    #                     if info_gain_per_traj > 0:
    #                         obj = info_gain_per_traj

    #                         if np.isclose(obj, best_obj):
    #                             pass
    #                         elif obj > best_obj:
    #                             best_obj = obj

    #                         if info_gain_per_traj > max_info_gain:
    #                             max_info_gain = info_gain_per_traj
    #                             # print("new max info: {}".format(max_info_gain))

    #             target_obj = obj_func_proportion * best_obj
    #             closest_obj_dist = float('inf')

    #             for env_idx, info_gains_per_env in enumerate(info_gains_record):
    #                 for traj_idx, info_gain_per_traj in enumerate(info_gains_per_env):
    #                     if info_gain_per_traj > 0:

    #                         obj = info_gain_per_traj

    #                         if np.isclose(abs(target_obj - obj), closest_obj_dist):
    #                             best_env_idxs.append(env_idx)
    #                             best_traj_idxs.append(traj_idx)
    #                         elif abs(target_obj - obj) < closest_obj_dist:
    #                             closest_obj_dist = abs(obj - target_obj)

    #                             best_env_idxs = [env_idx]
    #                             best_traj_idxs = [traj_idx]

    #         if max_info_gain < min_info_gain:
    #             no_info_flag = True
    #         elif max_info_gain == min_info_gain:
    #             no_info_flag = True
    #             # print(colored('Max info gain is equal to min info gain....!', 'red'))
    #         else:
    #             best_env_idx, best_traj_idx = ps_helpers.optimize_visuals(data_loc, best_env_idxs, best_traj_idxs, traj_record, summary)

    #     # print("current max info: {}".format(max_info_gain))
    #     # no need to continue search for demonstrations if none of them will improve the human's understanding
    #     if no_info_flag:
    #         # if no variables had been filtered out, then there are no more informative demonstrations to be found
    #         if not np.any(variable_filter):
    #             break
    #         else:
    #             # no more informative demonstrations with this variable filter, so update it
    #             variable_filter, nonzero_counter = BEC_helpers.update_variable_filter(nonzero_counter)
    #             # print(colored('Did not find any informative demonstrations.', 'red'))
    #             # print('variable filter: {}'.format(variable_filter))
    #             continue

    #     best_traj = traj_record[best_env_idx][best_traj_idx]

    #     filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
    #     with open(filename, 'rb') as f:
    #         wt_vi_traj_env = pickle.load(f)
    #     best_mdp = wt_vi_traj_env[0][1].mdp
    #     best_mdp.set_init_state(best_traj[0][0]) # for completeness
    #     min_BEC_constraints_running.extend(min_env_constraints_record[best_env_idx][best_traj_idx])
    #     min_BEC_constraints_running = BEC_helpers.remove_redundant_constraints(min_BEC_constraints_running, weights, step_cost_flag)
        
    #     # # check and update unit
    #     # if (running_variable_filter == variable_filter).all():
    #     #     unit.append([best_mdp, best_traj, (best_env_idx, best_traj_idx), min_env_constraints_record[best_env_idx][best_traj_idx], variable_filter, sample_human_models])
    #     # else:
    #     #     summary.append(unit)
    #     #     unit = [[best_mdp, best_traj, (best_env_idx, best_traj_idx), min_env_constraints_record[best_env_idx][best_traj_idx], variable_filter, sample_human_models]]
    #     #     running_variable_filter = variable_filter.copy()

    #     summary.append([best_mdp, best_traj, (best_env_idx, best_traj_idx), min_env_constraints_record[best_env_idx][best_traj_idx], variable_filter, sample_human_models])

    #     visited_env_traj_idxs.append((best_env_idx, best_traj_idx))

    #     particles.update(min_env_constraints_record[best_env_idx][best_traj_idx])

    #     # print(colored('Max infogain: {}'.format(max_info_gain), 'blue'))
    #     with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
    #         myfile.write('Max infogain: {}\n'.format(max_info_gain))
    #         myfile.write('\n')

    #     # print(colored('entropy: {}'.format(particles.calc_entropy()), 'blue'))


    #     # this method doesn't always finish, so save the summary along the way (for each completed unit)
    #     with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
    #         pickle.dump((summary, min_BEC_constraints_running, visited_env_traj_idxs, particles), f)

    # return summary, min_BEC_constraints_running, visited_env_traj_idxs, particles


    #######################################################################################
    ##################          Using pre-computed human models          ##################








# def obtain_team_summary_counterfactuals(data_loc, demo_strategy, team_models, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag, pool, team_size, n_human_models, consistent_state_count,
#                                 BEC_summary = None, n_train_demos=3, team_prior={}, downsample_threshold=float("inf"), consider_human_models_jointly=True, c=0.001, obj_func_proportion=1):

#     # TODO: This is still a work in progress

#     summary = []
#     visited_env_traj_idxs = []


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

#         # # print('Initial sample human models for person {} : {}'.format(p_id, sample_human_models))
        
#         if len(sample_human_models) == 0:
#             # print(colored("Likely cannot reduce the BEC further through additional demonstrations. Returning.", 'red'))
#             return summary, visited_env_traj_idxs

#         info_gains_record = []

#         # print("Length of summary: {}".format(summary_count))
#         with open('models/' + data_loc + '/teams_demo_gen_log.txt', 'a') as myfile:
#             myfile.write('Length of summary: {}\n'.format(summary_count))

#         for model_idx, human_model in enumerate(sample_human_models):
#             # print(colored('Model #: {}'.format(model_idx), 'red'))
#             # print(colored('Model val: {}'.format(human_model), 'red'))

#             with open('models/' + data_loc + '/teams_demo_gen_log.txt', 'a') as myfile:
#                 myfile.write('Model #: {}\n'.format(model_idx))
#                 myfile.write('Model val: {}\n'.format(human_model))

#             # based on the human's current model, obtain the information gain generated when comparing to the agent's
#             # optimal trajectories in each environment (human's corresponding optimal trajectories and constraints
#             # are saved for reference later)
#             # print("Obtaining counterfactual information gains:")

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


#         # print("Combining the most limiting constraints across human models:")
#         pool.restart()
#         args = [(i, len(sample_human_models), data_loc, summary_count, weights, step_cost_flag, variable_filter,
#                 traj_record[i], min_BEC_constraints_running, None) for
#                 i in range(len(traj_record))]
#         info_gains_record, min_env_constraints_record, n_diff_constraints_record, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = zip(
#             *pool.imap(BEC.combine_limiting_constraints_IG, tqdm(args)))
#         pool.close()
#         pool.join()
#         pool.terminate()

#         with open('models/' + data_loc + '/teams_info_gains_joint' + str(summary_count) + '.pickle', 'wb') as f:
#             pickle.dump(info_gains_record, f)

#         differing_constraint_count = 1          # number of constraints in the running human model that would differ after showing a particular demonstration
#         max_differing_constraint_count = max(list(itertools.chain(*n_diff_constraints_record)))
#         # print("max_differing_constraint_count: {}".format(max_differing_constraint_count))
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
#                                     # print("new max info: {}".format(max_info_gain))
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
#                                     # print("new max info: {}".format(max_info_gain))

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

#         # print("current max info: {}".format(max_info_gain))
        

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


#         # print(colored('Max infogain: {}'.format(max_info_gain), 'blue'))
#         with open('models/' + data_loc + '/teams_demo_gen_log.txt', 'a') as myfile:
#             myfile.write('Max infogain: {}\n'.format(max_info_gain))
#             myfile.write('\n')

#     # add any remaining demonstrations
#     summary.append(unit)


#     return summary, visited_env_traj_idxs