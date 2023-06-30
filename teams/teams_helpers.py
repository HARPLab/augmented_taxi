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
from policy_summarization import particle_filter as pf
from policy_summarization import probability_utils as p_utils
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = '1.0'
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'large'

import random



##########################################################




def calc_common_knowledge(team_knowledge, team_size, weights, step_cost_flag):


    for i in range(team_size):
        member_id = 'p' + str(i+1)
        if i==0:
            constraints = team_knowledge[member_id].copy()
        else:
            constraints.extend(team_knowledge[member_id])

    joint_constraints = BEC_helpers.remove_redundant_constraints(constraints, weights, step_cost_flag)

    return joint_constraints



def calc_joint_knowledge(team_knowledge, team_size, weights, step_cost_flag):

    # TODO: For place holder returns the constraints of member p1
    


    return team_knowledge['p1']



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

                

def compute_counterfactuals_team(args):
    data_loc, model_idx, env_idx, w_human_normalized, env_filename, trajs_opt, particles, min_BEC_constraints_running, step_cost_flag, summary_len, variable_filter, consider_human_models_jointly = args

    with open(env_filename, 'rb') as f:
        wt_vi_traj_env = pickle.load(f)

    agent = wt_vi_traj_env[0][1]
    weights = agent.mdp.weights

    human = copy.deepcopy(agent)
    mdp = human.mdp
    mdp.weights = w_human_normalized
    vi_human = ValueIteration(mdp, sample_rate=1)
    vi_human.run_vi()

    # only consider counterfactual trajectories from human models whose value iteration have converged
    if vi_human.stabilized:
        best_human_trajs_record_env = []
        constraints_env = []
        info_gain_env = []
        human_rewards_env = []
        overlap_in_opt_and_counterfactual_traj_env = []

        for traj_idx, traj_opt in enumerate(trajs_opt):
            constraints = []

            # b) contrast differing expected feature counts for each state-action pair along the agent's optimal trajectory
            best_human_trajs_record = []
            best_human_reward = 0
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

                # only store the reward of the full trajectory
                if sas_idx == 0:
                    best_human_reward = cur_best_reward
                    traj_opt_feature_count = mu_sa
                constraints.append(mu_sa - best_reward_features)
                best_human_trajs_record.append(best_human_traj)            


            if len(constraints) > 0:
                constraints = BEC_helpers.remove_redundant_constraints(constraints, weights, step_cost_flag)

            # print('Reduced set of constraints for model ', model_idx, 'after trajectory', traj_idx)


            # don't consider environments that convey information about a variable you don't currently wish to convey
            skip_demo = False
            if np.any(variable_filter):
                for constraint in constraints:
                    if abs(variable_filter.dot(constraint.T)[0, 0]) > 0:
                        # conveys information about variable designated to be filtered out, so block this demonstration
                        skip_demo = True

                # also prevent showing a trajectory that contains feature counts of a feature to be filtered out
                if abs(variable_filter.dot(traj_opt_feature_count.T)[0, 0]) > 0:
                    skip_demo = True

            if not skip_demo:
                if particles is not None:
                    info_gain = particles.calc_info_gain(constraints)
                else:
                    info_gain = BEC_helpers.calculate_information_gain(min_BEC_constraints_running, constraints,
                                                                       weights, step_cost_flag)
            else:
                info_gain = 0

            human_rewards_env.append(best_human_reward)
            best_human_trajs_record_env.append(best_human_trajs_record)
            constraints_env.append(constraints)
            info_gain_env.append(info_gain)

            if not consider_human_models_jointly:
                # you should only consider the overlap for the first counterfactual human trajectory (as opposed to
                # counterfactual trajectories that could've arisen from states after the first state)
                overlap_pct = BEC_helpers.calculate_counterfactual_overlap_pct(best_human_trajs_record[0], traj_opt)

                overlap_in_opt_and_counterfactual_traj_env.append(overlap_pct)

    # else just populate with dummy variables
    else:
        best_human_trajs_record_env = [[[]] for i in range(len(trajs_opt))]
        constraints_env = [[] for i in range(len(trajs_opt))]
        info_gain_env = [0 for i in range(len(trajs_opt))]
        if not consider_human_models_jointly:
            overlap_in_opt_and_counterfactual_traj_env = [float('inf') for i in range(len(trajs_opt))]
        human_rewards_env = [np.array([[0]]) for i in range(len(trajs_opt))]

    if summary_len is not None:
        with open('models/' + data_loc + '/teams_counterfactual_data_' + str(summary_len) + '/model' + str(model_idx) +
                  '/cf_data_env' + str(env_idx).zfill(5) + '.pickle', 'wb') as f:
            pickle.dump((best_human_trajs_record_env, constraints_env, human_rewards_env), f)

    if consider_human_models_jointly:
        return info_gain_env
    else:
        return info_gain_env, overlap_in_opt_and_counterfactual_traj_env



def combine_limiting_constraints_IG(args):
    '''
    Summary: combine the most limiting constraints across all potential human models for each potential demonstration
    '''
    env_idx, n_sample_human_models, data_loc, curr_summary_len, weights, step_cost_flag, variable_filter,\
    trajs_opt, min_BEC_constraints_running, particles = args

    info_gains_record = []
    min_env_constraints_record = []
    all_env_constraints = []
    n_diff_constraints = []               # number of constraints in the running human model that would differ after showing a particular demonstration

    # jointly consider the constraints generated by suboptimal trajectories by each human model
    for model_idx in range(n_sample_human_models):
        with open('models/' + data_loc + '/teams_counterfactual_data_' + str(curr_summary_len) + '/model' + str(
                model_idx) + '/cf_data_env' + str(
            env_idx).zfill(5) + '.pickle', 'rb') as f:
            best_human_trajs_record_env, constraints_env, human_rewards_env = pickle.load(f)
        all_env_constraints.append(constraints_env)

    all_env_constraints_joint = [list(itertools.chain.from_iterable(i)) for i in zip(*all_env_constraints)]
    
    # for each possible demonstration in each environment, find the non-redundant constraints across all human models
    # and use that to calculate the information gain for that demonstration
    for traj_idx in range(len(all_env_constraints_joint)):
        if len(all_env_constraints_joint[traj_idx]) > 1:
            min_env_constraints = BEC_helpers.remove_redundant_constraints(all_env_constraints_joint[traj_idx],
                                                                           weights, step_cost_flag)
        else:
            min_env_constraints = all_env_constraints_joint[traj_idx]

        min_env_constraints_record.append(min_env_constraints)

        # don't consider environments that convey information about a variable you don't currently wish to convey
        skip_demo = False
        if np.any(variable_filter):
            for constraint in min_env_constraints:
                if abs(variable_filter.dot(constraint.T)[0, 0]) > 0:
                    # conveys information about variable designated to be filtered out, so block this demonstration
                    skip_demo = True

                # also prevent showing a trajectory that contains feature counts of a feature to be filtered out
                if abs(variable_filter.dot(constraint.T)[0, 0]) > 0:
                    skip_demo = True

        if not skip_demo:
            if particles is not None:
                ig = particles_team[member_id].calc_info_gain(min_env_constraints)
            else:
                ig = BEC_helpers.calculate_information_gain(min_BEC_constraints_running, min_env_constraints, weights, step_cost_flag)
            info_gains_record.append(ig)

            hypothetical_constraints = min_BEC_constraints_running.copy()
            hypothetical_constraints.extend(min_env_constraints)
            hypothetical_constraints = BEC_helpers.remove_redundant_constraints(hypothetical_constraints,
                                                                                weights, step_cost_flag)
            overlapping_constraint_count = 0
            for arr1 in min_BEC_constraints_running:
                for arr2 in hypothetical_constraints:
                    if np.array_equal(arr1, arr2):
                        overlapping_constraint_count += 1
                        break

            max_diff = abs(len(hypothetical_constraints) - overlapping_constraint_count)

            n_diff_constraints.append(max_diff)
        else:
            info_gains_record.append(0)
            n_diff_constraints.append(0)

    # obtain the counterfactual human trajectories that could've given rise to the most limiting constraints and
    # how much it overlaps the agent's optimal trajectory
    human_counterfactual_trajs = [[] for i in range(len(min_env_constraints_record))]
    overlap_in_opt_and_counterfactual_traj = [[] for i in range(len(min_env_constraints_record))]
    overlap_in_opt_and_counterfactual_traj_avg = []

    return info_gains_record, min_env_constraints_record, n_diff_constraints, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs



def check_and_update_variable_filter(min_subset_constraints_record = None, variable_filter = None, nonzero_counter = None, initialize_filter_flag = False, no_info_flag = False):
    
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



def find_ascending_individual_knowledge(team_prior):

    # TODO for later;  currently hardcoded a random order)

    return ['p1', 'p3', 'p2']



def obtain_team_summary_counterfactuals(data_loc, demo_strategy, team_models, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag, pool, team_size, n_human_models, consistent_state_count,
                                BEC_summary = None, n_train_demos=3, team_prior={}, downsample_threshold=float("inf"), consider_human_models_jointly=True, c=0.001, obj_func_proportion=1):

    # TODO: This is still a work in progress




    # prior knowledge
    if demo_strategy =='individual_knowledge_low':
        ind_knowledge_ascending = find_ascending_individual_knowledge(team_prior)
        prior = team_prior[ind_knowledge_ascending[0]].copy()
    elif demo_strategy == 'individual_knowledge_high':
        ind_knowledge_ascending = find_ascending_individual_knowledge(team_prior)
        prior = team_prior[ind_knowledge_ascending[len(ind_knowledge_ascending) - 1]].copy()
    elif demo_strategy == 'joint_knowledge':
        prior = calc_joint_knowledge(team_prior)
    elif demo_strategy == 'common_knowledge':
        prior = calc_common_knowledge(team_prior)

    min_BEC_constraints_running = prior.copy() 


    ## demo generation loop
    while summary_count < n_train_demos:
        
        if summary_count == 0:
            if demo_strategy == 'individual_knowledge_low':
                sample_human_models = team_models[ind_knowledge_ascending[0]].copy()
            elif demo_strategy == 'individual_knowledge_high':
                sample_human_models = team_models[ind_knowledge_ascending[len(ind_knowledge_ascending) - 1]].copy()
            else:
                sample_human_models = BEC_helpers.sample_human_models_uniform(min_BEC_constraints_running, n_human_models)
        else:
            sample_human_models = BEC_helpers.sample_human_models_uniform(min_BEC_constraints_running, n_human_models)

        # print('Initial sample human models for person {} : {}'.format(p_id, sample_human_models))
        
        if len(sample_human_models) == 0:
            print(colored("Likely cannot reduce the BEC further through additional demonstrations. Returning.", 'red'))
            return summary, visited_env_traj_idxs

        info_gains_record = []

        print("Length of summary: {}".format(summary_count))
        with open('models/' + data_loc + '/teams_demo_gen_log.txt', 'a') as myfile:
            myfile.write('Length of summary: {}\n'.format(summary_count))

        for model_idx, human_model in enumerate(sample_human_models):
            print(colored('Model #: {}'.format(model_idx), 'red'))
            print(colored('Model val: {}'.format(human_model), 'red'))

            with open('models/' + data_loc + '/teams_demo_gen_log.txt', 'a') as myfile:
                myfile.write('Model #: {}\n'.format(model_idx))
                myfile.write('Model val: {}\n'.format(human_model))

            # based on the human's current model, obtain the information gain generated when comparing to the agent's
            # optimal trajectories in each environment (human's corresponding optimal trajectories and constraints
            # are saved for reference later)
            print("Obtaining counterfactual information gains:")

            cf_data_dir = 'models/' + data_loc + '/teams_counterfactual_data_' + str(summary_count) + '/model' + str(model_idx)
            os.makedirs(cf_data_dir, exist_ok=True)

            pool.restart()
            args = [(data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]), traj_record[i], None, min_BEC_constraints_running, step_cost_flag, summary_count, variable_filter, consider_human_models_jointly) for i in range(len(traj_record))]
            info_gain_envs = list(tqdm(pool.imap(compute_counterfactuals_team, args), total=len(args)))
            pool.close()
            pool.join()
            pool.terminate()

            info_gains_record.append(info_gain_envs)

        with open('models/' + data_loc + '/teams_info_gains_' + str(summary_count) + '.pickle', 'wb') as f:
            pickle.dump(info_gains_record, f)


        # do a quick check of whether there's any information to be gained from any of the trajectories
        no_info_flag = check_info_gain(info_gains_record, consistent_state_count)

        # no need to continue search for demonstrations if none of them will improve the human's understanding
        if no_info_flag:
            break


        print("Combining the most limiting constraints across human models:")
        pool.restart()
        args = [(i, len(sample_human_models), data_loc, summary_count, weights, step_cost_flag, variable_filter,
                traj_record[i], min_BEC_constraints_running, None) for
                i in range(len(traj_record))]
        info_gains_record, min_env_constraints_record, n_diff_constraints_record, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = zip(
            *pool.imap(combine_limiting_constraints_IG, tqdm(args)))
        pool.close()
        pool.join()
        pool.terminate()

        with open('models/' + data_loc + '/teams_info_gains_joint' + str(summary_count) + '.pickle', 'wb') as f:
            pickle.dump(info_gains_record, f)

        differing_constraint_count = 1          # number of constraints in the running human model that would differ after showing a particular demonstration
        max_differing_constraint_count = max(list(itertools.chain(*n_diff_constraints_record)))
        print("max_differing_constraint_count: {}".format(max_differing_constraint_count))
        no_info_flag = True
        max_info_gain = 1

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

                    # we're still in the same unit so try and optimize visuals wrt other demonstrations in this unit
                    best_env_idx, best_traj_idx = optimize_visuals_team(data_loc, best_env_idxs, best_traj_idxs, traj_record, unit)
                    no_info_flag = False
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
                    # we're still in the same unit so try and optimize visuals wrt other demonstrations in this unit
                    best_env_idx, best_traj_idx = optimize_visuals_team(data_loc, best_env_idxs, best_traj_idxs, traj_record, unit)
                    no_info_flag = False

        with open('models/' + data_loc + '/teams_best_env_idxs' + str(summary_count) + '.pickle', 'wb') as f:
            pickle.dump((best_env_idx, best_traj_idx, best_env_idxs, best_traj_idxs), f)

        print("current max info: {}".format(max_info_gain))
        

        if no_info_flag:
            break

        best_traj = traj_record[best_env_idx][best_traj_idx]

        filename = mp_helpers.lookup_env_filename(data_loc, best_env_idx)
        with open(filename, 'rb') as f:
            wt_vi_traj_env = pickle.load(f)
        best_mdp = wt_vi_traj_env[0][1].mdp
        best_mdp.set_init_state(best_traj[0][0]) # for completeness
        min_BEC_constraints_running.extend(min_env_constraints_record[best_env_idx][best_traj_idx])
        min_BEC_constraints_running = BEC_helpers.remove_redundant_constraints(min_BEC_constraints_running, weights, step_cost_flag)
        if (running_variable_filter == variable_filter).all():
            unit.append([demo_strategy, p_id, best_mdp, best_traj, (best_env_idx, best_traj_idx), min_env_constraints_record[best_env_idx][best_traj_idx], variable_filter, sample_human_models])
            summary_count += 1
        else:
            summary.append(unit)

            unit = [[demo_strategy, p_id, best_mdp, best_traj, (best_env_idx, best_traj_idx), min_env_constraints_record[best_env_idx][best_traj_idx], variable_filter, sample_human_models]]
            running_variable_filter = variable_filter.copy()
            summary_count += 1
        
        visited_env_traj_idxs.append((best_env_idx, best_traj_idx))


        print(colored('Max infogain: {}'.format(max_info_gain), 'blue'))
        with open('models/' + data_loc + '/teams_demo_gen_log.txt', 'a') as myfile:
            myfile.write('Max infogain: {}\n'.format(max_info_gain))
            myfile.write('\n')

    # add any remaining demonstrations
    summary.append(unit)


    return summary, visited_env_traj_idxs



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



def particles_for_demo_strategy(demo_strategy, team_knowledge, n_particles, teammate_idx=0):


    # prior knowledge
    if demo_strategy =='individual_knowledge_low':
        ind_knowledge_ascending = find_ascending_individual_knowledge(team_knowledge)
        prior = team_knowledge[ind_knowledge_ascending[teammate_idx]].copy()
    elif demo_strategy == 'individual_knowledge_high':
        ind_knowledge_ascending = find_ascending_individual_knowledge(team_knowledge)
        prior = team_knowledge[ind_knowledge_ascending[len(ind_knowledge_ascending) - teammate_idx - 1]].copy()
    elif demo_strategy == 'joint_knowledge':
        prior = calc_joint_knowledge(team_knowledge)
    elif demo_strategy == 'common_knowledge':
        prior = calc_common_knowledge(team_knowledge)


    # particles for human models+
    particle_positions = BEC_helpers.sample_human_models_uniform([], n_particles)
    particles = pf.Particles(particle_positions)
    particles.update(prior)


    return prior, particles




def obtain_team_summary(data_loc, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count,
                        n_train_demos, prior, particles_demo):


    summary_variant = 'counterfactual'
    # summary_variant = 'particle-filter'



    # obtain demo summary
    if summary_variant == 'counterfactual':
        current_BEC_summary, visited_env_traj_idxs = BEC.obtain_summary_counterfactual(data_loc, summary_variant, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count,
                                                                n_train_demos, prior=prior, downsample_threshold=float("inf"), consider_human_models_jointly=True, c=0.001, obj_func_proportion=1)
        
        # update the particle filter model according to the generated summary
        print('Len BEC summary: ', len(current_BEC_summary))

        for unit in current_BEC_summary:
            for summary in unit:
                # print('summary:', summary)
                particles_demo.update(summary[3])
        


    elif summary_variant == 'particle-filter':
        current_BEC_summary, visited_env_traj_idxs, particles_demo = BEC.obtain_summary_particle_filter(data_loc, particles_demo, summary_variant, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count,
                       n_train_demos=np.inf, prior=prior, downsample_threshold=float("inf"), consider_human_models_jointly=True, c=0.001, obj_func_proportion=1, min_info_gain=0.01)

    # save current summary temporarily
    if len(current_BEC_summary) > 0:
            with open('models/' + data_loc + '/current_BEC_summary_trial.pickle', 'wb') as f:
                pickle.dump((current_BEC_summary, visited_env_traj_idxs, particles_demo), f)


    return current_BEC_summary, visited_env_traj_idxs, particles_demo




def sample_team_pf(team_size, n_particles, weights, step_cost_flag, team_prior=None):

    particles_team = {}
    
    # particles for individual team members
    for i in range(team_size):
        member_id = 'p' + str(i+1)
        particles_team[member_id] = pf.Particles(BEC_helpers.sample_human_models_uniform([], n_particles))

        if team_prior is not None:
            particles_team[member_id].update(team_prior[member_id])
    
    # particles for aggregated team knowledge
    team_knowledgde_aggregates = ['common_knowledge', 'joint_knowledge']
    for agg in team_knowledgde_aggregates:
        particles_team[agg] = pf.Particles(BEC_helpers.sample_human_models_uniform([], n_particles))

        if team_prior is not None:
            if agg == 'common_knowledge':
                team_prior['common_knowledge'] = calc_common_knowledge(team_prior, team_size, weights, step_cost_flag)
                particles_team[agg].update(team_prior['common_knowledge'])
            elif agg == 'joint_knowledge':
                team_prior['joint_knowledge'] = calc_joint_knowledge(team_prior, team_size, weights, step_cost_flag)
                particles_team[agg].update(team_prior['joint_knowledge'])


    return team_prior, particles_team



def downsample_team_models_pf(particles_team, n_models):

    
    sampled_team_models = {}
    sampled_team_model_weights = {}

    for member_id in particles_team:
        sampled_human_model_idxs = []
        particles_team[member_id].cluster()
        if len(particles_team[member_id].cluster_centers) > n_models:
            # if there are more clusters than number of sought human models, return the spherical centroids of the top n
            # most frequently counted cluster indexes selected by systematic resampling
            while len(sampled_human_model_idxs) < n_models:
                indexes = p_utils.systematic_resample(particles_team[member_id].cluster_weights)
                unique_idxs, counts = np.unique(indexes, return_counts=True)
                # order the unique indexes via their frequency
                unique_idxs_sorted = [x for _, x in sorted(zip(counts, unique_idxs), reverse=True)]

                for idx in unique_idxs_sorted:
                    # add new unique indexes to the human models that will be considered for counterfactual reasoning
                    if idx not in sampled_human_model_idxs:
                        sampled_human_model_idxs.append(idx)

                    if len(sampled_human_model_idxs) == n_models:
                        break

            sampled_human_models = [particles_team[member_id].cluster_centers[i] for i in sampled_human_model_idxs]
            sampled_human_model_weights = np.array([particles_team[member_id].cluster_weights[i] for i in sampled_human_model_idxs])
            sampled_human_model_weights /= np.sum(sampled_human_model_weights)  # normalize
        elif len(particles_team[member_id].cluster_centers) == n_models:
            sampled_human_models = particles_team[member_id].cluster_centers
            sampled_human_model_weights = np.array(particles_team[member_id].cluster_weights) # should already be normalized
        else:
            # if there are fewer clusters than number of sought human models, use systematic sampling to determine how many
            # particles from each cluster to return (using the k-cities algorithm to ensure that they are diverse)
            indexes = p_utils.systematic_resample(particles_team[member_id].cluster_weights, N=n_models)
            unique_idxs, counts = np.unique(indexes, return_counts=True)

            sampled_human_models = []
            sampled_human_model_weights = []
            for j, unique_idx in enumerate(unique_idxs):
                # particles of this cluster
                clustered_particles = particles_team[member_id].positions[np.where(particles_team[member_id].cluster_assignments == unique_idx)]
                clustered_particles_weights = particles_team[member_id].weights[np.where(particles_team[member_id].cluster_assignments == unique_idx)]

                # use the k-cities algorithm to obtain a diverse sample of weights from this cluster
                pairwise = metrics.pairwise.euclidean_distances(clustered_particles.reshape(-1, 3))
                select_idxs = BEC_helpers.selectKcities(pairwise.shape[0], pairwise, counts[j])
                sampled_human_models.extend(clustered_particles[select_idxs])
                sampled_human_model_weights.extend(clustered_particles_weights[select_idxs])

            sampled_human_model_weights = np.array(sampled_human_model_weights)
            sampled_human_model_weights /= np.sum(sampled_human_model_weights)

        # update member models
        sampled_team_models[member_id] = sampled_human_models
        sampled_team_model_weights[member_id] = sampled_human_model_weights
    

    return sampled_team_models, sampled_team_model_weights



def show_demonstrations(summary, particles_demo, mdp_class, weights, visualize_pf_transition=False):
    # TODO: WIP
    

    for unit_idx, unit in enumerate(summary):
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
            print('Constraint ', n, 'for this unit: ', subunit[3])
            n += 1

            # update particle filter with demonstration's constraint
            particles_demo.update(subunit[3])
            
            # visualize the updated particle filter
            if visualize_pf_transition:
                visualize_transition(subunit[3], particles_demo, mdp_class, weights)
    


    return unit_constraints, running_variable_filter_unit





def visualize_transition(constraints, particles, mdp_class, weights=None, fig=None):

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

    # plot particles before and after the constraints
    particles.plot(fig=fig, ax=ax1, plot_prev=True)
    particles.plot(fig=fig, ax=ax3)

    # plot the constraints
    ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints)
    poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    for constraints in [constraints]:
        BEC_viz.visualize_planes(constraints, fig=fig, ax=ax2)
    for constraints in [constraints]:
        BEC_viz.visualize_planes(constraints, fig=fig, ax=ax1)
    for constraints in [constraints]:
        BEC_viz.visualize_planes(constraints, fig=fig, ax=ax3)
    BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax2, plot_ref_sphere=False, alpha=0.75)

    label_axes(ax1, mdp_class, weights)
    label_axes(ax2, mdp_class, weights)
    label_axes(ax3, mdp_class, weights)

    # New: Add what constraints are being shown in the demo to the plot
    x_loc = 0.5
    y_loc = 0.1
    for cnst in constraints:
        fig.text(x_loc, y_loc, str(cnst), fontsize=30)
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


def visualize_BEC_area(constraints):

    
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    for constraints in [constraints]:
        BEC_viz.visualize_planes(constraints, fig=fig, ax=ax1)

    plt.show()