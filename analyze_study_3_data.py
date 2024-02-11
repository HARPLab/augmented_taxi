import params
import dill as pickle
import numpy as np
import policy_summarization.BEC_helpers as BEC_helpers
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from policy_summarization import policy_summarization_helpers
import json
from collections import defaultdict
from simple_rl.utils import mdp_helpers
import policy_summarization.multiprocessing_helpers as mp_helpers
import os
import warnings
import pandas as pd
import dill as pickle
import copy
from termcolor import colored
import seaborn as sns

import sage.all
import sage.geometry.polyhedron.base as Polyhedron

import teams.teams_helpers as team_helpers
from teams import particle_filter_team as pf_team
from simple_rl.utils import make_mdp
import params_team as params
import policy_summarization.BEC_visualization as BEC_viz
from policy_summarization import BEC

warnings.simplefilter(action='ignore', category=FutureWarning)

'''
For managing and processing data related to the user study 
'''


def extract_study_data():

    path = 'data'

    try:
        with open(path + '/user_data.pickle', 'rb') as f:
            user_data = pickle.load(f)
    except:

        with open(path + '/dfs_f23_processed.pickle', 'rb') as f:
            dfs_users_processed, dfs_trials_processed, dfs_domain_processed  = pickle.load(f)

        # dfs_users_processed.to_csv(path + '/dfs_users_processed.csv')
        dfs_trials_processed.to_csv(path + '/dfs_trials_processed.csv')
        # dfs_domain_processed.to_csv(path + '/dfs_domain_processed.csv')

        columns_to_include = ['user_id', 'interaction_type', 'loop_condition', 'is_opt_response', 'unpickled_moves', 'domain', 'unpickled_mdp_parameters', 'coordinates', \
                            'unpickled_human_model_pf_pos', 'unpickled_human_model_pf_weights', 'unpickled_reward_ft_weights', 'opt_traj_reward', \
                            'test_difficulty', 'tag', 'age', 'gender', 'unpickled_ethnicity', 'education']
        
        # only taxi domain
        # dfs_trials_processed_at = dfs_trials_processed[(dfs_trials_processed['domain'] == 'at')]
        # study_data = dfs_trials_processed_at[columns_to_include]
        # dfs_trials_processed_at.to_csv(path + '/dfs_trials_processed_at.csv')
        # with open(path + '/dfs_trials_processed_at.pickle', 'wb') as f:
        #     pickle.dump(dfs_trials_processed_at, f)

        study_data = dfs_trials_processed[columns_to_include]  # all test domains

        # save study data
        with open(path + '/study_data.pickle', 'wb') as f:
            pickle.dump(study_data, f)

        study_data.to_csv(path + '/study_data.csv') # save as csv for easy manual debugging
        

        print('study_data: ', study_data)


        # print(dfs_trials_processed_at)


        unique_user_ids = study_data['user_id'].unique()
        N_interactions = []
        N_final_correct = []
        user_ids_low_learners = []
        user_ids_high_learners = []
        N_interactions_low_learners = []
        N_interactions_high_learners = []

        user_data = pd.DataFrame()
        
        for user_id in unique_user_ids:
            
            # calculate number of interactions
            N_interactions_user = len(study_data[study_data['user_id'] == user_id]) - 6
            N_interactions.append(N_interactions_user) # remove 6 final tests

            N_final_correct_user_at = len(study_data[(study_data['user_id'] == user_id) & (study_data['interaction_type'] == 'final test') \
                                                     & (study_data['is_opt_response'] == 1) & (study_data['domain'] == 'at')])
            N_final_correct_user_sb = len(study_data[(study_data['user_id'] == user_id) & (study_data['interaction_type'] == 'final test') \
                                                    & (study_data['is_opt_response'] == 1) & (study_data['domain'] == 'sb')])
            
            
            N_final_correct.append(N_final_correct_user_at+N_final_correct_user_sb)

            if N_final_correct_user_at + N_final_correct_user_sb  > 7:  # 2/3rd right answer --> high learner
                user_ids_high_learners.append(user_id)
                N_interactions_high_learners.append(N_interactions_user)
                learner_type = 'high'
            else:
                user_ids_low_learners.append(user_id)
                N_interactions_low_learners.append(N_interactions_user)
                learner_type = 'low'

            # save user data
            user_data_dict = {'user_id': user_id, 'N_interactions': N_interactions_user, 'N_final_correct_at': N_final_correct_user_at, 'N_final_correct_sb': N_final_correct_user_sb, \
                              'loop_condition': study_data[study_data['user_id'] == user_id]['loop_condition'].unique()[0], 'learner_type': learner_type}

            user_data = user_data.append(user_data_dict, ignore_index=True)


        
        # save user data
        with open(path + '/user_data.pickle', 'wb') as f:
            pickle.dump(user_data, f)

        user_data.to_csv(path + '/user_data.csv')
        
    
    # valid_data_idx = np.where(np.array(N_interactions) != 0)[0]
    # print('valid_data_idx: ', len(valid_data_idx))

    # valid_N_interactions = np.array(N_interactions)[valid_data_idx]
    # valid_N_final_correct = np.array(N_final_correct)[valid_data_idx]
    # print('valid_N_interactions: ', valid_N_interactions)
    # print('N_interactions: ', N_interactions[valid_data_idx], 'len unique_user_ids: ', len(unique_user_ids[valid_data_idx]), 'len N_interactions: ', len(N_interactions[valid_data_idx]))

    ### plots

    ## Interactions
    # fig, ax = plt.subplots(ncols=3)
    # ax[0].hist(valid_N_interactions, bins='auto')
    # ax[0].set_title('Histogram of learning interactions for each user')
    # ax[0].set_xlabel('Number of interactions')
    # ax[0].set_ylabel('Count')
    # ax[0].grid(True)

    # ax[1].hist(np.array(N_interactions_low_learners)[np.where(np.array(N_interactions_low_learners) != 0)[0]], bins='auto')
    # ax[1].set_title('Number of interactions for low leaners')
    # ax[1].set_xlabel('Number of interactions')
    # ax[1].set_ylabel('Count')
    # ax[1].grid(True)

    # ax[2].hist(np.array(N_interactions_high_learners)[np.where(np.array(N_interactions_high_learners) != 0)[0]], bins='auto')
    # ax[2].set_title('Number of interactions for high leaners')
    # ax[2].set_xlabel('Number of interactions')
    # ax[2].set_ylabel('Count')
    # ax[2].grid(True)

    # fig2, ax2 = plt.subplots()
    # ax2.hist(valid_N_final_correct, bins='auto')
    # ax2.set_title('Histogram of final correct responses for each user')
    # ax2.set_xlabel('Number of correct responses')
    # ax2.set_ylabel('Count')
    # ax2.grid(True)
    
    # plt.show()






    return 1

######################
def get_constraints():


    return 1




def run_sim_trials(loop_condition, interaction_data, domain, initial_learning_factor, vars_filename_prefix):

    # sim params
    learner_update_type = 'no_noise'
    max_learning_factor = params.max_learning_factor
    learning_factor = copy.deepcopy(initial_learning_factor)
    viz_flag = False
    
    # simulated filename
    vars_filename_prefix = vars_filename_prefix + '_' + initial_learning_factor 

    # load summary data
    path = 'models_user_study/augmented_taxi2'

    with open(path + '/base_constraints.pickle', 'rb') as f:
        policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(
            f)
        
    with open(path + '/BEC_summary.pickle', 'rb') as f:
        summary_data, visited_env_traj_idxs, particles = pickle.load(f)

    user_id = interaction_data['user_id']

    # # trial variables
    # demo_constraints = trial_data['unit_constraints']
    # min_demo_constraints = trial_data['min_KC_constraints']
    # test_constraints = trial_data['test_constraints']
    # test_responses_team = trial_data['test_constraints_team']
    # loop_count = trial_data['loop_count']
    # team_response_models = trial_data['team_response_models']
    # kc_id_list = trial_data['knowledge_comp_id']
    # team_learning_factor = trial_data['team_learning_factor']


    # initialize (simulated) teacher and learner particle filters
    learner_pf = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_learning_factor = learning_factor, team_prior = params.team_prior, pf_flag='learner', vars_filename=vars_filename_prefix, model_type = learner_update_type)

    # initialize dataframes to save probability data
    simulated_interaction_data = pd.DataFrame()


    # prior interaction data
    if domain == 'at':
        prior_test_constraints = [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]  # constraints of the first concept/KC
        min_BEC_constraints = [np.array([[1, 1, 0]]), np.array([[-1,  0,  2]]), np.array([[ 0, -1, -4]])]

    # prior probability of correct response/learning
    learner_pf.calc_particles_probability(prior_test_constraints)
    probability_correct_response_prior = learner_pf.particles_prob_correct

    learner_pf.calc_particles_probability(min_BEC_constraints)
    prop_particles_BEC = learner_pf.particles_prob_correct


    prior_interaction_data_dict = {'user_id': user_id, 'loop_condition': interaction_data['loop_condition'].iloc[0], 'loop_id': 0, 'demo_id': 0, 'kc_id': 1, 'learning_factor': learning_factor, \
                            'interaction_type': 'prior', 'test_constraints': prior_test_constraints, 'prob_correct_response': probability_correct_response_prior, 'prop_particles_BEC': prop_particles_BEC, \
                            'learner_pf_pos': learner_pf.positions, 'learner_pf_weights': learner_pf.weights}

    simulated_interaction_data = simulated_interaction_data.append(prior_interaction_data_dict, ignore_index=True)


    # simulate teaching loop
    loop_id = 1
    demo_id = 1
    for unit_idx, unit in enumerate(summary_data):
        unit_constraints = []
        running_variable_filter = unit[0][4]

        if domain == 'at':
            current_kc_id = unit_idx + 1
        
        # reset team learning factor for a new KC
        learning_factor = copy.deepcopy(initial_learning_factor)
        print('learning_factor reset to initial_team_learning_factor for new KC: ', current_kc_id, '. LF: ', learning_factor)


        # simulate showing demo
        for subunit in unit:
            unit_constraints.extend(subunit[3])
            
            # update particle filter for each demo
            learner_pf.update(subunit[3], learning_factor, plot_title = 'Learner belief after demo: ' + str(demo_id) + ' for unit: ' + str(unit_idx), viz_flag = viz_flag, vars_filename=vars_filename_prefix, model_type = learner_update_type)

            learner_pf.calc_particles_probability(subunit[3])
            prob_particles_correct = learner_pf.particles_prob_correct

            interaction_data_dict = {'user_id': user_id, 'loop_condition': interaction_data['loop_condition'].iloc[0], 'loop_id': loop_id, 'demo_id': demo_id, 'kc_id': current_kc_id, 'interaction_type': 'demo', \
                                     'test_constraints': test_constraints, 'pf_pos': pf_pos, 'pf_weights': pf_weights, 'prob_correct_response': prob_correct_response, 'prop_particles_BEC': prop_particles_BEC}
            # save interaction data
            interaction_data = interaction_data.append(interaction_data_dict, ignore_index=True)

            # update
            loop_id += 1
            demo_id += 1


        # update particle filter for demos of each concept/unit
        # learner_pf.update(unit_constraints, learning_factor, plot_title = 'Learner belief after unit', viz_flag = viz_flag, vars_filename=vars_filename_prefix, model_type = learner_update_type)
        

        





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # simulate teaching-learning interaction process
    for loop_id in range(len(loop_count)):
        print(colored('loop_id: ' + str(loop_id), 'red'))

        
        # if len(interesting_interactions) > 0:
        #     if loop_id+1 in interesting_interactions:
        #         viz_flag = True
        #     else:
        #         viz_flag = False

        # loop data
        demo_loop = demo_constraints.iloc[loop_id]
        min_demo_loop = min_demo_constraints.iloc[loop_id]
        test_loop = test_constraints.iloc[loop_id]
        test_responses_loop = test_responses_team.iloc[loop_id]
        team_models_loop = team_response_models.iloc[loop_id]
        current_kc_id = kc_id_list.iloc[loop_id]

        # check and reset team learning factor
        if loop_id > 0 and current_kc_id > prev_kc_id:
            team_learning_factor = copy.deepcopy(initial_team_learning_factor)
            print('team_learning_factor reset to initial_team_learning_factor for new KC: ', current_kc_id, '. TLF: ', team_learning_factor)

        # learner_pf_actual_after_demo_loop = learner_pf_actual_after_demo.iloc[loop_id]
        # prob_pf_actual_learner_before_test_loop = prob_pf_actual_learner_before_test_read_data.iloc[loop_id]
        # team_learning_factor_loop = team_learning_factor.iloc[loop_id]

        # initialize dicts to save probability data
        ## current PF (w test response and wo feedback)
        prob_pf_teacher_before_demo_dict = debug_calc_prob_mass_correct_side(min_demo_loop, teacher_pf, [params.default_learning_factor_teacher]*params.team_size)
        prob_pf_teacher_before_demo_dict['prob_type'] = 'teacher_before_demo'
        prob_pf_teacher_before_demo_dict['loop_id'] = loop_id+1
        prob_pf_teacher_before_demo_dict['constraints'] = min_demo_loop
        prob_pf_teacher_before_demo_dict['demo_strategy'] = demo_strategy
        prob_pf_teacher_before_demo_dict['team_composition'] = team_composition

        prob_pf_learner_before_demo_dict = debug_calc_prob_mass_correct_side(min_demo_loop, learner_pf, team_learning_factor)
        prob_pf_learner_before_demo_dict['prob_type'] = 'learner_before_demo'
        prob_pf_learner_before_demo_dict['loop_id'] = loop_id+1
        prob_pf_learner_before_demo_dict['constraints'] = min_demo_loop
        prob_pf_learner_before_demo_dict['demo_strategy'] = demo_strategy
        prob_pf_learner_before_demo_dict['team_composition'] = team_composition

        ## learner PF without 

        # update demo dataframes
        prob_pf = prob_pf.append(prob_pf_teacher_before_demo_dict, ignore_index=True)
        prob_pf = prob_pf.append(prob_pf_learner_before_demo_dict, ignore_index=True)

        # test dict
        prob_pf_teacher_before_test_dict = {'loop_id': loop_id+1,  'prob_type': 'teacher_before_test', 'demo_strategy': demo_strategy, 'team_composition': team_composition}
        prob_pf_learner_before_test_dict = {'loop_id': loop_id+1,  'prob_type': 'learner_before_test', 'demo_strategy': demo_strategy, 'team_composition': team_composition}
        prob_pf_teacher_after_test_dict = {'loop_id': loop_id+1,  'prob_type': 'teacher_after_test', 'demo_strategy': demo_strategy, 'team_composition': team_composition}
        prob_pf_teacher_after_feedback_dict = {'loop_id': loop_id+1,  'prob_type': 'teacher_after_feedback', 'demo_strategy': demo_strategy, 'team_composition': team_composition}
        prob_pf_learner_after_feedback_dict = {'loop_id': loop_id+1,  'prob_type': 'learner_after_feedback', 'demo_strategy': demo_strategy, 'team_composition': team_composition}

        # test constraints
        test_loop_extended = []
        for test_id in range(len(test_loop)):
            test_loop_extended.extend(test_loop[test_id])
        print('test_loop_extended: ', test_loop_extended)


        # update teacher and learner particle filters for demonstrations
        for p_id in range(params.team_size):
            member_id = 'p' + str(p_id+1)

            ### update teacher and learner particle filters for demonstrations
            print('Teacher update after demo for member: ', member_id)
            teacher_pf[member_id].update(demo_loop, teacher_learning_factor[p_id], plot_title = 'Simulated interaction No.' + str(loop_id +1) + '. Teacher belief after demo for member ' + member_id, viz_flag = viz_flag, vars_filename=vars_filename_prefix, model_type = params.teacher_update_model_type)
            # teacher_pf[member_id].calc_particles_probability(demo_loop)
            # prob_pf_teacher_after_demo_dict[member_id] = teacher_pf[member_id].particles_prob_correct

            print('Learner update after demo for member: ', member_id)
            learner_pf[member_id].update(demo_loop, team_learning_factor[p_id], plot_title = 'Simulated interaction No.' + str(loop_id +1) + '. Learner belief after demo for member ' + member_id, viz_flag = viz_flag, vars_filename=vars_filename_prefix, model_type = learner_update_type)
            # learner_pf[member_id].calc_particles_probability(demo_loop)
            # prob_pf_learner_after_demo_dict[member_id] = learner_pf[member_id].particles_prob_correct

        # update prob after demos
        prob_pf_teacher_after_demo_dict = debug_calc_prob_mass_correct_side(min_demo_loop, teacher_pf, [params.default_learning_factor_teacher]*params.team_size)
        prob_pf_teacher_after_demo_dict['loop_id'] = loop_id+1
        prob_pf_teacher_after_demo_dict['constraints'] = min_demo_loop
        prob_pf_teacher_after_demo_dict['prob_type'] = 'teacher_after_demo'
        prob_pf_teacher_after_demo_dict['demo_strategy'] = demo_strategy
        prob_pf_teacher_after_demo_dict['team_composition'] = team_composition
        

        prob_pf_learner_after_demo_dict = debug_calc_prob_mass_correct_side(min_demo_loop, learner_pf, team_learning_factor)
        prob_pf_learner_after_demo_dict['loop_id'] = loop_id+1
        prob_pf_learner_after_demo_dict['constraints'] = min_demo_loop
        prob_pf_learner_after_demo_dict['prob_type'] = 'learner_after_demo'
        prob_pf_learner_after_demo_dict['demo_strategy'] = demo_strategy
        prob_pf_learner_after_demo_dict['team_composition'] = team_composition

        # update demo dataframes
        prob_pf = prob_pf.append(prob_pf_teacher_after_demo_dict, ignore_index=True)
        prob_pf = prob_pf.append(prob_pf_learner_after_demo_dict, ignore_index=True)

        # plot sampled models
        print('team_models_loop: ', team_models_loop)
        if viz_flag:
            plot_title = 'Simulated interaction No.' + str(loop_id +1) + '. Human models sampled for test'
            sim_helpers.plot_sampled_models(learner_pf, test_loop_extended, team_models_loop, 1, plot_title = plot_title, vars_filename=vars_filename_prefix)
            # plot_title = 'Actual interaction No.' + str(loop_id +1) + '. Human models sampled for test'
            # sim_helpers.plot_sampled_models(learner_pf_actual_after_demo_loop, test_loop_extended, team_models_loop, 1, plot_title = plot_title, vars_filename=vars_filename_prefix)
            
            # if loop_id == 3:
            #     break
        print('test_responses_loop: ', test_responses_loop)



        # ### update teacher and learner particle filters for tests
            
        # # Method 2: update particle filter (after all test responses)
        N_tests = len(test_loop)
        
        
        for p_id in range(params.team_size):
            member_id = 'p' + str(p_id+1)
            lf_var = member_id + '_lf'
            
            if N_tests == 1:
                all_test_responses = test_responses_loop[p_id]
            else:
                all_test_responses = []
                for test_id in range(N_tests):
                    all_test_responses.append(test_responses_loop[p_id][test_id])
            print('all_test_responses: ', all_test_responses)


            # probabilities before test
            teacher_pf[member_id].calc_particles_probability(test_loop_extended)
            prob_pf_teacher_before_test_dict[member_id] = teacher_pf[member_id].particles_prob_correct
            prob_pf_teacher_before_test_dict['constraints'] = test_loop_extended
            prob_pf_teacher_before_test_dict[lf_var] = params.default_learning_factor_teacher

            learner_pf[member_id].calc_particles_probability(test_loop_extended)
            prob_pf_learner_before_test_dict[member_id] = learner_pf[member_id].particles_prob_correct
            prob_pf_learner_before_test_dict['constraints'] = test_loop_extended
            prob_pf_learner_before_test_dict[lf_var] = team_learning_factor[p_id]


            # update particle filter for test response (only for teacher)
            print('Teacher update after tests for member: ', member_id)
            teacher_pf[member_id].update(all_test_responses, teacher_learning_factor[p_id], model_type = params.teacher_update_model_type,  plot_title = 'Simulated interaction No.' + str(loop_id +1) + '. Teacher belief after test for member ' + member_id, viz_flag = viz_flag, vars_filename=vars_filename_prefix)

            # probabibilites after test
            teacher_pf[member_id].calc_particles_probability(test_loop_extended)
            prob_pf_teacher_after_test_dict[member_id] = teacher_pf[member_id].particles_prob_correct
            prob_pf_teacher_after_test_dict['constraints'] = test_loop_extended
            prob_pf_teacher_after_test_dict['test_response'] = all_test_responses
            prob_pf_teacher_after_test_dict[lf_var] = params.default_learning_factor_teacher


            ## update particle filter for feedback
            teacher_feedback_lf = []
            learner_feedback_lf = []

            # 
            # for test_id in range(N_tests):
                # if (all_test_responses[test_id] == test_loop_extended[test_id]).all():
                #     teacher_feedback_lf.append(min(1.1*params.default_learning_factor_teacher, max_learning_factor))
                #     learner_feedback_lf.append(min(1.05*team_learning_factor[p_id], max_learning_factor))
                # else:
                #     teacher_feedback_lf.append(min(1.05*params.default_learning_factor_teacher, max_learning_factor))
                #     learner_feedback_lf.append(min(1.05*team_learning_factor[p_id], max_learning_factor))
            
            all_tests_correct_flag = True
            for test_id in range(N_tests):
                if not (all_test_responses[test_id] == test_loop_extended[test_id]).all():
                    all_tests_correct_flag = False
                    break
            
            # update learning factor before feedback
            if feedback_flag:
                if all_tests_correct_flag:
                    team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_factor_delta[p_id, 0], max_learning_factor)
                else:
                    team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_factor_delta[p_id, 1], max_learning_factor)

            if review_flag:
                print('Teacher update after feedback for member: ', member_id)
                teacher_pf[member_id].update(test_loop_extended, teacher_learning_factor[p_id], model_type = params.teacher_update_model_type,  plot_title = 'Simulated interaction No.' + str(loop_id +1) + '. Teacher belief after corrective feedback for member ' + member_id, viz_flag = viz_flag, vars_filename=vars_filename_prefix)

                print('Learner update after feedback for member: ', member_id)
                learner_pf[member_id].update(test_loop_extended, team_learning_factor[p_id], model_type = learner_update_type, plot_title = 'Simulated interaction No.' + str(loop_id +1) + '. Learner belief after corrective feedback for member ' + member_id, viz_flag = viz_flag, vars_filename=vars_filename_prefix)

            # # update learning factor after feedback
            # if all_tests_correct_flag:
            #     team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 0], max_learning_factor)
            # else:
            #     team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 1], max_learning_factor)

            # probabilities after feedback
            teacher_pf[member_id].calc_particles_probability(test_loop_extended)
            prob_pf_teacher_after_feedback_dict[member_id] = teacher_pf[member_id].particles_prob_correct
            prob_pf_teacher_after_feedback_dict['constraints'] = test_loop_extended
            prob_pf_teacher_after_feedback_dict[lf_var] = teacher_feedback_lf


            learner_pf[member_id].calc_particles_probability(test_loop_extended)
            prob_pf_learner_after_feedback_dict[member_id] = learner_pf[member_id].particles_prob_correct
            prob_pf_learner_after_feedback_dict['constraints'] = test_loop_extended
            prob_pf_learner_after_feedback_dict[lf_var] = learner_feedback_lf
        
        #########

            # response type and update team learning factor
            print('all_test_responses: ', all_test_responses, 'test_loop_extended: ', test_loop_extended)
            prob_pf_learner_before_test_dict['N_tests'] = N_tests
            prob_pf_teacher_before_test_dict['N_tests'] = N_tests
            prob_pf_teacher_after_test_dict['N_tests'] = N_tests
            
            tests_correct_flag = True
            for test_id in range(N_tests):
                
                if (all_test_responses[test_id] == test_loop_extended[test_id]).all():
                    # team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 0], max_learning_factor)
                    var_name = 'response_type_' + member_id
                    if test_id == 0:
                        prob_pf_learner_before_test_dict[var_name] = ['correct']
                        prob_pf_teacher_before_test_dict[var_name] = ['correct']
                        prob_pf_teacher_after_test_dict[var_name] = ['correct']
                        prob_pf_teacher_after_feedback_dict[var_name] = ['correct']
                        prob_pf_learner_after_feedback_dict[var_name] = ['correct']
                    else:
                        prob_pf_learner_before_test_dict[var_name].append('correct')
                        prob_pf_teacher_before_test_dict[var_name].append('correct')
                        prob_pf_teacher_after_test_dict[var_name].append('correct')
                        prob_pf_teacher_after_feedback_dict[var_name].append('correct')
                        prob_pf_learner_after_feedback_dict[var_name].append('correct')
                else:
                    # team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 1], max_learning_factor)
                    tests_correct_flag = False
                    var_name = 'response_type_' + member_id
                    if test_id == 0:
                        prob_pf_learner_before_test_dict[var_name] = ['incorrect']
                        prob_pf_teacher_before_test_dict[var_name] = ['incorrect']
                        prob_pf_teacher_after_test_dict[var_name] = ['incorrect']
                        prob_pf_teacher_after_feedback_dict[var_name] = ['incorrect']
                        prob_pf_learner_after_feedback_dict[var_name] = ['incorrect']
                    else:
                        prob_pf_learner_before_test_dict[var_name].append('incorrect')
                        prob_pf_teacher_before_test_dict[var_name].append('incorrect')
                        prob_pf_teacher_after_test_dict[var_name].append('incorrect')
                        prob_pf_teacher_after_feedback_dict[var_name].append('incorrect')
                        prob_pf_learner_after_feedback_dict[var_name].append('incorrect')

            # # update learning factor when all tests are correct
            # if tests_correct_flag:
            #     team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 0], max_learning_factor)
            # else:
            #     team_learning_factor[p_id] = min(team_learning_factor[p_id] + team_learning_rate[p_id, 1], max_learning_factor)

        # update test dataframes
        prob_pf = prob_pf.append(prob_pf_teacher_before_test_dict, ignore_index=True)
        prob_pf = prob_pf.append(prob_pf_learner_before_test_dict, ignore_index=True)
        prob_pf = prob_pf.append(prob_pf_teacher_after_test_dict, ignore_index=True)
        prob_pf = prob_pf.append(prob_pf_teacher_after_feedback_dict, ignore_index=True)
        prob_pf = prob_pf.append(prob_pf_learner_after_feedback_dict, ignore_index=True)
        
        # update KC
        prev_kc_id = current_kc_id
        

    #################
    # save dataframes final
        
    prob_pf.to_csv(path + '/' + vars_filename_prefix + '_prob_pf.csv')
    with open(path + '/' + vars_filename_prefix + '_prob_pf.pickle', 'wb') as f:
        pickle.dump(prob_pf, f)
    ####################

    # plot simulated probability data
    # f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(15,10))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # # Method 1
    # for col_id in range(3):  
    #     member_id = 'p' + str(col_id+1)
    #     sns.lineplot(prob_pf_learner_before_test, x=prob_pf_learner_before_test.index, y = member_id, hue = 'test_id', ax=ax[col_id], errorbar=('se', 1), err_style="band").set(title='Simulated Prob. correct test response member'+ member_id)

    # # Method 2
    # for col_id in range(3):  
    #     member_id = 'p' + str(col_id+1)
    #     sns.lineplot(prob_pf_learner_before_test, x=prob_pf_learner_before_test.index, y = member_id, ax=ax[col_id], errorbar=('se', 1), err_style="band").set(title='Simulated Prob. correct test response member'+ member_id)
    # if params.save_plots_flag:
    #     plt.savefig(path + '/' + vars_filename_prefix + '_simulated_probability.png')

    # # plot actual probability data
    # f2, ax2 = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(15,10))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # # for col_id in range(3):  
    # #     member_id = 'p' + str(col_id+1)
    # #     sns.lineplot(prob_pf_actual_learner_before_test, x=prob_pf_actual_learner_before_test.index, y = member_id, hue = 'test_id', ax=ax2[col_id], errorbar=('se', 1), err_style="band").set(title='Actual Prob. correct test response member'+ member_id)

    # for col_id in range(3):  
    #     member_id = 'p' + str(col_id+1)
    #     sns.lineplot(prob_pf_actual_learner_before_test, x=prob_pf_actual_learner_before_test.index, y = member_id, ax=ax2[col_id], errorbar=('se', 1), err_style="band").set(title='Actual Prob. correct test response member'+ member_id)

    # if params.save_plots_flag:
    #     plt.savefig(path + '/' + vars_filename_prefix + '_actual_probability.png')

    # plt.show()
    

    
        


    return 1




def run_MC_sims(user_data, study_data):

    unique_user_ids = user_data['user_id'].unique()

    uf_list = np.arange(0.55, 1, 0.02)

    for id, user_data_row in user_data.iterrows():
        user_id = user_data_row['user_id']
        N_interactions = user_data_row['N_interactions']
        learner_type = user_data_row['learner_type']
        loop_condition = user_data_row['loop_condition']

        user_study_data = study_data[study_data['user_id'] == user_id]

        for uf in uf_list:
            run_sim_trials(loop_condition, user_study_data, uf)
       
        break


    return 1
###################

def analyze_learning_dynamics(user_data, study_data):

    ## compile interaction data

    # unpickle data
    # study_data['up_human_model_pf_pos'] = study_data['human_model_pf_pos'].apply(lambda x: pickle.loads(x))
    # study_data['up_human_model_pf_weights'] = study_data['human_model_pf_weights'].apply(lambda x: pickle.loads(x))
    # study_data['up_mdp_parameters'] = study_data['mdp_parameters'].apply(lambda x: pickle.loads(x))
    # study_data['up_moves'] = study_data['moves'].apply(lambda x: pickle.loads(x))

    # calculate the probability of correct response for prior knowledge
    _, particles_prior_team = team_helpers.sample_team_pf(1, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, teacher_learning_factor = [0.8], prior_lf = 0.8, team_prior = params.team_prior, model_type = 'no_noise')
    particles_prior = copy.deepcopy(particles_prior_team['p1'])
    prior_test_constraints = [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]
    particles_prior.calc_particles_probability(prior_test_constraints)
    probability_correct_response_prior = particles_prior.particles_prob_correct

    # fixed test constraints
    test_constraints_domain = {'at': {  1: [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], \
                                        2: [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], \
                                        3: [np.array([[1, 1, 0]])]}
                        }
    
    min_BEC_constraints = {'at': [np.array([[1, 1, 0]]), np.array([[-1,  0,  2]]), np.array([[ 0, -1, -4]])]}



    # save the known interaction data - id, interaction id, etc
    interaction_data = pd.DataFrame()

    study_data = study_data[study_data['domain'] == 'at']  # use only augmented taxi domain for now

    user_list = user_data['user_id'].unique()
    for user_id in user_list:
        user_study_data = study_data[study_data['user_id'] == user_id]
        
        # identify demo and kc ids
        kc_list = []
        cur_kc_id = 0
        demo_id_list = []
        cur_demo_id = 0
        prev_interaction_type = ''
        for int_id in range(len(user_study_data)):
            cur_interaction_type = user_study_data['interaction_type'].iloc[int_id]
            
            if 'demo' == cur_interaction_type and 'demo' != prev_interaction_type:
                cur_kc_id += 1

            if 'demo' in cur_interaction_type:
                cur_demo_id += 1
            
            if cur_interaction_type == 'final test':
                kc_list.append(-1)
            else:
                kc_list.append(cur_kc_id)
            
            if 'demo' not in cur_interaction_type:
                demo_id_list.append(-1)
            else:
                demo_id_list.append(cur_demo_id)

            prev_interaction_type = cur_interaction_type
            

        print(kc_list)
        print(demo_id_list)

        # prior data
        prior_interaction_data_dict = {'user_id': user_id, 'loop_condition': user_study_data['loop_condition'].iloc[0], 'interaction_id': 0, 'loop_id': 0, 'demo_id': 0, 'kc_id': 1, \
                                 'interaction_type': 'prior', 'test_constraints': prior_test_constraints, 'prob_correct_response': probability_correct_response_prior}
        interaction_data = interaction_data.append(prior_interaction_data_dict, ignore_index=True)


        # interaction data
        domain = user_study_data['domain'].iloc[0]
        prev_interaction_type = ''  
        for int_id in range(len(user_study_data)):
            cur_interaction_type = user_study_data['interaction_type'].iloc[int_id]
            
            # # skip if final test
            # if cur_interaction_type == 'final_test':
            #     continue

            cur_kc_id = kc_list[int_id]
            cur_demo_id = demo_id_list[int_id]
            mdp_params = user_study_data['unpickled_mdp_parameters'].iloc[int_id]   ## need MDP parameters to identify the type of final test (whether evaluates mud only or both mud and battery)

            # get test constrainst for this KC/final test
            if cur_kc_id != -1:
                test_constraints = test_constraints_domain[domain][cur_kc_id]

                # create a PF object from position and weights and calculate the probability of correct response
                pf_pos = user_study_data['unpickled_human_model_pf_pos'].iloc[int_id]
                pf_weights = user_study_data['unpickled_human_model_pf_weights'].iloc[int_id]
                particles_interaction = pf_team.Particles_team(pf_pos, 0.8) # do not use PF for updating, rather create a new PF object for every state change
                particles_interaction.weights = pf_weights
            else:
                # find all min_BEC_constraints for the final tests!
                if len(mdp_params['hotswap_station']) > 0:
                    max_kc = 3
                else:
                    max_kc = 1
                print(mdp_params['hotswap_station'], max_kc)

                all_test_constraints = []
                for id in range(1, max_kc+1):
                    all_test_constraints.extend(test_constraints_domain[domain][id])
                test_constraints = BEC_helpers.remove_redundant_constraints(all_test_constraints, params.weights['val'], params.step_cost_flag)

                # use the previous PF (retains already); final tests are at the end, so the PF will be that of the last diagnostic test!
            
            

            print('interaction type: ', cur_interaction_type, '. test_constraints:', test_constraints)

            # probability of correct response
            particles_interaction.calc_particles_probability(test_constraints)
            prob_correct_response =  particles_interaction.particles_prob_correct

            # proportion of particles within BEC area
            particles_interaction.calc_particles_probability(min_BEC_constraints[domain])
            prop_particles_BEC = particles_interaction.particles_prob_correct


            ## compile interaction data
            interaction_data_dict = {'user_id': user_id, 'loop_condition': user_study_data['loop_condition'].iloc[int_id], 'loop_id': int_id+1, 'demo_id': cur_demo_id, 'kc_id': cur_kc_id, 'interaction_type': cur_interaction_type, \
                                     'test_constraints': test_constraints, 'pf_pos': pf_pos, 'pf_weights': pf_weights, 'prob_correct_response': prob_correct_response, 'prop_particles_BEC': prop_particles_BEC}
            # save interaction data
            interaction_data = interaction_data.append(interaction_data_dict, ignore_index=True)
            
            
            
            
            # # ## mdp setting and plot response trajectory           
            # if cur_interaction_type == 'final test':
            #     # get demo, test or response constraints from mdp params and moves
            #     # mdp_params = user_study_data['unpickled_mdp_parameters'].iloc[int_id]
            #     mdp_domain = 'augmented_taxi2' if study_data[study_data['user_id'] == user_id]['domain'].iloc[0] == 'at' else 'skateboard2'
            #     mdp_params['weights'] = params.mdp_parameters['weights']
            #     mdp_user = make_mdp.make_custom_mdp(mdp_domain, mdp_params)

            #     moves_list = user_study_data['unpickled_moves'].iloc[int_id]
                
            #     print('mdp_user: ', mdp_user)
            #     print('moves_list: ', moves_list)
            #     print(mdp_params)

            #     mdp_user.reset()
            #     trajectory = []
            #     cur_state = mdp_user.get_init_state()

            #     for idx in range(len(moves_list)):
            #         # assumes that the user study only allows actions that change the state of the MDP
            #         reward, next_state = mdp_user.execute_agent_action(moves_list[idx])
            #         trajectory.append((cur_state, moves_list[idx], next_state))

            #         # deepcopy occurs within transition function
            #         cur_state = next_state

            #     print('trajectory: ', trajectory)
            #     print('reward features: ', mdp_user.accumulate_reward_features(trajectory))

            #     mdp_user.visualize_trajectory(trajectory)

            # # human_reward = mdp_user.weights.dot(mdp_user.accumulate_reward_features(trajectory).T)
            # #################
            

    # save interaction data
    with open('data/interaction_data.pickle', 'wb') as f:
        pickle.dump(interaction_data, f)
    
    interaction_data.to_csv('data/interaction_data.csv')



    return 1



def plot_learning_dynamics(user_data):

    def squarest_rectangle(area):
        # Initialize variables to store the dimensions of the squarest rectangle
        length = 1
        width = area
        
        # Initialize the minimum difference between length and width
        min_difference = abs(length - width)

        while min_difference > 2:

            area += 1
        
            # Iterate over all possible lengths up to the square root of the area
            for l in range(1, int(area**0.5) + 1):
                # Check if the length divides the area evenly
                if area % l == 0:
                    w = area // l  # Calculate the corresponding width
                    
                    # Calculate the difference between length and width
                    difference = abs(l - w)
                    
                    # Update dimensions if the current rectangle is "squarer"
                    if difference < min_difference:
                        length = l
                        width = w
                        min_difference = difference

        return length, width
    

    with open('data/interaction_data.pickle', 'rb') as f:
        interaction_data = pickle.load(f)
    
    # interaction_data.to_csv('data/interaction_data.csv')

    user_list = user_data['user_id'].unique()

    # interesting_user = [30, 42, 16, 18, 5, 32, 76, 147, 200, 228, 68, 108]

    interesting_user = [30]

    # 30 - Closed Loop; Low learner; few interactions
    # 42 - Closed Loop; Low learner; many interactions
    # 16 - Closed Loop; Low  learner; worst performance
    # 18 - Closed Loop; Low learner; significant domain difference (6/6 in at, 1/6 in skateboard)
    # 5 - Closed Loop; High learner; few interactions
    # 32 - Closed Loop; High learner; many interactions
    # 76 - Closed Loop; High learner; high performance
    # 147 - Open loop; Low learner; low performance
    # 200 - Open loop; Low learner; significant domain difference (6/6 in at, 0/6 in skateboard)
    # 228 - Open loop; Low learner; high performance
    # 68 - Open loop; High learner; low performance
    # 108 - Open loop; High learner; high performance


    # interesting_user = [1]

    color_dict = {'demo': 'blue', 'remedial demo': 'purple', 'diagnostic test': 'red',  'remedial test': 'pink', 'diagnostic feedback': 'yellow', 'remedial feedback': 'orange', 'final test': 'green'}

    # # plot learning dynamics (after final demos just before a test/remedial test)
    for user_id in interesting_user:
        # cur_user_data = interaction_data[(interaction_data['user_id'] == user_id) & ( (interaction_data['interaction_type'] == 'demo') | (interaction_data['interaction_type'] == 'remedial demo') | (interaction_data['interaction_type'] == 'prior')) ]
        # cur_user_data = interaction_data[(interaction_data['user_id'] == user_id) & (interaction_data['interaction_type'] != 'final test')]
        cur_user_data = interaction_data[(interaction_data['user_id'] == user_id)]
        print(cur_user_data)
        loop_condition = cur_user_data['loop_condition'].iloc[0]
        kc_id_list = cur_user_data['kc_id'].unique()
        
        user_plot_data = pd.DataFrame()
        cur_comb_demo_id = 0
        
        concept_end_id = []
        for kc_id in kc_id_list:
            # user_kc_data = cur_user_data[(cur_user_data['kc_id'] == kc_id) & ((cur_user_data['demo_id'] == max_demo_id_in_kc) | (cur_user_data['interaction_type'] == 'remedial demo') | (cur_user_data['interaction_type'] == 'prior') )]

            user_kc_data = cur_user_data[(cur_user_data['kc_id'] == kc_id)]

            print('user_kc_data: ', user_kc_data) 

            combined_demo_id_list = [cur_comb_demo_id + id for id in range(len(user_kc_data))]
            cur_comb_demo_id = combined_demo_id_list[-1] + 1
            user_kc_data['combined_demo_id'] = combined_demo_id_list

            concept_end_id.append(user_kc_data['combined_demo_id'].iloc[-1])
            
            print('combined_demo_id_list: ', combined_demo_id_list)
            print('user_kc_data: ', user_kc_data) 
            user_plot_data = user_plot_data.append(user_kc_data, ignore_index=True)

        print('user_plot_data: ', user_plot_data)


        # plot learning dynamics - probability of correct response
        plt.figure(user_id)
        sns.lineplot(data=user_plot_data, x='combined_demo_id', y='prob_correct_response', color = 'black').set(title='Learning dynamics-conceptwise for user: ' + str(user_id) + '. Condition: ' + str(loop_condition))
        plt.ylim([0, 1])
        for id, row in user_plot_data.iterrows():
            if row['interaction_type'] != 'prior':
                # plt.axvline(x=row['combined_demo_id'], color='red', linestyle='--')
                plt.axvspan(user_plot_data['combined_demo_id'].iloc[id-1], row['combined_demo_id'], alpha=0.2, color=color_dict[row['interaction_type']])

        for id in concept_end_id:
            plt.axvline(x=id, color='black', linestyle='--')

        
        # plot learning dynamics - proportion of particles within BEC area
        plt.figure(user_id+100)
        sns.lineplot(data=user_plot_data, x='combined_demo_id', y='prop_particles_BEC', color = 'black').set(title='Learning dynamics-overall for user: ' + str(user_id) + '. Condition: ' + str(loop_condition))
        plt.ylim([0, 1])
        for id, row in user_plot_data.iterrows():
            if row['interaction_type'] != 'prior':
                # plt.axvline(x=row['combined_demo_id'], color='red', linestyle='--')
                plt.axvspan(user_plot_data['combined_demo_id'].iloc[id-1], row['combined_demo_id'], alpha=0.2, color=color_dict[row['interaction_type']])
        
        for id in concept_end_id:
            plt.axvline(x=id, color='black', linestyle='--')


        # plot particle distribution
        fig = plt.figure()
        
        row_len, col_len = squarest_rectangle(len(user_plot_data))
        print('row_len: ', row_len, '. col_len: ', col_len)
        axs = np.array([fig.add_subplot(row_len, col_len, i+1, projection='3d') for i in range(len(user_plot_data))])   
        
        row_id, col_id = 0, 0
        for id, row in user_plot_data.iterrows(): 

            if row['interaction_type'] == 'prior':
                _, particles_all = team_helpers.sample_team_pf(1, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, teacher_learning_factor = [0.8], prior_lf = 0.8, team_prior = params.team_prior, model_type = 'no_noise')
                particles = copy.deepcopy(particles_all['p1'])
                # print('particle weights: ', particles.weights)
            else:
                particles = pf_team.Particles_team(row['pf_pos'], 0.8)
                # print('row[pf_weights]: ', row['pf_weights'])
                particles.weights = row['pf_weights']
            
            test_constraints = row['test_constraints'] 
            particles.calc_particles_probability(test_constraints)

            print('id: ', id, '. test_constraints: ', test_constraints, '. prob_correct_response: ', particles.particles_prob_correct)
            print('row_id: ', row_id, '. col_id: ', col_id)
            particles.plot(fig=fig, ax=axs[id])
            BEC_viz.visualize_planes(test_constraints, fig=fig, ax=axs[id])
            ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(test_constraints)
            poly = Polyhedron.Polyhedron(ieqs=ieqs)
            BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=axs[id], plot_ref_sphere=False, alpha=0.75)
            axs[id].set_title('id ' + str(id) + ': ' + row['interaction_type'])

            if col_id < col_len-1:
                col_id += 1
            else:
                col_id = 0
                row_id += 1
        
        fig.suptitle('Particle distribution for user: ' + str(user_id) + '. Condition: ' + str(loop_condition))

        
        # debug
        # actual_test_response = [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]
        # sampled_test_response = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1]

        # plt.figure()

        # plt.plot(np.arange(len(actual_test_response)), actual_test_response)
        # plt.plot(np.arange(len(sampled_test_response)), sampled_test_response)
        # plt.xlabel('Test no')
        # plt.ylabel('Response')
        # plt.legend(['Actual', 'Sampled'])

    plt.show()

###############
    
def read_summary_data(path, file_list):

    with open(path + '/base_constraints.pickle', 'rb') as f:
        policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(
            f)
        
    data_loc = 'augmented_taxi2'


    # # initiliaze particles
    # learner_uf = 0.8

    # _, particles_all = team_helpers.sample_team_pf(1, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_learning_factor = [learner_uf], pf_flag = 'learner', prior_lf = 0.8, team_prior = params.team_prior, model_type = 'no_noise', '')
    # particles = copy.deepcopy(particles_all['p1'])


    for file in file_list:
        with open(path + '/' + file, 'rb') as f:
            summary_data, visited_env_traj_idxs, particles = pickle.load(f)
            # summary_data, visited_env_traj_idxs = pickle.load(f)
        
        demo_ids_list = [subunit[2] for unit in summary_data for subunit in unit ]
        constraints_list = [subunit[3] for unit in summary_data for subunit in unit ]
        
        running_visited_env_traj_idxs = []

        # demo_ids_list = []
        # for unit in summary_data:
            # for subunit in unit:
            #     constraints = subunit[3]
            #     variable_filter = subunit[4]
            #     if len(demo_ids_list) == 0:
            #         demo_ids_list = [subunit[2]]
            #     else:
            #         demo_ids_list = demo_ids_list.append(subunit[2])
        
        # visualize demos
        # for unit in summary_data:
        #     team_helpers.show_demonstrations(unit,[],[],[],[],viz_flag=True)
        # team_helpers.show_demonstrations(summary_data[0],[],[],[],[],viz_flag=True)

        # generate test demos
        for unit_idx, unit in enumerate(summary_data):

            unit_constraints = []
            running_variable_filter = unit[0][4]

            for subunit in unit:
                unit_constraints.extend(subunit[3])

            min_constraints = BEC_helpers.remove_redundant_constraints(unit_constraints, params.weights['val'], params.step_cost_flag)

            preliminary_tests, visited_env_traj_idxs = BEC.obtain_diagnostic_tests(data_loc, unit, visited_env_traj_idxs, min_constraints, min_subset_constraints_record, traj_record, traj_features_record, running_visited_env_traj_idxs, mdp_features_record)

            for test in preliminary_tests:
                test_constraints = test[3]
                print('unit_idx: ', unit_idx, 'test_constraints: ', test_constraints)

        
        
        print('file: ', file, '. demo_ids: ', demo_ids_list, '. constraints_list: ', constraints_list, '. visited_env_traj_idxs: ', visited_env_traj_idxs)
                

    return 1

###################
    



##################


if __name__ == "__main__":

    # # find human response distribution
    # extract_study_data()
    ########

    ## load study data
    path = 'data'
    with open(path + '/user_data.pickle', 'rb') as f:
        user_data = pickle.load(f)
    with open(path + '/study_data.pickle', 'rb') as f:
        study_data = pickle.load(f)

    # ## Run Monte Carlo simulations


    # run_MC_sims(user_data, study_data)
    ##########

    ## analyze learning dynamics
    # analyze_learning_dynamics(user_data, study_data)


    ## plot learning dynamics
    # plot_learning_dynamics(user_data)
        
    
    # ## (Debug) read summary data
    path = 'models_user_study/augmented_taxi2'
    # # file_list = ['BEC_summary.pickle', 'BEC_summary_2.pickle', 'BEC_summary_3.pickle']
    # # file_list = ['BEC_summary_open.pickle', 'BEC_summary_open_2.pickle', 'BEC_summary_open_3.pickle']
    file_list = ['BEC_summary.pickle']
    read_summary_data(path, file_list)



