import params
import dill as pickle
import numpy as np
import policy_summarization.BEC_helpers as BEC_helpers
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans
from policy_summarization import policy_summarization_helpers
import json
from collections import defaultdict
from simple_rl.utils import mdp_helpers
import policy_summarization.multiprocessing_helpers as mp_helpers
import os
import warnings
import pandas as pd
import copy
from termcolor import colored
import seaborn as sns
from scipy.stats import norm, halfnorm

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
    


def run_sim_trials(params, study_id, run_id, interaction_data, domain, initial_learning_factor, learning_factor_delta, learner_update_type, viz_flag=False, vars_filename_prefix = ''):

    # load data
    path = 'models_user_study/augmented_taxi2'
    user_id = interaction_data['user_id'].iloc[0]
    max_kc = 3
    run_sim_flag = True
    insert_feedback_flag = True
    # simulated filename
    vars_filename_prefix = vars_filename_prefix + '_' + domain + '_user_' + str(user_id) + '_study_' + str(study_id) + '_run_' + str(run_id)

    try:
        with open(path + '/' + vars_filename_prefix + 'random_name' + '_simulated_interaction_data.pickle', 'rb') as f:
            simulated_interaction_data = pickle.load(f)


    except:

        if domain == 'at':
            mdp_domain = 'augmented_taxi2'
        else:
            mdp_domain = 'skateboard2'

        print('mdp_domain: ', mdp_domain)

        # sim params
        max_learning_factor = params.max_learning_factor
        learning_factor = copy.deepcopy(initial_learning_factor)
        
        if insert_feedback_flag:
            # update interaction data - insert a diagnostic feedback for each test (even when they get it correct)

            # Create an empty DataFrame with columns from the existing DataFrame
            interaction_data_updated_list = []
            delta_loop_id = 0
            print('Updating interaction data....')
            interaction_data.reset_index(inplace=True)
            init_id_flag = True
            for id, row in interaction_data.iterrows():
                
                # check if new row should be inserted for diagnostic feedback when responses are correct
                if (id < len(interaction_data) - 1) and row['interaction_type'] != 'final test':
                    print('Not Last interactiorn or final test...', len(interaction_data), id, row['interaction_type'], interaction_data['interaction_type'].iloc[id+1])
                    
                    if 'test' in row['interaction_type'] and 'feedback' not in interaction_data['interaction_type'].iloc[id+1]:
                        print('Inserting feedback row...')
                        delta_loop_id += 1
                        current_new_row = copy.deepcopy(row)
                        
                        # update current test row to have particles from previous interaction (demo or previous test)
                        current_new_row['pf_pos'] = interaction_data['pf_pos'].iloc[id-1]
                        current_new_row['pf_weights'] = interaction_data['pf_weights'].iloc[id-1]
                        # current_new_row['prob_correct_response'] = interaction_data['prob_correct_response'].iloc[id-1]
                        # current_new_row['prop_particles_BEC'] = interaction_data['prop_particles_BEC'].iloc[id-1]
                        interaction_data_updated_list.append(current_new_row.to_dict()) 
                        
                        # update row to be the new row that represents feedback
                        if 'diagnostic' in row['interaction_type']:
                            row['interaction_type'] = 'diagnostic feedback'
                        elif 'remedial' in row['interaction_type']:
                            row['interaction_type'] = 'remedial feedback'
                        else:
                            print('Error: incorrect interaction type for feedback: ', row['interaction_type'])
                            break
                        
                        row['loop_id'] = row['loop_id'] + delta_loop_id
                        interaction_data_updated_list.append(row.to_dict())

                    else:
                        print('Nofeedback row..', id, row['interaction_type'], interaction_data['interaction_type'].iloc[id+1])
                        # append current row to interaction data
                        row['loop_id'] = row['loop_id'] + delta_loop_id
                        interaction_data_updated_list.append(row.to_dict())

                else:
                    # append current row to interaction data
                    print('Last intearction or final test..', id, row['interaction_type'])

                    row['loop_id'] = row['loop_id'] + delta_loop_id
                    interaction_data_updated_list.append(row.to_dict())

            interaction_data_updated = pd.DataFrame(interaction_data_updated_list)
            # print('interaction_data_updated: ', interaction_data_updated)
        else:
            interaction_data_updated = copy.deepcopy(interaction_data)
        
        

        with open(path + '/base_constraints.pickle', 'rb') as f:
            policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(f)
            
        with open(path + '/BEC_summary.pickle', 'rb') as f:
            summary_data, visited_env_traj_idxs, particles = pickle.load(f)

        if run_sim_flag:
            # initialize (simulated) learner particle filters
            if study_id==0:
                all_learner_pf = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, prior_lf=learning_factor, team_learning_factor = learning_factor, team_prior = params.team_prior, pf_flag='learner', vars_filename=vars_filename_prefix, model_type = learner_update_type)
            else:
                all_learner_pf = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_learning_factor = learning_factor, team_prior = params.team_prior, pf_flag='learner', vars_filename=vars_filename_prefix, model_type = learner_update_type)
            learner_pf = copy.deepcopy(all_learner_pf['p1'])
        
        # initialize dataframes to save probability data
        simulated_interaction_data_list = []

        # prior interaction data
        if domain == 'at':
            prior_test_constraints = [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]  # constraints of the first concept/KC
            all_test_constraints = {  1: [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], \
                                        2: [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], \
                                        3: [np.array([[1, 1, 0]])]}
            min_BEC_constraints = [np.array([[1, 1, 0]]), np.array([[-1,  0,  2]]), np.array([[ 0, -1, -4]])]
        else:
            prior_test_constraints = [np.array([[ 0, -2, -1]]), np.array([[0, 5, 2]])]  # constraints of the first concept/KC
            all_test_constraints = { 1: [np.array([[ 0, -2, -1]]), np.array([[0, 5, 2]])], \
                                    2: [np.array([[-6,  0, -5]]), np.array([[4, 0, 3]])], \
                                    3: [np.array([[-6,  4, -3]]), np.array([[5, 2, 5]]), np.array([[ 3, -3,  1]])]}
            min_BEC_constraints = [np.array([[5, 2, 5]]), np.array([[ 3, -3,  1]]), np.array([[-6,  4, -3]])]


        # simulate teaching loop
        prev_kc_id = 1
        demo_id = 1
        unit_constraints = [np.array([[0, 0, -1]])]

        mdp_params = []
        moves = []
        test_response_type = []

        for loop_id, loop_data in interaction_data_updated.iterrows():
            # print('loop_data: ', loop_data)
            current_kc_id = loop_data['kc_id']
            current_interaction_type = str(loop_data['interaction_type'])
            is_opt_response = loop_data['is_opt_response']

            if current_interaction_type != 'final test' and current_kc_id <= max_kc:
                test_constraints = all_test_constraints[current_kc_id]
            else:
                test_constraints = min_BEC_constraints

            next_interaction_type = interaction_data_updated['interaction_type'].iloc[loop_id+1] if loop_id+1 < len(interaction_data_updated) else 'final test'

            # if run sim
            if run_sim_flag:
                if current_kc_id > prev_kc_id:
                    learning_factor = copy.deepcopy(initial_learning_factor)
                    print('learning_factor reset to initial_team_learning_factor for new KC: ', current_kc_id, '. LF: ', learning_factor)

            # study constraints
            study_constraints = []
            if 'prior' not in current_interaction_type and 'final test' not in current_interaction_type:
                for cnst in loop_data['mdp_params']['constraints']:
                    study_constraints.append(np.array(cnst))
                # print('study_constraints: ', study_constraints)
            
            # Prior
            if run_sim_flag:
                if current_interaction_type == 'prior':
                    test_constraints = prior_test_constraints
                    # teacher model of the learner
                    _, all_teacher_pf = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, prior_lf=[0.7], teacher_learning_factor = [0.7], team_prior = params.team_prior, pf_flag='teacher', vars_filename=vars_filename_prefix)
                    teacher_pf_interaction = copy.deepcopy(all_teacher_pf['p1'])

                else:
                    # teacher model of the learner
                    teacher_pf_interaction = pf_team.Particles_team(loop_data['pf_pos'], 0.7) # do not use PF for updating, rather create a new PF object for every state change
                    teacher_pf_interaction.weights = loop_data['pf_weights']

            else:
                if current_interaction_type == 'prior':
                    test_constraints = prior_test_constraints


            # Demos
            if current_interaction_type == 'demo':
                
                ## get demo constraints from BEC summary
                # unit = summary_data[current_kc_id-1]               
                # if len(unit) > 1:
                #     if next_interaction_type == 'demo':
                #         subunit = unit[0]
                #     else:
                #         subunit = unit[1]
                # else:
                #     subunit = unit[0]
                # unit_constraints = subunit[3]

                ## get demo constraints from actual study demo constraints
                unit_constraints = copy.deepcopy(study_constraints) 

                
                # update particle filter for each demo
                if run_sim_flag:
                    learner_pf.update(unit_constraints, learning_factor, plot_title = 'Learner belief after demo: ' + str(demo_id) + ' for KC: ' + str(current_kc_id), viz_flag = viz_flag, vars_filename=vars_filename_prefix, model_type = learner_update_type)

                # update
                demo_id += 1

                # reset test response variables
                test_response_type = []
                test_response_constraints = []

            # Nothing changes during diagnostic test except for the constraints
            if current_interaction_type == 'diagnostic test':
                print('diagnostic test....')
                test = copy.deepcopy(loop_data)

                # print('mdp_params: ', test['mdp_params'])
                mdp_params = test['mdp_params']
                moves = test['moves']
                mdp_params['weights'] = params.mdp_parameters['weights']
                mdp_user = make_mdp.make_custom_mdp(mdp_domain, mdp_params)
                opt_moves = mdp_params['all_opt_actions'][0]

                # find human rewards
                mdp_user.reset()
                trajectory = []
                cur_state = mdp_user.get_init_state()

                for idx in range(len(moves)):
                    # assumes that the user study only allows actions that change the state of the MDP
                    reward, next_state = mdp_user.execute_agent_action(moves[idx])
                    trajectory.append((cur_state, moves[idx], next_state))

                    # deepcopy occurs within transition function
                    cur_state = next_state

                human_feature_count = mdp_user.accumulate_reward_features(trajectory, discount=True)
                # print('trajectory: ', trajectory, 'reward features: ', human_feature_count)


                # find optimal rewards
                mdp_user.reset()
                trajectory = []
                cur_state = mdp_user.get_init_state()

                for idx in range(len(opt_moves)):
                    # assumes that the user study only allows actions that change the state of the MDP
                    reward, next_state = mdp_user.execute_agent_action(opt_moves[idx])
                    trajectory.append((cur_state, opt_moves[idx], next_state))

                    # deepcopy occurs within transition function
                    cur_state = next_state

                opt_feature_count = mdp_user.accumulate_reward_features(trajectory, discount=True)
                # print('trajectory: ', trajectory, 'opt features: ', opt_feature_count)


                if (human_feature_count == opt_feature_count).all():
                    unit_constraints = copy.deepcopy(study_constraints)
                    test_response_type = ['correct']
                    failed_BEC_constraint = []
                else:
                    failed_BEC_constraint = opt_feature_count - human_feature_count
                    unit_constraints = copy.deepcopy([-failed_BEC_constraint])
                    test_response_type = ['incorrect']


            # Diagnostic Feedback
            if current_interaction_type == 'diagnostic feedback':
                # Diagnostic Feedback (given after test in any case!)
                unit_constraints = copy.deepcopy(study_constraints)
                
                print('response_type_team: ', test_response_type)
                response_flag = True
                for response_type in test_response_type:
                    if response_type == 'incorrect':
                        response_flag = False

                if run_sim_flag:
                    if response_flag:
                        learning_factor[0] = min(learning_factor[0] + learning_factor_delta[0], max_learning_factor)
                    else:
                        learning_factor[0] = min(learning_factor[0] + learning_factor_delta[1], max_learning_factor)

                    # updated learner model with corrective feedback - feedback is based on the specific test constraint and not KC's overall test constraints
                    plot_title =  ' Learner after corrective feedback for KC ' + str(current_kc_id)
                    learner_pf.update(unit_constraints, learning_factor, plot_title = plot_title, viz_flag = viz_flag, vars_filename = vars_filename_prefix, model_type = learner_update_type)


            # Remedial Demos (follow feedback and diagnostic test)
            if current_interaction_type == 'remedial demo':
                if len(failed_BEC_constraint) == 0:
                    unit_constraints = copy.deepcopy(study_constraints) #get from the actual study remedial demo constraints
                else:
                    unit_constraints = copy.deepcopy([failed_BEC_constraint]) #use the simulated learner's failed BEC constraint
                print('unit_constraints: ', unit_constraints)
                
                if run_sim_flag:
                    plot_title =  'Learner belief after remedial demo ' + str(demo_id) + ' for KC ' + str(current_kc_id)
                    learner_pf.update(unit_constraints, learning_factor, plot_title = plot_title, viz_flag = viz_flag, vars_filename=vars_filename_prefix, model_type = learner_update_type)
                demo_id += 1

                # reset test response variables
                test_response_type = []

            # Nothing changes during test

            # Remedial feedback
            if current_interaction_type == 'remedial test':
                test = copy.deepcopy(loop_data)

                mdp_params = test['mdp_params']
                moves = test['moves']
                mdp_params['weights'] = params.mdp_parameters['weights']
                mdp_user = make_mdp.make_custom_mdp(mdp_domain, mdp_params)
                opt_moves = mdp_params['all_opt_actions'][0]

                # find human rewards
                mdp_user.reset()
                trajectory = []
                cur_state = mdp_user.get_init_state()

                for idx in range(len(moves)):
                    # assumes that the user study only allows actions that change the state of the MDP
                    reward, next_state = mdp_user.execute_agent_action(moves[idx])
                    trajectory.append((cur_state, moves[idx], next_state))

                    # deepcopy occurs within transition function
                    cur_state = next_state

                human_feature_count = mdp_user.accumulate_reward_features(trajectory, discount=True)
                # print('remedial trajectory: ', trajectory, 'remedial reward features: ', human_feature_count)

                # find optimal rewards
                mdp_user.reset()
                trajectory = []
                cur_state = mdp_user.get_init_state()

                for idx in range(len(opt_moves)):
                    # assumes that the user study only allows actions that change the state of the MDP
                    reward, next_state = mdp_user.execute_agent_action(opt_moves[idx])
                    trajectory.append((cur_state, opt_moves[idx], next_state))
                    # deepcopy occurs within transition function
                    cur_state = next_state

                opt_feature_count = mdp_user.accumulate_reward_features(trajectory, discount=True)
                # print('remedial trajectory: ', trajectory, 'remedial opt features: ', opt_feature_count)


                if (human_feature_count == opt_feature_count).all():
                    unit_constraints = copy.deepcopy(study_constraints)
                    test_response_type = ['correct']
                    failed_BEC_constraint = []
                else:
                    failed_BEC_constraint = opt_feature_count - human_feature_count
                    unit_constraints = copy.deepcopy([-failed_BEC_constraint])
                    test_response_type = ['incorrect']


            if current_interaction_type == 'remedial feedback':
                unit_constraints = copy.deepcopy(study_constraints)
                # Remedial Feedback (given after test in any case!)
                print('remedial response: ', test_response_type)
                response_flag = True
                for response_type in test_response_type:
                    if response_type == 'incorrect':
                        response_flag = False

                if run_sim_flag:
                    if response_flag:
                        learning_factor[0] = min(learning_factor[0] + learning_factor_delta[0], max_learning_factor)
                    else:
                        learning_factor[0] = min(learning_factor[0] + learning_factor_delta[1], max_learning_factor)

                    # updated learner model with corrective feedback. Note that here we use the study/unit constraints as remedial feedback can be specific to the demo that was shown in the remedial demo and maynot relate to the KC test constraints.
                    plot_title =  ' Learner after remedial feedback for KC ' + str(current_kc_id)
                    learner_pf.update(unit_constraints, learning_factor, plot_title = plot_title, viz_flag = viz_flag, vars_filename = vars_filename_prefix, model_type = learner_update_type)


            # Final test
            # Nothing changes in the learner model since they do not get a feedback after these tests. Test constraints are set to the minimum BEC constraints at the beginning for final tests.
            if current_interaction_type == 'final test':
                test_constraints = min_BEC_constraints
                
                test = copy.deepcopy(loop_data)
                mdp_params = test['mdp_params']
                moves = test['moves']

            if run_sim_flag:
                # calculate learner probabilities
                learner_pf.calc_particles_probability(test_constraints)  
                prob_particles_correct = learner_pf.particles_prob_correct

                learner_pf.calc_particles_probability(min_BEC_constraints)
                prop_particles_BEC = learner_pf.particles_prob_correct

                # calculate teacher probabilities
                teacher_pf_interaction.calc_particles_probability(test_constraints)
                prob_particles_correct_teacher = teacher_pf_interaction.particles_prob_correct

                teacher_pf_interaction.calc_particles_probability(min_BEC_constraints)
                prop_particles_BEC_teacher = teacher_pf_interaction.particles_prob_correct

                # calculate info gain
                entropy_learner_pf = learner_pf.calc_entropy()
                entropy_teacher_pf = teacher_pf_interaction.calc_entropy()

                # save current interaction data
                
                interaction_data_dict = {'user_id': user_id, 'loop_condition': interaction_data_updated['loop_condition'].iloc[0], 'initial_learning_factor': initial_learning_factor, 'learning_factor_delta': learning_factor_delta, \
                                        'loop_id': loop_id, 'demo_id': demo_id, 'kc_id': current_kc_id, 'interaction_type': current_interaction_type, 'interaction_constraints': unit_constraints, 'test_constraints': test_constraints, 'study_constraints': study_constraints,\
                                        'learning_factor': learning_factor[0], 'is_opt_response': is_opt_response, 'prob_correct_response': prob_particles_correct, 'prop_particles_BEC': prop_particles_BEC, \
                                        'prob_correct_response_teacher': prob_particles_correct_teacher, 'prob_particles_BEC_teacher': prop_particles_BEC_teacher, 'learner_pf_pos': learner_pf.positions, 'learner_pf_weights': learner_pf.weights, \
                                        'teacher_pf_pos': teacher_pf_interaction.positions, 'teacher_pf_weights': teacher_pf_interaction.weights, 'entropy_learner_pf': entropy_learner_pf, 'entropy_teacher_pf': entropy_teacher_pf}
            
            else:
                interaction_data_dict = {'user_id': user_id, 'loop_condition': interaction_data_updated['loop_condition'].iloc[0], 'initial_learning_factor': initial_learning_factor, 'learning_factor_delta': learning_factor_delta, \
                                        'loop_id': loop_id, 'demo_id': demo_id, 'kc_id': current_kc_id, 'interaction_type': current_interaction_type, 'interaction_constraints': unit_constraints, 'test_constraints': test_constraints, 'study_constraints': study_constraints,\
                                        'is_opt_response': is_opt_response, 'test_response_type': test_response_type, 'mdp_params': mdp_params, 'moves': moves}
                
            
            simulated_interaction_data_list.append(interaction_data_dict)

            # update 
            prev_kc_id = current_kc_id
            prev_interaction_type = current_interaction_type

        simulated_interaction_data = pd.DataFrame(simulated_interaction_data_list)

        if run_sim_flag:
            # save data
            with open(path + '/' + vars_filename_prefix + '_simulated_interaction_data.pickle', 'wb') as f:
                pickle.dump(simulated_interaction_data, f)

            simulated_interaction_data.to_csv(path + '/' + vars_filename_prefix + '_simulated_interaction_data.csv')
        else:
            return simulated_interaction_data

    
    # ############    Plots    #############################################

    # color_dict = {'demo': 'blue', 'remedial demo': 'purple', 'diagnostic test': 'red',  'remedial test': 'pink', 'diagnostic feedback': 'yellow', 'remedial feedback': 'orange', 'final test': 'green'}

    colors = sns.color_palette("colorblind", 7).as_hex()
    # colors = ','.join(color_pal)
    # print('colors: ', colors)

    color_dict = {'demo': str(colors[0]), 'remedial demo': str(colors[1]), 'diagnostic test': str(colors[2]),  'remedial test': str(colors[3]), 'diagnostic feedback': str(colors[4]), 'remedial feedback': str(colors[5]), 'final test': str(colors[6])}
    # print('color_dict: ', color_dict)

    # plot simulated probability data
    # f, ax = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(15,10))
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # print('interaction data interactions: ', simulated_interaction_data['interaction_type'])

    loop_condition = simulated_interaction_data['loop_condition'].iloc[0]
    kc_id_list = simulated_interaction_data['kc_id'].unique()
    
    user_plot_data = pd.DataFrame()
    cur_comb_demo_id = 0
    
    concept_end_id = []
    max_kc_in_list = max(kc_id_list)
    for kc_id in kc_id_list:
        user_kc_data = simulated_interaction_data[(simulated_interaction_data['kc_id'] == kc_id)]
        combined_demo_id_list = [cur_comb_demo_id + id for id in range(len(user_kc_data))]
        if kc_id != max_kc_in_list:
            cur_comb_demo_id = combined_demo_id_list[-1]
        else:
            cur_comb_demo_id = combined_demo_id_list[-1] + 1
        user_kc_data['combined_demo_id'] = combined_demo_id_list

        concept_end_id.append(user_kc_data['combined_demo_id'].iloc[-1])
        
        # print('combined_demo_id_list: ', combined_demo_id_list)
        # print('user_kc_data: ', user_kc_data) 
        user_plot_data = pd.concat([user_plot_data, user_kc_data], ignore_index=True)
    
    # print('concept_end_id: ', concept_end_id)
    # print('user_plot_data interactions: ', user_plot_data['interaction_type'], 'demo_id: ', user_plot_data['combined_demo_id'])

    # info gain
    user_plot_data['info_gain_learner'] = np.diff(user_plot_data['entropy_learner_pf'], n=1, prepend=-1)
    user_plot_data['info_gain_teacher'] = np.diff(user_plot_data['entropy_teacher_pf'], n=1, prepend=-1)


    # plots
    plt.figure(user_id, figsize=(16, 12))
    ax0 = plt.gca()
    # ax_dup = ax0.twinx()
    linethickness = 6

    # plot concept-wise learning dynamics - probability of correct response
    for index in range(len(kc_id_list)):
        kc_id = kc_id_list[index]
        if kc_id > 0:
            user_kc_data = user_plot_data[(user_plot_data['kc_id'] == kc_id)]
            user_kc_data['beta'] = user_kc_data['learning_factor']
            if index != len(kc_id_list)-2:
                sns.lineplot(data=user_kc_data, x='combined_demo_id', y='prob_correct_response', ax=ax0, legend=False, linewidth=linethickness, color = str(colors[0]))
                # sns.lineplot(data=user_kc_data, x='combined_demo_id', y='prob_correct_response_teacher', ax=ax0, color = str(colors[1]))
                sns.lineplot(data=user_kc_data, x='combined_demo_id', y='beta', ax=ax0, legend=False, linewidth=linethickness, color = str(colors[2]))
            else:
                sns.lineplot(data=user_kc_data, x='combined_demo_id', y='prob_correct_response', ax=ax0, legend=False, color = str(colors[0]), linewidth=linethickness, label='Concept knowledge - learner')
                # sns.lineplot(data=user_kc_data, x='combined_demo_id', y='prob_correct_response_teacher', ax=ax0, color = str(colors[1]), label='Concept knowledge - teacher model of learner')
                sns.lineplot(data=user_kc_data, x='combined_demo_id', y='beta', color = str(colors[2]), ax=ax0, legend=False, linewidth=linethickness, label='Understanding factor')

    user_plot_data_no_duplicates = user_plot_data.drop_duplicates(subset=['combined_demo_id'])
    user_plot_data_no_duplicates = user_plot_data_no_duplicates.reset_index(drop=True)
    print('No duplicate interactions: ', user_plot_data_no_duplicates['interaction_type'])

    # plot overall learning dynamics - probability of correct response
    sns.lineplot(data=user_plot_data_no_duplicates, x='combined_demo_id', y='prop_particles_BEC', ax=ax0, legend=False, linewidth=linethickness, color = str(colors[3]), label = 'Total knowledge - learner')
    # sns.lineplot(data=user_plot_data_no_duplicates, x='combined_demo_id', y='prob_particles_BEC_teacher', ax=ax0, color = str(colors[4]), label = 'Total knowledge - teacher model of learner')
    # sns.lineplot(data=user_plot_data_no_duplicates, x='combined_demo_id', y='info_gain_learner', ax=ax_dup, color = str(colors[5]), label = 'Info gain - learner')
    # sns.lineplot(data=user_plot_data_no_duplicates, x='combined_demo_id', y='info_gain_teacher', ax=ax_dup, color = str(colors[6]), label = 'Info gain - teacher model of learner')



    # plot actual human learner responses to tests
    test_data_list = []
    for id, row in user_plot_data_no_duplicates.iterrows():
        if 'test' in row['interaction_type']:
            test_data_list.append(row)
    test_data = pd.DataFrame(test_data_list)

    markers_list = {1: 'o', 0: 'x'}
    # color_list = {1: 'green', 0: 'red'}
    test_data_correct = test_data[test_data['is_opt_response'] == 1]
    test_data_incorrect = test_data[test_data['is_opt_response'] == 0]
    print('test_data_correct: ', test_data_correct['combined_demo_id'], test_data_correct['is_opt_response'])
    print('test_data_incorrect: ', test_data_incorrect['combined_demo_id'], test_data_incorrect['is_opt_response'])
    sns.scatterplot(data=test_data_correct, x='combined_demo_id', y='is_opt_response', marker = 'o', ax=ax0, color='green', s=400, label = 'Correct response - learner')
    sns.scatterplot(data=test_data_incorrect, x='combined_demo_id', y='is_opt_response', marker = 'x', ax=ax0, linewidth = 3, color='red', s=200, label = 'Incorrect response - learner')
    
    
    
    for id, row in user_plot_data_no_duplicates.iterrows():
        print('id:', id)
        if row['interaction_type'] != 'prior':
            # plt.axvline(x=row['combined_demo_id'], color='red', linestyle='--')
            plt.axvspan(user_plot_data_no_duplicates['combined_demo_id'].iloc[id-1], row['combined_demo_id'], alpha=0.2, color=color_dict[row['interaction_type']])
            plt.text(row['combined_demo_id']-0.5, 0.3, row['interaction_type'], rotation=90, fontsize=28, weight="bold")

    for id in concept_end_id:
        plt.axvline(x=id, color='black', linestyle='--', linewidth=2)


    
    plt.title('Learning dynamics for user: ' + str(user_id) + '. Condition: ' + str(loop_condition) + ' Study id: ' + str(study_id) + ' Run id: ' + str(run_id) + '.')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.legend(title = '', loc='best')
    # plt.xlabel('Interaction number')
    ax0.set_ylabel('Prob. knowledge - Sum product of particle weights within BEC area')
    # ax_dup.set_ylabel('Understanding factor', fontsize=28)
    ax0.set_xlabel('Interaction number', fontsize=28)

    # ax0.legend(fontsize='large')
    # ax_dup.legend(fontsize='large')
    # ax0.legend(loc='upper left')
    # ax_dup.legend(loc='upper right')

    ax0.set_ylim([-0.05, 1.4])
    # ax_dup.set_ylim([-0.05, 1.4])

    # Get the current y-ticks
    current_y_ticks = ax0.get_yticks()

    # Filter y-ticks to remove values beyond max_y_value
    filtered_y_ticks = [tick for tick in current_y_ticks if tick <= 1.05 and tick >= -0.05]

    # Set the filtered y-ticks
    ax0.set_yticks(filtered_y_ticks)
    # ax_dup.set_yticks(filtered_y_ticks)
    
    ax0.yaxis.set_tick_params(labelsize=28)
    ax0.xaxis.set_tick_params(labelsize=28)
    # ax_dup.yaxis.set_tick_params(labelsize=28)

    # lines1, labels1 = ax0.get_legend_handles_labels()
    # lines2, labels2 = ax_dup.get_legend_handles_labels()
    # ax0.legend(lines1 + lines2, labels1 + labels2, loc='best')


    if viz_flag:
        plot_title = vars_filename_prefix + '_simulated_interaction_data.png'
        plt.savefig(path + '/' + plot_title)


    # #################################

    # # plot particle distribution
    # fig = plt.figure(figsize=(18, 14))
    # fig2 = plt.figure(figsize=(18, 14))
    
    # row_len, col_len = squarest_rectangle(len(user_plot_data))
    # print('row_len: ', row_len, '. col_len: ', col_len)
    # axs = np.array([fig.add_subplot(row_len, col_len, i+1, projection='3d') for i in range(len(user_plot_data))])   
    # axs2 = np.array([fig2.add_subplot(row_len, col_len, i+1, projection='3d') for i in range(len(user_plot_data))])
    
    # row_id, col_id = 0, 0
    # for id, row in user_plot_data_no_duplicates.iterrows(): 

    #     test_constraints = row['test_constraints'] 

    #     learner_particles = pf_team.Particles_team(row['learner_pf_pos'], 0.8)
    #     learner_particles.weights = row['learner_pf_weights']

    #     teacher_particles = pf_team.Particles_team(row['teacher_pf_pos'], 0.8)
    #     teacher_particles.weights = row['teacher_pf_weights']
        
    #     learner_particles.calc_particles_probability(test_constraints)
    #     learner_pcr = learner_particles.particles_prob_correct

    #     teacher_particles.calc_particles_probability(test_constraints)
    #     teacher_pcr = teacher_particles.particles_prob_correct

    #     learner_particles.plot(fig=fig, ax=axs[id])
    #     BEC_viz.visualize_planes(test_constraints, fig=fig, ax=axs[id])
    #     ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(test_constraints)
    #     poly = Polyhedron.Polyhedron(ieqs=ieqs)
    #     BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=axs[id], plot_ref_sphere=False, alpha=0.75)
    #     axs[id].set_title('id ' + str(id) + ': ' + row['interaction_type'], fontsize=11)

    #     teacher_particles.plot(fig=fig2, ax=axs2[id])
    #     BEC_viz.visualize_planes(test_constraints, fig=fig2, ax=axs2[id])
    #     BEC_viz.visualize_spherical_polygon(poly, fig=fig2, ax=axs2[id], plot_ref_sphere=False, alpha=0.75)
    #     axs2[id].set_title('id ' + str(id) + ': ' + row['interaction_type'], fontsize=11)


    #     if col_id < col_len-1:
    #         col_id += 1
    #     else:
    #         col_id = 0
    #         row_id += 1
    
    # fig.suptitle('User: ' + str(user_id) + '. Condition: ' + str(loop_condition))
    # fig2.suptitle('Teacher model of user: ' + str(user_id) + '. Condition: ' + str(loop_condition))

    # if viz_flag:
    #     fig.savefig(path + '/' + vars_filename_prefix + '_learner_particles.png')
    #     fig2.savefig(path + '/' + vars_filename_prefix + '_teacher_particles.png')



    plt.show()


    
        


#############################################




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


def analyze_learning_dynamics(user_data, study_data, file_prefix=''):

    file_prefix = 'interaction_data_' + file_prefix

    max_kc = 3

    try:
        with open('data/' + file_prefix + 'random.pickle', 'wb') as f:
            interaction_data = pickle.load(f)
    except:

        # # calculate the probability of correct response for prior knowledge
        # _, particles_prior_team = team_helpers.sample_team_pf(1, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, teacher_learning_factor = [0.8], prior_lf = 0.8, team_prior = params.team_prior, model_type = 'no_noise')
        # particles_prior = copy.deepcopy(particles_prior_team['p1'])
        # prior_test_constraints = [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]
        # particles_prior.calc_particles_probability(prior_test_constraints)
        # probability_correct_response_prior = particles_prior.particles_prob_correct

        # fixed test constraints
        test_constraints_domain = {'at': {  1: [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], \
                                            2: [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], \
                                            3: [np.array([[1, 1, 0]])]},
                                   'sb': {1: [np.array([[ 0, -2, -1]]), np.array([[0, 5, 2]])], \
                                          2: [np.array([[-6,  0, -5]]), np.array([[4, 0, 3]])], \
                                          3: [np.array([[-6,  4, -3]]), np.array([[5, 2, 5]]), np.array([[ 3, -3,  1]])]}
                            }
        
        min_BEC_constraints = {'at': [np.array([[1, 1, 0]]), np.array([[-1,  0,  2]]), np.array([[ 0, -1, -4]])], \
                               'sb': [np.array([[5, 2, 5]]), np.array([[ 3, -3,  1]]), np.array([[-6,  4, -3]])]}
        
        initial_variable_filter = {'at': [[0, 1, 0]], 'sb': [[]]}


        # save the known interaction data - id, interaction id, etc
        interaction_data = pd.DataFrame()

        
        user_list = user_data['user_id'].unique()
        # domain_list = study_data['domain'].unique()
        domain_list = ['at']

        for user_id in user_list:
            user_study_data_all_domains = study_data[study_data['user_id'] == user_id]
            if user_study_data_all_domains['domain'].iloc[0] == 'at':
                domain_order = 'at--sb'
            else:
                domain_order = 'sb--at'

            for domain in domain_list:
                
                user_study_data = user_study_data_all_domains[user_study_data_all_domains['domain'] == domain]  # use only augmented taxi domain for now
                user_study_data = user_study_data.reset_index(drop=True)

                # print('user_id: ', user_id, 'user_study_data: ', user_study_data)
                
                if len(user_study_data) > 0:
                    
                    print('user study len before reindex: ', len(user_study_data))

                    # if condition is partial loop
                    cur_loop_cond = user_study_data['loop_condition'].iloc[0]
                    if cur_loop_cond == 'pl':
                        reindex = []
                        cur_index = -1
                        # print('user_study_data: ', user_study_data, 'len: ', len(user_study_data))
                        for int_id in range(len(user_study_data)-1):
                            if int_id > 1 and int_id < len(user_study_data) - 2:
                                if user_study_data['interaction_type'].iloc[int_id] == 'demo' and user_study_data['interaction_type'].iloc[int_id+1] == 'diagnostic test' and user_study_data['interaction_type'].iloc[int_id+2] == 'final test':
                                    
                                    cur_index += 2
                                    reindex.append(cur_index)

                                    cur_index -= 1
                                    reindex.append(cur_index)
                                    cur_index += 1
                                    # print('int_id:', int_id, 'type reindex: ', type(reindex), 'reindex: ', reindex)

                                else:
                                    cur_index += 1
                                    reindex.append(cur_index)

                                    # print('int_id:', int_id, 'type reindex2: ', type(reindex), 'reindex: ', reindex)
                            else:
                                cur_index += 1
                                reindex.append(cur_index)

                                # print('int_id:', int_id, 'type reindex3: ', type(reindex), 'reindex: ', reindex)
                        
                        print('reindex: ', reindex)
                        user_study_data = user_study_data.reindex(reindex)

                        print('user study len after reindex: ', len(user_study_data))

                    # identify demo and kc ids
                    kc_list = []
                    cur_kc_id = 0
                    demo_id_list = []
                    cur_demo_id = 0
                    prev_interaction_type = ''
                    print('user_id:', user_id)
                    for int_id in range(len(user_study_data)):
                        cur_interaction_type = user_study_data['interaction_type'].iloc[int_id]
                        
                        # check for new KC
                        mdp_params = user_study_data['unpickled_mdp_parameters'].iloc[int_id]
                        print('mdp_params: ', mdp_params)
                        demo_constraints = mdp_params['constraints']
                        for cnst_id in range(len(demo_constraints)):
                            if cnst_id == 0:
                                cur_variable_filter = [1 if i == 0 else 0 for i in demo_constraints[cnst_id][0]]
                            else:
                                var_filter = [1 if i == 0 else 0 for i in demo_constraints[cnst_id][0]]
                                print('var_filter: ', var_filter, 'cur_variable_filter: ', cur_variable_filter)
                                cur_variable_filter = [cur_variable_filter[i] and var_filter[i] for i in range(len(cur_variable_filter))]
                        print('int_id: ', int_id, 'cur_variable_filter: ', cur_variable_filter)
                        

                        if ('demo' == cur_interaction_type and 'demo' != prev_interaction_type):
                            cur_kc_id += 1
                        
                        if domain == 'at':
                            if cur_variable_filter[1] == 1:
                                cur_kc_id = 1
                            elif cur_variable_filter[0] == 1:
                                cur_kc_id = 2
                            elif cur_variable_filter[2] == 1:
                                cur_kc_id = 3
                            else:
                                cur_kc_id = 4 # final demos of entire BEC constraints for open loop conditions
                        else:
                            print(KeyError('Variable filters for this Domain not configured yet!'))
                        

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
                        prev_variable_filter = cur_variable_filter
                        

                    print('kc_list:', kc_list)
                    # print(demo_id_list)
                    ##########################

                    # prior data
                    prior_interaction_data_dict = {'user_id': user_id, 'loop_condition': user_study_data['loop_condition'].iloc[0], 'loop_id': 0, 'demo_id': 0, 'kc_id': 1, \
                                            'interaction_type': 'prior', 'domain': domain, 'domain_order': domain_order}
                    interaction_data = interaction_data.append(prior_interaction_data_dict, ignore_index=True)
                    ##########################

                    # interaction data
                    prev_kc_id = 0  
                    for int_id in range(len(user_study_data)):
                        cur_interaction_type = user_study_data['interaction_type'].iloc[int_id]
                        is_opt_response = user_study_data['is_opt_response'].iloc[int_id]
                        
                        cur_kc_id = kc_list[int_id]
                        cur_demo_id = demo_id_list[int_id]
                        mdp_params = user_study_data['unpickled_mdp_parameters'].iloc[int_id]   ## need MDP parameters to identify the type of final test (whether evaluates mud only or both mud and battery)
                        moves = user_study_data['unpickled_moves'].iloc[int_id]

                        # check for new KC
                        if cur_kc_id != prev_kc_id and cur_kc_id > 1:
                            if cur_kc_id != -1 and cur_kc_id <= max_kc:
                                test_constraints = test_constraints_domain[domain][cur_kc_id]
                            else:
                                test_constraints = min_BEC_constraints[domain]

                            # # create a PF object based on the last interaction data
                            pf_pos = user_study_data['unpickled_human_model_pf_pos'].iloc[int_id-1]
                            pf_weights = user_study_data['unpickled_human_model_pf_weights'].iloc[int_id-1]
                            # particles_interaction = pf_team.Particles_team(pf_pos, 0.8) # do not use PF for updating, rather create a new PF object for every state change
                            # particles_interaction.weights = pf_weights

                            # # probability of correct response
                            # particles_interaction.calc_particles_probability(test_constraints)
                            # prob_correct_response =  particles_interaction.particles_prob_correct

                            # # proportion of particles within BEC area
                            # particles_interaction.calc_particles_probability(min_BEC_constraints[domain])
                            # prop_particles_BEC = particles_interaction.particles_prob_correct

                            ## compile interaction data 
                            # interaction_data_dict = {'user_id': user_id, 'loop_condition': user_study_data['loop_condition'].iloc[int_id], 'loop_id': int_id, 'demo_id': cur_demo_id, 'kc_id': cur_kc_id, 'interaction_type': 'prior_' + str(cur_kc_id), \
                            #                         'test_constraints': test_constraints, 'pf_pos': pf_pos, 'pf_weights': pf_weights, 'prob_correct_response': prob_correct_response, 'prop_particles_BEC': prop_particles_BEC, \
                            #                             'domain': domain, 'domain_order': domain_order}
                            
                            interaction_data_dict = {'user_id': user_id, 'loop_condition': user_study_data['loop_condition'].iloc[int_id], 'loop_id': int_id, 'demo_id': cur_demo_id, 'kc_id': cur_kc_id, 'interaction_type': 'prior_' + str(cur_kc_id), \
                                                    'test_constraints': test_constraints, 'pf_pos': pf_pos, 'pf_weights': pf_weights, \
                                                        'domain': domain, 'domain_order': domain_order}
                            
                            
                            # save interaction data
                            interaction_data = interaction_data.append(interaction_data_dict, ignore_index=True)




                        # get test constrainst for this KC/final test
                        if (cur_kc_id != -1) and (cur_kc_id <= max_kc):
                            test_constraints = test_constraints_domain[domain][cur_kc_id]

                            # create a PF object from position and weights and calculate the probability of correct response
                            pf_pos = user_study_data['unpickled_human_model_pf_pos'].iloc[int_id]
                            pf_weights = user_study_data['unpickled_human_model_pf_weights'].iloc[int_id]
                            # particles_interaction = pf_team.Particles_team(pf_pos, 0.8) # do not use PF for updating, rather create a new PF object for every state change
                            # particles_interaction.weights = pf_weights
                        else:
                            # final tests (kc=-1); always consider all KCs for final tests for consistency even though the individual tests might evaluate digfferent concepts
                            all_test_constraints = []
                            for id in range(1, max_kc+1):
                                all_test_constraints.extend(test_constraints_domain[domain][id])
                            test_constraints = BEC_helpers.remove_redundant_constraints(all_test_constraints, params.weights['val'], params.step_cost_flag)

                            # use the previous PF (retains already); final tests are at the end, so the PF will be that of the last diagnostic test!
                        
                        # update loop vars
                        prev_kc_id = cur_kc_id


                        print('interaction type: ', cur_interaction_type, '. test_constraints:', test_constraints)

                        # # probability of correct response
                        # particles_interaction.calc_particles_probability(test_constraints)
                        # prob_correct_response =  particles_interaction.particles_prob_correct

                        # # proportion of particles within BEC area
                        # particles_interaction.calc_particles_probability(min_BEC_constraints[domain])
                        # prop_particles_BEC = particles_interaction.particles_prob_correct


                        ## compile interaction data
                        interaction_data_dict = {'user_id': user_id, 'loop_condition': user_study_data['loop_condition'].iloc[int_id], 'loop_id': int_id+1, 'demo_id': cur_demo_id, 'kc_id': cur_kc_id, 'interaction_type': cur_interaction_type, \
                                                'test_constraints': test_constraints, 'pf_pos': pf_pos, 'pf_weights': pf_weights, 'mdp_params': mdp_params, \
                                                'moves': moves, 'is_opt_response': is_opt_response, 'domain': domain, 'domain_order': domain_order}
                        
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
        
        with open('data/' + file_prefix + '.pickle', 'wb') as f:
            pickle.dump(interaction_data, f)
        
        interaction_data.to_csv('data/' + file_prefix + '.csv')

###################




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
    domain = 'at'

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
        cur_user_data = interaction_data[(interaction_data['user_id'] == user_id) & (interaction_data['domain'] == domain)]
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
            user_plot_data = pd.concat([user_plot_data, user_kc_data], ignore_index=True)

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
        sns.lineplot(data=user_plot_data, x='combined_demo_id', y='is_opt_response', color = 'blue')
        
        plt.ylim([0, 1])
        for id, row in user_plot_data.iterrows():
            if row['interaction_type'] != 'prior':
                # plt.axvline(x=row['combined_demo_id'], color='red', linestyle='--')
                plt.axvspan(user_plot_data['combined_demo_id'].iloc[id-1], row['combined_demo_id'], alpha=0.2, color=color_dict[row['interaction_type']])
        
        for id in concept_end_id:
            plt.axvline(x=id, color='black', linestyle='--')


        # # plot particle distribution
        # fig = plt.figure()
        
        # row_len, col_len = squarest_rectangle(len(user_plot_data))
        # print('row_len: ', row_len, '. col_len: ', col_len)
        # axs = np.array([fig.add_subplot(row_len, col_len, i+1, projection='3d') for i in range(len(user_plot_data))])   
        
        # row_id, col_id = 0, 0
        # for id, row in user_plot_data.iterrows(): 

        #     if row['interaction_type'] == 'prior':
        #         _, particles_all = team_helpers.sample_team_pf(1, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, teacher_learning_factor = [0.8], prior_lf = 0.8, team_prior = params.team_prior, model_type = 'no_noise')
        #         particles = copy.deepcopy(particles_all['p1'])
        #         # print('particle weights: ', particles.weights)
        #     else:
        #         particles = pf_team.Particles_team(row['pf_pos'], 0.8)
        #         # print('row[pf_weights]: ', row['pf_weights'])
        #         particles.weights = row['pf_weights']
            
        #     test_constraints = row['test_constraints'] 
        #     particles.calc_particles_probability(test_constraints)

        #     print('id: ', id, '. test_constraints: ', test_constraints, '. prob_correct_response: ', particles.particles_prob_correct)
        #     print('row_id: ', row_id, '. col_id: ', col_id)
        #     particles.plot(fig=fig, ax=axs[id])
        #     BEC_viz.visualize_planes(test_constraints, fig=fig, ax=axs[id])
        #     ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(test_constraints)
        #     poly = Polyhedron.Polyhedron(ieqs=ieqs)
        #     BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=axs[id], plot_ref_sphere=False, alpha=0.75)
        #     axs[id].set_title('id ' + str(id) + ': ' + row['interaction_type'])

        #     if col_id < col_len-1:
        #         col_id += 1
        #     else:
        #         col_id = 0
        #         row_id += 1
        
        # fig.suptitle('Particle distribution for user: ' + str(user_id) + '. Condition: ' + str(loop_condition))

        
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
    
def read_summary_data(path, data_loc, file_list):

    with open(path + '/base_constraints.pickle', 'rb') as f:
        policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(f)
    
    with open(path + '/BEC_constraints.pickle', 'rb') as f:
        min_BEC_constraints, BEC_lengths_record = pickle.load(f)

    


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
                print('unit_idx: ', unit_idx, 'test_constraints: ', test_constraints, 'running_variable_filter: ', running_variable_filter)

        
        
        print('file: ', file, '. demo_ids: ', demo_ids_list, '. constraints_list: ', constraints_list, '. visited_env_traj_idxs: ', visited_env_traj_idxs)
                
        print('min_constraints: ', min_constraints, 'min_BEC_constraints: ', min_BEC_constraints)
    return 1

###################


def find_mislabeled_data(path):
    
    with open(path + '/interaction_data.pickle', 'rb') as f:
        interaction_data = pickle.load(f)
    
    with open(path + '/user_data.pickle', 'rb') as f:
        all_user_data = pickle.load(f)

    unique_user_ids = all_user_data['user_id'].unique()

    mislabeled_flag = np.zeros(len(unique_user_ids))
    for user_no in range(len(unique_user_ids)):
        user_id = unique_user_ids[user_no]
        user_interaction_data = interaction_data[interaction_data['user_id'] == user_id]
        loop_condition = user_interaction_data['loop_condition'].iloc[0]

        # int_no = user_interaction_data.index[0]
        int_no = 0

        # cl should have a remedial feedback and test after a diagnostic feedback
        if loop_condition == 'cl':
            
            print(int_no, user_interaction_data.index[-1])
            while int_no < len(user_interaction_data)-1:
                if user_interaction_data['interaction_type'].iloc[int_no] == 'diagnostic test' and user_interaction_data['is_opt_response'].iloc[int_no] == 0:
                    if user_interaction_data['interaction_type'].iloc[int_no+1] == 'diagnostic feedback' and 'remedial' not in user_interaction_data['interaction_type'].iloc[int_no+2]:
                        mislabeled_flag[user_no] = 1
                        break
                int_no += 1

        
        if loop_condition == 'pl' or loop_condition == 'open':
            diagnostic_test_flag = 0
            while int_no < len(user_interaction_data):
                if user_interaction_data['interaction_type'].iloc[int_no] == 'diagnostic test':
                    diagnostic_test_flag = 1
                    break
                int_no += 1
            
            # pl should have a diagnostic test
            if loop_condition == 'pl' and diagnostic_test_flag == 0:
                mislabeled_flag[user_no] = 1

            # ol should not have a diagnostic test
            if loop_condition == 'open' and diagnostic_test_flag == 1:
                mislabeled_flag[user_no] = 1

    all_user_data['mislabeled_flag'] = mislabeled_flag

    with open(path + '/user_data_w_flag.pickle', 'wb') as f:
        pickle.dump(all_user_data, f)

    all_user_data.to_csv(path + '/user_data_w_flag.csv')
#####
    
def plot_perf_dist(path):

    with open(path + '/user_data_w_flag.pickle', 'rb') as f:
        all_user_data = pickle.load(f)
    
    # filter out mislabeled data
    user_data = all_user_data[all_user_data['mislabeled_flag'] == 0]
        
    # relabel mislabeled data
    # user_data = all_user_data.copy()

    # for id in range(len(user_data)):
    #     if all_user_data['mislabeled_flag'].iloc[id] == 1:
    #         if all_user_data['loop_condition'].iloc[id] == 'pl':
    #            user_data['loop_condition'].iloc[id] = 'open'
            
    #         if all_user_data['loop_condition'].iloc[id] == 'open':
    #             user_data['loop_condition'].iloc[id] = 'pl'



    user_data = user_data[(user_data['loop_condition'] != 'wt') & (user_data['loop_condition'] != 'wtcl')]
    sns.histplot(data=user_data, x='N_final_correct_at', kde=True, stat = 'proportion')

    # fig, ax = plt.subplots(1,3)
    # loop_conds_list = ['open', 'pl', 'cl']

    # for id in range(len(loop_conds_list)):
    #     loop_cond = loop_conds_list[id]
    #     user_data_loop = user_data[user_data['loop_condition'] == loop_cond]
    #     sns.histplot(data=user_data_loop, x='N_final_correct_at', kde=True, stat = 'proportion', ax=ax[id])
    #     ax[loop_conds_list.index(loop_cond)].set_title('Condition: ' + loop_cond)

    plt.show()


##################



def data_prep_grid_search(path):

    with open(path + '/processed_interaction_data.pickle', 'rb') as f:
        interaction_data = pickle.load(f)
    
    with open(path + '/user_data_w_flag.pickle', 'rb') as f:
        all_user_data = pickle.load(f)

    unique_user_ids = all_user_data['user_id'].unique()

    prepared_interaction_data = pd.DataFrame()
    for user_no in range(len(unique_user_ids)):
        user_id = unique_user_ids[user_no]
        user_interaction_data = interaction_data[interaction_data['user_id'] == user_id]

        interaction_constraints = []
        test_constraints = []
        interaction_types = []
        mdp_params = []
        moves = []
        kc_id = []
        test_response_type = []
        is_opt_response = []

        for int_no in range(len(user_interaction_data)):
            kc_id.append(user_interaction_data['kc_id'].iloc[int_no])
            mdp_params.append(user_interaction_data['mdp_params'].iloc[int_no])
            moves.append(user_interaction_data['moves'].iloc[int_no])
            interaction_constraints.append(user_interaction_data['interaction_constraints'].iloc[int_no])
            interaction_types.append(user_interaction_data['interaction_type'].iloc[int_no])
            is_opt_response.append(user_interaction_data['is_opt_response'].iloc[int_no])
            test_response_type.append(user_interaction_data['test_response_type'].iloc[int_no])
            test_constraints.append(user_interaction_data['test_constraints'].iloc[int_no])

        prepared_interaction_data_dict = {'user_id': user_id, 'kc_id': kc_id, 'interaction_constraints': interaction_constraints, 'interaction_types': interaction_types, \
                                          'is_opt_response': is_opt_response, 'test_response_type': test_response_type, 'test_constraints': test_constraints}

        prepared_interaction_data = prepared_interaction_data.append(prepared_interaction_data_dict, ignore_index=True)

    with open(path + '/prepared_interaction_data.pickle', 'wb') as f:
        pickle.dump(prepared_interaction_data, f)
    
    prepared_interaction_data.to_csv(path + '/prepared_interaction_data.csv')

    return 1
#########################


def simulate_objective(learning_params):

    mdp_domain = 'augmented_taxi2'
    viz_flag = True
    learner_update_type = 'no_noise'

    # sim params
    max_learning_factor = params.max_learning_factor
    initial_learning_factor = copy.deepcopy(learning_params['initial_learning_factor'])
    learning_factor_delta = copy.deepcopy(learning_params['learning_factor_delta'])

    # initialize (simulated) learner particle filters
    initial_learner_pf = copy.deepcopy(all_learner_pf['p1'])

    # prior interaction data
    prior_test_constraints = [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])]  # constraints of the first concept/KC
    all_test_constraints = {  1: [np.array([[ 1,  0, -4]]), np.array([[-1,  0,  2]])], \
                                2: [np.array([[0, 1, 2]]), np.array([[ 0, -1, -4]])], \
                                3: [np.array([[1, 1, 0]])]}
    min_BEC_constraints = [np.array([[1, 1, 0]]), np.array([[-1,  0,  2]]), np.array([[ 0, -1, -4]])]

    
    # initialize dataframes to save probability data
    simulated_interaction_data = pd.DataFrame()


    # simulate teaching loop
    prev_kc_id = 1
    demo_id = 1
    unit_constraints = [np.array([[0, 0, -1]])]
    # objective = 0
    objective = []

    learning_factor = copy.deepcopy(initial_learning_factor)

    # plotting
    color_dict = {'demo': 'blue', 'remedial demo': 'purple', 'diagnostic test': 'red',  'remedial test': 'pink', 'diagnostic feedback': 'yellow', 'remedial feedback': 'orange', 'final test': 'green'}
    


    for user_id, user_interaction_data in prepared_interaction_data.iterrows():

        vars_filename_prefix = 'param_fit_user_' + str(user_id) + 'init_lf_' + str(initial_learning_factor) + 'lf_delta_' + str(learning_factor_delta)
        N_final_tests_correct = 0

        print('user_id:', user_id, '.Len user_data:', len(user_interaction_data['kc_id']))

        # params to plot
        user_interaction_type = []
        user_learning_factor = []
        user_prob_BEC = []
        user_prob_KC = []
        user_loop_id = []
        concept_end_id = []
        plot_id = 0
        for loop_id in range(len(user_interaction_data['kc_id'])):
            
            # print('user_data_kcid:', user_data['kc_id'])
            current_kc_id = user_interaction_data['kc_id'][loop_id]
            current_interaction_type = user_interaction_data['interaction_types'][loop_id]
            # is_opt_response = user_data['is_opt_response'][loop_id]
            current_interaction_constraints = user_interaction_data['interaction_constraints'][loop_id]
            current_test_constraints = user_interaction_data['test_constraints'][loop_id]   

            # Not needed for now!
            # if current_interaction_type != 'final test' and current_kc_id <= max_kc:
            #     test_constraints = all_test_constraints[current_kc_id]
            # else:
            #     test_constraints = min_BEC_constraints

            if current_kc_id > prev_kc_id:
                learning_factor = copy.deepcopy(initial_learning_factor)
                # print('New KC! Resetting learning factor to initial value: ', learning_factor)
                concept_end_id.append(plot_id-1)

            # updates for various interaction types
            # Prior
            if current_interaction_type == 'prior':
                learner_pf = copy.deepcopy(initial_learner_pf)
            
            # Demo
            if current_interaction_type == 'demo':
                learner_pf.update(current_interaction_constraints, learning_factor, plot_title = 'Learner belief after demo. Interaction ID:  ' + str(loop_id) + ' for KC: ' + str(current_kc_id), viz_flag = viz_flag, vars_filename=vars_filename_prefix, model_type = learner_update_type)


            # Diagnostic Test
            if current_interaction_type == 'diagnostic test':
                # Nothing changes
                response_type = user_interaction_data['test_response_type'][loop_id][0]


            # Diagnostic Feedback
            if current_interaction_type == 'diagnostic feedback':
                # print('response_type: ', response_type)
                if response_type == 'correct':
                    learning_factor[0] = min(learning_factor[0] + learning_factor_delta[0], max_learning_factor)
                elif response_type == 'incorrect':
                    learning_factor[0] = min(learning_factor[0] + learning_factor_delta[1], max_learning_factor)
                else:
                    RuntimeError('Invalid response type')

                # updated learner model with corrective feedback
                plot_title =  ' Learner after corrective feedback for KC ' + str(current_kc_id)
                learner_pf.update(current_interaction_constraints, learning_factor, plot_title = plot_title, viz_flag = viz_flag, vars_filename = vars_filename_prefix, model_type = learner_update_type)

            # Remedial Demo
            if current_interaction_type == 'remedial demo':
                plot_title =  'Learner belief after remedial demo. Interaction ID: ' + str(loop_id) + ' for KC ' + str(current_kc_id)
                learner_pf.update(current_interaction_constraints, learning_factor, plot_title = plot_title, viz_flag = viz_flag, vars_filename=vars_filename_prefix, model_type = learner_update_type)
                
            # Remedial Test
            if current_interaction_type == 'remedial test':
                response_type = user_interaction_data['test_response_type'][loop_id][0]

            # Remedial Feedback
            if current_interaction_type == 'remedial feedback':
                # print('response_type: ', response_type)
                if response_type == 'correct':
                    learning_factor[0] = min(learning_factor[0] + learning_factor_delta[0], max_learning_factor)
                elif response_type == 'incorrect':
                    learning_factor[0] = min(learning_factor[0] + learning_factor_delta[1], max_learning_factor)
                else:
                    RuntimeError('Invalid response type')

                # updated learner model with corrective feedback
                plot_title =  ' Learner after remedial feedback for KC ' + str(current_kc_id)
                learner_pf.update(current_interaction_constraints, learning_factor, plot_title = plot_title, viz_flag = viz_flag, vars_filename = vars_filename_prefix, model_type = learner_update_type)

            # Final Test Performance
            if current_interaction_type == 'final test':
                if user_interaction_data['is_opt_response'][loop_id] == 1:
                    N_final_tests_correct += 1

            if 'prior_' not in current_interaction_type:
                # calculate probability of correct response
                learner_pf.calc_particles_probability(current_test_constraints)
                prop_particles_KC = learner_pf.particles_prob_correct


                learner_pf.calc_particles_probability(min_BEC_constraints)
                prop_particles_BEC = learner_pf.particles_prob_correct
                # print('loop_id: ', loop_id, 'interaction: ', current_interaction_type, 'prop_particles_BEC: ', prop_particles_BEC)

                # update loop vars
                user_interaction_type.append(current_interaction_type)
                user_learning_factor.append(learning_factor[0])
                user_prob_BEC.append(prop_particles_BEC)
                user_prob_KC.append(prop_particles_KC)
                user_loop_id.append(plot_id)
                plot_id += 1

            # update loop kcid
            prev_kc_id = current_kc_id

        
        # plot user learning dynamics
        user_plot_data = pd.DataFrame({'loop_id': user_loop_id, 'interaction_type': user_interaction_type, 'learning_factor': user_learning_factor, 'prob_KC': user_prob_KC, 'prob_BEC': user_prob_BEC})
            
        plt.figure(user_id)
        ax0 = plt.gca()
        print('user_id: ', user_id, 'user_plot_data:', user_plot_data)
        sns.lineplot(data=user_plot_data, x='loop_id', y='prob_BEC', ax=ax0, color = 'blue').set(title='Obj. func. learning dynamics for user: ' + str(user_id))
        sns.lineplot(data=user_plot_data, x='loop_id', y='prob_KC', ax=ax0, color = 'brown')
        sns.lineplot(data=user_plot_data, x='loop_id', y='learning_factor', ax=ax0, color = 'green')

            
        for id, row in user_plot_data.iterrows():
            print('id:', id)
            if row['interaction_type'] != 'prior':
                plt.axvspan(user_plot_data['loop_id'].iloc[id-1], row['loop_id'], alpha=0.2, color=color_dict[row['interaction_type']])
                plt.text(row['loop_id']-0.5, 0.3, row['interaction_type'], rotation=90, fontsize=12, weight="bold")

        for id in concept_end_id:
            plt.axvline(x=id, color='black', linestyle='--', linewidth=2)
        
        plt.show()

        # calculate final probability
        learner_pf.calc_particles_probability(min_BEC_constraints)
        prop_particles_BEC = learner_pf.particles_prob_correct

        # final test performance
        test_perf = N_final_tests_correct/6

        # update objective function
        # objective += np.abs(prop_particles_BEC - test_perf)
        objective.append(np.abs(prop_particles_BEC-test_perf))

        print('user_id: ', user_id, 'N_final_tests_correct: ', N_final_tests_correct, 'prop_particles_BEC: ', prop_particles_BEC, 'test_perf: ', test_perf, 'objective: ', prop_particles_BEC - test_perf)

        

    # return objective/len(prepared_interaction_data)
        
    return objective




######################


def analyze_grid_search():


    try:
        with open('data/simulation/sim_experiments/parameter_estimation/grid_search_results.pickle', 'rb') as f:
            grid_search_results = pickle.load(f)

    except:

        path = 'data/simulation/sim_experiments/parameter_estimation/current'

        files = os.listdir(path)

        calib_vars_list = []
        for learner_type in ['low', 'high']:
            for file in files:
                if learner_type in file:
                    print('Reading file: ', file, '...')
                    with open(path + '/' + file, 'rb') as f:
                        learner_type, dataset_type, study_id, run_id, u, delta_c, delta_i, objective, prop_particles_BEC_list = pickle.load(f)
                        objective = objective/len(prop_particles_BEC_list)
                    calib_dict = {'learner_type': learner_type, 'dataset_type': dataset_type, 'study_id': study_id, 'run_id': run_id, 'u': np.round(u,2), 'delta_c': np.round(delta_c,2), 'delta_i': np.round(delta_i,2), 'objective': np.round(objective,2), 'prop_particles_BEC_list': prop_particles_BEC_list}
                    calib_vars_list.append(calib_dict)


        grid_search_results = pd.DataFrame(calib_vars_list)

        with open('data/simulation/sim_experiments/parameter_estimation/grid_search_results.pickle', 'wb') as f:
            pickle.dump(grid_search_results, f)

        grid_search_results.to_csv('data/simulation/sim_experiments/parameter_estimation/grid_search_results.csv')
    ###########################
        
    # plot grid search results performance
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
    learner_list = ['low', 'high']
    
    for i in range(len(learner_list)):
        learner = learner_list[i]
        grid_search_results_subset = grid_search_results[grid_search_results['learner_type'] == learner]
        sns.histplot(data=grid_search_results_subset, x='objective', kde=True, ax=axs2[i]).set(title='Learner type: ' + learner)

    #     # # Fit a Gaussian distribution
    #     # mu, std = norm.fit(grid_search_results_subset['objective'])
    #     # xmin, xmax = axs[i].get_xlim()
    #     # x = np.linspace(xmin, xmax, 100)
    #     # p = norm.pdf(x, mu, std)

    #     # # Overlay the Gaussian distribution on the histogram
    #     # axs[i].plot(x, p, 'k', linewidth=2)

    #########################
    # plot grid search parameters
        
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    learner_list = ['low', 'high']
    vars_list = ['u', 'delta_c', 'delta_i']
    max_objective_list = [0.2, 0.2]
    params_ranges = [[0.5, 0.9], [0.0, 0.2], [0.0, 0.2]]
    params_diff = [0.05, 0.02, 0.02]


    for i in range(len(learner_list)):
        for j in range(len(vars_list)):
            learner = learner_list[i]
            var = vars_list[j]
            grid_search_results_subset = grid_search_results[(grid_search_results['learner_type'] == learner) & (grid_search_results['objective'] <= max_objective_list[i])]
            
            
            # Labels
            if var == 'u':
                var_label = r'$\beta_0$'
            elif var == 'delta_c':
                var_label = r'$\delta \beta_c$'
            elif var == 'delta_i':
                var_label = r'$\delta \beta_i$'
            
            
            
            sns.histplot(data=grid_search_results_subset, x=var, stat='density', binrange=params_ranges[j], binwidth=params_diff[j], ax=axs[i, j]).set(title='Learner type: ' + learner + '. Var: ' + var_label)

            # Fit a Gaussian distribution
            xmin, xmax = axs[i, j].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            if var == 'u':
                if learner == 'low':
                    mu, std = norm.fit(grid_search_results_subset[var]-0.04)
                    p = norm.pdf(x, mu, std)
                else:
                    mu, std = norm.fit(grid_search_results_subset[var])
                    p = norm.pdf(x, mu, std)
            else:
                if var == 'delta_c':
                    mu, std = halfnorm.fit(grid_search_results_subset[var])
                    p = halfnorm.pdf(x, mu, std)
                else:
                    mu, std = halfnorm.fit(max(grid_search_results_subset[var])-grid_search_results_subset[var])
                    p = np.flip(halfnorm.pdf(x, mu, std))


            print('learner: ', learner, '. var: ', var, '. mu: ', mu, '. std: ', std)

            # Overlay the Gaussian distribution on the histogram
            axs[i, j].plot(x, p, 'k', linewidth=2)  

            # Label axes
            axs[i, j].set_xlabel(var_label)


    

    plt.show()

    return 1


#########################

if __name__ == "__main__":

    # # find human response distribution
    # extract_study_data()
    ########

    # ## load study data
    # path = 'data'
    # with open(path + '/user_data.pickle', 'rb') as f:
    #     user_data = pickle.load(f)
    # with open(path + '/study_data.pickle', 'rb') as f:
    #     study_data = pickle.load(f)

    # ## Run Monte Carlo simulations


    # run_MC_sims(user_data, study_data)
    ##########


    # path = 'data'
    # with open(path + '/user_data_w_flag.pickle', 'rb') as f:
    #     all_user_data = pickle.load(f)

        
    # user_data = all_user_data[all_user_data['mislabeled_flag'] == 0]

    # user_id = 30
    # user_data_trial = user_data[user_data['user_id'] == user_id]

    # ## plot learning dynamics
    # plot_learning_dynamics(user_data_trial)
        
    ############################
    # # ## (Debug) read summary data
    # path = 'models_user_study/skateboard2'
    # # # file_list = ['BEC_summary.pickle', 'BEC_summary_2.pickle', 'BEC_summary_3.pickle']
    # # # file_list = ['BEC_summary_open.pickle', 'BEC_summary_open_2.pickle', 'BEC_summary_open_3.pickle']
    # file_list = ['BEC_summary.pickle']
    # data_loc  = 'skateboard2'
    # read_summary_data(path, data_loc, file_list)


    #################
    # 1 - simply; first user
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
    # 102 - Open loop; High learner; low performance
    # 108 - Open loop; High learner; high performance
    # 29, 51, 54 - Perfect learner; closed loop; all final test correct 
    # 159 - Open loop; low learner
    # 181 - Open loop; high learner
    # 218 - Partial loop; high learner
    # 135 - Partial loop; low learner
        
    #############################
        
    # # analyze learning dynamics
    # path = 'data'
    # with open(path + '/user_data_w_flag.pickle', 'rb') as f:
    #     all_user_data = pickle.load(f)
    # with open(path + '/study_data.pickle', 'rb') as f:
    #     study_data = pickle.load(f)

    # user_data = all_user_data[all_user_data['mislabeled_flag'] == 0]

    # # user_id = 218
    # # user_data_trial = user_data[user_data['user_id'] == user_id]
    # # study_data_trial = study_data[study_data['user_id'] == user_id]
    # analyze_learning_dynamics(user_data, study_data, file_prefix = 'w_mdp')

    ###########################

    # # ##run learner model
    # study_id = 4
    # initial_learning_factor = [0.80]  #default: 0.92 (used in Mike's study)
    # learning_factor_delta = [0.035, 0.068] #default: 0.0, 0.0

    # run_id = 1
    # user_id = 30

    # learner_update_type = 'no_noise'
    # domain = 'at'
    # path = 'data'


    # # filename = 'interaction_data_' + str(user_id) + '.pickle'
    # filename = 'interaction_data_w_mdp.pickle'

    # with open(path + '/' + filename, 'rb') as f:
    #     all_interaction_data = pickle.load(f)
    # # interaction_data = all_interaction_data[(all_interaction_data['user_id'] == user_id) & (all_interaction_data['domain'] == domain)]
    # interaction_data = all_interaction_data[(all_interaction_data['user_id'] == user_id)]


    # params.max_learning_factor = 1.0
    # params.team_size = 1

    # # print('interaction_data: ', interaction_data)

    # run_sim_trials(params, study_id, run_id, interaction_data, domain, initial_learning_factor, learning_factor_delta, learner_update_type, viz_flag=False, vars_filename_prefix = 'study_3_simulation_' + learner_update_type)

    # ###############################
        
    # path = 'data'
    # find_mislabeled_data(path)
    # plot_perf_dist(path)

    # ###########
    # path = 'data'

    # params.max_learning_factor = 1.0
    # domain = 'at'

    # with open(path + '/user_data_w_flag.pickle', 'rb') as f:
    #     all_user_data = pickle.load(f)
    
    # user_data = all_user_data[all_user_data['mislabeled_flag'] == 0]
    # unique_user_ids = user_data['user_id'].unique()
    # # unique_user_ids = [5, 30]

    # # with open(path + '/study_data.pickle', 'rb') as f:
    # #     study_data = pickle.load(f)

    # with open(path + '/interaction_data_w_mdp.pickle', 'rb') as f:
    #     all_interaction_data = pickle.load(f)
    # # interaction_data = all_interaction_data[(all_interaction_data['user_id'] == user_id) & (all_interaction_data['domain'] == domain)]


    # sim_data = pd.DataFrame()
    # for user_id in unique_user_ids:
    #     user_interaction_data = all_interaction_data[(all_interaction_data['user_id'] == user_id) & (all_interaction_data['domain'] == domain)]
    #     sim_study_data_user = run_sim_trials(params, 1, 1, user_interaction_data, domain, [], [], [], viz_flag=False, vars_filename_prefix = '')
    #     sim_data = sim_data.append(sim_study_data_user, ignore_index=True)

    # with open(path + '/processed_interaction_data.pickle', 'wb') as f:
    #     pickle.dump(sim_data, f)
    
    # sim_data.to_csv(path + '/processed_interaction_data.csv')

    ##############################

    # path = 'data'
    # data_prep_grid_search(path)

    # ####################################
    # function to give output
    # params.team_size = 1
    # params.max_learning_factor = 1.0

    # all_learner_pf = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_learning_factor = [0.8], team_prior = params.team_prior, pf_flag='learner', model_type = 'no_noise')

    # with open('data/prepared_interaction_data.pickle', 'rb') as f:
    #     prepared_interaction_data_full = pickle.load(f)

    # with open('data/user_data_w_flag.pickle', 'rb') as f:
    #     all_user_data = pickle.load(f)

    # # user_data = all_user_data[(all_user_data['mislabeled_flag'] == 0) & (all_user_data['loop_condition'] != 'wt') & (all_user_data['loop_condition'] != 'wtcl') & (all_user_data['N_final_correct_at'] == 2)]
    
    # user_data = all_user_data[(all_user_data['mislabeled_flag'] == 0) & (all_user_data['loop_condition'] != 'wt') & (all_user_data['loop_condition'] != 'wtcl') & \
    #                           ((all_user_data['N_final_correct_at'] == 2) | (all_user_data['N_final_correct_at'] == 3) | (all_user_data['N_final_correct_at'] == 4))]

    
    
    # unique_user_ids = user_data['user_id'].unique()
    # print('user_data: ', user_data)

    # prepared_interaction_data = pd.DataFrame()

    # for user_id in unique_user_ids:
    #     prepared_interaction_data = pd.concat([prepared_interaction_data, prepared_interaction_data_full[prepared_interaction_data_full['user_id'] == user_id]], ignore_index=True)
    # #  [0.5992593795130855, 0.1630325868102005, 0.03176307063688731]
    # #   [0.6685487563750678, 0.0]
    # #  [0.6568400939216162, 0.04]
    # # [0.7456719191538574, 0.04] for low learner full set
    # # [0.6624425805340786, 0.02, 0.02] for test; 100 func evals
    # learning_params = {'initial_learning_factor': [0.8], 'learning_factor_delta': [0.035, 0.068]}

    # objective = simulate_objective(learning_params)

    # print('Objective: ', objective)

    #################################

    # ## analyze grid search
    analyze_grid_search()