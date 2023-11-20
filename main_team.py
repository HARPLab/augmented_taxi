# Python imports.
import sys
import dill as pickle
import numpy as np
import copy
from termcolor import colored
from multiprocessing import Process, Queue, Pool
import sage.all
import sage.geometry.polyhedron.base as Polyhedron
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import itertools

# Other imports.
# sys.path.append("simple_rl")
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
import matplotlib as mpl
import teams.teams_helpers as team_helpers
import simulation.sim_helpers as sim_helpers
import teams.utils_teams as utils_teams
mpl.rcParams['figure.facecolor'] = '1.0'
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'large'

import random
import pandas as pd
from numpy.linalg import norm
from analyze_sim_data import run_analysis_script


############################################






def calc_expected_learning(team_knowledge_expected, kc_id, kc_reset_flag, particles_team, min_BEC_constraints, new_constraints, params, loop_count, viz_flag = False):

    print(colored('Calculating expected learning for KC ' + str(kc_id) + 'in loop ' + str(loop_count) + '...', 'blue'))
    # # print('Current expected team knowledge: ', team_knowledge_expected)
    # # print('New constraints: ', new_constraints)
    team_knowledge_expected = team_helpers.update_team_knowledge(team_knowledge_expected, kc_id, kc_reset_flag, new_constraints, params.team_size,  params.weights['val'], params.step_cost_flag)

    # # print('Updated expected team knowledge: ', team_knowledge_expected)

    p = 1
    while p <= params.team_size:
        member_id = 'p' + str(p)
        # print('PF update of ', member_id)
        particles_team[member_id].update(new_constraints)
        
        if viz_flag:
            team_helpers.visualize_transition(new_constraints, particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Expected knowledge change for set ' + str(loop_count+1) + ' for player ' + member_id, plot_filename ='ek_p' + str(p) + '_loop_' + str(loop_count+1))
        p += 1
            
    # Update common knowledge model
    # # print('PF update of common_knowledge')
    particles_team['common_knowledge'].update(new_constraints)
    # if viz_flag:
    #     team_helpers.visualize_transition(new_constraints, particles_team_expected['common_knowledge'], params.mdp_class, params.weights['val'], text = 'Expected knowledge change for set' + str(loop_count+1) + ' for common knowledge', plot_filename ='ek_ck_loop_' + str(loop_count+1))
    

    # Update joint knowledge model
    # Method 1: Use complete joint knowledge of team
    # particles_team_expected['joint_knowledge'].update_jk(team_knowledge_expected['joint_knowledge'])

    # Method 2: Use new joint knowledge of team
    new_constraints_team = []
    for p in range(params.team_size):
        new_constraints_team.append(new_constraints)
    # # print('PF update of joint_knowledge')
    particles_team['joint_knowledge'].update_jk(new_constraints_team)

    # if viz_flag:
    #     team_helpers.visualize_transition(new_constraints_team, particles_team_expected['joint_knowledge'], params.mdp_class, params.weights['val'], text = 'Expected knowledge change for set ' + str(loop_count+1) + ' for joint knowledge',  knowledge_type = 'joint_knowledge', plot_filename ='ek_jk_loop_' + str(loop_count+1))

    # # print('min_BEC_constraints: ', min_BEC_constraints)
    # # print('Expected unit knowledge after seeing unit demonstrations: ', team_helpers.calc_knowledge_level(team_knowledge_expected, new_constraints) )
    # # print('Expected absolute knowledge after seeing unit demonstrations: ', team_helpers.calc_knowledge_level(team_knowledge_expected, min_BEC_constraints) )


    return team_knowledge_expected, particles_team




def run_remedial_loop(failed_BEC_constraints_tuple, particles_team, team_knowledge, min_subset_constraints_record, env_record, traj_record, traj_features_record, test_history, visited_env_traj_idxs, running_variable_filter_unit, mdp_features_record, consistent_state_count, particles_demo, pool, viz_flag = False, experiment_type = 'simulated'):
    human_history = []

    # Method 1: Generate remedial demonstration from the combined failed BEC constraints of the team
    # # print("Here is a remedial demonstration that might be helpful")
    # min_failed_constraints = BEC_helpers.remove_redundant_constraints(failed_BEC_constraints_team, params.weights['val'], params.step_cost_flag)

    # # print('min_failed_constraints:', min_failed_constraints)
    
    # remedial_demos, visited_env_traj_idxs = team_helpers.obtain_remedial_demos_tests(params.data_loc['BEC'], BEC_summary[-1], visited_env_traj_idxs, min_failed_constraints, min_subset_constraints_record, traj_record, traj_features_record, running_variable_filter_unit, mdp_features_record)
    # # # print(remedial_demos[0])

    # # TODO: Show remedial demonstration
    # for demo in remedial_demos:
    #     demo[0].visualize_trajectory(demo[1])
    #     test_history.extend(demo)

    #     particles_demo.update([demo[3]])
    #     # # print('remedial demo :', demo)
    #     # # print('remedial demo constraints:', demo[3])
    #     if visualize_pf_transition:
    #         team_helpers.visualize_transition(demo[3], particles_demo, params.mdp_class, params.weights['val'])

    #     with open('models/' + params.data_loc['BEC'] + '/remedial_instruction.pickle', 'wb') as f:
    #         pickle.dump(demo, f)


    # Method 2: Generate remedial demonstration for each individual

    for idx, failed_BEC_cnst_tuple in enumerate(failed_BEC_constraints_tuple):
        

        member_id = failed_BEC_cnst_tuple[0]
        
        remedial_instruction, visited_env_traj_idxs = BEC.obtain_remedial_demonstrations(params.data_loc['BEC'], pool, particles_team[member_id], params.BEC['n_human_models'], failed_BEC_cnst_tuple[1], min_subset_constraints_record, env_record, traj_record, traj_features_record, test_history, visited_env_traj_idxs, running_variable_filter_unit, mdp_features_record, consistent_state_count, params.weights['val'], params.step_cost_flag, n_human_models_precomputed= params.BEC['n_human_models_precomputed'])
        remedial_mdp, remedial_traj, _, remedial_constraint, _ = remedial_instruction[0]
        if viz_flag:
            remedial_mdp.visualize_trajectory(remedial_traj)
        test_history.extend(remedial_instruction)

        particles_demo.update([remedial_constraint])
        if viz_flag:
            BEC_viz.visualize_pf_transition([remedial_constraint], particles_demo, params.mdp_class, params.weights['val'])


        with open('models/' + params.data_loc['BEC'] + '/remedial_instruction.pickle', 'wb') as f:
            pickle.dump(remedial_instruction, f)

        
        if experiment_type == 'simulated':
            N_remedial_tests = random.randint(1, 3)
            # print('N_remedial_tests: ', N_remedial_tests)
        remedial_resp_no = 0

        # print("Here is a remedial test to see if you've correctly learned the lesson")
        remedial_test_end = False
        while (not remedial_test_end) and (remedial_resp_no < N_remedial_tests): #Note: To be updated. Do not show remedial until they get it right, rather show it until there is common knowledge and joint knowledge
            # print('Still inside while loop...')
            remedial_test, visited_env_traj_idxs = BEC.obtain_remedial_demonstrations(params.data_loc['BEC'], pool,
                                                                                                particles_demo,
                                                                                                params.BEC['n_human_models'],
                                                                                                failed_BEC_cnst_tuple[1],
                                                                                                min_subset_constraints_record,
                                                                                                env_record,
                                                                                                traj_record,
                                                                                                traj_features_record,
                                                                                                test_history,
                                                                                                visited_env_traj_idxs,
                                                                                                running_variable_filter_unit,
                                                                                                mdp_features_record,
                                                                                                consistent_state_count,
                                                                                                params.weights['val'],
                                                                                                params.step_cost_flag, type='testing', n_human_models_precomputed=params.BEC['n_human_models_precomputed'])

            remedial_mdp, remedial_traj, remedial_env_traj_tuple, remedial_constraint, _ = remedial_test[0]
            test_history.extend(remedial_test)


            if experiment_type == 'simulated':
                if remedial_resp_no == N_remedial_tests-1:
                    remedial_response = 'correct'
                else:
                    remedial_response = 'mixed' # Note: We assume that one person always gets the remedial test correct and the other person gets it wrong (only for N=2)
                human_traj_team, human_history = sim_helpers.get_human_response_old(remedial_env_traj_tuple[0], remedial_constraint, remedial_traj, human_history, team_knowledge, team_size = params.team_size, response_distribution = remedial_response)
                remedial_resp_no += 1
                # print('Simulating human response for remedial test... Remedial Response no: ', remedial_resp_no, '. Response type: ', remedial_response)

            
            remedial_constraints_team = []
            # Show the same test for each person and get test responses of each person in the team
            p = 1
            while p <= params.team_size:
                member_id = 'p' + str(p)

                if experiment_type == 'simulated':
                    human_traj = human_traj_team[p-1]
                else:
                    human_traj, human_history = remedial_mdp.visualize_interaction(
                        keys_map=params.keys_map)  # the latter is simply the gridworld locations of the agent


                human_feature_count = remedial_mdp.accumulate_reward_features(human_traj, discount=True)
                opt_feature_count = remedial_mdp.accumulate_reward_features(remedial_traj, discount=True)

                if (human_feature_count == opt_feature_count).all():
                    # print("You got the remedial test correct")
                    particles_demo.update([remedial_constraint])
                    remedial_constraints_team.append([remedial_constraint])
                    if viz_flag:
                        team_helpers.visualize_transition([remedial_constraint], particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Remedial Test ' + str(remedial_resp_no) + ' for player ' + member_id)

                else:
                    failed_BEC_constraint = opt_feature_count - human_feature_count
                    # # print('Optimal traj: ', remedial_traj)
                    # # print('Human traj: ', human_traj)
                    if viz_flag:
                        # print("You got the remedial test wrong. Here's the correct answer")
                        # print("Failed BEC constraint: {}".format(failed_BEC_constraint))
                        remedial_mdp.visualize_trajectory_comparison(remedial_traj, human_traj)
                    # else:
                        # print("You got the remedial test wrong. Failed BEC constraint: {}".format(failed_BEC_constraint))

                    particles_demo.update([-failed_BEC_constraint])
                    remedial_constraints_team.append([-failed_BEC_constraint])
                    if viz_flag:
                        # BEC_viz.visualize_pf_transition([-failed_BEC_constraint], particles_demo, params.mdp_class, params.weights['val'])
                        team_helpers.visualize_transition([-failed_BEC_constraint], particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Remedial Test ' + str(remedial_resp_no) + ' for player ' + member_id)


                p += 1
        
        # Update team knowledge
        # opposing_constraints_flag, _, _ = team_helpers.check_opposing_constraints(remedial_constraints_team)
        non_intersecting_constraints_flag, _ = team_helpers.check_for_non_intersecting_constraints(remedial_constraints_team, params.weights['val'], params.step_cost_flag)
        # print('Opposing constraints remedial loop? ', opposing_constraints_flag)

        if not non_intersecting_constraints_flag:
            remedial_test_end = True
            remedial_constraints_team_expanded = []
            for test_constraints in remedial_constraints_team:
                remedial_constraints_team_expanded.extend(test_constraints)
            
            # Update individual belief
            p = 1
            while p <= params.team_size:
                member_id = 'p' + str(p)
                team_knowledge = team_helpers.update_team_knowledge(team_knowledge, remedial_constraints_team[p-1], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = [member_id])
                particles_team[member_id].update(remedial_constraints_team[p-1])
                particles_team[member_id].knowledge_update(team_knowledge[member_id])
                p += 1

            if viz_flag:
                team_helpers.visualize_transition(remedial_constraints_team[p-1], particles_team[member_id], params.mdp_class, params.weights['val'], text = 'After Remedial Test for player ' + member_id)

            # Update team beliefs
            team_knowledge = team_helpers.update_team_knowledge(team_knowledge, [], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = ['common_knowledge', 'joint_knowledge'])
            particles_team['common_knowledge'].update(remedial_constraints_team_expanded)
            particles_team['common_knowledge'].knowledge_update(team_knowledge['common_knowledge'])
            particles_team['joint_knowledge'].update_jk(remedial_constraints_team)
            particles_team['joint_knowledge'].knowledge_update(team_knowledge['joint_knowledge'])

    return test_history, visited_env_traj_idxs, particles_team, team_knowledge, particles_demo, remedial_constraints_team_expanded, remedial_resp_no




# def run_reward_teaching(params, pool, demo_strategy = 'common', experiment_type = 'simulated', response_distribution_list = ['correct']*10, run_no = 1, vars_to_save = None):
def run_reward_teaching(params, pool, sim_params, demo_strategy = 'common_knowledge', experiment_type = 'simulated',  team_likelihood_correct_response = 0.5*np.ones(params.team_size) ,  team_learning_rate = np.hstack((0.2*np.ones([params.team_size, 1]), -0.1*np.ones([params.team_size, 1]))), run_no = 1, viz_flag=[False, False, False], vars_filename = 'var_to_save'):

    ## Initialize variables
    initial_likelihood_vars = {'ilcr': np.copy(team_likelihood_correct_response), 'rlcr': np.copy(team_learning_rate)}
    
    plt.rcParams['figure.figsize'] = [10, 6]
    BEC_summary, visited_env_traj_idxs, min_BEC_constraints_running, prior_min_BEC_constraints_running = [], [], copy.deepcopy(params.prior), copy.deepcopy(params.prior)
    summary_count, prev_summary_len = 0, 0
    obj_func_prop = 1.0 
    demo_viz_flag, test_viz_flag, knowledge_viz_flag = viz_flag

    # # hard-coded for the current reward weights and pre-computed counterfactuals
    # all_unit_constraints = [np.array([[-2, -1,  0]]), np.array([[1, 1, 0]]), np.array([[-1,  0,  2]])]
    all_unit_constraints = []

    #####################################
    # get optimal policies
    ps_helpers.obtain_env_policies(params.mdp_class, params.data_loc['BEC'], np.expand_dims(params.weights['val'], axis=0), params.mdp_parameters, pool)

    # get base constraints for all the environments and demonstrations
    try:
        with open('models/' + params.data_loc['BEC'] + '/team_base_constraints.pickle', 'rb') as f:
            policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(f)
    except:
        # use policy BEC to extract constraints
        policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = BEC.extract_constraints(params.data_loc['BEC'], params.BEC['BEC_depth'], params.step_cost_flag, pool, print_flag=True)
        with open('models/' + params.data_loc['BEC'] + '/team_base_constraints.pickle', 'wb') as f:
            pickle.dump((policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count), f)

    # get BEC constraints
    try:
        with open('models/' + params.data_loc['BEC'] + '/team_BEC_constraints.pickle', 'rb') as f:
            min_BEC_constraints, BEC_lengths_record = pickle.load(f)
    except:
        min_BEC_constraints, BEC_lengths_record = BEC.extract_BEC_constraints(policy_constraints, min_subset_constraints_record, env_record, params.weights['val'], params.step_cost_flag, pool)

        with open('models/' + params.data_loc['BEC'] + '/team_BEC_constraints.pickle', 'wb') as f:
            pickle.dump((min_BEC_constraints, BEC_lengths_record), f)
    
    ########################

    # sample particles for human models
    team_prior, particles_team = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_prior = params.team_prior)
    team_knowledge = copy.deepcopy(team_prior) # team_prior calculated from team_helpers.sample_team_pf also has the aggregated knowledge from individual priors
    # particles_team_expected = particles_team
    team_knowledge_expected = copy.deepcopy(team_knowledge)

    print(colored('min_BEC_constraints for this run: ', 'red'))
    print(min_BEC_constraints)

    # unit (set of knowledge components) selection
    variable_filter, nonzero_counter, teaching_complete_flag = team_helpers.check_and_update_variable_filter(min_subset_constraints_record, initialize_filter_flag=True)

    # for testing design choices (is remedial demo needed)
    remedial_test_flag = False
    demo_vars_template = {'run_no': 1,
                         'demo_strategy': None,
                         'knowledge_id': None,
                          'variable_filter': None,
                          'knowledge_comp_id': None,
                          'loop_count': None,  
                            'summary_count': None,
                            'min_BEC_constraints': None,
                            'unit_constraints': None,
                            'all_unit_constraints': None,  
                            'demo_ids': None,
                            'team_knowledge_expected': None,
                            'particles_team_expected': None,
                            'unit_knowledge_level_expected': None,
                            'BEC_knowledge_level_expected': None,
                            # 'response_type': None,
                            # 'response_distribution': None,
                            'test_constraints': None,
                            'test_constraints_team': None,
                            'opposing_constraints_count': 0,
                            'final_remedial_constraints': None,
                            'N_remedial_tests': None,
                            'team_knowledge': None,
                            'particles_team': None,
                            'unit_knowledge_level': None,
                            'BEC_knowledge_level': None,
                            'unit_knowledge_area': None,
                            'all_unit_constraints_area': None,
                            'BEC_knowledge_area': None,
                            'pf_reset_count': np.zeros(params.team_size+2, dtype=int),
                            'likelihood_correct_response': np.zeros(params.team_size, dtype=int)
                            }
    

    try:
        with open('models/augmented_taxi2/' + vars_filename + '_' + str(run_no) + '.pickle', 'rb') as f:
            vars_to_save = pickle.load(f)
    except:
        vars_to_save = pd.DataFrame(columns=demo_vars_template.keys())


    # print('Demo strategy: ', demo_strategy)

    # initialize human models for demo generation
    knowledge_id, particles_demo = team_helpers.particles_for_demo_strategy(demo_strategy, team_knowledge, particles_team, params.team_size, params.weights['val'], params.step_cost_flag, params.BEC['n_particles'], min_BEC_constraints)

    ################################################
    
    # Initialize teaching loop variables
    loop_count, next_unit_loop_id, resp_no = 0, 0, 0
    unit_learning_goal_reached, next_kc_flag = False, False
    kc_id = 1 # knowledge component id or variable filter id
    
    # Teaching-testing loop
    while not teaching_complete_flag:

        # reset loop varibales
        opposing_constraints_count, non_intersecting_constraints_count, N_remedial_tests = 0, 0, 0
        remedial_constraints_team_expanded = []
        if next_kc_flag:
            # next_unit_loop_id += 1
            team_likelihood_correct_response = np.copy(initial_likelihood_vars['ilcr'])
            team_learning_rate = np.copy(initial_likelihood_vars['rlcr'])
            kc_id += 1
            next_kc_flag = False

        if 'individual' in demo_strategy:
            ind_knowledge_ascending = team_helpers.find_ascending_individual_knowledge(team_knowledge, min_BEC_constraints) # based on absolute individual knowledge
            if demo_strategy =='individual_knowledge_low':
                knowledge_id_new = ind_knowledge_ascending[0]
            elif demo_strategy == 'individual_knowledge_high':
                knowledge_id_new = ind_knowledge_ascending[len(ind_knowledge_ascending) - 1]

            if knowledge_id_new != knowledge_id:
                knowledge_id = knowledge_id_new
        
        # update demo particles
        print(colored('Updating demo particles with particles for: ', 'red'), knowledge_id)
        particles_demo = copy.deepcopy(particles_team[knowledge_id])

        # Obtain BEC summary for a new unit (skip if its the 1st unit)
        # print('Summary count: ', summary_count)
        if summary_count == 0:
            # print(params.data_loc['BEC'] )
            try:
                # # print('Trying to open existing BEC summary file...')
                with open('models/' + params.data_loc['BEC'] + '/BEC_summary_initial_2.pickle', 'rb') as f:
                    BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = pickle.load(f)
            except:
                print(colored('Starting summary generation for 1st unit..', 'blue'))
                BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = team_helpers.obtain_team_summary(params.data_loc['BEC'], min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, params.weights['val'], params.step_cost_flag, 
                                                                                                                    pool, params.BEC['n_human_models'], consistent_state_count, params.BEC['n_train_demos'], particles_demo, knowledge_id, variable_filter, nonzero_counter, BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, obj_func_proportion = obj_func_prop )
                # # print('Ended summary generation for 1st unit..')
                if len(BEC_summary) > 0:
                    with open('models/' + params.data_loc['BEC'] + '/BEC_summary_initial.pickle', 'wb') as f:
                        pickle.dump((BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo), f)
            
            
        else:
            print(colored('Starting summary generation for unit no.  ', 'blue') + str(loop_count + 1) )
            BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = team_helpers.obtain_team_summary(params.data_loc['BEC'], min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, params.weights['val'], params.step_cost_flag, 
                                                                                                                pool, params.BEC['n_human_models'], consistent_state_count, params.BEC['n_train_demos'], particles_demo, knowledge_id, variable_filter, nonzero_counter, BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, obj_func_proportion = obj_func_prop)
            # # print('Ended summary generation for unit no.  ' + str(loop_count + 1) )
        


        # check for any new summary
        print('Prev summary len: ', prev_summary_len)
        print('Current summary len: ', len(BEC_summary))

        if len(BEC_summary) > prev_summary_len:
            print('Showing demo....')
            unit_constraints, demo_ids, running_variable_filter_unit = team_helpers.show_demonstrations(BEC_summary[-1], particles_demo, params.mdp_class, params.weights['val'], loop_count, viz_flag = demo_viz_flag)

            # print('Knowledge component / Variable filter:', variable_filter)
            if (variable_filter != running_variable_filter_unit).any():
                RuntimeError('running variable filter does not match:', running_variable_filter_unit)

            print(colored('Unit constraints: ', 'red'))
            print(unit_constraints)

            # obtain the constraints conveyed by the unit's demonstrations
            min_KC_constraints = BEC_helpers.remove_redundant_constraints(unit_constraints, params.weights['val'], params.step_cost_flag)

            # For debugging. Visualize the expected particles transition
            team_knowledge_expected, particles_team = calc_expected_learning(team_knowledge_expected, kc_id, True, particles_team, min_BEC_constraints, min_KC_constraints, params, loop_count, viz_flag=knowledge_viz_flag)
            
            ## Conduct tests for the unit
            # obtain the diagnostic tests that will test the human's understanding of the unit's constraints
            # print('Getting diagnostic tests for unit ' + str(loop_count) + '...')
            # # print('visited_env_traj_idxs: ', visited_env_traj_idxs)
            preliminary_tests, visited_env_traj_idxs = team_helpers.obtain_diagnostic_tests(params.data_loc['BEC'], BEC_summary[-1], visited_env_traj_idxs, min_KC_constraints, min_subset_constraints_record, traj_record, traj_features_record, running_variable_filter_unit, mdp_features_record)

            
            # query the human's response to the diagnostic tests
            test_no = 1
            all_tests_constraints = []
            test_constraints_team = []
            kc_reset_flag = True  # flag to reset the KC constraints in the team knowledge
            for test in preliminary_tests:
                print('Test no ', test_no, ' out of ', len(preliminary_tests), 'for unit ', loop_count)
                response_category_team = []
                failed_BEC_constraints_tuple = []
                test_mdp = test[0]
                opt_traj = test[1]
                env_idx, traj_idx = test[2]
                test_constraints = copy.deepcopy(test[3])
                all_tests_constraints.append(test_constraints)
                test_history = [test] # to ensure that remedial demonstrations and tests are visually simple/similar and complex/different, respectively

                print('Test constraints: ', test_constraints)

                if experiment_type == 'simulated':
                    
                    # print('Simulating human response... Response no: ', resp_no)
                    # human_traj_team, human_history = sim_helpers.get_human_respons_old(env_idx, test_constraints, opt_traj, human_history, team_knowledge, team_size = params.team_size, response_distribution = response_distribution_list[resp_no])
                    
                    human_traj_team = []
                    response_type_team = []
                    # team_likelihood_correct_response_temp = [1, 0.1]
                    for i in range(params.team_size):
                        print('Simulating response for player ', i+1, 'for constraint', test_constraints[0], 'with likelihood of correct response ', team_likelihood_correct_response[i])

                        human_traj, response_type = sim_helpers.get_human_response(env_idx, test_constraints[0], opt_traj, team_likelihood_correct_response[i])
                        # print(colored('Got human response!', 'green'))
                        # human_traj, response_type = sim_helpers.get_human_response(env_idx, test_constraints[0], opt_traj, team_likelihood_correct_response_temp[i]) # for testing effects of specific response combinations
                        human_traj_team.append(human_traj)
                        response_type_team.append(response_type)

                        # update likelihood of correct response
                        if response_type == 'correct':
                            # print('Current team likelihood correct response: ', team_likelihood_correct_response)
                            team_likelihood_correct_response[i] = max(min(1, team_likelihood_correct_response[i] + team_learning_rate[i, 0]), sim_params['min_correct_likelihood']) # have a minimum likelihood so that people can still learn
                            # print('Updating LCR (correct) for player ', i+1, 'to ', team_likelihood_correct_response[i], 'using learning rate ', team_learning_rate[i, 0])
                            print("Sampled a correct response!")
                        else:
                            # print('Current team likelihood correct response: ', team_likelihood_correct_response)
                            team_likelihood_correct_response[i] = max(min(1, team_likelihood_correct_response[i] + team_learning_rate[i, 1]), sim_params['min_correct_likelihood']) # have a minimum likelihood so that people can still learn
                            # print('Updating LCR (incorrect) for player ', i+1, 'to ', team_likelihood_correct_response[i], 'using learning rate ', team_learning_rate[i, 1])
                            print("Sampled an incorrect response!")

                        # if len(test_constraints) > 1:
                            # print(colored('Multiple test constraints!', 'red'))
                            # print('Test constraints: ', test_constraints)
                    
                    resp_no += 1
                    # print('Opt traj len: ', len(opt_traj))

                        


                # Show the same test for each person and get test responses of each person in the team
                p = 1
                while p <= params.team_size:
                    member_id = 'p' + str(p)
                    print("Here is a diagnostic test for this unit for player ", p)

                    if experiment_type != 'simulated':
                        # print('Running manual test for response type: ', experiment_type)
                        human_traj, _ = test_mdp.visualize_interaction(keys_map=params.keys_map) # the latter is simply the gridworld locations of the agent
                    else:
                        human_traj = human_traj_team[p-1]
                        if test_viz_flag:
                            test_mdp.visualize_trajectory(human_traj)


                    human_feature_count = test_mdp.accumulate_reward_features(human_traj, discount=True)
                    opt_feature_count = test_mdp.accumulate_reward_features(opt_traj, discount=True)

                    if (human_feature_count == opt_feature_count).all():
                        # if len(test_constraints_team) < test_no:
                        if test_no == 1:
                            print('Test constraints: ', test_constraints)
                            test_constraints_team.append(copy.deepcopy(test_constraints))
                            print(colored('Correct response! ', 'green'))
                            print('Test constraints team: ', test_constraints_team, 'for test no: ', test_no)
                        else:
                            print('Test constraints: ', test_constraints)
                            test_constraints_team[p-1].extend(copy.deepcopy(test_constraints))
                            print('Test constraints team: ', test_constraints_team, 'for test no: ', test_no)
                            print(colored('Correct response!!! ', 'green'))

                        response_category_team.append('correct')

                        print(colored('Current team knowledge for member id: ' + str(member_id), 'blue'))
                        print(team_knowledge)
                        print('test_constraints: ', test_constraints)
                        print('kc_reset_flag: ', kc_reset_flag)
                        team_knowledge = team_helpers.update_team_knowledge(team_knowledge, kc_id, kc_reset_flag, test_constraints, params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = [member_id])
                        
                        print(colored('Updated team knowledge for member id: ' + str(member_id), 'blue'))
                        print(team_knowledge)

                        prev_pf_reset_count = particles_team[member_id].pf_reset_count
                        print('Test constraints for PF update: ', test_constraints)
                        particles_team[member_id].update(test_constraints) # update individual knowledge based on test respons
                        
                        if particles_team[member_id].pf_reset_count > prev_pf_reset_count:
                            # reset knowledge to mirror particle reset
                            # print('Resetting constraints.. Previous constraints in KC: ', team_knowledge[member_id][kc_id])
                            reset_index = [i for i in range(len(team_knowledge[member_id][kc_id])) if (team_knowledge[member_id][kc_id][i] == particles_team[member_id].reset_constraint).all()]
                            # print('Reset index: ', reset_index)
                            team_knowledge[member_id][kc_id] = team_knowledge[member_id][kc_id][reset_index[0]:]
                            # print('New constraints: ', team_knowledge[member_id][kc_id])
                        
                        print("You got the diagnostic test right")
                        if knowledge_viz_flag:
                            team_helpers.visualize_transition(test_constraints, particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Test ' + str(test_no) + ' of Unit ' + str(loop_count) + ' for player ' + member_id, plot_filename = 'ak_test_' + str(test_no) + '_p' + member_id)

                    else:
                        failed_BEC_constraint = opt_feature_count - human_feature_count
                        failed_BEC_constraints_tuple.append([member_id, failed_BEC_constraint])
                        unit_learning_goal_reached = False
                        if test_no == 1:
                            test_constraints_team.append(copy.deepcopy([-failed_BEC_constraint]))
                            print(colored('Incorrect response! ', 'red'))
                            print('Test constraints team: ', test_constraints_team, 'for test no: ', test_no)
                        else:
                            test_constraints_team[p-1].extend(copy.deepcopy([-failed_BEC_constraint]))
                            print(colored('Incorrect response!!! ', 'red'))
                            print('Test constraints team: ', test_constraints_team, 'for test no: ', test_no)
                        
                        response_category_team.append('incorrect')
                    
                        print(colored('Current team knowledge for member id: ' + str(member_id), 'blue'))
                        print(team_knowledge)
                        team_knowledge = team_helpers.update_team_knowledge(team_knowledge, kc_id, kc_reset_flag, [-failed_BEC_constraint], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = [member_id])
                        
                        print(colored('Updated team knowledge for member id: ' + str(member_id), 'blue'))
                        print(team_knowledge)

                        prev_pf_reset_count = particles_team[member_id].pf_reset_count
                        
                        particles_team[member_id].update([-failed_BEC_constraint])

                        if particles_team[member_id].pf_reset_count > prev_pf_reset_count:
                            # reset knowledge to mirror particle reset
                            # print('Resetting constraints.. Previous constraints in KC: ', team_knowledge[member_id][kc_id])
                            # print('Constraint that reset particle filter: ', particles_team[member_id].reset_constraint)
                            # print('particles_team[member_id].reset_constraint: ', particles_team[member_id].reset_constraint)
                            reset_index = [i for i in range(len(team_knowledge[member_id][kc_id])) if (team_knowledge[member_id][kc_id][i] == particles_team[member_id].reset_constraint).all()]
                            # print('Reset index: ', reset_index)
                            team_knowledge[member_id][kc_id] = team_knowledge[member_id][kc_id][reset_index[0]:]
                            # print('New constraints: ', team_knowledge[member_id])
                        
                        print("You got the diagnostic test wrong. Failed BEC constraint: {}".format(failed_BEC_constraint))

                        if knowledge_viz_flag:
                            team_helpers.visualize_transition([-failed_BEC_constraint], particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Test ' + str(test_no) + ' of Unit ' + str(loop_count) + ' for player ' + member_id, plot_filename = 'ak_test_' + str(test_no) + '_p' + member_id)
                        
                        # Correct trajectory
                        if demo_viz_flag:
                            # print('Here is the correct trajectory')
                            test_mdp.visualize_trajectory_comparison(opt_traj, human_traj)

                        
                        # # print('Opt traj: ', opt_traj)
                        # # print('Human traj: ', human_traj)

                    # update 
                    particles_team[member_id].knowledge_update(team_knowledge[member_id])
                    p += 1
                
                # Update test number
                test_no += 1
                kc_reset_flag = False

            ###  Completed simulating team response for all tests  ###


            ###########
            # Update team knowledge belief based on the set of tests
            if params.team_size > 1:
                ## Check if there are non-intersecting constraints and go to fall-back teaching behavior
                print('test_constraints_team for recent test: ', test_constraints_team)
                test_constraints_team_expanded = []
                for ind_test_constraints in test_constraints_team:
                    test_constraints_team_expanded.extend(ind_test_constraints)
                print('test_constraints_team_expanded: ', test_constraints_team_expanded)

                # # check for opposing constraints (since remove redundant constraints gives the perpendicular axes for opposing constraints)
                # opposing_constraints_flag, opposing_constraints_count, opposing_idx = team_helpers.check_opposing_constraints(test_constraints_team_expanded, opposing_constraints_count)
                # # # print('Opposing constraints normal loop? ', opposing_constraints_flag)
                # # Assign majority rules and update common knowledge and joint knowledge accordingly
                # if opposing_constraints_flag:
                #     test_constraints_team_expanded = team_helpers.majority_rules_opposing_team_constraints(opposing_idx, test_constraints_team_expanded, response_category_team)
                #     print('Majority rules for opposing contraints...')

                non_intersecting_constraints_flag, non_intersecting_constraints_count = team_helpers.check_for_non_intersecting_constraints(test_constraints_team_expanded, params.weights['val'], params.step_cost_flag, non_intersecting_constraints_count)
                # print('Non-intersecting constraints normal loop? ', non_intersecting_constraints_flag)
                # Assign majority rules and update common knowledge and joint knowledge accordingly
                if non_intersecting_constraints_flag:
                    test_constraints_team_expanded, intersecting_constraints = team_helpers.majority_rules_non_intersecting_team_constraints(test_constraints_team, params.weights['val'], params.step_cost_flag, test_flag = True)
                    print('Majority rules for non intersecting contraints...')
                    print('Team constraints team expanded after processing: ', test_constraints_team_expanded)
                
                # if there are no constraints after majority rules non intersecting constraints! just an additional check so that simulation does not stop
                if len(test_constraints_team_expanded) == 0:
                    teaching_complete_flag = True


                # double check again
                # opposing_constraints_flag, opposing_constraints_count, opposing_idx = team_helpers.check_opposing_constraints(test_constraints_team_expanded, opposing_constraints_count)
                non_intersecting_constraints_flag, non_intersecting_constraints_count = team_helpers.check_for_non_intersecting_constraints(test_constraints_team_expanded, params.weights['val'], params.step_cost_flag, non_intersecting_constraints_count)


                if not non_intersecting_constraints_flag:
                    
                    ## update common knowledge manually since it could have been updated by the majority rules function
                    test_cnst_team_updated = []
                    for cnst in test_constraints_team_expanded:
                        test_cnst_team_updated.append(cnst)

                    min_new_common_knowledge = BEC_helpers.remove_redundant_constraints(test_constraints_team_expanded, params.weights['val'], params.step_cost_flag)
                    
                    print('Updating common knowledge...')
                    print('min_new_common_knowledge: ', min_new_common_knowledge)

                    print(colored('Updated team knowledge for common knowledge: ', 'blue'))
                    print(team_knowledge)

                    # if kc_id == len(team_knowledge['common_knowledge']):
                    #     # # print('Appendin common knowledge...')
                    #     team_knowledge['common_knowledge'].append(min_new_common_knowledge)
                    # elif kc_reset_flag:
                    #     # # print('Reset common knowledge...')
                    #     team_knowledge['common_knowledge'][kc_id] = copy.deepcopy(min_new_common_knowledge)
                    # else:
                    #     # # print('Extend common knowledge..')
                    #     team_knowledge['common_knowledge'][kc_id].extend(min_new_common_knowledge)
                   
                    if kc_id == len(team_knowledge['common_knowledge']):
                        # # print('Appendin common knowledge...')
                        team_knowledge['common_knowledge'].append(min_new_common_knowledge)
                    else:
                        team_knowledge['common_knowledge'][kc_id] = copy.deepcopy(min_new_common_knowledge)


                    print(colored('Updated team knowledge for common knowledge: ', 'blue'))
                    print(team_knowledge)
                    

                    ## update joint knowledge
                    team_knowledge = team_helpers.update_team_knowledge(team_knowledge, kc_id, kc_reset_flag, [], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = ['joint_knowledge'])
                    

                    print(colored('Updated team knowledge for joint knowledge: ', 'blue'))
                    print(team_knowledge)

                    particles_team['common_knowledge'].knowledge_update(team_knowledge['common_knowledge'])
                    particles_team['joint_knowledge'].knowledge_update(team_knowledge['joint_knowledge'])

                    # Update particle filters of common and joint knowledge
                    prev_pf_reset_count = particles_team['common_knowledge'].pf_reset_count
                    # print('PF update of commong knowledge with constraints: ', min_new_common_knowledge)
                    particles_team['common_knowledge'].update(min_new_common_knowledge)
                    if particles_team['common_knowledge'].pf_reset_count > prev_pf_reset_count:
                        # reset knowledge to mirror particle reset
                        # print('Resetting constraints.. Previous constraints common knowledge: ', team_knowledge['common_knowledge'])
                        # reset_index = [i for i in range(len(team_knowledge['common_knowledge'])) if (team_knowledge['common_knowledge'][i] == particles_team['common_knowledge'].reset_constraint).all()]
                        reset_index = [i for i in range(len(team_knowledge['common_knowledge'][kc_id])) if (team_knowledge['common_knowledge'][kc_id][i] == particles_team['common_knowledge'].reset_constraint).all()]
                        # print('Reset index: ', reset_index, 'previous reset count: ', prev_pf_reset_count, 'current reset count: ', particles_team['common_knowledge'].pf_reset_count, 'reset constraint: ', particles_team['common_knowledge'].reset_constraint)
                        team_knowledge['common_knowledge'][kc_id] = team_knowledge['common_knowledge'][kc_id][reset_index[0]:]
                        # print('New constraints: ', team_knowledge['common_knowledge'])
                    
                    prev_pf_reset_count = particles_team['joint_knowledge'].pf_reset_count
                    # print('PF update of joint knowledge with constraints: ', test_constraints_team)
                    particles_team['joint_knowledge'].update_jk(test_constraints_team)
                    
                # if particles_team['joint_knowledge'].pf_reset_count > prev_pf_reset_count:
                    # reset knowledge to mirror particle reset
                    # print(colored('Joint knowledge reset!! But constraints not reset! ', 'red'))
                #     reset_index = team_knowledge['joint_knowledge'].index(particles_team['joint_knowledge'].reset_constraint.all())
                #     team_knowledge['joint_knowledge'] = team_knowledge['joint_knowledge'][reset_index:]
                #     # print('New constraints: ', team_knowledge['joint_knowledge'])


                # if knowledge_viz_flag:
                #     # print('test_constraints_team_expanded: ', test_constraints_team_expanded)
                #     team_helpers.visualize_transition(test_constraints_team_expanded, particles_team['common_knowledge'], params.mdp_class, params.weights['val'], text = 'Test ' + str(test_no) + ' of Unit ' + str(loop_count) + ' for common knowledge', plot_filename = 'ak_test_' + str(test_no) + '_ck')
                #     team_helpers.visualize_transition(test_constraints_team, particles_team['joint_knowledge'], params.mdp_class, params.weights['val'], text = 'Test ' + str(test_no) + ' of Unit ' + str(loop_count) + ' for joint knowledge',  knowledge_type = 'joint_knowledge', plot_filename = 'ak_test_' + str(test_no) + '_jk')
            
            # else:
                # raise RuntimeError('Still getting opposing or non intersecting constraints!')
                # print(colored('Still getting opposing or non intersecting constraints!', 'red'))
                # print('Opposing constraints flag: ', opposing_constraints_flag, 'Non intersecting constraints flag: ', non_intersecting_constraints_flag)
            
            else:
                team_knowledge['common_knowledge'] = copy.deepcopy(team_knowledge['p1'])
                team_knowledge = team_helpers.update_team_knowledge(team_knowledge, kc_id, kc_reset_flag, [], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = ['joint_knowledge'])
                    
                        

            ##########################################################################

            loop_count += 1

            # Check if unit knowledge is sufficient to move on to the next unit (unit learning goal reached)
            print('team_knowledge: ', team_knowledge)
            print('min_KC_constraints: ', min_KC_constraints)
            unit_learning_goal_reached = team_helpers.check_unit_learning_goal_reached(team_knowledge, min_KC_constraints, kc_id)
            print(colored('unit_learning_goal_reached: ', 'blue'), unit_learning_goal_reached)

            # Evaluate if next unit should be checked
            next_unit_flag = False
            # if loop_count - next_unit_loop_id > params.max_KC_loops or unit_learning_goal_reached:
            if unit_learning_goal_reached:
                next_unit_flag = True
                next_unit_loop_id = loop_count
                print(colored('prior_min_BEC_constraints_running: ', 'green'), prior_min_BEC_constraints_running)
                prior_min_BEC_constraints_running = copy.deepcopy(min_BEC_constraints_running)  # update last_min_BEC_cnst_id since the constraints have been learned
                print(colored('updated prior_min_BEC_constraints_running: ', 'green'), prior_min_BEC_constraints_running)
            else:
                # demo generation loop continues for the current KC. reset min_BEC_constraints_running
                print(colored('min_BEC_constraints_running including unlearned ones: ', 'green'), min_BEC_constraints_running)
                min_BEC_constraints_running = copy.deepcopy(prior_min_BEC_constraints_running)
                print(colored('updated min_BEC_constraints_running: ', 'green'), min_BEC_constraints_running)


            

            # update variables
            loop_vars = copy.deepcopy(demo_vars_template)
            loop_vars['run_no'] = run_no
            loop_vars['demo_strategy'] = demo_strategy
            loop_vars['knowledge_id'] = knowledge_id
            loop_vars['variable_filter'] = variable_filter
            loop_vars['knowledge_comp_id'] = kc_id
            loop_vars['loop_count'] = loop_count
            loop_vars['summary_count'] = summary_count
            loop_vars['min_BEC_constraints'] = copy.deepcopy(min_BEC_constraints)
            loop_vars['unit_constraints'] = copy.deepcopy(unit_constraints)
            loop_vars['all_unit_constraints'] = copy.deepcopy(all_unit_constraints)
            loop_vars['demo_ids'] = copy.deepcopy(demo_ids)
            loop_vars['team_knowledge_expected'] = copy.deepcopy(team_knowledge_expected)
            loop_vars['unit_knowledge_level_expected'] = team_helpers.calc_knowledge_level(team_knowledge_expected, min_KC_constraints, kc_id_list = [kc_id], plot_flag = False, fig_title = 'Expected Unit knowledge level for var filter: ' + str(variable_filter) )
            loop_vars['BEC_knowledge_level_expected'] = team_helpers.calc_knowledge_level(team_knowledge_expected, min_BEC_constraints, plot_flag = False, fig_title = 'Expected BEC knowledge level after var filter: ' + str(variable_filter))
            # loop_vars['response_type'] = response_type
            # loop_vars['response_distribution'] = response_distribution_list[resp_no-1]
            loop_vars['test_constraints'] = copy.deepcopy(all_tests_constraints)
            loop_vars['test_constraints_team'] = copy.deepcopy(test_constraints_team)
            loop_vars['opposing_constraints_count'] = opposing_constraints_count  
            loop_vars['final_remedial_constraints'] = copy.deepcopy(remedial_constraints_team_expanded)
            loop_vars['N_remedial_tests'] = N_remedial_tests
            loop_vars['team_knowledge'] = copy.deepcopy(team_knowledge)
            loop_vars['particles_team'] = copy.deepcopy(particles_team)
            loop_vars['unit_knowledge_level'] = team_helpers.calc_knowledge_level(team_knowledge, min_KC_constraints, particles_team = particles_team, kc_id_list = [kc_id], plot_flag = False, fig_title = 'Actual Unit knowledge level for var filter: ' + str(variable_filter))
            loop_vars['BEC_knowledge_level'] = team_helpers.calc_knowledge_level(team_knowledge, min_BEC_constraints, particles_team = particles_team, plot_flag = False, fig_title = 'Actual BEC knowledge level expected after var filter: ' + str(variable_filter))
            loop_vars['unit_knowledge_area'] = BEC_helpers.calc_solid_angles([min_KC_constraints])
            # loop_vars['all_unit_constraints_area'] = BEC_helpers.calc_solid_angles([all_unit_constraints])
            loop_vars['all_unit_constraints_area'] = []
            loop_vars['BEC_knowledge_area'] = BEC_helpers.calc_solid_angles([min_BEC_constraints])
            loop_vars['initial_likelihood_vars'] = copy.deepcopy(initial_likelihood_vars)
            loop_vars['likelihood_correct_response'] = copy.deepcopy(team_likelihood_correct_response)


            print(colored('Unit knowledge level: ' + str(loop_vars['unit_knowledge_level']), 'red'))
            print(colored('BEC knowledge level: ' + str(loop_vars['BEC_knowledge_level']), 'red'))
            



            for i in range(len(team_knowledge)):
                if i < params.team_size:
                    loop_vars['pf_reset_count'][i] = particles_team['p' + str(i+1)].pf_reset_count
                elif i == len(team_knowledge) - 2:
                    loop_vars['pf_reset_count'][i] = particles_team['common_knowledge'].pf_reset_count
                elif i == len(team_knowledge) - 1:
                    loop_vars['pf_reset_count'][i] = particles_team['joint_knowledge'].pf_reset_count
                # else:
                    # print('Wrong index for pf_reset_count!')
                 
            vars_to_save = vars_to_save.append(loop_vars, ignore_index=True)
            # # print('unit_knowledge_level_expected: ', loop_vars['unit_knowledge_level_expected'])
            # # print('unit_knowledge_level: ', loop_vars['unit_knowledge_level'])
            # # print('BEC_knowledge_level_expected: ', loop_vars['BEC_knowledge_level_expected'])
            # # print('BEC_knowledge_level: ', loop_vars['BEC_knowledge_level'])
            
            # save vars so far (within one session)
            vars_to_save.to_csv('models/' + params.data_loc['BEC'] + '/' + vars_filename + '_' + str(run_no) + '.csv', index=False)

            with open('models/augmented_taxi2/' + vars_filename + '_' + str(run_no) + '.pickle', 'wb') as f:
                pickle.dump(vars_to_save, f)


            # debug - plot loop variables
            # print('Visualizing team knowledge constraints for this teaching loop...')
            if knowledge_viz_flag:
                # team_helpers.visualize_team_knowledge_constraints(min_BEC_constraints, team_knowledge, loop_vars['unit_knowledge_level'], loop_vars['BEC_knowledge_level'], params.mdp_class, fig=None, weights=None, text=None)
                fig = team_helpers.visualize_team_knowledge_constraints(team_knowledge, params.weights['val'], params.step_cost_flag, particles_team = particles_team, min_unit_constraints = min_BEC_constraints, plot_filename = 'team_knowledge_constraints', fig_title = 'team_knowledge_constraints')
        


            # Update variable filter
            if next_unit_flag:
                variable_filter, nonzero_counter, teaching_complete_flag = team_helpers.check_and_update_variable_filter(variable_filter = variable_filter, nonzero_counter = nonzero_counter)
                
                # if knowledge was added to current KC then add KC id
                if kc_id == len(team_knowledge['p1'])-1:
                    next_kc_flag = True
                    obj_func_prop = 1.0
                    print(colored('kc_id: ' + str(kc_id), 'red') )
                    print('len(team_knowledge[p1]): ', len(team_knowledge['p1']), 'next_kc_flag: ', next_kc_flag)
                
                prev_summary_len = len(BEC_summary)

                print(colored('Moving to next unit. Updated variable filter: ', 'blue'), variable_filter, '. Teaching_complete_flag: ', teaching_complete_flag)

                # print(colored('Saving session data for run {}.'.format(run_no), 'blue'))
                # save vars so far (end of session)
                vars_to_save.to_csv('models/' + params.data_loc['BEC'] + '/' + vars_filename + '_' + str(run_no) + '.csv', index=False)
                # print('Saved sim data len: ', vars_to_save.shape[0])
                with open('models/augmented_taxi2/' + vars_filename + '_' + str(run_no) + '.pickle', 'wb') as f:
                    pickle.dump(vars_to_save, f)

            else:
            
                # update variables for next summary but same KC
                prev_summary_len = len(BEC_summary)

                # # Update demo particles for next iteration based on actual team knowledge and the demo strategy - now being done at the beginning of the loop
                # print(colored('Updating demo particles (EOL) with particles for: ', 'red'), knowledge_id)
                # particles_demo = copy.deepcopy(particles_team[knowledge_id])

                # Update expected knowledge with actual knowledge for next iteration
                # particles_team_expected = copy.deepcopy(particles_team)
                team_knowledge_expected = copy.deepcopy(team_knowledge)

                if loop_count - next_unit_loop_id > params.loop_threshold_demo_simplification:
                    obj_func_prop = 0.5     # reduce informativeness from demos if they have not learned
                    print(colored('updated obj_func_prop: ' + str(obj_func_prop), 'red') )

                # check if number of max interactions sets have been reached
                if loop_count > params.max_loops:
                    print(colored('Maximum teaching interactions reached! ', 'red'))
                    teaching_complete_flag = True

        else:
            # Update Variable filter and move to next unit, if applicable, if no demos are available for this unit.
            # print(colored('No new summaries for this unit...!!', 'red'))
            unit_learning_goal_reached = True

            # maybe double check knowledge metric
            # unit_learning_goal_reached2 = team_helpers.check_unit_learning_goal_reached(team_knowledge, min_unit_constraints)
            # # print('Measured unit learning goal staus: ', unit_learning_goal_reached2)
            
            # Update variable filter
            if unit_learning_goal_reached:
                variable_filter, nonzero_counter, teaching_complete_flag = team_helpers.check_and_update_variable_filter(variable_filter = variable_filter, nonzero_counter = nonzero_counter)
                
                # if knowledge was added to current KC then add KC id
                if kc_id == len(team_knowledge['p1'])-1:
                    next_kc_flag = True
                    obj_func_prop = 1.0
                    print(colored('updated obj_func_prop for new KC: ' + str(obj_func_prop), 'red') )

            if teaching_complete_flag:
                print(colored('Teaching completed for run: ', 'blue'), run_no,'. Saving session data...')
                
                vars_to_save.to_csv('models/' + params.data_loc['BEC'] + '/' + vars_filename + '_' + str(run_no) + '.csv', index=False)
                # print('Saved sim data len: ', vars_to_save.shape[0])
                with open('models/augmented_taxi2/' + vars_filename + '_' + str(run_no) + '.pickle', 'wb') as f:
                    pickle.dump(vars_to_save, f)

    # return vars_to_save






if __name__ == "__main__":
    pool = Pool(min(params.n_cpu, 60))
    os.makedirs('models/' + params.data_loc['base'], exist_ok=True)
    os.makedirs('models/' + params.data_loc['BEC'], exist_ok=True)


    ## run_reward_teaching
    # run_reward_teaching(params, pool, demo_strategy = 'common_knowledge', experiment_type = 'simulated', response_distribution_list = ['mixed', 'correct', 'mixed', 'incorrect', 'correct', 'correct', 'correct', 'correct'], run_no = 1, viz_flag=True, vars_filename = 'workshop_data')
    # vars_to_save = run_reward_teaching(params, pool)
    
    # viz_flag = [demo_viz, test_viz, pf_knowledge_viz]
    sim_params = {'min_correct_likelihood': 0.5}
    run_reward_teaching(params, pool, sim_params, demo_strategy = 'individual_knowledge_low', experiment_type = 'simulated', run_no = 1, viz_flag=[True, True, True], vars_filename = 'new_hlm_4')

    
    pool.close()
    pool.join()



# save files
    # if len(BEC_summary) > 0:
    #     with open('models/' + data_loc + '/teams_BEC_summary.pickle', 'wb') as f:
    #         pickle.dump((BEC_summary, visited_env_traj_idxs, particles_team), f)










