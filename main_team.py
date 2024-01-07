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

plt.rcParams['figure.figsize'] = [15, 10]

import random
import pandas as pd
from numpy.linalg import norm
from analyze_sim_data import run_analysis_script


############################################

def initialize_loop_vars():

    demo_vars_template = {'run_no': 1,
                        'demo_strategy': None,
                        'sampling_condition': None,
                        'team_composition': None,
                        'initial_team_learning_factor': np.zeros(params.team_size, dtype=int),

                        'knowledge_id': None,
                        'variable_filter': None,
                        'knowledge_comp_id': None,
                        'loop_count': None,  
                        'summary_count': None,
                        'min_BEC_constraints': None,
                        'unit_constraints': None,
                        'min_KC_constraints': None,
                        'all_unit_constraints': None,  
                        'demo_ids': None,
                        
                        'team_learning_factor': np.zeros(params.team_size, dtype=int),
                        'test_constraints': None,
                        'test_constraints_team': None,
                        'team_knowledge_expected': None,
                        'particles_team_teacher_expected': None,
                        'unit_knowledge_level_expected': None,
                        'BEC_knowledge_level_expected': None,
                        'opposing_constraints_count': 0,
                        # 'final_remedial_constraints': None,
                        # 'N_remedial_tests': None,
                        'team_knowledge': None,
                        'particles_team_teacher': None,
                        'particles_team_learner': None,
                        'unit_knowledge_level': None,
                        'BEC_knowledge_level': None,
                        'unit_knowledge_area': None,
                        # 'all_unit_constraints_area': None,
                        'BEC_knowledge_area': None,
                        'pf_reset_count': np.zeros(params.team_size+2, dtype=int),
                        
                        'team_response_models': None,
                        # 'particles_prob_teacher_after_demo': None,
                        # 'particles_prob_learner_after_demo': None,
                        'particles_prob_teacher_before_test': None,
                        'particles_prob_learner_before_test': None,
                        # 'particles_prob_teacher_after_test': None,
                        # 'particles_prob_learner_after_test': None,
                        # 'cluster_prob_learner_demo': None,
                        # 'particles_prob_teacher_test': None,
                        # 'particles_prob_learner_test': None,

                        'sim_status': None
                        }
    
    return copy.deepcopy(demo_vars_template)



# def run_reward_teaching(params, pool, demo_strategy = 'common', experiment_type = 'simulated', response_distribution_list = ['correct']*10, run_no = 1, vars_to_save = None):
# def run_reward_teaching(params, pool, sim_params, demo_strategy = 'common_knowledge', experiment_type = 'simulated',  team_likelihood_correct_response = 0.5*np.ones(params.team_size) ,  team_learning_rate = np.hstack((0.2*np.ones([params.team_size, 1]), -0.1*np.ones([params.team_size, 1]))), obj_func_prop = 1.0, run_no = 1, viz_flag=[False, False, False], vars_filename = 'var_to_save'):



def get_optimal_policies(pool):

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
    
    print(colored('min_BEC_constraints for this run: ', 'red'), min_BEC_constraints)

    return policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count, min_BEC_constraints, BEC_lengths_record




def run_reward_teaching(params, pool, sim_params, demo_strategy = 'common_knowledge', experiment_type = 'simulated',  initial_team_learning_factor = 0.65*np.ones([params.team_size, 1]), 
                        team_learning_rate = np.hstack((0.05*np.ones([params.team_size, 1]), 0*np.ones([params.team_size, 1]))), obj_func_prop = 1.0, run_no = 1, viz_flag=[False, False, False], \
                        vars_filename = 'var_to_save', response_sampling_condition = 'particles', team_composition = None, learner_update_type = 'no_noise'):

    ####### Initialize variables ########################

    ## Initialize run variables
    team_learning_factor = copy.deepcopy(initial_team_learning_factor)
    max_learning_factor = params.max_learning_factor
    demo_viz_flag, test_viz_flag, knowledge_viz_flag = viz_flag
    
    BEC_summary, visited_env_traj_idxs, min_BEC_constraints_running, prior_min_BEC_constraints_running = [], [], copy.deepcopy(params.prior), copy.deepcopy(params.prior)
    summary_count, prev_summary_len = 0, 0
    all_unit_constraints = []
    
    
    ## Initialize teaching variables
    # particle filter models for individual and team knowledge of teachers and individual knowledge of learners; initialize expected and actual team knowledge
    team_prior, particles_team_teacher = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_prior = params.team_prior, vars_filename=vars_filename)
    team_knowledge = copy.deepcopy(team_prior) # team_prior calculated from team_helpers.sample_team_pf also has the aggregated knowledge from individual priors
    particles_team_learner = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_learning_factor = team_learning_factor, team_prior = params.team_prior, pf_flag='learner', vars_filename=vars_filename)
    team_knowledge_expected = copy.deepcopy(team_knowledge)
    
    # initialize/load variables to save
    loop_vars = initialize_loop_vars()
    try:
        with open('models/augmented_taxi2/' + vars_filename + '_' + str(run_no) + '.pickle', 'rb') as f:
            vars_to_save = pickle.load(f)
    except:
        vars_to_save = pd.DataFrame(columns=loop_vars.keys())

    ## Initialize teaching loop variables
    loop_count, next_unit_loop_id, resp_no = 0, 0, 0
    unit_learning_goal_reached, next_kc_flag = False, False   # learning flags
    kc_id = 1 # knowledge component id / variable filter id
    
    
    ##### initialize debugging variables
    sampled_points_history, team_learning_factor_history, response_history, member, constraint_history, test_history, constraint_flag_history = [], [], [], [], [], [], []
    update_id_history, update_sequence_history, skip_model_history, cluster_id_history, point_probability, prob_initial_history, prob_reweight_history = [], [], [], [], [], [], []
    prob_resample_history, resample_flag_history, particles_learner_prob_demo_history, particles_learner_prob_test_history, resample_noise_history = [], [], [], [], [] 

    ########################

    ### Get/calculate optimal policies
    policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count, min_BEC_constraints, BEC_lengths_record = get_optimal_policies(pool)
    
    # unit (knowledge component/concept) initialization
    variable_filter, nonzero_counter, teaching_complete_flag = team_helpers.check_and_update_variable_filter(min_subset_constraints_record, initialize_filter_flag=True)
    # initialize human models for demo generation
    knowledge_id, particles_demo = team_helpers.particles_for_demo_strategy(demo_strategy, team_knowledge, particles_team_teacher, params.team_size, params.weights['val'], params.step_cost_flag, params.BEC['n_particles'], min_BEC_constraints)

    
    ########################

    ### Teaching-testing loop
    
    while not teaching_complete_flag:
        
        ####### Reset variables for each loop #######
        # reset variables for each interaction round/loop
        opposing_constraints_count, non_intersecting_constraints_count, N_remedial_tests = 0, 0, 0
        
        # reset variables for new KC
        if next_kc_flag:
            kc_id += 1
            next_kc_flag = False
            team_learning_factor = copy.deepcopy(initial_team_learning_factor) 

        # reset which individual knowledge to generate demonstrations for this loop in case of "individual" demo strategy
        if 'individual' in demo_strategy:
            ind_knowledge_ascending = team_helpers.find_ascending_individual_knowledge(team_knowledge, min_BEC_constraints) # based on absolute individual knowledge
            if demo_strategy =='individual_knowledge_low':
                knowledge_id_new = ind_knowledge_ascending[0]
            elif demo_strategy == 'individual_knowledge_high':
                knowledge_id_new = ind_knowledge_ascending[len(ind_knowledge_ascending) - 1]

            if knowledge_id_new != knowledge_id:
                knowledge_id = knowledge_id_new

        # update demo PF model with the teacher's estimate of appropriate team/individual PF model based on the demo strategy
        # print(colored('Updating demo particles with particles for: ', 'red'), knowledge_id)
        particles_demo = copy.deepcopy(particles_team_teacher[knowledge_id])

        ################################################
        
        ### Obtain BEC summary/demos for a new KC #######
        if summary_count == 0:
            try:
                with open('models/' + params.data_loc['BEC'] + '/BEC_summary_initial_2.pickle', 'rb') as f:
                    BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = pickle.load(f)
            except:
                print(colored('Starting summary generation for 1st unit..', 'blue'))
                BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = team_helpers.obtain_team_summary(params.data_loc['BEC'], min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, params.weights['val'], params.step_cost_flag, 
                                                                                                                    pool, params.BEC['n_human_models'], consistent_state_count, params.BEC['n_train_demos'], particles_demo, knowledge_id, variable_filter, nonzero_counter, BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, obj_func_proportion = obj_func_prop, vars_filename =vars_filename )
                if len(BEC_summary) > 0:
                    with open('models/' + params.data_loc['BEC'] + '/BEC_summary_initial.pickle', 'wb') as f:
                        pickle.dump((BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo), f)
        else:
            print(colored('Starting summary generation for unit no.  ', 'blue') + str(loop_count + 1) )
            BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = team_helpers.obtain_team_summary(params.data_loc['BEC'], min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, params.weights['val'], params.step_cost_flag, 
                                                                                                                pool, params.BEC['n_human_models'], consistent_state_count, params.BEC['n_train_demos'], particles_demo, knowledge_id, variable_filter, nonzero_counter, BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, obj_func_proportion = obj_func_prop, vars_filename=vars_filename)        

            ## if there are no new demos; reuse the old ones
            if len(BEC_summary) == prev_summary_len:
                # check if it's for the same KC as previous interaction
                if (variable_filter != running_variable_filter_unit).any():
                    print(colored('No new demos generated. Reusing last set of demos..', 'red'))
                    BEC_summary.append(BEC_summary[-1])



        ################################################
        ### Go into showing demos and tests if a new summary has been generated
        
        print('Prev summary len: ', prev_summary_len, 'Current summary len: ', len(BEC_summary))

        if len(BEC_summary) > prev_summary_len:

            
            # print('Showing demo....')
            unit_constraints, demo_ids, running_variable_filter_unit = team_helpers.show_demonstrations(BEC_summary[-1], particles_demo, params.mdp_class, params.weights['val'], loop_count, viz_flag = demo_viz_flag)
            print(colored('Unit constraints for this set of demonstrations: ' + str(unit_constraints), 'red'))

            # check if variable filter matches the running variable filter
            if (variable_filter != running_variable_filter_unit).any():
                # print('Knowledge component / Variable filter:', variable_filter)
                RuntimeError('Running variable filter does not match:', running_variable_filter_unit)

            
            # obtain the constraints conveyed by the unit's demonstrations
            min_KC_constraints = BEC_helpers.remove_redundant_constraints(unit_constraints, params.weights['val'], params.step_cost_flag)

            # calculate expected learning
            team_knowledge_expected, particles_team_teacher, pf_update_args = team_helpers.calc_expected_learning(team_knowledge_expected, kc_id, particles_team_teacher, min_BEC_constraints, min_KC_constraints, params, loop_count, kc_reset_flag=True, viz_flag=knowledge_viz_flag, vars_filename=vars_filename)
            
            # Simulate learning by team members
            particles_team_learner = team_helpers.simulate_team_learning(kc_id, particles_team_learner, min_KC_constraints, params, loop_count, viz_flag=knowledge_viz_flag, learner_update_type = learner_update_type, vars_filename=vars_filename)

            ############ debug - update probabilities after seeing demos (this is the distribution from which a response will be sampled for the learner)
            particles_teacher_prob_after_demo = {}
            particles_learner_prob_after_demo = {}
            for i in range(params.team_size):
                member_id = 'p' + str(i+1)
                particles_teacher_prob_after_demo[member_id] = particles_team_teacher[member_id].particles_prob_correct
                particles_learner_prob_after_demo[member_id] = particles_team_learner[member_id].particles_prob_correct
                # print(colored('particles_teacher_prob_demo for player ' + member_id + ': ' + str(particles_teacher_prob_demo[member_id]) + '. particles_learner_prob_demo for player ' + member_id + ': ' + str(particles_learner_prob_demo[member_id]), 'green'))
            ############################################################

            ### Generate tests for the unit and sample responses

            # obtain the diagnostic tests that will test the human's understanding of the unit's constraints
            # random.shuffle(min_KC_constraints) # shuffle the order of the constraints so that it's not always the same; use it for the actual user study
            preliminary_tests, visited_env_traj_idxs = team_helpers.obtain_diagnostic_tests(params.data_loc['BEC'], BEC_summary[-1], visited_env_traj_idxs, min_KC_constraints, min_subset_constraints_record, traj_record, traj_features_record, running_variable_filter_unit, mdp_features_record)

            ### query the human's response to the diagnostic tests
            test_no = 1
            all_tests_constraints, test_constraints_team = [], []
            prob_teacher_before_testing, prob_learner_before_testing, prob_teacher_after_testing, prob_learner_after_testing = [], [], [], []
            kc_reset_flag = True  # flag to reset the KC constraints in the team knowledge
            human_opt_trajs_all_tests_team = {}
            response_type_all_tests_team = {}
            human_model_weight_team = {}

            ###### get human responses to the diagnostic tests
            # Method 1: Sample human models for all tests
            if params.response_generation_type == 'All_tests':

                for i in range(params.team_size):
                    member_id = 'p' + str(i+1)

                    prob_initial, prob_reweight, prob_resample, resample_flag, noise_measures = pf_update_args[i]

                    args = loop_count, member_id, [], sampled_points_history, response_history, member, constraint_history, constraint_flag_history, update_id_history, skip_model_history, cluster_id_history, point_probability, \
                            team_learning_factor_history, particles_learner_prob_after_demo[member_id], [], particles_learner_prob_demo_history, particles_learner_prob_test_history, prob_initial, prob_reweight, prob_resample, resample_flag, prob_initial_history, \
                            prob_reweight_history, prob_resample_history, resample_flag_history, [], update_sequence_history, noise_measures, resample_noise_history

                    human_model_weight_all_tests, human_opt_trajs_all_tests, response_type_all_tests, sampled_points_history, response_history, member, constraint_history, constraint_flag_history, update_id_history, skip_model_history, cluster_id_history, point_probability, team_learning_factor_history, particles_learner_prob_demo_history, \
                           particles_learner_prob_test_history, prob_initial, prob_reweight, prob_resample, resample_flag, prob_initial_history, prob_reweight_history, prob_resample_history, resample_flag_history, update_sequence_history, resample_noise_history = sim_helpers.get_human_response_all_tests(particles_team_learner[member_id], preliminary_tests, team_learning_factor[i], args)
                    
                    human_opt_trajs_all_tests_team[member_id] = human_opt_trajs_all_tests
                    response_type_all_tests_team[member_id] = response_type_all_tests
                    human_model_weight_team[member_id] = human_model_weight_all_tests
            #####################

            ##### simulate response for each test
            for test in preliminary_tests:
        
                response_category_team = []
                failed_BEC_constraints_tuple = []
                test_mdp = test[0]
                opt_traj = test[1]
                test_mdp.set_init_state(opt_traj[0][0])
                env_idx, traj_idx = test[2]
                test_constraints = copy.deepcopy(test[3])
                all_tests_constraints.append(test_constraints)
                # test_history = [test] # to ensure that remedial demonstrations and tests are visually simple/similar and complex/different, respectively

                print('Interaction No: ', loop_count+1, ' Test no ', test_no, ' out of ', len(preliminary_tests), 'for KC ', kc_id, '. Test constraints: ', test_constraints)
                
                ### save files for easy debugging
                # with open('models/augmented_taxi2/test_mdp.pickle', 'wb') as f:
                #     pickle.dump(test_mdp, f)
                # with open('models/augmented_taxi2/test_constraints.pickle', 'wb') as f:
                #     pickle.dump(test_constraints, f)
                # with open('models/augmented_taxi2/opt_traj.pickle', 'wb') as f:
                #     pickle.dump(opt_traj, f)
                # with open('models/augmented_taxi2/env_idx.pickle', 'wb') as f:
                #     pickle.dump(env_idx, f)
                ################

                if experiment_type == 'simulated':
                
                    for i in range(params.team_size):
                        print('Simulating response for player ', i+1, 'for constraint', test_constraints[0])

                        member_id = 'p' + str(i+1)
                        
                        ## for debugging
                        prob_initial, prob_reweight, prob_resample, resample_flag, noise_measures = pf_update_args[i]

                        args = loop_count, member_id, test_constraints, sampled_points_history, response_history, member, constraint_history, constraint_flag_history, update_id_history, skip_model_history, cluster_id_history, point_probability, \
                            team_learning_factor_history, particles_learner_prob_after_demo[member_id], [], particles_learner_prob_demo_history, particles_learner_prob_test_history, prob_initial, prob_reweight, prob_resample, resample_flag, prob_initial_history, \
                            prob_reweight_history, prob_resample_history, resample_flag_history, [], update_sequence_history, noise_measures, resample_noise_history

                        # Method 2: Sample human models for individual tests
                        if params.response_generation_type == 'Individual_tests':

                            # human_traj, response_type = sim_helpers.get_human_response(response_sampling_condition, env_idx, particles_team_learner[member_id], opt_traj, test_constraints, team_learning_factor[i])

                            ## for debugging
                            human_model_weight, human_traj, response_type, sampled_points_history, response_history, member, constraint_history, constraint_flag_history, update_id_history, skip_model_history, cluster_id_history, point_probability, team_learning_factor_history, particles_learner_prob_demo_history, \
                            particles_learner_prob_test_history, prob_initial, prob_reweight, prob_resample, resample_flag, prob_initial_history, prob_reweight_history, prob_resample_history, resample_flag_history, update_sequence_history, resample_noise_history = sim_helpers.get_human_response_each_test(response_sampling_condition, env_idx, particles_team_learner[member_id], opt_traj, test_constraints, team_learning_factor[i], args)

                            if test_no == 1:
                                human_opt_trajs_all_tests_team[member_id] = []
                                response_type_all_tests_team[member_id] = []
                                human_model_weight_team[member_id] = []

                            human_opt_trajs_all_tests_team[member_id].append(human_traj)
                            response_type_all_tests_team[member_id].append(response_type)
                            human_model_weight_team[member_id].append(human_model_weight)
                            
                        
                        elif params.response_generation_type == 'All_tests':

                            human_traj = human_opt_trajs_all_tests_team[member_id][test_no-1]
                            response_type = response_type_all_tests_team[member_id][test_no-1]
                            human_model_weight = human_model_weight_team[member_id][test_no-1]

                        # sampled responses
                        print('human model weight: ', human_model_weight, 'human_traj len: ', len(human_traj), 'response_type: ', response_type)



                    resp_no += 1
                        
                ################################################

                ### plot sampled human models
                if knowledge_viz_flag:
                    all_tests_constraints_expanded = [item for tc in all_tests_constraints for item in tc]
                    # print('all_tests_constraints: ', all_tests_constraints, 'all_tests_constraints_expanded: ', all_tests_constraints_expanded, 'human_model_weight_team: ', human_model_weight_team)
                    plot_title = 'Interaction No.' + str(loop_count +1) + '. Human models for test ' + str(test_no) + ' of KC ' + str(kc_id)
                    sim_helpers.plot_sampled_models(particles_team_learner, all_tests_constraints_expanded, human_model_weight_team, test_no, plot_title = plot_title, vars_filename = vars_filename)
                    # print('human_opt_trajs_all_tests: ', human_opt_trajs_all_tests)

                ################################

                ## show demos and simulate responses for human team members
                particles_prob_teacher_before_test, particles_prob_learner_before_test, particles_prob_teacher_after_test, particles_prob_learner_after_test = {}, {}, {}, {}
                
                p = 1
                # Show the same test for each person and get test responses of each person in the team
                while p <= params.team_size:
                    member_id = 'p' + str(p)
                    # print("Here is a diagnostic test for this unit for player ", p)

                    # update probability of particles before test
                    particles_team_teacher[member_id].calc_particles_probability(test_constraints)
                    particles_team_learner[member_id].calc_particles_probability(test_constraints)
                    particles_prob_teacher_before_test[member_id] = particles_team_teacher[member_id].particles_prob_correct
                    particles_prob_learner_before_test[member_id] = particles_team_learner[member_id].particles_prob_correct

                    # get response trajectory
                    if experiment_type != 'simulated':
                        # print('Test for response type: ', experiment_type)
                        human_traj, _ = test_mdp.visualize_interaction(keys_map=params.keys_map) # the latter is simply the gridworld locations of the agent
                    else:
                        # print('human_opt_trajs_all_tests_team: ', human_opt_trajs_all_tests_team)
                        human_traj = human_opt_trajs_all_tests_team[member_id][test_no-1]
                        if test_viz_flag:
                            test_mdp.visualize_trajectory(human_traj)
                    
                    print('human_traj len: ', len(human_traj))
                    human_feature_count = test_mdp.accumulate_reward_features(human_traj, discount=True)
                    opt_feature_count = test_mdp.accumulate_reward_features(opt_traj, discount=True)

                    # for correct response
                    if (human_feature_count == opt_feature_count).all():
                        if test_no == 1:
                            test_constraints_team.append(copy.deepcopy(test_constraints))
                        else:
                            test_constraints_team[p-1].extend(copy.deepcopy(test_constraints))
                        print('Test constraints: ', test_constraints, '. Test constraints team: ', test_constraints_team, 'for test no: ', test_no)
                        
                        # update team knowledge
                        # print(colored('Current team knowledge for member id ' + str(member_id) + ': ' + str(team_knowledge), 'blue'))
                        # print('test_constraints: ', test_constraints, 'kc_reset_flag: ', kc_reset_flag)
                        team_knowledge = team_helpers.update_team_knowledge(team_knowledge, kc_id, kc_reset_flag, test_constraints, params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = [member_id])
                        # print(colored('Updated team knowledge for member id  ' + str(member_id) + ': ' + str(team_knowledge), 'blue'))

                        # update teacher model
                        plot_title = 'Interaction No.' + str(loop_count +1) + '. Teacher belief for player ' + member_id + ' after test ' + str(test_no) + ' of Unit ' + str(kc_id)
                        particles_team_teacher[member_id].update(test_constraints, plot_title = plot_title, viz_flag = knowledge_viz_flag, vars_filename = vars_filename) # update individual knowledge based on test response

                        # update learner model
                        plot_title = 'Interaction No.' + str(loop_count +1) + '. Learner belief for player ' + member_id + ' after test ' + str(test_no) + ' of Unit ' + str(kc_id)
                        particles_team_learner[member_id].update(test_constraints, learning_factor=team_learning_factor[p-1], plot_title = plot_title, model_type = learner_update_type, viz_flag = knowledge_viz_flag, vars_filename = vars_filename) # update individual knowledge based on test response
                        
                        # update learner factor - after demonstrating learning or its lack (only if updating after each individual test and sampling a new human model after each individual test)
                        if params.response_generation_type == 'Individual_tests':
                            team_learning_factor[p-1] = min(team_learning_factor[p-1] + team_learning_rate[p-1, 0], max_learning_factor) # update learning parameter
                        

                        if knowledge_viz_flag:
                            plot_title = 'Interaction No.' + str(loop_count+1) +'Simulated knowledge change for player ' + member_id + ' after test ' + str(test_no) + ' of KC' + str(kc_id)
                            team_helpers.visualize_transition(test_constraints, particles_team_learner[member_id], params.mdp_class, params.weights['val'], text = plot_title, vars_filename = vars_filename)

                        # reset knowledge to mirror particle reset
                        prev_pf_reset_count = particles_team_teacher[member_id].pf_reset_count
                        if particles_team_teacher[member_id].pf_reset_count > prev_pf_reset_count:
                            reset_index = [i for i in range(len(team_knowledge[member_id][kc_id])) if (team_knowledge[member_id][kc_id][i] == particles_team_teacher[member_id].reset_constraint).all()]
                            # print('Resetting constraints.. Previous constraints in KC: ', team_knowledge[member_id][kc_id], 'Reset index: ', reset_index)
                            team_knowledge[member_id][kc_id] = team_knowledge[member_id][kc_id][reset_index[0]:]
                            # print('New constraints: ', team_knowledge[member_id][kc_id])
                        
                        print(colored("You got the diagnostic test right!", 'green'))
                        response_category_team.append('correct')
                        if knowledge_viz_flag:
                            plot_title = 'Interaction No.' + str(loop_count +1) + '. After test ' + str(test_no) + ' of KC ' + str(kc_id) + ' for player ' + member_id
                            team_helpers.visualize_transition(test_constraints, particles_team_teacher[member_id], params.mdp_class, params.weights['val'], text = plot_title, vars_filename = vars_filename)

                    else:
                        
                        failed_BEC_constraint = opt_feature_count - human_feature_count
                        failed_BEC_constraints_tuple.append([member_id, failed_BEC_constraint])
                        unit_learning_goal_reached = False
                        if test_no == 1:
                            test_constraints_team.append(copy.deepcopy([-failed_BEC_constraint]))
                        else:
                            test_constraints_team[p-1].extend(copy.deepcopy([-failed_BEC_constraint]))
                        print('Test constraints: ', [-failed_BEC_constraint], 'Test constraints team: ', test_constraints_team, 'for test no: ', test_no)
                        
                        
                        # update team knowledge
                        # print(colored('Current team knowledge for member id ' + str(member_id) + ': ' + str(team_knowledge), 'blue'))
                        team_knowledge = team_helpers.update_team_knowledge(team_knowledge, kc_id, kc_reset_flag, [-failed_BEC_constraint], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = [member_id])
                        # print(colored('Updated team knowledge for member id ' + str(member_id) + ': ' + str(team_knowledge), 'blue'))

                        # update teacher model
                        plot_title =  'Interaction No.' + str(loop_count +1) + '. Teacher belief for player ' + member_id + ' after test ' + str(test_no) + ' of KC ' + str(kc_id)
                        particles_team_teacher[member_id].update([-failed_BEC_constraint], plot_title = plot_title, viz_flag = knowledge_viz_flag, vars_filename = vars_filename)

                        # updated learner model
                        plot_title =  'Interaction No.' + str(loop_count +1) + '. Learner belief for player ' + member_id + ' after test ' + str(test_no) + ' of KC ' + str(kc_id)
                        particles_team_learner[member_id].update([-failed_BEC_constraint], learning_factor=team_learning_factor[p-1], plot_title = plot_title, model_type = learner_update_type, viz_flag = knowledge_viz_flag, vars_filename = vars_filename)       
                        
                        # update learner factor - after demonstrating learning or its lack (only if updating after each individual test and sampling a new human model after each individual test)
                        if params.response_generation_type == 'Individual_tests':
                            team_learning_factor[p-1] = min(team_learning_factor[p-1] + team_learning_rate[p-1, 1], max_learning_factor) # update learning parameter
                        
                        if knowledge_viz_flag:
                            plot_title = 'Interaction No.' + str(loop_count+1) +'Simulated knowledge change for player ' + member_id + ' after test ' + str(test_no) + ' of KC' + str(kc_id)
                            team_helpers.visualize_transition([-failed_BEC_constraint], particles_team_learner[member_id], params.mdp_class, params.weights['val'], text = plot_title, vars_filename = vars_filename)

                        
                        # reset knowledge to mirror particle reset
                        prev_pf_reset_count = particles_team_teacher[member_id].pf_reset_count
                        if particles_team_teacher[member_id].pf_reset_count > prev_pf_reset_count:
                            # print('Resetting constraints.. Previous constraints in KC: ', team_knowledge[member_id][kc_id])
                            # print('Constraint that reset particle filter: ', particles_team_teacher[member_id].reset_constraint)
                            reset_index = [i for i in range(len(team_knowledge[member_id][kc_id])) if (team_knowledge[member_id][kc_id][i] == particles_team_teacher[member_id].reset_constraint).all()]
                            # print('Reset index: ', reset_index)
                            team_knowledge[member_id][kc_id] = team_knowledge[member_id][kc_id][reset_index[0]:]
                            # print('New constraints: ', team_knowledge[member_id])
                        
                        print("You got the diagnostic test wrong. Failed BEC constraint: {}".format(failed_BEC_constraint))
                        response_category_team.append('incorrect')
                        if knowledge_viz_flag:
                            plot_title = 'Interaction No.' + str(loop_count+1) + '. After test ' + str(test_no) + ' of KC ' + str(kc_id) + ' for player ' + member_id
                            team_helpers.visualize_transition([-failed_BEC_constraint], particles_team_teacher[member_id], params.mdp_class, params.weights['val'], text = plot_title, vars_filename = vars_filename)
                        
                        # Correct trajectory
                        if demo_viz_flag:
                            print('Here is the correct trajectory...')
                            test_mdp.visualize_trajectory_comparison(opt_traj, human_traj)
                        

                    ####################################
                    
                    # update team knowlegde in PF model
                    # print(colored('team_knowledge for member ' + str(member_id) + ': ' + str(team_knowledge[member_id]), 'green'))
                    particles_team_teacher[member_id].knowledge_update(team_knowledge[member_id])


                    ########## debugging - calculate proportion of particles that are within the BEC for teacher and learner
                    particles_team_teacher[member_id].calc_particles_probability(test_constraints)
                    particles_team_learner[member_id].calc_particles_probability(test_constraints)
                    particles_prob_teacher_after_test[member_id] = particles_team_teacher[member_id].particles_prob_correct
                    particles_prob_learner_after_test[member_id] = particles_team_learner[member_id].particles_prob_correct

                    # print(colored('particles_prob_teacher_test for player : ' + member_id +   str(particles_prob_teacher_test[member_id]) + '. particles_prob_learner_test for player : ' + member_id + str(particles_prob_learner_test[member_id]), 'green'))
                    ##########################

                    # update team player id
                    p += 1
                
                # update probability for current test
                prob_teacher_before_testing.append(particles_prob_teacher_before_test)
                prob_learner_before_testing.append(particles_prob_learner_before_test)
                prob_teacher_after_testing.append(particles_prob_teacher_after_test)
                prob_learner_after_testing.append(particles_prob_learner_after_test)


                # Update test number
                test_no += 1
                kc_reset_flag = False
            ##############################  Completed simulating team response
            
            ######  update learning factor based on all test responses for the next set of interaction
            if params.response_generation_type == 'All_tests':    
                for test_id in range(len(preliminary_tests)):
                    for i in range(params.team_size):
                        member_id = 'p' + str(i+1)
                        if response_type_all_tests_team[member_id][test_id] == 'correct':
                            team_learning_factor[i] = min(team_learning_factor[i] + team_learning_rate[i, 0], max_learning_factor)
                        else:
                            team_learning_factor[i] = min(team_learning_factor[i] + team_learning_rate[i, 1], max_learning_factor)
            
            #######################################

            
            #### Update team knowledge belief based on the set of tests
            if params.team_size > 1:
                
                ## Check if there are non-intersecting constraints and go to fall-back teaching behavior
                # print('test_constraints_team for recent test: ', test_constraints_team)
                test_constraints_team_expanded = []
                for ind_test_constraints in test_constraints_team:
                    test_constraints_team_expanded.extend(ind_test_constraints)
                # print('test_constraints_team_expanded: ', test_constraints_team_expanded)
                non_intersecting_constraints_flag, non_intersecting_constraints_count = team_helpers.check_for_non_intersecting_constraints(test_constraints_team_expanded, params.weights['val'], params.step_cost_flag, non_intersecting_constraints_count)
                # print('Non-intersecting constraints normal loop? ', non_intersecting_constraints_flag)
                
                ## Assign majority rules and update common knowledge and joint knowledge accordingly
                if non_intersecting_constraints_flag:
                    test_constraints_team_expanded, intersecting_constraints = team_helpers.majority_rules_non_intersecting_team_constraints(test_constraints_team, params.weights['val'], params.step_cost_flag, test_flag = True)
                    # print('Majority rules for non intersecting contraints... Team constraints team expanded after processing: ', test_constraints_team_expanded)
                
                # if there are no constraints after majority rules non intersecting constraints! just an additional check so that simulation does not stop
                if len(test_constraints_team_expanded) == 0:
                    RuntimeError('No constraints after majority rules non intersecting constraints!')

                # double check again
                non_intersecting_constraints_flag, non_intersecting_constraints_count = team_helpers.check_for_non_intersecting_constraints(test_constraints_team_expanded, params.weights['val'], params.step_cost_flag, non_intersecting_constraints_count)

                ## update team knowledge based on the test responses of the team
                if not non_intersecting_constraints_flag:
                
                    ## update common knowledge manually since it could have been updated by the majority rules function
                    test_cnst_team_updated = []
                    for cnst in test_constraints_team_expanded:
                        test_cnst_team_updated.append(cnst)

                    min_new_common_knowledge = BEC_helpers.remove_redundant_constraints(test_constraints_team_expanded, params.weights['val'], params.step_cost_flag)
                    
                    # print('Updating common knowledge...')
                    if kc_id == len(team_knowledge['common_knowledge']):
                        # # print('Appendin common knowledge...')
                        team_knowledge['common_knowledge'].append(min_new_common_knowledge)
                    else:
                        team_knowledge['common_knowledge'][kc_id] = copy.deepcopy(min_new_common_knowledge)
                    # print(colored('Updated team knowledge for common knowledge: ' + str(team_knowledge['common_knowledge']), 'blue'))
                    

                    ## update joint knowledge
                    team_knowledge = team_helpers.update_team_knowledge(team_knowledge, kc_id, kc_reset_flag, [], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = ['joint_knowledge'])
                    print(colored('Updated team knowledge for joint knowledge: ' + str(team_knowledge['joint_knowledge']), 'blue'))

                    # update team knowlwedge in PF model
                    particles_team_teacher['common_knowledge'].knowledge_update(team_knowledge['common_knowledge'])
                    particles_team_teacher['joint_knowledge'].knowledge_update(team_knowledge['joint_knowledge'])

                    # Update particle filters of common and joint knowledge
                    
                    # print('PF update of commong knowledge with constraints: ', min_new_common_knowledge)
                    plot_title = 'Interaction No.' + str(loop_count +1) + '. Teacher belief for common knowledge after tests of KC ' + str(kc_id)
                    particles_team_teacher['common_knowledge'].update(min_new_common_knowledge, plot_title = plot_title, viz_flag = knowledge_viz_flag, vars_filename = vars_filename)

                    # reset knowledge to mirror particle reset
                    prev_pf_reset_count = particles_team_teacher['common_knowledge'].pf_reset_count
                    if particles_team_teacher['common_knowledge'].pf_reset_count > prev_pf_reset_count:
                        # print('Resetting constraints.. Previous constraints common knowledge: ', team_knowledge['common_knowledge'])
                        # reset_index = [i for i in range(len(team_knowledge['common_knowledge'])) if (team_knowledge['common_knowledge'][i] == particles_team_teacher['common_knowledge'].reset_constraint).all()]
                        reset_index = [i for i in range(len(team_knowledge['common_knowledge'][kc_id])) if (team_knowledge['common_knowledge'][kc_id][i] == particles_team_teacher['common_knowledge'].reset_constraint).all()]
                        # print('Reset index: ', reset_index, 'previous reset count: ', prev_pf_reset_count, 'current reset count: ', particles_team_teacher['common_knowledge'].pf_reset_count, 'reset constraint: ', particles_team_teacher['common_knowledge'].reset_constraint)
                        team_knowledge['common_knowledge'][kc_id] = team_knowledge['common_knowledge'][kc_id][reset_index[0]:]
                        # print('New constraints: ', team_knowledge['common_knowledge'])
                    
                    prev_pf_reset_count = particles_team_teacher['joint_knowledge'].pf_reset_count
                    # print('PF update of joint knowledge with constraints: ', test_constraints_team)
                    particles_team_teacher['joint_knowledge'].update_jk(test_constraints_team)
                    
            
            else:
                ## update team knowledge based on the test responses of the team for intersecting constraints
                team_knowledge['common_knowledge'] = copy.deepcopy(team_knowledge['p1'])
                team_knowledge = team_helpers.update_team_knowledge(team_knowledge, kc_id, kc_reset_flag, [], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = ['joint_knowledge'])
                    
            ##########################################################################

            loop_count += 1

            # Check if unit knowledge is sufficient to move on to the next unit (unit learning goal reached)
            print('team_knowledge: ', team_knowledge, 'min_KC_constraints: ', min_KC_constraints)
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
            loop_vars = initialize_loop_vars()
            loop_vars['run_no'] = run_no
            loop_vars['demo_strategy'] = demo_strategy
            loop_vars['knowledge_id'] = knowledge_id
            loop_vars['variable_filter'] = variable_filter
            loop_vars['knowledge_comp_id'] = kc_id
            loop_vars['loop_count'] = loop_count
            loop_vars['summary_count'] = summary_count
            loop_vars['min_BEC_constraints'] = copy.deepcopy(min_BEC_constraints)
            loop_vars['unit_constraints'] = copy.deepcopy(unit_constraints)
            loop_vars['min_KC_constraints'] = copy.deepcopy(min_KC_constraints)
            loop_vars['all_unit_constraints'] = copy.deepcopy(all_unit_constraints)
            loop_vars['demo_ids'] = copy.deepcopy(demo_ids)
            loop_vars['team_knowledge_expected'] = copy.deepcopy(team_knowledge_expected)
            loop_vars['unit_knowledge_level_expected'] = team_helpers.calc_knowledge_level(team_knowledge_expected, min_KC_constraints, kc_id_list = [kc_id], plot_flag = False, fig_title = 'Expected Unit knowledge level for var filter: ' + str(variable_filter), vars_filename=vars_filename )
            loop_vars['BEC_knowledge_level_expected'] = team_helpers.calc_knowledge_level(team_knowledge_expected, min_BEC_constraints, plot_flag = False, fig_title = 'Expected BEC knowledge level after var filter: ' + str(variable_filter), vars_filename=vars_filename)
            # loop_vars['response_type'] = response_type
            # loop_vars['response_distribution'] = response_distribution_list[resp_no-1]
            loop_vars['test_constraints'] = copy.deepcopy(all_tests_constraints)
            loop_vars['test_constraints_team'] = copy.deepcopy(test_constraints_team)
            loop_vars['opposing_constraints_count'] = opposing_constraints_count  
            # loop_vars['final_remedial_constraints'] = copy.deepcopy(remedial_constraints_team_expanded)
            loop_vars['N_remedial_tests'] = N_remedial_tests
            loop_vars['team_knowledge'] = copy.deepcopy(team_knowledge)
            loop_vars['particles_team_teacher'] = copy.deepcopy(particles_team_teacher)
            loop_vars['particles_team_learner'] = copy.deepcopy(particles_team_learner)
            loop_vars['unit_knowledge_level'] = team_helpers.calc_knowledge_level(team_knowledge, min_KC_constraints, particles_team_teacher = particles_team_teacher, kc_id_list = [kc_id], plot_flag = False, fig_title = 'Actual Unit knowledge level for var filter: ' + str(variable_filter), vars_filename=vars_filename)
            loop_vars['BEC_knowledge_level'] = team_helpers.calc_knowledge_level(team_knowledge, min_BEC_constraints, particles_team_teacher = particles_team_teacher, plot_flag = False, fig_title = 'Actual BEC knowledge level expected after var filter: ' + str(variable_filter), vars_filename=vars_filename)
            loop_vars['unit_knowledge_area'] = BEC_helpers.calc_solid_angles([min_KC_constraints])
            # loop_vars['all_unit_constraints_area'] = BEC_helpers.calc_solid_angles([all_unit_constraints])
            # loop_vars['all_unit_constraints_area'] = []
            loop_vars['BEC_knowledge_area'] = BEC_helpers.calc_solid_angles([min_BEC_constraints])
            loop_vars['initial_team_learning_factor'] = copy.deepcopy(initial_team_learning_factor)
            loop_vars['team_learning_factor'] = copy.deepcopy(team_learning_factor)
            loop_vars['team_composition'] = team_composition
            # loop_vars['sampling_condition'] = response_sampling_condition
            loop_vars['particles_prob_learner_before_test'] = copy.deepcopy(prob_learner_before_testing)
            loop_vars['particles_prob_teacher_before_test'] = copy.deepcopy(prob_teacher_before_testing)
            # loop_vars['particles_prob_learner_after_test'] = copy.deepcopy(prob_learner_after_testing)
            # loop_vars['particles_prob_teacher_after_test'] = copy.deepcopy(prob_teacher_after_testing)
            # loop_vars['particles_prob_learner_demos'] = copy.deepcopy(particles_learner_prob_after_demo)
            # loop_vars['particles_prob_teacher_demos'] = copy.deepcopy(particles_teacher_prob_after_demo)
            loop_vars['sim_status'] = 'Running'
            loop_vars['team_response_models'] = human_model_weight_team

            for i in range(len(team_knowledge)):
                if i < params.team_size:
                    loop_vars['pf_reset_count'][i] = particles_team_teacher['p' + str(i+1)].pf_reset_count
                elif i == len(team_knowledge) - 2:
                    loop_vars['pf_reset_count'][i] = particles_team_teacher['common_knowledge'].pf_reset_count
                elif i == len(team_knowledge) - 1:
                    loop_vars['pf_reset_count'][i] = particles_team_teacher['joint_knowledge'].pf_reset_count


            # print(colored('Unit knowledge level: ' + str(loop_vars['unit_knowledge_level']), 'red'))
            # print(colored('BEC knowledge level: ' + str(loop_vars['BEC_knowledge_level']), 'red'))

            # debug - plot loop variables
            if knowledge_viz_flag:
            # print('Visualizing team knowledge constraints for this teaching loop...')
                plot_title =  'Interaction No.' + str(loop_count +1) + '. Team_knowledge_constraints'
                fig = team_helpers.visualize_team_knowledge_constraints(team_knowledge, params.weights['val'], params.step_cost_flag, particles_team_teacher = particles_team_teacher, min_unit_constraints = min_BEC_constraints, plot_filename = 'team_knowledge_constraints', fig_title = plot_title)
        

            # Update variable filter for next loop
            if next_unit_flag:
                variable_filter, nonzero_counter, teaching_complete_flag = team_helpers.check_and_update_variable_filter(variable_filter = variable_filter, nonzero_counter = nonzero_counter)
                print(colored('Moving to next unit. Updated variable filter: ', 'blue'), variable_filter, '. Teaching_complete_flag: ', teaching_complete_flag)
                
                # if knowledge was added to current KC then add KC id
                if kc_id == len(team_knowledge['p1'])-1:
                    next_kc_flag = True
                    obj_func_prop = 1.0
                    print(colored('kc_id: ' + str(kc_id) + 'len(team_knowledge): ' + str(len(team_knowledge['p1'])) + 'next_kc_flag: ' + str(next_kc_flag), 'red') )  

                if teaching_complete_flag:
                    loop_vars['sim_status'] = 'Teaching complete'

            else:
            
                # Update expected knowledge with actual knowledge for next iteration
                team_knowledge_expected = copy.deepcopy(team_knowledge)

                # update obj_func_prop (is ths needed?)
                if loop_count - next_unit_loop_id > params.loop_threshold_demo_simplification:
                    obj_func_prop = 0.5     # reduce informativeness from demos if they have not learned
                    # print(colored('updated obj_func_prop: ' + str(obj_func_prop), 'red') )
                    RuntimeError('obj_func_prop needs to be reduced since the team is not learning!')

                # check if number of max interactions sets have been reached
                if loop_count > params.max_loops:
                    print(colored('Maximum teaching interactions reached! ', 'red'))
                    teaching_complete_flag = True
                    loop_vars['sim_status'] = 'Max loops reached'


            prev_summary_len = len(BEC_summary)
            # append loop vars to dataframe
            vars_to_save = vars_to_save.append(loop_vars, ignore_index=True)

        
        else:
            # Update Variable filter and move to next unit, if applicable, if no demos are available for this unit.
            # print(colored('No new summaries for this unit...!!', 'red'))
            # unit_learning_goal_reached = True
            
            # Update variable filter
            # if unit_learning_goal_reached:
            variable_filter, nonzero_counter, teaching_complete_flag = team_helpers.check_and_update_variable_filter(variable_filter = variable_filter, nonzero_counter = nonzero_counter)
                
            # if knowledge was added to current KC then add KC id
            if kc_id == len(team_knowledge['p1'])-1:
                next_kc_flag = True
                obj_func_prop = 1.0
                # print(colored('updated obj_func_prop for new KC: ' + str(obj_func_prop), 'red') )

            # if teaching_complete_flag:
            vars_to_save['sim_status'].iloc[-1] = 'No new demos'  # update the last loop sim status to teaching complete


        ########
        # save vars so far (end of session)
        vars_to_save.to_csv('models/' + params.data_loc['BEC'] + '/' + vars_filename + '_' + str(run_no) + '.csv', index=False)
        with open('models/augmented_taxi2/' + vars_filename + '_' + str(run_no) + '.pickle', 'wb') as f:
            pickle.dump(vars_to_save, f)


        ## debugging - save data
        data_dict = {'update_id': update_id_history, 'member_id': member, 'learning_factor': team_learning_factor_history, 'model': sampled_points_history, 'point_probability': point_probability, 'skip_model': skip_model_history, \
                 'constraints': constraint_history, 'constraint_flag': constraint_flag_history, 'response': response_history, 'cluster_id': cluster_id_history, 'particles_learner_prob_demo_history': particles_learner_prob_demo_history, \
                'particles_learner_prob_test_history': particles_learner_prob_demo_history, 'prob_initial_history': prob_initial_history, 'prob_reweight_history': prob_reweight_history, 'prob_resample_history': prob_resample_history, \
                    'resample_flag_history': resample_flag_history, 'update_type': update_sequence_history, 'resample_noise': resample_noise_history}
    
        debug_data_response = pd.DataFrame(data=data_dict)

        debug_data_response.to_csv('models/' + params.data_loc['BEC'] + '/' + 'debug_data_' + vars_filename + '_' + str(run_no) + '.csv', index=False)

####################################################################################


if __name__ == "__main__":
    pool = Pool(min(params.n_cpu, 60))
    os.makedirs('models/' + params.data_loc['base'], exist_ok=True)
    os.makedirs('models/' + params.data_loc['BEC'], exist_ok=True)


    ## run_reward_teaching
    # run_reward_teaching(params, pool, demo_strategy = 'common_knowledge', experiment_type = 'simulated', response_distribution_list = ['mixed', 'correct', 'mixed', 'incorrect', 'correct', 'correct', 'correct', 'correct'], run_no = 1, viz_flag=True, vars_filename = 'workshop_data')
    # vars_to_save = run_reward_teaching(params, pool)
    
    # viz_flag = [demo_viz, test_viz, pf_knowledge_viz]
    sim_params = {'min_correct_likelihood': 0.5}
    initial_team_learning_factor = [0.5, 0.6, 0.7]
    run_reward_teaching(params, pool, sim_params, demo_strategy = 'individual_knowledge_low', experiment_type = 'simulated', run_no = 1, viz_flag=[False, False, True], vars_filename = '12_15_sim_debug', \
                        response_sampling_condition = 'particles', team_composition = None, initial_team_learning_factor = initial_team_learning_factor)

    
    pool.close()
    pool.join()



# save files
    # if len(BEC_summary) > 0:
    #     with open('models/' + data_loc + '/teams_BEC_summary.pickle', 'wb') as f:
    #         pickle.dump((BEC_summary, visited_env_traj_idxs, particles_team_teacher), f)










