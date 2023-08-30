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
import matplotlib as mpl
import teams.teams_helpers as team_helpers
mpl.rcParams['figure.facecolor'] = '1.0'
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'large'

import random
import pandas as pd


############################################






def calc_expected_learning(team_knowledge_expected, particles_team_expected, min_BEC_constraints, new_constraints, params, loop_count):

    print('Current expected team knowledge: ', team_knowledge_expected)
    print('New constraints: ', new_constraints)
    team_knowledge_expected = team_helpers.update_team_knowledge(team_knowledge_expected, new_constraints, params.team_size,  params.weights['val'], params.step_cost_flag)

    print('Updated expected team knowledge: ', team_knowledge_expected)

    p = 1
    while p <= params.team_size:
        member_id = 'p' + str(p)
        particles_team_expected[member_id].update(new_constraints)
        team_helpers.visualize_transition(new_constraints, particles_team_expected[member_id], params.mdp_class, params.weights['val'], text = 'Expected knowledge change for set ' + str(loop_count+1) + ' for player ' + member_id)  
        p += 1
            
    # Update common knowledge model
    particles_team_expected['common_knowledge'].update(new_constraints)
    team_helpers.visualize_transition(new_constraints, particles_team_expected['common_knowledge'], params.mdp_class, params.weights['val'], text = 'Expected knowledge change for set' + str(loop_count+1) + ' for common knowledge')
    


    # Update joint knowledge model
    # Method 1: Use complete joint knowledge of team
    # particles_team_expected['joint_knowledge'].update_jk(team_knowledge_expected['joint_knowledge'])

    # Method 2: Use new joint knowledge of team
    new_constraints_team = []
    for p in range(params.team_size):
        new_constraints_team.append(new_constraints)
    particles_team_expected['joint_knowledge'].update_jk(new_constraints_team)

    team_helpers.visualize_transition(new_constraints_team, particles_team_expected['joint_knowledge'], params.mdp_class, params.weights['val'], text = 'Expected knowledge change for set ' + str(loop_count+1) + ' for joint knowledge',  knowledge_type = 'joint_knowledge')

    print('min_BEC_constraints: ', min_BEC_constraints)
    print('Expected unit knowledge after seeing unit demonstrations: ', team_helpers.calc_knowledge_level(team_knowledge_expected, new_constraints) )
    print('Expected absolute knowledge after seeing unit demonstrations: ', team_helpers.calc_knowledge_level(team_knowledge_expected, min_BEC_constraints) )


    return team_knowledge_expected, particles_team_expected




def run_remedial_loop(failed_BEC_constraints_tuple, particles_team, team_knowledge, min_subset_constraints_record, env_record, traj_record, traj_features_record, test_history, visited_env_traj_idxs, running_variable_filter_unit, mdp_features_record, consistent_state_count, particles_demo, pool, viz_flag = False, response_type = 'simulated'):
    human_history = []

    # Method 1: Generate remedial demonstration from the combined failed BEC constraints of the team
    # print("Here is a remedial demonstration that might be helpful")
    # min_failed_constraints = BEC_helpers.remove_redundant_constraints(failed_BEC_constraints_team, params.weights['val'], params.step_cost_flag)

    # print('min_failed_constraints:', min_failed_constraints)
    
    # remedial_demos, visited_env_traj_idxs = team_helpers.obtain_remedial_demos_tests(params.data_loc['BEC'], BEC_summary[-1], visited_env_traj_idxs, min_failed_constraints, min_subset_constraints_record, traj_record, traj_features_record, running_variable_filter_unit, mdp_features_record)
    # # print(remedial_demos[0])

    # # TODO: Show remedial demonstration
    # for demo in remedial_demos:
    #     demo[0].visualize_trajectory(demo[1])
    #     test_history.extend(demo)

    #     particles_demo.update([demo[3]])
    #     # print('remedial demo :', demo)
    #     # print('remedial demo constraints:', demo[3])
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

        
        if response_type == 'simulated':
            N_remedial_tests = random.randint(1, 3)
            print('N_remedial_tests: ', N_remedial_tests)
        remedial_resp_no = 0

        print("Here is a remedial test to see if you've correctly learned the lesson")
        remedial_test_end = False
        while (not remedial_test_end) and (remedial_resp_no < N_remedial_tests): #Note: To be updated. Do not show remedial until they get it right, rather show it until there is common knowledge and joint knowledge
            print('Still inside while loop...')
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


            if response_type == 'simulated':
                if remedial_resp_no == N_remedial_tests-1:
                    remedial_response = 'correct'
                else:
                    remedial_response = 'mixed' # Note: We assume that one person always gets the remedial test correct and the other person gets it wrong (only for N=2)
                human_traj_team, human_history = get_human_response(remedial_env_traj_tuple[0], remedial_constraint, remedial_traj, human_history, team_size = params.team_size, response_distribution = remedial_response)
                remedial_resp_no += 1
                print('Simulating human response for remedial test... Remedial Response no: ', remedial_resp_no, '. Response type: ', remedial_response)

            
            remedial_constraints_team = []
            # Show the same test for each person and get test responses of each person in the team
            p = 1
            while p <= params.team_size:
                member_id = 'p' + str(p)

                if response_type == 'simulated':
                    human_traj = human_traj_team[p-1]
                else:
                    human_traj, human_history = remedial_mdp.visualize_interaction(
                        keys_map=params.keys_map)  # the latter is simply the gridworld locations of the agent


                human_feature_count = remedial_mdp.accumulate_reward_features(human_traj, discount=True)
                opt_feature_count = remedial_mdp.accumulate_reward_features(remedial_traj, discount=True)

                if (human_feature_count == opt_feature_count).all():
                    print("You got the remedial test correct")
                    particles_demo.update([remedial_constraint])
                    remedial_constraints_team.append([remedial_constraint])
                    if viz_flag:
                        team_helpers.visualize_transition([remedial_constraint], particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Remedial Test ' + str(remedial_resp_no) + ' for player ' + member_id)

                else:
                    failed_BEC_constraint = opt_feature_count - human_feature_count
                    # print('Optimal traj: ', remedial_traj)
                    # print('Human traj: ', human_traj)
                    if viz_flag:
                        print("You got the remedial test wrong. Here's the correct answer")
                        print("Failed BEC constraint: {}".format(failed_BEC_constraint))
                        remedial_mdp.visualize_trajectory_comparison(remedial_traj, human_traj)
                    else:
                        print("You got the remedial test wrong. Failed BEC constraint: {}".format(failed_BEC_constraint))

                    particles_demo.update([-failed_BEC_constraint])
                    remedial_constraints_team.append([-failed_BEC_constraint])
                    if viz_flag:
                        # BEC_viz.visualize_pf_transition([-failed_BEC_constraint], particles_demo, params.mdp_class, params.weights['val'])
                        team_helpers.visualize_transition([-failed_BEC_constraint], particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Remedial Test ' + str(remedial_resp_no) + ' for player ' + member_id)


                p += 1
        
        # Update team knowledge
        opposing_constraints_flag, _, _ = team_helpers.check_opposing_constraints(remedial_constraints_team)
        print('Opposing constraints remedial loop? ', opposing_constraints_flag)

        if not opposing_constraints_flag:
            remedial_test_end = True
            remedial_constraints_team_expanded = []
            for test_constraints in remedial_constraints_team:
                remedial_constraints_team_expanded.extend(test_constraints)
            
            p = 1
            while p <= params.team_size:
                member_id = 'p' + str(p)
                team_knowledge = team_helpers.update_team_knowledge(team_knowledge, remedial_constraints_team[p-1], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = [member_id])
                particles_team[member_id].update(remedial_constraints_team[p-1])
                p += 1

            if viz_flag:
                team_helpers.visualize_transition(remedial_constraints_team[p-1], particles_team[member_id], params.mdp_class, params.weights['val'], text = 'After Remedial Test for player ' + member_id)

            # Update common knowledge model
            team_knowledge = team_helpers.update_team_knowledge(team_knowledge, [], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = ['common_knowledge', 'joint_knowledge'])
            particles_team['common_knowledge'].update(remedial_constraints_team_expanded)
            particles_team['joint_knowledge'].update_jk(remedial_constraints_team)

    return test_history, visited_env_traj_idxs, particles_team, team_knowledge, particles_demo, remedial_constraints_team_expanded, remedial_resp_no



def get_human_response(env_idx, env_cnst, opt_traj, human_history, team_size = 2, response_distribution = 'correct'):

    human_traj = []
    cnst = []

    # a) find the sub_optimal responses
    BEC_depth_list = [1]

    filename = 'models/augmented_taxi2/gt_policies/wt_vi_traj_params_env' + str(env_idx).zfill(5) + '.pickle'
    
    with open(filename, 'rb') as f:
        wt_vi_traj_env = pickle.load(f)

    mdp = wt_vi_traj_env[0][1].mdp
    agent = FixedPolicyAgent(wt_vi_traj_env[0][1].policy)
    mdp.set_init_state(opt_traj[0][0])
    

    constraints_list_correct = []
    human_trajs_list_correct = []
    constraints_list_incorrect = []
    human_trajs_list_incorrect = []


    for BEC_depth in BEC_depth_list:
        # print('BEC_depth: ', BEC_depth)
        action_seq_list = list(itertools.product(mdp.actions, repeat=BEC_depth))

        traj_opt = mdp_helpers.rollout_policy(mdp, agent)
        # print('Optimal Trajectory length: ', len(traj_opt))
        traj_hyp = []

        for sas_idx in range(len(traj_opt)):
        
            # reward features of optimal action
            mu_sa = mdp.accumulate_reward_features(traj_opt[sas_idx:], discount=True)

            sas = traj_opt[sas_idx]
            cur_state = sas[0]
            # if sas_idx > 0:
            #     traj_hyp = traj_opt[:sas_idx-1]

            # currently assumes that all actions are executable from all states
            for action_seq in action_seq_list:
                if sas_idx > 0:
                    traj_hyp = traj_opt[:sas_idx-1]

                traj_hyp_human = mdp_helpers.rollout_policy(mdp, agent, cur_state=cur_state, action_seq=action_seq)
                traj_hyp.extend(traj_hyp_human)
                
                mu_sb = mdp.accumulate_reward_features(traj_hyp, discount=True)
                new_constraint = mu_sa - mu_sb

                count = sum(np.array_equal(new_constraint, arr) for arr in constraints_list_correct) + sum(np.array_equal(new_constraint, arr) for arr in constraints_list_incorrect)

                # if count < team_size: # one sample trajectory for each constriant is sufficient; but just for a variety gather one trajectory for each person for each constraint, if possible
                    # print('Hyp traj len: ', len(traj_hyp))
                    # print('new_constraint: ', new_constraint)
                if (new_constraint == np.array([0, 0, 0])).all():
                    constraints_list_correct.append(env_cnst)
                    human_trajs_list_correct.append(traj_opt) 
                else:
                    constraints_list_incorrect.append(new_constraint)
                    human_trajs_list_incorrect.append(traj_hyp)

           
    print('Constraints list correct: ', len(constraints_list_correct))
    print('Constraints list incorrect: ', len(constraints_list_incorrect))
    
    # b) find the counterfactual human responses
    sample_human_models = BEC_helpers.sample_human_models_uniform([], 8)

    for model_idx, human_model in enumerate(sample_human_models):

        mdp.weights = human_model
        vi_human = ValueIteration(mdp, sample_rate=1)
        vi_human.run_vi()

        if not vi_human.stabilized:
            skip_human_model = True
            print(colored('Human model ' + str(model_idx) + ' did not converge and skipping for response generation', 'red'))
        
        if not skip_human_model:
            human_opt_trajs = mdp_helpers.rollout_policy_recursive(vi_human.mdp, vi_human, cur_state, [])
            for human_opt_traj in human_opt_trajs:
                human_traj_rewards = mdp.accumulate_reward_features(human_opt_traj, discount=True)
                mu_sa = mdp.accumulate_reward_features(traj_opt, discount=True)
                new_constraint = mu_sa - human_traj_rewards

                count = sum(np.array_equal(new_constraint, arr) for arr in constraints_list_correct) + sum(np.array_equal(new_constraint, arr) for arr in constraints_list_incorrect)

                # if count < team_size:
                    # print('Hyp traj len: ', len(traj_hyp))
                    # print('new_constraint: ', new_constraint)
                if (new_constraint == np.array([0, 0, 0])).all():
                    constraints_list_correct.append(env_cnst)
                    human_trajs_list_correct.append(traj_opt) 
                else:
                    constraints_list_incorrect.append(new_constraint)
                    human_trajs_list_incorrect.append(human_opt_traj)

    print('Constraints list correct after human models: ', len(constraints_list_correct))
    print('Constraints list incorrect after human models: ', len(constraints_list_incorrect))
    

    # Currently coded for a team size of 2
    if response_distribution == 'correct':

        for i in range(team_size):
            random_index = random.randint(0, len(constraints_list_correct)-1)
            human_traj.append(human_trajs_list_correct[random_index])
            cnst.append(constraints_list_correct[random_index])

            constraints_list_correct.pop(random_index)
            human_trajs_list_correct.pop(random_index)
        
    elif response_distribution == 'incorrect':
        for i in range(team_size):
            random_index = random.randint(0, len(constraints_list_incorrect)-1)
            human_traj.append(human_trajs_list_incorrect[random_index])
            cnst.append(constraints_list_incorrect[random_index])

            constraints_list_incorrect.pop(random_index)
            human_trajs_list_incorrect.pop(random_index)

    elif response_distribution == 'mixed':
        constraints_list_correct_used = []
        for i in range(team_size):
            if i%2 == 0:
                print('len constraints_list_correct: ', len(constraints_list_correct))
                random_index = random.randint(0, len(constraints_list_correct)-1)
                human_traj.append(human_trajs_list_correct[random_index])
                cnst.append(constraints_list_correct[random_index])
                constraints_list_correct_used.append(constraints_list_correct[random_index])
                constraints_list_correct.pop(random_index)
                human_trajs_list_correct.pop(random_index)
            else:
                indices = [i for i in range(len(constraints_list_incorrect)) if np.array_equal(-constraints_list_incorrect[i], constraints_list_correct_used[-1])]
                print('constraints_list_correct_used: ', constraints_list_correct_used)
                print('opposing indices: ', indices, 'for constraint: ', constraints_list_correct_used[-1])
                print('constraints_list_incorrect: ', constraints_list_incorrect)
                if len(indices) > 0:    
                    random_index = random.choice(indices)    # opposing incorrect response
                else:
                    random_index = random.randint(0, len(constraints_list_incorrect)-1)  # random incorrect response
                
                human_traj.append(human_trajs_list_incorrect[random_index])
                cnst.append(constraints_list_incorrect[random_index])

                constraints_list_incorrect.pop(random_index)
                human_trajs_list_incorrect.pop(random_index)

    human_history.append((env_idx, human_traj, cnst))

    # print('N of human_traj: ', len(human_traj))

    # print('Visualizing human trajectory ....')
    # for ht in human_traj:
    #     print('human_traj len: ', len(ht))
    #     print('constraint: ', cnst[human_traj.index(ht)])
    #     mdp.visualize_trajectory(ht)

        # Later: Check that test responses are not getting repeated for the same environment
        # TODO: Check if the trajectories generated are valid - seems like diagonal moves are occuring sometimes



    return human_traj, human_history 




# def run_reward_teaching(params, pool, demo_strategy = 'common', response_type = 'simulated', response_distribution_list = ['correct']*10, run_no = 1, vars_to_save = None):
def run_reward_teaching(params, pool, demo_strategy = 'common_knowledge', response_type = 'simulated', response_distribution_list = ['correct']*10, run_no = 1, viz_flag=False, vars_filename = 'var_to_save'):

    #### Individual demos
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
    
    if viz_flag:
        print('Team prior: ', team_prior)
        # team_helpers.visualize_team_knowledge(particles_team, [], params.mdp_class, weights=params.weights['val'], text='Team prior')
        for know_id, know_type in enumerate(team_prior):
            team_helpers.visualize_transition([], particles_team[know_type], params.mdp_class, weights=params.weights['val'], knowledge_type = know_type)
                

    # unit (set of knowledge components) selection
    variable_filter, nonzero_counter, teaching_complete_flag = team_helpers.check_and_update_variable_filter(min_subset_constraints_record, initialize_filter_flag=True)

    
    BEC_summary = []
    visited_env_traj_idxs = []
    min_BEC_constraints_running = []
    summary_count = 0 # this is actually the number of demos and not the number of units
    team_knowledge = copy.deepcopy(team_prior) # team_prior calculated from team_helpers.sample_team_pf also has the aggregated knowledge from individual priors
    visualize_pf_transition = True
    prev_summary_len = 0

    plt.rcParams['figure.figsize'] = [10, 6]

    # for debugging
    particles_team_expected = copy.deepcopy(particles_team)
    team_knowledge_expected = copy.deepcopy(team_knowledge)

    # for testing design choices (is remedial demo needed)
    remedial_test_flag = False
    demo_vars_template = {'run_no': 1,
                         'demo_strategy': None,
                          'variable_filter': None,
                          'loop_count': None,  
                            'summary_count': None,
                            'min_BEC_constraints': None,
                            'unit_constraints': None,
                            'team_knowledge_expected': None,
                            'particles_team_expected': None,
                            'unit_knowledge_level_expected': None,
                            'BEC_knowledge_level_expected': None,
                            'response_type': None,
                            'response_distribution': None,
                            'test_constraints': None,
                            'opposing_constraints_count': 0,
                            'final_remedial_constraints': None,
                            'N_remedial_tests': None,
                            'team_knowledge': None,
                            'particles_team': None,
                            'unit_knowledge_level': None,
                            'BEC_knowledge_level': None,
                            }
    
    # Variables to save for subsequent analysis
    if run_no == 1:
        vars_to_save = pd.DataFrame(columns=demo_vars_template.keys())
    else:
        with open('models/augmented_taxi2/' + vars_filename + '.pickle', 'rb') as f:
                    vars_to_save = pickle.load(f)
        print('Previousaly saved sim data len: ', vars_to_save.shape[0])

    # if vars_to_save is None:
    #     vars_to_save = pd.DataFrame(columns=demo_vars_template.keys())


    print('Demo strategy: ', demo_strategy)

    # initialize human models for demo generation
    knowledge_id, particles_demo = team_helpers.particles_for_demo_strategy(demo_strategy, team_knowledge, particles_team, params.team_size, params.weights['val'], params.step_cost_flag, params.BEC['n_particles'], min_BEC_constraints)

    ################################################
    
    
    loop_count = 0
    next_unit_loop_id = 0
    unit_learning_goal_reached = False
    resp_no = 0
    human_history = []
    # WIP: Unitwise teaching-testing loop
    while not teaching_complete_flag:

        # reset loop varibales
        opposing_constraints_count = 0
        remedial_constraints_team_expanded = []
        N_remedial_tests = 0

        # # Sample particles from the corresponding human/team knowledge based on the demo strategy
        # if demo_reset_flag == True:
        #     knowledge_id, particles_demo = team_helpers.particles_for_demo_strategy(demo_strategy, team_knowledge, particles_team, params.team_size, params.weights['val'], params.step_cost_flag, params.BEC['n_particles'], min_BEC_constraints)


        # Obtain BEC summary for a new unit (skip if its the 1st unit)
        print('Summary count: ', summary_count)
        if summary_count == 0:
            print(params.data_loc['BEC'] )
            try:
                # print('Trying to open existing BEC summary file...')
                with open('models/' + params.data_loc['BEC'] + '/BEC_summary_initial_2.pickle', 'rb') as f:
                    BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = pickle.load(f)
            except:
                # print('Starting summary generation for 1st unit..')
                BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = team_helpers.obtain_team_summary(params.data_loc['BEC'], min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, params.weights['val'], params.step_cost_flag, 
                                                                                                                    pool, params.BEC['n_human_models'], consistent_state_count, params.BEC['n_train_demos'], particles_demo, variable_filter, nonzero_counter, BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs)
                # print('Ended summary generation for 1st unit..')
                if len(BEC_summary) > 0:
                    with open('models/' + params.data_loc['BEC'] + '/BEC_summary_initial.pickle', 'wb') as f:
                        pickle.dump((BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo), f)
            
            
        else:
            # print('Starting summary generation for unit no.  ' + str(loop_count + 1) )
            BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = team_helpers.obtain_team_summary(params.data_loc['BEC'], min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, params.weights['val'], params.step_cost_flag, 
                                                                                                                pool, params.BEC['n_human_models'], consistent_state_count, params.BEC['n_train_demos'], particles_demo, variable_filter, nonzero_counter, BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs)
            # print('Ended summary generation for unit no.  ' + str(loop_count + 1) )
        


        # TODO: Demo-test-remedial-test loop if new summary is obtained
        print('BEC summary length: ', len(BEC_summary))
        print('Previous summary length: ', prev_summary_len)

        # check for any new summary
        if len(BEC_summary) > prev_summary_len:
            unit_constraints, running_variable_filter_unit = team_helpers.show_demonstrations(BEC_summary[-1], particles_demo, params.mdp_class, params.weights['val'], visualize_pf_transition, loop_count)

            # print('Variable filter:', variable_filter)
            # print('running variable filter:', running_variable_filter_unit)

            # obtain the constraints conveyed by the unit's demonstrations
            min_unit_constraints = BEC_helpers.remove_redundant_constraints(unit_constraints, params.weights['val'], params.step_cost_flag)


            # For debugging. Visualize the expected particles transition
            team_knowledge_expected, particles_team_expected = calc_expected_learning(team_knowledge_expected, particles_team_expected, min_BEC_constraints, min_unit_constraints, params, loop_count)
            
            if viz_flag and min_unit_constraints is not None:
                for know_id, know_type in enumerate(team_knowledge_expected):
                    if know_type == 'joint_knowledge':
                        p = 1
                        plot_constraints = []
                        while p <= params.team_size:
                            p += 1
                            print(plot_constraints)
                            plot_constraints.append(min_unit_constraints)
                            print(plot_constraints)
                    else:
                        plot_constraints = min_unit_constraints.copy()
                    
                    team_helpers.visualize_transition(plot_constraints, particles_team_expected[know_type], params.mdp_class, weights=params.weights['val'], knowledge_type = know_type)
                # team_helpers.visualize_team_knowledge(particles_team_expected, params.mdp_class, weights=params.weights['val'], text='Team expected knowledge after unit ' + str(loop_count))

            ## Conduct tests for the unit
            # obtain the diagnostic tests that will test the human's understanding of the unit's constraints
            print('Getting diagnostic tests for unit ' + str(loop_count) + '...')
            # print('visited_env_traj_idxs: ', visited_env_traj_idxs)
            preliminary_tests, visited_env_traj_idxs = BEC.obtain_diagnostic_tests(params.data_loc['BEC'], BEC_summary[-1], visited_env_traj_idxs, min_unit_constraints, min_subset_constraints_record, traj_record, traj_features_record, running_variable_filter_unit, mdp_features_record)

            
            # query the human's response to the diagnostic tests
            test_no = 1
            for test in preliminary_tests:
                print('Test no ', test_no, ' out of ', len(preliminary_tests), 'for unit ', loop_count)
                test_constraints_team = []
                response_category_team = []
                failed_BEC_constraints_tuple = []

                test_mdp = test[0]
                opt_traj = test[1]
                env_idx, traj_idx = test[2]
                test_constraints = test[3]
                test_history = [test] # to ensure that remedial demonstrations and tests are visually simple/similar and complex/different, respectively

                if response_type == 'simulated':
                    # print('response_distribution_list: ', response_distribution_list)
                    print('Simulating human response... Response no: ', resp_no, '. Response type: ', response_distribution_list[resp_no])
                    human_traj_team, human_history = get_human_response(env_idx, test_constraints, opt_traj, human_history, team_size = params.team_size, response_distribution = response_distribution_list[resp_no])
                    resp_no += 1
                    print('Opt traj len: ', len(opt_traj))
                    for i in range(len(human_traj_team)):
                        print('Simulated  responses for player ', i+1, ' : ', len(human_traj_team[i]))


                # Show the same test for each person and get test responses of each person in the team
                p = 1
                while p <= params.team_size:
                    member_id = 'p' + str(p)
                    # print("Here is a diagnostic test for this unit for player ", p)

                    if response_type != 'simulated':
                        human_traj, human_history = test_mdp.visualize_interaction(keys_map=params.keys_map) # the latter is simply the gridworld locations of the agent
                    else:
                        human_traj = human_traj_team[p-1]

                    human_feature_count = test_mdp.accumulate_reward_features(human_traj, discount=True)
                    opt_feature_count = test_mdp.accumulate_reward_features(opt_traj, discount=True)

                    if (human_feature_count == opt_feature_count).all():
                        test_constraints_team.append(test_constraints)
                        response_category_team.extend('correct')
                        team_knowledge = team_helpers.update_team_knowledge(team_knowledge, test_constraints, params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = [member_id])
                        particles_team[member_id].update(test_constraints) # update individual knowledge based on test response
                        print("You got the diagnostic test right")
                        if viz_flag:
                            team_helpers.visualize_transition(test_constraints, particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Test ' + str(test_no) + ' of Unit ' + str(loop_count) + ' for player ' + member_id)

                    else:
                        failed_BEC_constraint = opt_feature_count - human_feature_count
                        failed_BEC_constraints_tuple.append([member_id, failed_BEC_constraint])
                        unit_learning_goal_reached = False
                        test_constraints_team.append([-failed_BEC_constraint])
                        response_category_team.extend('incorrect')
                    
                        # print('Current team knowledge: ', team_knowledge)
                        team_knowledge = team_helpers.update_team_knowledge(team_knowledge, [-failed_BEC_constraint], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = [member_id])
                        particles_team[member_id].update([-failed_BEC_constraint])
                        
                        if viz_flag:
                            print("You got the diagnostic test wrong. Here's the correct answer")
                            print("Failed BEC constraint: {}".format(failed_BEC_constraint))
                            team_helpers.visualize_transition([-failed_BEC_constraint], particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Test ' + str(test_no) + ' of Unit ' + str(loop_count) + ' for player ' + member_id)
                            # Correct trajectory
                            test_mdp.visualize_trajectory_comparison(opt_traj, human_traj)
                        else:
                            print("You got the diagnostic test wrong.")
                        
                        # print('Opt traj: ', opt_traj)
                        # print('Human traj: ', human_traj)

                    
                    p += 1
                
                
                ## Check if there are non-intersecting constraints and go to fall-back teaching behavior
                print('test_constraints_team: ', test_constraints_team)
                test_constraints_team_expanded = []
                for test_constraints in test_constraints_team:
                    test_constraints_team_expanded.extend(test_constraints)
                # print('test_constraints_team_expanded: ', test_constraints_team_expanded)

                # check for opposing constraints (since remove redundant constraints gives the perpendicular axes for opposing constraints)
                opposing_constraints_flag, opposing_constraints_count, opposing_idx = team_helpers.check_opposing_constraints(test_constraints_team_expanded, opposing_constraints_count)
                print('Opposing constraints normal loop? ', opposing_constraints_flag)
                
                if not opposing_constraints_flag:  # Note: indicates that there is no common knowledge and entire sphere is the joint knowledge
                    # Update common knowledge and joint knowledge (only if there are no opposing constraints)
                    team_knowledge = team_helpers.update_team_knowledge(team_knowledge, [], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = ['common_knowledge', 'joint_knowledge'])
                    particles_team['common_knowledge'].update(test_constraints_team_expanded)
                    particles_team['joint_knowledge'].update_jk(test_constraints_team)
                    if viz_flag:
                        print('test_constraints_team_expanded: ', test_constraints_team_expanded)
                        team_helpers.visualize_transition(test_constraints_team_expanded, particles_team['common_knowledge'], params.mdp_class, params.weights['val'], text = 'Test ' + str(test_no) + ' of Unit ' + str(loop_count) + ' for common knowledge')
                        team_helpers.visualize_transition(test_constraints_team, particles_team['joint_knowledge'], params.mdp_class, params.weights['val'], text = 'Test ' + str(test_no) + ' of Unit ' + str(loop_count) + ' for joint knowledge',  knowledge_type = 'joint_knowledge')
                else:
                    
                    ##########################################################################
                    # # Option 1: Assign zero common knowledge and go to remedial demonstrations
                    # ck = np.zeros((3,1), dtype=int)
                    # team_knowledge['common_knowledge'] = [ck.reshape(ck.shape[1], ck.shape[0])]
                    # team_knowledge = team_helpers.update_team_knowledge(team_knowledge, [], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = ['joint_knowledge'])
                    # # particles_team['common_knowledge'].reinitialize(BEC_helpers.sample_human_models_uniform([], params.BEC['n_particles'])) # though this does not represent zero knowledge intersection of knowledge
                    # # particles_team['joint_knowledge'].reinitialize(BEC_helpers.sample_human_models_uniform([], params.BEC['n_particles']))
                    # particles_team['common_knowledge'].update(team_knowledge['common_knowledge']) # Note that the common knowledge is updated based on the minimum common constraint which is [0, 0, 0]
                    # particles_team['joint_knowledge'].update_jk(test_constraints_team) # Note that the joint knowledge is updated based on the opposing constraints which in this case is the entire space.

                    # test_history, visited_env_traj_idxs, particles_team, team_knowledge, particles_demo, remedial_constraints_team_expanded, N_remedial_tests = run_remedial_loop(failed_BEC_constraints_tuple, particles_team, team_knowledge, min_subset_constraints_record, env_record, traj_record, traj_features_record, test_history, visited_env_traj_idxs, running_variable_filter_unit, mdp_features_record, consistent_state_count, particles_demo, pool, viz_flag=False, response_type = 'simulated')
                    # print('Remedial loop ended for unit: ', loop_count, ' test no: ', test_no)
                    
                    ##########################################################################
                    # Option 2: Assign majority rules and update common knowledge and joint knowledge accordingly
                    opp_idx_unique = []
                    for i in range(len(opposing_idx)):
                        opp_idx_unique.extend(x for x in opposing_idx[i] if x not in opp_idx_unique)

                    print('opp_idx_unique: ', opp_idx_unique)
                    opp_constraints = test_constraints_team_expanded[opp_idx_unique]
                    print('opp_constraints: ', opp_constraints)
                    resp_cat = response_category_team[opp_idx_unique]
                    print('resp_cat: ', resp_cat)
                    opp_set = set(opp_constraints)
                    max_count = 0
                    max_c = []
                    for opp_c in opp_set:
                        count_c = opp_constraints.count(opp_c)
                        if count_c > max_count:
                            max_count = count_c
                            max_c = opp_c
                        elif count_c == max_count:
                            max_c.extend(opp_c)

                    alternate_team_constraints = max_c  # majority rules
                    print('Majority_team_constraints: ', alternate_team_constraints, ' count: ', max_count)

                    # alternate_team_constraints = [opp_constraints[i] for i in len(range(resp_cat)) if resp_cat[i] == 'correct'] # correct constraint
                    # print('Correct_team_constraints: ', alternate_team_constraints)

                    team_constraints = team_knowledge['common_knowledge'].copy()
                    team_constraints.extend(alternate_team_constraints)
                    team_constraints = BEC_helpers.remove_redundant_constraints(team_constraints, params.weights['val'], params.step_cost_flag)
                    team_knowledge['common_knowledge'] = team_constraints.copy()

                    team_knowledge = team_helpers.update_team_knowledge(team_knowledge, [], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = ['joint_knowledge'])
                    particles_team['common_knowledge'].reset(team_knowledge['common_knowledge'][0]) # Note that the common knowledge is updated based on the minimum common constraint which is [0, 0, 0]
                    particles_team['common_knowledge'].update(team_knowledge['common_knowledge'][1:])
                    particles_team['joint_knowledge'].update_jk(test_constraints_team) # Note that the joint knowledge is updated based on the opposing constraints which in this case is the entire space.
                
                
                test_no += 1
                

            ##########################################################################

            loop_count += 1

            # TODO: check if unit knowledge is sufficient to move on to the next unit (unit learning goal reached)
            unit_learning_goal_reached = team_helpers.check_unit_learning_goal_reached(team_knowledge, min_unit_constraints)
            print('unit_learning_goal_reached: ', unit_learning_goal_reached)

            # Temp: Arbitarily set when next unit should be checked
            next_unit_flag = False
            if loop_count - next_unit_loop_id > 4 or unit_learning_goal_reached:
                next_unit_flag = True
                next_unit_loop_id = loop_count
            

            # update variables
            loop_vars = copy.deepcopy(demo_vars_template)
            loop_vars['run_no'] = run_no
            loop_vars['demo_strategy'] = demo_strategy
            loop_vars['variable_filter'] = variable_filter
            loop_vars['loop_count'] = loop_count
            loop_vars['summary_count'] = summary_count
            loop_vars['min_BEC_constraints'] = min_BEC_constraints
            loop_vars['unit_constraints'] = unit_constraints
            loop_vars['team_knowledge_expected'] = team_knowledge_expected
            loop_vars['particles_team_expected'] = particles_team_expected
            loop_vars['unit_knowledge_level_expected'] = team_helpers.calc_knowledge_level(team_knowledge_expected, min_unit_constraints)
            loop_vars['BEC_knowledge_level_expected'] = team_helpers.calc_knowledge_level(team_knowledge_expected, min_BEC_constraints)
            loop_vars['response_type'] = response_type
            loop_vars['response_distribution'] = response_distribution_list[resp_no-1]
            loop_vars['test_constraints'] = test_constraints_team_expanded
            loop_vars['opposing_constraints_count'] = opposing_constraints_count  
            loop_vars['final_remedial_constraints'] = remedial_constraints_team_expanded
            loop_vars['N_remedial_tests'] = N_remedial_tests
            loop_vars['team_knowledge'] = team_knowledge
            loop_vars['particles_team'] = particles_team
            loop_vars['unit_knowledge_level'] = team_helpers.calc_knowledge_level(team_knowledge, min_unit_constraints)
            loop_vars['BEC_knowledge_level'] = team_helpers.calc_knowledge_level(team_knowledge, min_BEC_constraints)    
                 
            vars_to_save = vars_to_save.append(loop_vars, ignore_index=True)
            print('unit_knowledge_level_expected: ', loop_vars['unit_knowledge_level_expected'])
            print('unit_knowledge_level: ', loop_vars['unit_knowledge_level'])
            
            # save vars so far (within one session)
            vars_to_save.to_csv('models/' + params.data_loc['BEC'] + '/' + vars_filename + '.csv', index=False)

            with open('models/augmented_taxi2/' + vars_filename + '.pickle', 'wb') as f:
                pickle.dump(vars_to_save, f)

            
            # Update variable filter
            if next_unit_flag:
                variable_filter, nonzero_counter, teaching_complete_flag = team_helpers.check_and_update_variable_filter(variable_filter = variable_filter, nonzero_counter = nonzero_counter)
                print('Moving to next unit. Updated variable filter: ', variable_filter)

            # if teaching_complete_flag:
                print(colored('Saving session data for run {}.'.format(run_no), 'blue'))

                # save vars so far (end of session)
                vars_to_save.to_csv('models/' + params.data_loc['BEC'] + '/' + vars_filename + '.csv', index=False)

                print('Saved sim data len: ', vars_to_save.shape[0])

                with open('models/augmented_taxi2/' + vars_filename + '.pickle', 'wb') as f:
                    pickle.dump(vars_to_save, f)
                
                prev_summary_len = len(BEC_summary)
            
            else:
            
                # update variables
                prev_summary_len = len(BEC_summary)

                # Update demo particles for next iteration based on actual team knowledge and the demo strategy
                particles_demo = copy.deepcopy(particles_team[knowledge_id])

                # Update expected knowledge with actual knowledge for next iteration
                particles_team_expected = copy.deepcopy(particles_team)
                team_knowledge_expected = copy.deepcopy(team_knowledge)


        else:
            # Update Variable filter and move to next unit, if applicable, if no demos are available for this unit.
            print(colored('No new summaries for this unit...!!', 'red'))
            unit_learning_goal_reached = True

            # maybe double check knowledge metric
            # unit_learning_goal_reached2 = team_helpers.check_unit_learning_goal_reached(team_knowledge, min_unit_constraints)
            # print('Measured unit learning goal staus: ', unit_learning_goal_reached2)
            
            # TODO: Update variable filter
            if unit_learning_goal_reached:
                variable_filter, nonzero_counter, teaching_complete_flag = team_helpers.check_and_update_variable_filter(variable_filter = variable_filter, nonzero_counter = nonzero_counter)

            if teaching_complete_flag:
                print(colored('Teaching completed for run: ', run_no,'. Saving session data...', 'blue'))
                
                # save vars so far (end of session)
                vars_to_save.to_csv('models/' + params.data_loc['BEC'] + '/vars_to_save.csv', index=False)
                
                print('Saved sim data len: ', vars_to_save.shape[0])    

                with open('models/augmented_taxi2/vars_to_save.pickle', 'wb') as f:
                    pickle.dump(vars_to_save, f)

    # return vars_to_save






if __name__ == "__main__":
    pool = Pool(min(params.n_cpu, 60))
    os.makedirs('models/' + params.data_loc['base'], exist_ok=True)
    os.makedirs('models/' + params.data_loc['BEC'], exist_ok=True)


    ## run_reward_teaching
    run_reward_teaching(params, pool, demo_strategy = 'common_knowledge', response_type = 'simulated', response_distribution_list = ['mixed', 'correct', 'mixed', 'incorrect', 'correct', 'correct', 'correct', 'correct'], run_no = 1, viz_flag=True, vars_filename = 'workshop_data')
    # vars_to_save = run_reward_teaching(params, pool)

    
    pool.close()
    pool.join()



# save files
    # if len(BEC_summary) > 0:
    #     with open('models/' + data_loc + '/teams_BEC_summary.pickle', 'wb') as f:
    #         pickle.dump((BEC_summary, visited_env_traj_idxs, particles_team), f)










