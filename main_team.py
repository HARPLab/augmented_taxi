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


    team_helpers.visualize_transition(new_constraints, particles_team_expected['joint_knowledge'], params.mdp_class, params.weights['val'], text = 'Expected knowledge change for set ' + str(loop_count+1) + ' for joint knowledge',  demo_strategy = 'joint_knowledge')

    print('min_BEC_constraints: ', min_BEC_constraints)
    print('Expected unit knowledge after seeing unit demonstrations: ', team_helpers.calc_knowledge_level(team_knowledge_expected, new_constraints) )
    print('Expected absolute knowledge after seeing unit demonstrations: ', team_helpers.calc_knowledge_level(team_knowledge_expected, min_BEC_constraints) )


    return team_knowledge_expected, particles_team_expected




def run_remedial_loop():

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


        # # # Method 2: Generate remedial demonstration for each individual
        # # p=1
        # # for failed_BEC_cnst in failed_BEC_constraints_team:
        # #     member_id = 'p' + str(p)
        # #     remedial_demos, visited_env_traj_idxs = BEC.obtain_remedial_demonstrations(params.data_loc['BEC'], pool, particles_team[member_id], params.BEC['n_human_models'], failed_BEC_cnst, min_subset_constraints_record, env_record, traj_record, traj_features_record, test_history, visited_env_traj_idxs, running_variable_filter_unit, mdp_features_record, consistent_state_count, params.weights['val'], params.step_cost_flag, n_human_models_precomputed= params.BEC['n_human_models_precomputed'])

        # #     # TODO: Show remedial demonstration
        # #     for demo in remedial_demos:
        # #         demo[0].visualize_trajectory(demo[1])
        # #         test_history.extend(demo)

        # #         particles_demo.update([demo[3]])
        # #         # print('remedial demo :', demo)
        # #         # print('remedial demo constraints:', demo[3])
        # #         if visualize_pf_transition:
        # #             team_helpers.visualize_transition(demo[3], particles_demo, params.mdp_class, params.weights['val'])

        # #         with open('models/' + params.data_loc['BEC'] + '/remedial_instruction.pickle', 'wb') as f:
        # #             pickle.dump(demo, f)




        # # remedial_instruction, visited_env_traj_idxs = BEC.obtain_remedial_demonstrations(params.data_loc['BEC'], pool, particles_demo, params.BEC['n_human_models'], failed_BEC_constraint, min_subset_constraints_record, env_record, traj_record, traj_features_record, test_history, visited_env_traj_idxs, running_variable_filter_unit, mdp_features_record, consistent_state_count, params.weights['val'], params.step_cost_flag, n_human_models_precomputed= params.BEC['n_human_models_precomputed'])
        # # remedial_mdp, remedial_traj, _, remedial_constraint, _ = remedial_instruction[0]
        # # remedial_mdp.visualize_trajectory(remedial_traj)
        # # test_history.extend(remedial_instruction)

        # # particles_demo.update([remedial_constraint])
        # # if visualize_pf_transition:
        # #     BEC_viz.visualize_pf_transition([remedial_constraint], particles_demo, params.mdp_class, params.weights['val'])

        # # with open('models/' + params.data_loc['BEC'] + '/remedial_instruction.pickle', 'wb') as f:
        # #     pickle.dump(remedial_instruction, f)




        #     # # TODO: Conduct remedial test
        #     # remedial_test_correct = False

        #     # print("Here is a remedial test to see if you've correctly learned the lesson")
        #     # while not remedial_test_correct:

        #     remedial_test, visited_env_traj_idxs = BEC.obtain_remedial_demonstrations(data_loc, pool,
        #                                                                                     particles,
        #                                                                                     n_human_models,
        #                                                                                     failed_BEC_constraint,
        #                                                                                     min_subset_constraints_record,
        #                                                                                     env_record,
        #                                                                                     traj_record,
        #                                                                                     traj_features_record,
        #                                                                                     test_history,
        #                                                                                     visited_env_traj_idxs,
        #                                                                                     running_variable_filter,
        #                                                                                     mdp_features_record,
        #                                                                                     consistent_state_count,
        #                                                                                     weights,
        #                                                                                     step_cost_flag, type='testing', n_human_models_precomputed=n_human_models_precomputed)

        #     remedial_mdp, remedial_traj, _, _, _ = remedial_test[0]
        #     test_history.extend(remedial_test)

        #     human_traj, human_history = remedial_mdp.visualize_interaction(
        #         keys_map=params.keys_map)  # the latter is simply the gridworld locations of the agent
        #     # with open('models/' + data_loc + '/human_traj.pickle', 'wb') as f:
        #     #     pickle.dump((human_traj, human_history), f)
        #     # with open('models/' + data_loc + '/human_traj.pickle', 'rb') as f:
        #     #     human_traj, human_history = pickle.load(f)

        #     human_feature_count = remedial_mdp.accumulate_reward_features(human_traj, discount=True)
        #     opt_feature_count = remedial_mdp.accumulate_reward_features(remedial_traj, discount=True)

        #     if (human_feature_count == opt_feature_count).all():
        #         print("You got the remedial test correct")
        #         remedial_test_correct = True

        #         particles.update([failed_BEC_constraint])
        #         if visualize_pf_transition:
        #             BEC_viz.visualize_pf_transition([failed_BEC_constraint], particles, mdp_class, weights)
        #     else:
        #         failed_BEC_constraint = opt_feature_count - human_feature_count
        #         print("You got the remedial test wrong. Here's the correct answer")
        #         print("Failed BEC constraint: {}".format(failed_BEC_constraint))
        #         remedial_mdp.visualize_trajectory_comparison(remedial_traj, human_traj)

        #         particles.update([-failed_BEC_constraint])
        #         if visualize_pf_transition:
        #             BEC_viz.visualize_pf_transition([-failed_BEC_constraint], particles, mdp_class, weights)

        # # TODO: Update human models and checking if the learning goal is met; if not, loop back to remedial test loop

    return 1


if __name__ == "__main__":
    pool = Pool(min(params.n_cpu, 60))
    os.makedirs('models/' + params.data_loc['base'], exist_ok=True)
    os.makedirs('models/' + params.data_loc['BEC'], exist_ok=True)

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

    # # Check for cached counterfactuals demos (Skip for now)
    # try:
    #     with open('models/' + params.data_loc['BEC'] + '/teams_BEC_summary.pickle', 'rb') as f:
    #         BEC_summary, visited_env_traj_idxs, particles_team = pickle.load(f)
    #     print('Reading cached BEC summary data')
    #     print(BEC_summary)
    # except:
    
    # print('Base constraints optimal trajectories:', traj_record)


    # print('Value for consistent state count is: ', consistent_state_count)


    # sample particles for human models
    team_prior, particles_team = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_prior = params.team_prior)
    
    print('Team prior: ', team_prior)
    team_helpers.visualize_team_knowledge(particles_team, params.mdp_class, weights=params.weights['val'], text='Team prior')

    for member_id in particles_team:
        print(colored('entropy of ' + str(member_id) + ': {}'.format(particles_team[member_id].calc_entropy()), 'blue'))
    
        # debug: visualize particles
        # BEC_viz.visualize_pf_transition(params.team_prior[member_id], particles_team[member_id], params.mdp_class, params.weights['val'])

    # debug:
    # for member_id in particles_team:
    #     BEC_viz.visualize_pf_transition(params.team_prior[member_id], pf.Particles(sampled_team_models[member_id]), params.mdp_class, params.weights['val'])

    # unit (set of knowledge components) selection
    variable_filter, nonzero_counter, teaching_complete_flag = team_helpers.check_and_update_variable_filter(min_subset_constraints_record, initialize_filter_flag=True)

    
    BEC_summary = []
    visited_env_traj_idxs = []
    min_BEC_constraints_running = []
    summary_count = 0 # this is actually the number of demos and not the number of units
    no_info_flag = False
    demo_reset_flag = False
    team_knowledge = copy.deepcopy(team_prior) # team_prior calculated from team_helpers.sample_team_pf also has the aggregated knowledge from individual priors
    visualize_pf_transition = True
    prev_summary_len = 0

    plt.rcParams['figure.figsize'] = [10, 6]

    # for debugging
    particles_team_expected = copy.deepcopy(particles_team)
    team_knowledge_expected = copy.deepcopy(team_knowledge)

    # for testing design choices (is remedial demo needed)
    remedial_test_flag = False
    demo_vars_template = {'demo_strategy': None,
                          'variable_filter': None,
                          'event_type': None,
                          'loop_count': None,  
                            'summary_count': None,
                            'unit_constraints': None,
                            'team_knowledge_expected': None,
                            'particles_team_expected': None,
                            'unit_knowledge_level_expected': None,
                            'BEC_knowledge_level_expected': None,
                            'test_constraints': None,
                            'team_knowledge': None,
                            'particles_team': None,
                            'unit_knowledge_level': None,
                            'BEC_knowledge_level': None
                            }
    

    vars_to_save = pd.DataFrame(columns=demo_vars_template.keys())

    # demo_strategy = params.demo_strategy
    
    # for quick testing
    demo_strategy = 'common_knowledge'
    # demo_strategy = 'joint_knowledge'
    # demo_strategy = 'individual_knowledge_low'
    # demo_strategy = 'individual_knowledge_high'

    print('Demo strategy: ', demo_strategy)

    # initialize human models for demo generation
    knowledge_id, particles_demo = team_helpers.particles_for_demo_strategy(demo_strategy, team_knowledge, particles_team, params.team_size, params.weights['val'], params.step_cost_flag, params.BEC['n_particles'], min_BEC_constraints)


    ################################################
    loop_count = 0
    next_unit_loop_id = 0
    unit_learning_goal_reached = True
    # WIP: Unitwise teaching-testing loop
    while not teaching_complete_flag:

        # # Sample particles from the corresponding human/team knowledge based on the demo strategy
        # if demo_reset_flag == True:
        #     knowledge_id, particles_demo = team_helpers.particles_for_demo_strategy(demo_strategy, team_knowledge, particles_team, params.team_size, params.weights['val'], params.step_cost_flag, params.BEC['n_particles'], min_BEC_constraints)


        # Obtain BEC summary for a new unit (skip if its the 1st unit)
        print('Summary count: ', summary_count)
        if summary_count == 0:
            print(params.data_loc['BEC'] )
            try:
                print('Trying to open existing BEC summary file...')
                with open('models/' + params.data_loc['BEC'] + '/BEC_summary_initial.pickle', 'rb') as f:
                    BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = pickle.load(f)
            except:
                print('Starting summary generation for 1st unit..')
                BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = team_helpers.obtain_team_summary(params.data_loc['BEC'], min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, params.weights['val'], params.step_cost_flag, 
                                                                                                                    pool, params.BEC['n_human_models'], consistent_state_count, params.BEC['n_train_demos'], particles_demo, variable_filter, nonzero_counter, BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs)
                print('Ended summary generation for 1st unit..')
                if len(BEC_summary) > 0:
                    with open('models/' + params.data_loc['BEC'] + '/BEC_summary_initial.pickle', 'wb') as f:
                        pickle.dump((BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo), f)
            
            
        else:
            print('Starting summary generation for unit no.  ' + str(loop_count + 1) )
            BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = team_helpers.obtain_team_summary(params.data_loc['BEC'], min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, params.weights['val'], params.step_cost_flag, 
                                                                                                                pool, params.BEC['n_human_models'], consistent_state_count, params.BEC['n_train_demos'], particles_demo, variable_filter, nonzero_counter, BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs)
            print('Ended summary generation for unit no.  ' + str(loop_count + 1) )
        


        # TODO: Demo-test-remedial-test loop if new summary is obtained
        print('BEC summary length: ', len(BEC_summary))
        print('Previous summary length: ', prev_summary_len)

        # check for any new summary
        if len(BEC_summary) > prev_summary_len:
            unit_constraints, running_variable_filter_unit = team_helpers.show_demonstrations(BEC_summary[-1], particles_demo, params.mdp_class, params.weights['val'], visualize_pf_transition, loop_count)

            print('Variable filter:', variable_filter)
            print('running variable filter:', running_variable_filter_unit)

            # obtain the constraints conveyed by the unit's demonstrations
            min_unit_constraints = BEC_helpers.remove_redundant_constraints(unit_constraints, params.weights['val'], params.step_cost_flag)


            # For debugging. Visualize the expected particles transition
            team_knowledge_expected, particles_team_expected = calc_expected_learning(team_knowledge_expected, particles_team_expected, min_BEC_constraints, min_unit_constraints, params, loop_count)

            # team_helpers.visualize_team_knowledge(particles_team_expected, params.mdp_class, weights=params.weights['val'], text='Team expected knowledge after unit ' + str(loop_count))

            ## Conduct tests for the unit
            # obtain the diagnostic tests that will test the human's understanding of the unit's constraints
            print('Getting diagnostic tests for unit ' + str(loop_count) + '...')
            print('visited_env_traj_idxs: ', visited_env_traj_idxs)
            preliminary_tests, visited_env_traj_idxs = BEC.obtain_diagnostic_tests(params.data_loc['BEC'], BEC_summary[-1], visited_env_traj_idxs, min_unit_constraints, min_subset_constraints_record, traj_record, traj_features_record, running_variable_filter_unit, mdp_features_record)

            
            # query the human's response to the diagnostic tests
            test_no = 1
            for test in preliminary_tests:
                test_constraints_team = []

                test_mdp = test[0]
                opt_traj = test[1]
                test_constraints = test[3]
                test_history = [test] # to ensure that remedial demonstrations and tests are visually simple/similar and complex/different, respectively

                # Show the same test for each person and get test responses of each person in the team
                p = 1
                while p <= params.team_size:
                    member_id = 'p' + str(p)
                    print("Here is a diagnostic test for this unit for player ", p)
                    human_traj, human_history = test_mdp.visualize_interaction(keys_map=params.keys_map) # the latter is simply the gridworld locations of the agent

                    human_feature_count = test_mdp.accumulate_reward_features(human_traj, discount=True)
                    opt_feature_count = test_mdp.accumulate_reward_features(opt_traj, discount=True)

                    if (human_feature_count == opt_feature_count).all():
                        print("You got the diagnostic test right")

                        test_constraints_team.append(test_constraints)
                        team_knowledge = team_helpers.update_team_knowledge(team_knowledge, test_constraints, params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = [member_id])
                        particles_team[member_id].update(test_constraints) # update individual knowledge based on test response
                        
                        if visualize_pf_transition:
                            team_helpers.visualize_transition(test_constraints, particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Test ' + str(test_no) + ' of Unit ' + str(loop_count) + ' for player ' + member_id)

                    else:
                        print("You got the diagnostic test wrong. Here's the correct answer")
                        failed_BEC_constraint = opt_feature_count - human_feature_count
                        print("Failed BEC constraint: {}".format(failed_BEC_constraint))

                        unit_learning_goal_reached = False

                        test_constraints_team.append([-failed_BEC_constraint])
                        print('Current team knowledge: ', team_knowledge)
                        team_knowledge = team_helpers.update_team_knowledge(team_knowledge, [-failed_BEC_constraint], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = [member_id])
                        particles_team[member_id].update([-failed_BEC_constraint])
                        
                        if visualize_pf_transition:
                            team_helpers.visualize_transition([-failed_BEC_constraint], particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Test ' + str(test_no) + ' of Unit ' + str(loop_count) + ' for player ' + member_id)

                        test_mdp.visualize_trajectory_comparison(opt_traj, human_traj)
                    
                    p += 1
                
                print('test_constraints_team: ', test_constraints_team)
                test_constraints_team_expanded = []
                for test_constraints in test_constraints_team:
                    test_constraints_team_expanded.extend(test_constraints)
                print('test_constraints_team_expanded: ', test_constraints_team_expanded)

                # Update common knowledge and joint knowledge
                team_knowledge = team_helpers.update_team_knowledge(team_knowledge, [], params.team_size, params.weights['val'], params.step_cost_flag, knowledge_to_update = ['common_knowledge', 'joint_knowledge'])
                particles_team['common_knowledge'].update(test_constraints_team_expanded)
                particles_team['joint_knowledge'].update_jk(test_constraints_team)
                if visualize_pf_transition:
                    print('test_constraints_team_expanded: ', test_constraints_team_expanded)
                    team_helpers.visualize_transition(test_constraints_team_expanded, particles_team['common_knowledge'], params.mdp_class, params.weights['val'], text = 'Test ' + str(test_no) + ' of Unit ' + str(loop_count) + ' for common knowledge')
                    team_helpers.visualize_transition(test_constraints_team_expanded, particles_team['joint_knowledge'], params.mdp_class, params.weights['val'], text = 'Test ' + str(test_no) + ' of Unit ' + str(loop_count) + ' for joint knowledge',  demo_strategy = 'joint_knowledge')

                
                test_no += 1
                
                ####################################################################
                # Remedial demo-test loop
                if remedial_test_flag: 
                    
                    x = 1
                    # run_remedial_loop()
                ####################################################################
        
            
            ## Alternate method that does not use remedial demo-test loop
            if not remedial_test_flag:               
                
                loop_count += 1

                # TODO: check if unit knowledge is sufficient to move on to the next unit (unit learning goal reached)
                # unit_learning_goal_reached = team_helpers.check_unit_learning_goal_reached(team_knowledge, min_unit_constraints)


                # Temp: Arbitarily set when next unit should be checked
                if loop_count - next_unit_loop_id > 1:
                    unit_learning_goal_reached = True
                    next_unit_loop_id = loop_count
                

                print('unit_learning_goal_reached: ', unit_learning_goal_reached)


                # update variables
                loop_vars = copy.deepcopy(demo_vars_template)
                loop_vars['demo_strategy'] = demo_strategy
                loop_vars['variable_filter'] = variable_filter
                loop_vars['loop_count'] = loop_count
                loop_vars['summary_count'] = summary_count
                loop_vars['unit_constraints'] = unit_constraints
                loop_vars['team_knowledge_expected'] = team_knowledge_expected
                loop_vars['particles_team_expected'] = particles_team_expected
                loop_vars['unit_knowledge_level_expected'] = team_helpers.calc_knowledge_level(team_knowledge_expected, unit_constraints)
                loop_vars['BEC_knowledge_level_expected'] = team_helpers.calc_knowledge_level(team_knowledge_expected, min_BEC_constraints)
                loop_vars['test_constraints'] = test_constraints
                loop_vars['team_knowledge'] = team_knowledge
                loop_vars['particles_team'] = particles_team
                loop_vars['unit_knowledge_level'] = team_helpers.calc_knowledge_level(team_knowledge, unit_constraints)
                loop_vars['BEC_knowledge_level'] = team_helpers.calc_knowledge_level(team_knowledge, min_BEC_constraints)           

                vars_to_save = vars_to_save.append(loop_vars, ignore_index=True)
                print('unit_knowledge_level_expected: ', loop_vars['unit_knowledge_level_expected'])
                print('unit_knowledge_level: ', loop_vars['unit_knowledge_level'])
                # save vars so far
                vars_to_save.to_csv('models/' + params.data_loc['BEC'] + '/vars_to_save.csv', index=False)

                
                # Update variable filter
                if unit_learning_goal_reached:
                    variable_filter, nonzero_counter, teaching_complete_flag = team_helpers.check_and_update_variable_filter(variable_filter = variable_filter, nonzero_counter = nonzero_counter)
                    print('Should be the updated variable filter: ', variable_filter)

                if teaching_complete_flag:
                    print(colored('Teaching completed...', 'blue'))
                    break
                
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
                print(colored('Teaching completed...', 'blue'))
                break


# save files
    # if len(BEC_summary) > 0:
    #     with open('models/' + data_loc + '/teams_BEC_summary.pickle', 'wb') as f:
    #         pickle.dump((BEC_summary, visited_env_traj_idxs, particles_team), f)










