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


############################################






def calc_expected_learning(team_knowledge_expected, particles_team_expected, min_BEC_constraints, new_constraints, params, summary_count):


    team_knowledge_expected = team_helpers.update_team_knowledge(team_knowledge_expected, new_constraints, params.team_size,  params.weights['val'], params.step_cost_flag)

    p = 1
    while p <= params.team_size:
        member_id = 'p' + str(p)
        particles_team_expected[member_id].update(new_constraints)
        team_helpers.visualize_transition(new_constraints, particles_team_expected[member_id], params.mdp_class, params.weights['val'], text = 'Expected knowledge change for unit ' + str(summary_count+1) + ' for player ' + member_id)  
        p += 1
            
    # Update common knowledge model
    particles_team_expected['common_knowledge'].update(new_constraints)
    team_helpers.visualize_transition(new_constraints, particles_team_expected['common_knowledge'], params.mdp_class, params.weights['val'], text = 'Expected knowledge change for unit' + str(summary_count+1) + ' for common knowledge')
    
    # Update joint knowledge model
    particles_team_expected['joint_knowledge'].update_jk(team_knowledge_expected['joint_knowledge'])
    team_helpers.visualize_transition(new_constraints, particles_team_expected['joint_knowledge'], params.mdp_class, params.weights['val'], text = 'Expected knowledge change for unit ' + str(summary_count+1) + ' for joint knowledge',  demo_strategy = 'joint_knowledge')


    print('Expected knowledge after seeing unit demonstrations: ', team_helpers.calc_knowledge_level(team_knowledge_expected, min_BEC_constraints) )


    return team_knowledge_expected, particles_team_expected










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


    print('Value for consistent state count is: ', consistent_state_count)


    # sample particles for human models
    team_prior, particles_team = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_prior = params.team_prior)
    
    print('Team prior: ', team_prior)

    n = 1
    for member_id in particles_team:
        print(colored('entropy of p' + str(n) + ': {}'.format(particles_team[member_id].calc_entropy()), 'blue'))
        n += 1
    
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
    summary_count = 0
    summary_id = 0
    no_info_flag = False
    demo_reset_flag = False
    team_knowledge = team_prior.copy() # team_prior calculated from team_helpers.sample_team_pf also has the aggregated knowledge from individual priors
    visualize_pf_transition = True

    # for debugging
    particles_team_expected = particles_team.copy()
    print('particles_team_expected: ', particles_team_expected)
    team_knowledge_expected = team_knowledge.copy()


    # demo_strategy = params.demo_strategy
    
    demo_strategy = 'common_knowledge'
    # demo_strategy = 'joint_knowledge'
    # demo_strategy = 'individual_knowledge_low'
    # demo_strategy = 'individual_knowledge_high'

    print('Demo strategy: ', demo_strategy)


    # TODO: Unitwise teaching-testing loop
    while not teaching_complete_flag:

        # TODO: Sample particles from the corresponding human/team knowledge based on the demo strategy
        # print(team_knowledge)
        if summary_count == 0 or demo_reset_flag == True:
            particles_demo = team_helpers.particles_for_demo_strategy(demo_strategy, team_knowledge, particles_team, params.team_size, params.weights['val'], params.step_cost_flag, params.BEC['n_particles'], min_BEC_constraints)

        # print(particles_demo)
        # particles_test = particles_demo.copy()

        prev_summary_len = len(BEC_summary)

        #### Individual demos
        # TODO: obtain a BEC demonstration for this unit
        try:
            with open('models/' + params.data_loc['BEC'] + '/BEC_summary.pickle', 'rb') as f:
                BEC_summary, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = pickle.load(f)
        except:
            BEC_summary, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo = team_helpers.obtain_team_summary(params.data_loc['BEC'], min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, params.weights['val'], params.step_cost_flag, 
                                                                                                              pool, params.BEC['n_human_models'], consistent_state_count, params.BEC['n_train_demos'], prior, particles_demo, variable_filter, nonzero_counter, BEC_summary, summary_count, min_BEC_constraints_running, visited_env_traj_idxs)
            # # save the first set of summaries for quick testing
            # if len(BEC_summary) > 0 and prev_summary_len == 0:
            #     with open('models/' + params.data_loc['BEC'] + '/BEC_summary.pickle', 'wb') as f:
            #         pickle.dump((BEC_summary, min_BEC_constraints_running, visited_env_traj_idxs, particles_demo), f)    
            # break
        ###############


        # TODO: Demo-test-remedial-test loop if new summary is obtained
        # print(BEC_summary)

        if len(BEC_summary) > prev_summary_len:
            unit_constraints, running_variable_filter_unit = team_helpers.show_demonstrations(BEC_summary[-1], particles_demo, params.mdp_class, params.weights['val'], visualize_pf_transition, summary_count)


            print('Variable filter:', variable_filter)
            print('running variable filter:', running_variable_filter_unit)

            # For debugging. Visualize the expected particles
            team_knowledge_expected, particles_team_expected = calc_expected_learning(team_knowledge_expected, particles_team_expected, min_BEC_constraints, unit_constraints, params, summary_count)


            # Conduct tests
            # obtain the constraints conveyed by the unit's demonstrations
            min_constraints = BEC_helpers.remove_redundant_constraints(unit_constraints, params.weights['val'], params.step_cost_flag)
            # obtain the diagnostic tests that will test the human's understanding of the unit's constraints
            preliminary_tests, visited_env_traj_idxs = BEC.obtain_diagnostic_tests(params.data_loc['BEC'], BEC_summary[-1], visited_env_traj_idxs, min_constraints, min_subset_constraints_record, traj_record, traj_features_record, running_variable_filter_unit, mdp_features_record)
            # print(preliminary_tests[0])

            test_constraints_team = []

            
            # query the human's response to the diagnostic tests
            for test in preliminary_tests:
                
                test_mdp = test[0]
                opt_traj = test[1]
                test_constraints = test[3]
                test_history = [test] # to ensure that remedial demonstrations and tests are visually simple/similar and complex/different, respectively

                # TEMP: show the same test for each person and get test responses of each person in the team
                p = 1
                while p <= params.team_size:
                    member_id = 'p' + str(p)
                    print("Here is a diagnostic test for this unit for player ", p)
                    human_traj, human_history = test_mdp.visualize_interaction(keys_map=params.keys_map) # the latter is simply the gridworld locations of the agent

                    human_feature_count = test_mdp.accumulate_reward_features(human_traj, discount=True)
                    opt_feature_count = test_mdp.accumulate_reward_features(opt_traj, discount=True)

                    if (human_feature_count == opt_feature_count).all():
                        print("You got the diagnostic test right")

                        particles_team[member_id].update(test_constraints) # update individual knowledge based on test response
                        if visualize_pf_transition:
                            team_helpers.visualize_transition(test_constraints, particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Test of Unit ' + str(summary_count) + ' for player ' + member_id)

                    else:
                        print("You got the diagnostic test wrong. Here's the correct answer")
                        failed_BEC_constraint = opt_feature_count - human_feature_count
                        print("Failed BEC constraint: {}".format(failed_BEC_constraint))
                        test_constraints_team.append([-failed_BEC_constraint])

                        particles_team[member_id].update([-failed_BEC_constraint])
                        if visualize_pf_transition:
                            team_helpers.visualize_transition([-failed_BEC_constraint], particles_team[member_id], params.mdp_class, params.weights['val'], text = 'Test of Unit ' + str(summary_count + 1) + ' for player ' + member_id)

                        test_mdp.visualize_trajectory_comparison(opt_traj, human_traj)
                    
                    p += 1
                
                print('test_constraints_team: ', test_constraints_team)
                test_constraints_team_expanded = [x for x in test_constraints_team]
                print('test_constraints_team_expanded: ', test_constraints_team_expanded)

                # Update common knowledge model
                particles_team['common_knowledge'].update(test_constraints_team_expanded)
                if visualize_pf_transition:
                    team_helpers.visualize_transition(test_constraints_team_expanded, particles_team['common_knowledge'], params.mdp_class, params.weights['val'], text = 'Test of Unit ' + str(summary_count + 1) + ' for common knowledge')
                
                # Update joint knowledge model
                particles_team['joint_knowledge'].update_jk(test_constraints_team)
                if visualize_pf_transition:
                    team_helpers.visualize_transition(test_constraints_team_expanded, particles_team['joint_knowledge'], params.mdp_class, params.weights['val'], text = 'Test of Unit ' + str(summary_count + 1) + ' for joint knowledge',  demo_strategy = 'joint_knowledge')


                # # update knowledge for strategy
                # particles_test.update( [-x for x in failed_BEC_constraints_team])
                # if visualize_pf_transition:
                #         team_helpers.visualize_transition([ -x for x in failed_BEC_constraints_team], particles_demo, params.mdp_class, params.weights['val'])




                #####################################################################

                # Remedial demo-test loop

                # TODO: Generate remedial demonstration

                # if remedial_test_flag: 

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

                #####################################################################
        


        summary_count += 1

        # Update demo particles for next iteration
        if demo_strategy == 'common_knowledge' or 'joint_knowledge':
            particles_demo = particles_team[demo_strategy]
        

        print('particles_demo: ', particles_demo)


        # TODO: Update variable filter
        if no_info_flag:
            variable_filter, nonzero_counter, teaching_complete_flag = team_helpers.check_and_update_variable_filter(variable_filter = variable_filter, nonzero_counter = nonzero_counter, no_info_flag = no_info_flag)


        # Temp to stop loop
        teaching_complete_flag = True
        print(colored('Teaching completed...', 'blue'))
        


# save files
    # if len(BEC_summary) > 0:
    #     with open('models/' + data_loc + '/teams_BEC_summary.pickle', 'wb') as f:
    #         pickle.dump((BEC_summary, visited_env_traj_idxs, particles_team), f)










