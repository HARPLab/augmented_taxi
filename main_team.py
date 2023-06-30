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
from policy_summarization import particle_filter as pf
import matplotlib as mpl
import teams.teams_helpers as team_helpers
mpl.rcParams['figure.facecolor'] = '1.0'
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'large'

import random


############################################







    



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
    
    # sample particles for human models
    team_prior, particles_team = team_helpers.sample_team_pf(params.team_size, params.BEC['n_particles'], params.weights['val'], params.step_cost_flag, team_prior = params.team_prior)

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
    tt_loop_counter = 0
    no_info_flag = False
    demo_reset_flag = False
    team_knowledge = team_prior.copy() # team_prior calculated from team_helpers.sample_team_pf also has the aggregated knowledge from individual priors
    visualize_pf_transition = True

    # TODO: Teaching-testing loop
    while not teaching_complete_flag:

        # TODO: Choose particles based on demo strategy
        if tt_loop_counter == 0 or demo_reset_flag == True:
            prior, particles_demo = team_helpers.particles_for_demo_strategy(params.demo_strategy, team_knowledge, params.BEC['n_particles'])



        #### Individual demos
        # TODO: obtain a BEC demonstration for this unit
        try:
            with open('models/' + params.data_loc['BEC'] + '/current_BEC_summary_trial.pickle', 'rb') as f:
                current_BEC_summary, current_visited_env_traj_idxs, particles_demo = pickle.load(f)
        except:
            current_BEC_summary, current_visited_env_traj_idxs, particles_demo = team_helpers.obtain_team_summary(params.data_loc['BEC'], min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, params.weights['val'], params.step_cost_flag, 
                                                                                                              pool, params.BEC['n_human_models'], consistent_state_count, params.BEC['n_train_demos'], prior, particles_demo)
               
        # append to BEC summary
        BEC_summary.append(current_BEC_summary)
        visited_env_traj_idxs.append(current_visited_env_traj_idxs)
        ###############


        # TODO: Show demonstrations
        unit_constraints, running_variable_filter_unit = team_helpers.show_demonstrations(current_BEC_summary, particles_demo, params.mdp_class, params.weights['val'], visualize_pf_transition)
    
    
        # Conduct tests
        # obtain the constraints conveyed by the unit's demonstrations
        min_constraints = BEC_helpers.remove_redundant_constraints(unit_constraints, params.weights['val'], params.step_cost_flag)
        # obtain the diagnostic tests that will test the human's understanding of the unit's constraints
        preliminary_tests, visited_env_traj_idxs = BEC.obtain_diagnostic_tests(params.data_loc['BEC'], current_BEC_summary[0], current_visited_env_traj_idxs, min_constraints, min_subset_constraints_record, traj_record, traj_features_record, running_variable_filter_unit, mdp_features_record)
        # print(preliminary_tests[0])

        

        
        # query the human's response to the diagnostic tests
        for test in preliminary_tests:
            
            test_mdp = test[0]
            opt_traj = test[1]
            test_constraints = test[3]
            test_history = [test] # to ensure that remedial demonstrations and tests are visually simple/similar and complex/different, respectively

            # TEMP: show the same test for each person and get test responses of each person in the team
            for p_id in range(params.team_size):
                print("Here is a diagnostic test for this unit for player ", p_id)
                human_traj, human_history = test_mdp.visualize_interaction(keys_map=params.keys_map) # the latter is simply the gridworld locations of the agent

                human_feature_count = test_mdp.accumulate_reward_features(human_traj, discount=True)
                opt_feature_count = test_mdp.accumulate_reward_features(opt_traj, discount=True)

                if (human_feature_count == opt_feature_count).all():
                    print("You got the diagnostic test right")

                    particles_demo.update(test_constraints)
                    if visualize_pf_transition:
                        team_helpers.visualize_transition(test_constraints, particles_demo, params.mdp_class, params.weights['val'])

                else:
                    print("You got the diagnostic test wrong. Here's the correct answer")
                    failed_BEC_constraint = opt_feature_count - human_feature_count
                    print("Failed BEC constraint: {}".format(failed_BEC_constraint))

                    particles_demo.update([-failed_BEC_constraint])
                    if visualize_pf_transition:
                        team_helpers.visualize_transition([-failed_BEC_constraint], particles_demo, params.mdp_class, params.weights['val'])

                    test_mdp.visualize_trajectory_comparison(opt_traj, human_traj)


        # TODO: Update human models and checking if the learning goal is met



        # Remedial demo-test loop

            # TODO: Generate remedial demonstration
            # print("Here is a remedial demonstration that might be helpful")

            # remedial_instruction, visited_env_traj_idxs = BEC.obtain_remedial_demonstrations(params.data_loc['BEC'], pool, particles_demo, params.BEC['n_human_models'], failed_BEC_constraint, min_subset_constraints_record, env_record, traj_record, traj_features_record, test_history, visited_env_traj_idxs, running_variable_filter_unit, mdp_features_record, consistent_state_count, params.weights['val'], params.step_cost_flag, n_human_models_precomputed= params.BEC['n_human_models_precomputed'])
            # remedial_mdp, remedial_traj, _, remedial_constraint, _ = remedial_instruction[0]
            # remedial_mdp.visualize_trajectory(remedial_traj)
            # test_history.extend(remedial_instruction)

            # particles_demo.update([remedial_constraint])
            # if visualize_pf_transition:
            #     BEC_viz.visualize_pf_transition([remedial_constraint], particles_demo, params.mdp_class, params.weights['val'])

            # with open('models/' + params.data_loc['BEC'] + '/remedial_instruction.pickle', 'wb') as f:
            #     pickle.dump(remedial_instruction, f)

            # TODO: Show remedial demonstration


            # Remedial test loop

            # TODO: Conduct remedial test

            # TODO: Update human models and checking if the learning goal is met; if not, loop back to remedial test loop







        # TODO: Update variable filter
        if no_info_flag:
            variable_filter, nonzero_counter, teaching_complete_flag = team_helpers.check_and_update_variable_filter(variable_filter = variable_filter, nonzero_counter = nonzero_counter, no_info_flag = no_info_flag)

        # Temp to stop loop
        teaching_complete_flag = True
        


# save files
    # if len(BEC_summary) > 0:
    #     with open('models/' + data_loc + '/teams_BEC_summary.pickle', 'wb') as f:
    #         pickle.dump((BEC_summary, visited_env_traj_idxs, particles_team), f)










