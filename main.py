#!/usr/bin/env python
'''
-----------HEADER-----------
This variation of augmented taxi 2 is used to create an augmented taxi 2 
with multiple passengers that a player can click on in order to choose which 
one they would like to path plan towards.

This code mainly uses the simple_rl framework
most of the folders including BEC are not being used in this code and are only kept
in case future development may require their use.
'''
# Python imports.
import sys
import dill as pickle
import numpy as np
import copy
from termcolor import colored
#import sage.all
#import sage.geometry.polyhedron.base as Polyhedron
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import itertools
from collections import defaultdict
from multiprocessing import Process, Queue, Pool

# Other imports.
sys.path.append("simple_rl")
import params
from simple_rl.agents import FixedPolicyAgent
from simple_rl.planning import ValueIteration
from simple_rl.utils import make_mdp
#from policy_summarization import bayesian_IRL
#from policy_summarization import policy_summarization_helpers as ps_helpers
#from policy_summarization import BEC
#import policy_summarization.multiprocessing_helpers as mp_helpers
#from simple_rl.utils import mdp_helpers
#import policy_summarization.BEC_helpers as BEC_helpers
#import policy_summarization.BEC_visualization as BEC_viz
#from policy_summarization import particle_filter as pf
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = '1.0'
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'large'

#currently only thing being used basically
#=[{'x': 4, 'y': 1, 'dest_x': 3, 'dest_y': 2, 'in_taxi': 0}]
def generate_agent(mdp_class, data_loc, mdp_parameters, passengers=[{'x': 4, 'y': 1, 'dest_x': 3, 'dest_y': 2, 'in_taxi': 0}], visualize=False):
    '''try:
        with open('models/' + data_loc + '/vi_agent.pickle', 'rb') as f:
            mdp_agent, vi_agent = pickle.load(f)
    except:
        cf_data_dir = 'models/' + data_loc
        os.makedirs(cf_data_dir, exist_ok=True)
    '''

    

    mdp_vi_dict = {}
    combined_mdp_passengers = []
    #print(mdp_parameters['passengers'])
    for i in passengers:
        combined_mdp_passengers.append(i)
        mdp_parameters['passengers'] = [i]
        #print(mdp_parameters['passengers'])
        temp_mdp_agent = make_mdp.make_custom_mdp(mdp_class, mdp_parameters)
        mdp_vi_dict[(i['x'],i['y'])]=(temp_mdp_agent)
        #print("mdp_vi_dict:")
        #print((i['x'],i['y']))

    #print(mdp_list)
    #print(vi_agent_list)
    '''
        with open('models/' + data_loc + '/vi_agent.pickle', 'wb') as f:
            pickle.dump((mdp_agent, vi_agent), f)
    '''

    

    # Visualize agent
    if visualize:
        mdp_parameters['passengers'] = combined_mdp_passengers
        combined_mdp_agent = make_mdp.make_custom_mdp(mdp_class, mdp_parameters)
        #print(combined_mdp_agent.get_passengers())
        cell_coords = combined_mdp_agent.visualize_state(combined_mdp_agent.cur_state)
        combined_mdp_agent.reset()

        print(cell_coords)
        if not cell_coords == None:
            #print(cell_coords)
            mdp_agent = mdp_vi_dict[cell_coords]
            #test case: mdp_agent.visualize_state(mdp_agent.cur_state)
            
            vi_agent = ValueIteration(mdp_agent, sample_rate=1)
            vi_agent.run_vi()
            #print(mdp_agent.get_passengers())
            fixed_agent = FixedPolicyAgent(vi_agent.policy)
            mdp_agent.visualize_agent(fixed_agent)
            mdp_agent.reset()  # reset the current state to the initial state
        
        

        #agent visualization:
        '''
        combined_vi_agent = ValueIteration(combined_mdp_agent, sample_rate=1)
        combined_vi_agent.run_vi()
        combined_fixed_agent = FixedPolicyAgent(combined_vi_agent.policy)
        combined_mdp_agent.visualize_agent(combined_fixed_agent, augmented_inputs=True)
        combined_mdp_agent.reset()  # reset the current state to the initial state
        '''
        
        '''
        for mdp,vi_agent in mdp_list,vi_agent_list:
            fixed_agent = FixedPolicyAgent(vi_agent.policy)
            mdp_agent.visualize_agent(fixed_agent, augmented_inputs=True)
            mdp_agent.reset()  # reset the current state to the initial state
            from simple_rl.tasks.taxi.taxi_visualizer import _draw_augmented_state
            mdp_agent.visualize_interaction(keys_map=params.keys_map)
        '''


#functions from old code
'''
def obtain_summary(mdp_class, data_loc, mdp_parameters, weights, step_cost_flag, summary_variant, pool, n_train_demos, BEC_depth, n_human_models, n_particles, prior, posterior, obj_func_proportion, hardcode_envs=False):
    #print(f"data_loc: {data_loc}\nmdp_params: {mdp_parameters}\nweights: {weights}\nn_human_models: {n_human_models}\nn_train_demos: {n_train_demos}\n\n")
    if hardcode_envs:
        # using 4 hardcoded environments
        ps_helpers.obtain_env_policies(mdp_class, data_loc, np.expand_dims(weights, axis=0), mdp_parameters, pool, hardcode_envs=True)
        vi_traj_triplets = []
        for i in range(4):
            env_filename = mp_helpers.lookup_env_filename(data_loc, i)

            with open(env_filename, 'rb') as f:
                wt_vi_traj_env = pickle.load(f)

            mdp = wt_vi_traj_env[0][1].mdp
            agent = wt_vi_traj_env[0][1]
            weights = mdp.weights
            trajectory = mdp_helpers.rollout_policy(mdp, agent)

            vi_traj_triplets.append((i, agent, trajectory))
    else:
        ps_helpers.obtain_env_policies(mdp_class, data_loc, np.expand_dims(weights, axis=0), mdp_parameters, pool)
    try:
        with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
            policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(f)
    except:
        if hardcode_envs:
            # use demo BEC to extract constraints
            policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = BEC.extract_constraints(data_loc, BEC_depth, step_cost_flag, pool, vi_traj_triplets=vi_traj_triplets, print_flag=True)
        else:
            # use policy BEC to extract constraints
            policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = BEC.extract_constraints(data_loc, BEC_depth, step_cost_flag, pool, print_flag=True)
        with open('models/' + data_loc + '/base_constraints.pickle', 'wb') as f:
            pickle.dump((policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count), f)

    try:
        with open('models/' + data_loc + '/BEC_constraints.pickle', 'rb') as f:
            min_BEC_constraints, BEC_lengths_record = pickle.load(f)
    except:
        min_BEC_constraints, BEC_lengths_record = BEC.extract_BEC_constraints(policy_constraints, min_subset_constraints_record, env_record, weights, step_cost_flag, pool)

        with open('models/' + data_loc + '/BEC_constraints.pickle', 'wb') as f:
            pickle.dump((min_BEC_constraints, BEC_lengths_record), f)

    try:
        with open('models/' + data_loc + '/BEC_summary.pickle', 'rb') as f:
            BEC_summary, visited_env_traj_idxs, particles = pickle.load(f)
    except:
        # SCOT_summary = BEC.obtain_SCOT_summaries(data_loc, summary_variant, min_BEC_constraints, BEC_lengths_record, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag)
        # initialize particle filter
        particle_positions = BEC_helpers.sample_human_models_uniform([], n_particles)
        particles = pf.Particles(particle_positions)
        particles.update(prior)
        print(colored('entropy: {}'.format(particles.calc_entropy()), 'blue'))

        if summary_variant == 'particle_filter':
            BEC_summary, visited_env_traj_idxs, particles = BEC.obtain_summary_particle_filter(data_loc, particles, summary_variant, min_subset_constraints_record,
                                           min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool,
                                           n_human_models, consistent_state_count)

        elif summary_variant == 'proposed' or summary_variant == 'counterfactual_only':
            BEC_summary, visited_env_traj_idxs = BEC.obtain_summary_counterfactual(data_loc, summary_variant, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, mdp_features_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count, n_train_demos=n_train_demos, prior=prior, obj_func_proportion=obj_func_proportion)
        elif summary_variant == 'feature_only' or summary_variant == 'baseline':
            BEC_summary = BEC.obtain_summary(data_loc, summary_variant, min_BEC_constraints, BEC_lengths_record, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag, n_train_demos=n_train_demos)
        else:
            raise AssertionError("Unknown summary variant.")

        if summary_variant != 'particle_filter':
            # update the particle filter model according to the generated summary
            for unit in BEC_summary:
                for summary in unit:
                    particles.update(summary[3])

        if len(BEC_summary) > 0:
            with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
                pickle.dump((BEC_summary, visited_env_traj_idxs, particles), f)

    # BEC.visualize_summary(BEC_summary, weights, step_cost_flag)
    #
    # for summary in BEC_summary:
    #     best_mdp = summary[0]
    #     best_traj = summary[1]
    #
    #     with open('models/augmented_taxi/info_gains_' + str(0) + '. pickle', 'rb') as f:
    #         info_gains_record = pickle.load(f)
    #
    #     with open('models/' + data_loc + '/counterfactual_data_' + str(0) + '/model' + str(
    #             select_model) + '/cf_data_env' + str(
    #             best_env_idx).zfill(5) + '.pickle', 'rb') as f:
    #         best_human_trajs_record_env, constraints_env = pickle.load(f)
    #

    # # constraint visualization
    # BEC_summary = list(itertools.chain(*BEC_summary))
    # constraints_record = prior
    # for summary in BEC_summary:
    #     constraints_record.extend(summary[3])
    #     # constraints_record = summary[3]
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.set_facecolor('white')
    #     ax.xaxis.pane.fill = False
    #     ax.yaxis.pane.fill = False
    #     ax.zaxis.pane.fill = False
    #
    #     solid_angle = BEC_helpers.calc_solid_angles([constraints_record])[0]
    #     print(solid_angle)
    #
    #     ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_record)
    #     poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    #
    #     min_constraints = BEC_helpers.remove_redundant_constraints(constraints_record, weights, step_cost_flag)
    #     print(min_constraints)
    #     for constraints in [min_constraints]:
    #         BEC_viz.visualize_planes(constraints, fig=fig, ax=ax)
    #
    #     # visualizing uninformed prior
    #     # ieqs2 = BEC_helpers.constraints_to_halfspace_matrix_sage([[]])
    #     # poly2 = Polyhedron.Polyhedron(ieqs=ieqs2)
    #     # BEC_viz.visualize_spherical_polygon(poly2, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)
    #     #
    #     # visualize spherical polygon
    #     BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)
    #
    #     # ieqs_posterior = BEC_helpers.constraints_to_halfspace_matrix_sage(posterior)
    #     # poly_posterior = Polyhedron.Polyhedron(ieqs=ieqs_posterior)  # automatically finds the minimal H-representation
    #     # BEC_viz.visualize_spherical_polygon(poly_posterior, fig=fig, ax=ax, plot_ref_sphere=False, color='g')
    #
    #     ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='r', s=100)
    #     if mdp_class == 'augmented_taxi2':
    #         ax.set_xlabel('$\mathregular{w_0}$: Mud')
    #         ax.set_ylabel('$\mathregular{w_1}$: Recharge')
    #     elif mdp_class == 'two_goal2':
    #         ax.set_xlabel('X: Goal 1 (grey)')
    #         ax.set_ylabel('Y: Goal 2 (green)')
    #     else:
    #         ax.set_xlabel('X: Goal')
    #         ax.set_ylabel('Y: Skateboard')
    #     ax.set_zlabel('$\mathregular{w_2}$: Action')
    #
    #     ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    #     ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    #     ax.set_zticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    #
    #     # if matplotlib.get_backend() == 'TkAgg':
    #     #     mng = plt.get_current_fig_manager()
    #     #     mng.resize(*mng.window.maxsize())
    #
    #     plt.show()

    # # particle filter visualization
    # from numpy.random import seed
    # seed(2)
    #
    # particle_positions = BEC_helpers.sample_human_models_uniform([], n_particles)
    # particles = pf.Particles(particle_positions)
    #
    # constraints_running = prior
    #
    # # print(particles.calc_info_gain(prior))
    # particles.update(prior)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_facecolor('white')
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    #
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    #
    # particles.plot(fig=fig, ax=ax)
    # BEC_viz.visualize_planes(constraints_running, fig=fig, ax=ax)
    #
    # # visualize spherical polygon
    # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_running)
    # poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)
    #
    # # visualize the ground truth weight
    # w = np.array([[-3, 3.5, -1]])  # toll, hotswap station, step cost
    # w_normalized = w / np.linalg.norm(w[0, :], ord=2)
    # ax.scatter(w_normalized[0, 0], w_normalized[0, 1], w_normalized[0, 2], marker='o', c='r', s=100)
    #
    # plt.show()
    #
    # for j, summary in enumerate(BEC_summary):
    #     print(j)
    #
    #     constraints = summary[3]
    #
    #     constraints_running.extend(constraints)
    #     constraints_running = BEC_helpers.remove_redundant_constraints(constraints_running, None, False)
    #
    #     particles.update(constraints)
    #     print('Entropy: {}'.format(particles.calc_entropy()))
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.set_facecolor('white')
    #     ax.xaxis.pane.fill = False
    #     ax.yaxis.pane.fill = False
    #     ax.zaxis.pane.fill = False
    #
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('z')
    #
    #     particles.plot(fig=fig, ax=ax)
    #     BEC_viz.visualize_planes(constraints_running, fig=fig, ax=ax)
    #
    #     # visualize spherical polygon
    #     ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_running)
    #     poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    #     BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)
    #
    #     # visualize the ground truth weight
    #     w = np.array([[-3, 3.5, -1]])  # toll, hotswap station, step cost
    #     w_normalized = w / np.linalg.norm(w[0, :], ord=2)
    #     ax.scatter(w_normalized[0, 0], w_normalized[0, 1], w_normalized[0, 2], marker='o', c='b', s=100)
    #
    #     plt.show()
    #     checking what is being returned
    return BEC_summary, visited_env_traj_idxs, particles
'''
'''
def obtain_test_environments(mdp_class, data_loc, mdp_parameters, weights, BEC_params, step_cost_flag, n_human_models, prior, posterior, summary=None, use_counterfactual=True, visualize_test_env=False):
    
    Summary: Correlate the difficulty of a test environment with the generalized area of the BEC region obtain by the
    corresponding optimal demonstration. Return the desired number and difficulty of test environments (to be given
    to the human to test his understanding of the agent's policy).
    
    # use generalized area of the BEC region to select test environments
    try:
        with open('models/' + data_loc + '/test_environments.pickle', 'rb') as f:
            test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers = pickle.load(f)
    except:
        ps_helpers.obtain_env_policies(mdp_class, data_loc, np.expand_dims(weights, axis=0), mdp_parameters, pool)

        try:
            with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
                policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(f)
        except:
            # use policy BEC to extract constraints
            policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, consistent_state_count = BEC.extract_constraints(
                data_loc, step_cost_flag, pool, print_flag=True)
            with open('models/' + data_loc + '/base_constraints.pickle', 'wb') as f:
                pickle.dump((policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record,
                             consistent_state_count), f)

        try:
            with open('models/' + data_loc + '/BEC_constraints.pickle', 'rb') as f:
                min_BEC_constraints, BEC_lengths_record = pickle.load(f)
        except:
            min_BEC_constraints, BEC_lengths_record = BEC.extract_BEC_constraints(policy_constraints,
                                                                                  min_subset_constraints_record,
                                                                                  env_record, weights, step_cost_flag,
                                                                                  pool)

            with open('models/' + data_loc + '/BEC_constraints.pickle', 'wb') as f:
                pickle.dump((min_BEC_constraints, BEC_lengths_record), f)

        equal_prior_posterior = True
        counterfactual_folder_idx = 0
        if len(prior) == len(posterior):
            for j, posterior_constraint in enumerate(posterior):
                if not np.array_equal(posterior_constraint, prior[j]):
                    equal_prior_posterior = False
                    counterfactual_folder_idx = -1
        else:
            equal_prior_posterior = False
            counterfactual_folder_idx = -1

        try:
            with open('models/' + data_loc + '/BEC_constraints_counterfactual.pickle', 'rb') as f:
                min_subset_constraints_record_counterfactual = pickle.load(f)

            with open('models/' + data_loc + '/BEC_lengths_counterfactual.pickle', 'rb') as f:
                BEC_lengths_record_counterfactual = pickle.load(f)

        except:
            if not equal_prior_posterior:
                sample_human_models = BEC_helpers.sample_human_models_uniform(posterior, n_human_models)
                print("Obtaining counterfactual data for human models sampled from the posterior: ")
                for model_idx, human_model in enumerate(sample_human_models):
                    cf_data_dir = 'models/' + data_loc + '/counterfactual_data_-1/model' + str(model_idx)
                    os.makedirs(cf_data_dir, exist_ok=True)

                    print(colored('Model #: {}'.format(model_idx), 'red'))
                    print(colored('Model val: {}'.format(human_model), 'red'))

                    # assuming that I'm considering human models jointly
                    n_processed_envs = len(os.listdir(cf_data_dir))
                    args = [
                        (data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]),
                         traj_record[i], posterior, step_cost_flag, counterfactual_folder_idx, None,
                         True) for i in range(n_processed_envs, len(traj_record))]
                    _ = list(tqdm(pool.imap(BEC.compute_counterfactuals, args), total=len(args)))

            try:
                with open('models/' + data_loc + '/BEC_constraints_counterfactual.pickle', 'rb') as f:
                    min_subset_constraints_record_counterfactual = pickle.load(f)
            except:
                args = [(data_loc, i, min_subset_constraints_record[i], n_human_models, counterfactual_folder_idx, weights, step_cost_flag) for i in range(len(min_subset_constraints_record))]
                # combine the human counterfactual and one-step deviation constraints
                min_subset_constraints_record_counterfactual = list(tqdm(pool.imap(BEC_helpers.combine_counterfactual_constraints, args), total=len(args)))

                with open('models/' + data_loc + '/BEC_constraints_counterfactual.pickle', 'wb') as f:
                    pickle.dump(min_subset_constraints_record_counterfactual, f)

            # take the overlap of the human posterior with BEC of suboptimal trajectories of one-step deviation
            args = [(i, n_human_models, min_subset_constraints, prior, posterior, data_loc, counterfactual_folder_idx, weights, traj_record[i], step_cost_flag, pool)
                    for i, min_subset_constraints in enumerate(min_subset_constraints_record_counterfactual)]

            print("Obtaining overlap in BEC area between posterior human model and potential test demonstrations: ")
            BEC_lengths_record_counterfactual = list(tqdm(pool.imap(BEC.overlap_demo_BEC_and_human_posterior, args), total=len(args)))
            with open('models/' + data_loc + '/BEC_lengths_counterfactual.pickle', 'wb') as f:
                pickle.dump(BEC_lengths_record_counterfactual, f)

        if use_counterfactual:
            test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers = \
                ps_helpers.obtain_test_environments(data_loc, min_subset_constraints_record_counterfactual, env_record, traj_record, weights, BEC_lengths_record_counterfactual, BEC_params['n_test_demos'], BEC_params['test_difficulty'], step_cost_flag, summary=summary)
        else:
            test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = \
                ps_helpers.obtain_test_environments(data_loc, min_subset_constraints_record, env_record,
                                                    traj_record, weights, BEC_lengths_record,
                                                    BEC_params['n_test_demos'], BEC_params['test_difficulty'],
                                                    step_cost_flag, summary=summary)

        with open('models/' + data_loc + '/test_environments.pickle', 'wb') as f:
            pickle.dump((test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers), f)

    if visualize_test_env:
        BEC.visualize_test_envs(posterior, test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers, weights,
                                step_cost_flag)
    return test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints
'''
'''
def simulate_teaching_loop(mdp_class, BEC_summary, visited_env_traj_idxs, particles_summary, pool, prior, n_particles, n_human_models, n_human_models_precomputed, data_loc, weights, step_cost_flag, visualize_pf_transition=False):
    # todo: maybe pass in some of these objects later
    print(f"mdp_class: {mdp_class}\nvisited_env_traj_idxs: {visited_env_traj_idxs}\npool: {pool}\nprior: {prior}\nn_particles: {n_particles}\n n_human_models: {n_human_models}\nn_human_models_precomputed: {n_human_models_precomputed}\ndata_loc: {data_loc}\nweights: {weights}\nstep_cost_flag: {step_cost_flag}\nvisualize_pf_transition: {visualize_pf_transition}\n")
    with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
        policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(f)
    #print(f"min_subset_constraint_record: {min_subset_constraints_record}\nenv_record: {env_record}\ntraj_record: {traj_record}\ntraj_features_record: {traj_features_record}\nmdp_features_record: {mdp_features_record}\nconsistent_state_count: {consistent_state_count}\n")
    # initialize a particle filter model of human
    particle_positions = BEC_helpers.sample_human_models_uniform([], n_particles)
    particles = pf.Particles(particle_positions)
    particles.update(prior)

    # run through the pre-selected units
    for unit_idx, unit in enumerate(BEC_summary):
        print("Here are the demonstrations for this unit")
        unit_constraints = []
        running_variable_filter = unit[0][4]

        # show each demonstration that is part of this unit
        for subunit in unit:
            subunit[0].visualize_trajectory(subunit[1])
            unit_constraints.extend(subunit[3])

            # update particle filter with demonstration's constraint
            particles.update(subunit[3])
            # visualize the updated particle filter
            if visualize_pf_transition:
                BEC_viz.visualize_pf_transition(subunit[3], particles, mdp_class, weights)

        # obtain the constraints conveyed by the unit's demonstrations
        min_constraints = BEC_helpers.remove_redundant_constraints(unit_constraints, weights, step_cost_flag)
        # obtain the diagnostic tests that will test the human's understanding of the unit's constraints
        preliminary_tests, visited_env_traj_idxs = BEC.obtain_diagnostic_tests(data_loc, unit, visited_env_traj_idxs, min_constraints, min_subset_constraints_record, traj_record, traj_features_record, running_variable_filter, mdp_features_record)
        print(preliminary_tests[0])

        # with open('models/' + data_loc + '/preliminary_tests.pickle', 'wb') as f:
        #     pickle.dump((preliminary_tests, visited_env_traj_idxs), f)
        # with open('models/' + data_loc + '/preliminary_tests.pickle', 'rb') as f:
        #     preliminary_tests, visited_env_traj_idxs = pickle.load(f)

        # query the human's response to the diagnostic tests
        for test in preliminary_tests:
            test_mdp = test[0]
            opt_traj = test[1]
            test_constraints = test[3]
            test_history = [test] # to ensure that remedial demonstrations and tests are visually simple/similar and complex/different, respectively

            print("Here is a diagnostic test for this unit")
            human_traj, human_history = test_mdp.visualize_interaction(keys_map=params.keys_map) # the latter is simply the gridworld locations of the agent
            # with open('models/' + data_loc + '/human_traj.pickle', 'wb') as f:
            #     pickle.dump((human_traj, human_history), f)
            # with open('models/' + data_loc + '/human_traj.pickle', 'rb') as f:
            #     human_traj, human_history = pickle.load(f)

            human_feature_count = test_mdp.accumulate_reward_features(human_traj, discount=True)
            opt_feature_count = test_mdp.accumulate_reward_features(opt_traj, discount=True)

            if (human_feature_count == opt_feature_count).all():
                print("You got the diagnostic test right")

                particles.update(test_constraints)
                if visualize_pf_transition:
                    BEC_viz.visualize_pf_transition(test_constraints, particles, mdp_class, weights)

            else:
                print("You got the diagnostic test wrong. Here's the correct answer")
                failed_BEC_constraint = opt_feature_count - human_feature_count
                print("Failed BEC constraint: {}".format(failed_BEC_constraint))

                particles.update([-failed_BEC_constraint])
                if visualize_pf_transition:
                    BEC_viz.visualize_pf_transition([-failed_BEC_constraint], particles, mdp_class, weights)

                test_mdp.visualize_trajectory_comparison(opt_traj, human_traj)

                print("Here is a remedial demonstration that might be helpful")

                remedial_instruction, visited_env_traj_idxs = BEC.obtain_remedial_demonstrations(data_loc, pool, particles, n_human_models, failed_BEC_constraint, min_subset_constraints_record, env_record, traj_record, traj_features_record, test_history, visited_env_traj_idxs, running_variable_filter, mdp_features_record, consistent_state_count, weights, step_cost_flag, n_human_models_precomputed=n_human_models_precomputed)
                remedial_mdp, remedial_traj, _, remedial_constraint, _ = remedial_instruction[0]
                remedial_mdp.visualize_trajectory(remedial_traj)
                test_history.extend(remedial_instruction)

                particles.update([remedial_constraint])
                if visualize_pf_transition:
                    BEC_viz.visualize_pf_transition([remedial_constraint], particles, mdp_class, weights)

                with open('models/' + data_loc + '/remedial_instruction.pickle', 'wb') as f:
                    pickle.dump(remedial_instruction, f)

                remedial_test_correct = False

                print("Here is a remedial test to see if you've correctly learned the lesson")
                while not remedial_test_correct:

                    remedial_test, visited_env_traj_idxs = BEC.obtain_remedial_demonstrations(data_loc, pool,
                                                                                                     particles,
                                                                                                     n_human_models,
                                                                                                     failed_BEC_constraint,
                                                                                                     min_subset_constraints_record,
                                                                                                     env_record,
                                                                                                     traj_record,
                                                                                                     traj_features_record,
                                                                                                     test_history,
                                                                                                     visited_env_traj_idxs,
                                                                                                     running_variable_filter,
                                                                                                     mdp_features_record,
                                                                                                     consistent_state_count,
                                                                                                     weights,
                                                                                                     step_cost_flag, type='testing', n_human_models_precomputed=n_human_models_precomputed)

                    remedial_mdp, remedial_traj, _, _, _ = remedial_test[0]
                    test_history.extend(remedial_test)

                    human_traj, human_history = remedial_mdp.visualize_interaction(
                        keys_map=params.keys_map)  # the latter is simply the gridworld locations of the agent
                    # with open('models/' + data_loc + '/human_traj.pickle', 'wb') as f:
                    #     pickle.dump((human_traj, human_history), f)
                    # with open('models/' + data_loc + '/human_traj.pickle', 'rb') as f:
                    #     human_traj, human_history = pickle.load(f)

                    human_feature_count = remedial_mdp.accumulate_reward_features(human_traj, discount=True)
                    opt_feature_count = remedial_mdp.accumulate_reward_features(remedial_traj, discount=True)

                    if (human_feature_count == opt_feature_count).all():
                        print("You got the remedial test correct")
                        remedial_test_correct = True

                        particles.update([failed_BEC_constraint])
                        if visualize_pf_transition:
                            BEC_viz.visualize_pf_transition([failed_BEC_constraint], particles, mdp_class, weights)
                    else:
                        failed_BEC_constraint = opt_feature_count - human_feature_count
                        print("You got the remedial test wrong. Here's the correct answer")
                        print("Failed BEC constraint: {}".format(failed_BEC_constraint))
                        remedial_mdp.visualize_trajectory_comparison(remedial_traj, human_traj)

                        particles.update([-failed_BEC_constraint])
                        if visualize_pf_transition:
                            BEC_viz.visualize_pf_transition([-failed_BEC_constraint], particles, mdp_class, weights)
'''
'''
def analyze_prev_study_tests(domain, BEC_summary, visited_env_traj_idxs, particles_summary, pool, prior, n_particles, n_human_models, n_human_models_precomputed, data_loc, weights, step_cost_flag, visualize_pf_transition=True):
    with open('filtered_human_responses.pickle', 'rb') as f:
        filtered_human_traj_dict, filtered_mdp_dict, filtered_count_dict, filtered_opt_reward_dict, filtered_human_reward_dict, filtered_opt_traj_dict = pickle.load(
            f)

    with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
        policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(
            f)

    # go through potentially duplicate trajectories and visualize them
    print('domain: {}'.format(domain))

    model_low = prior.copy()
    if domain == 'augmented_taxi2':
        model_low.extend([np.array([[1, 0, -4]]), np.array([[-1, 0, 2]])])
        model_medium = model_low.copy()
        model_medium.extend([np.array([[0, -1, -4]]), np.array([[0, 1, 2]])])
        model_high = model_medium.copy()
        model_high.extend([np.array([[1, 1, 0]])])
    elif domain == 'colored_tiles':
        model_low.extend([np.array([[1, 0, -8]]), np.array([[-1, 0, 6]])])
        model_medium = model_low.copy()
        model_medium.extend([np.array([[0, -1, 4]]), np.array([[0, 1, -6]])])
        model_high = model_medium.copy()
        model_high.extend([np.array([[1, -1, -2]])])
    elif domain == 'skateboard2':
        model_low.extend([np.array([[-6, 0, 1]]), np.array([[9, 0, -2]])])
        model_medium = model_low.copy()
        model_medium.extend([np.array([[0, -2, 1]]), np.array([[0, 5, -3]])])
        model_high = model_medium.copy()
        model_high.extend([np.array([[-6, 4, -1]]), np.array([[8, 1, -2]]), np.array([[5, 2, -2]])])

    for difficulty in filtered_human_traj_dict[domain].keys():
    # for difficulty in ['low']:

        if difficulty == 'low':
            human_model_constraints = model_low
        elif difficulty == 'medium':
            human_model_constraints = model_medium
        else:
            human_model_constraints = model_high

        print('difficulty: {}'.format(difficulty))

        # initialize a particle filter model of human
        particle_positions = BEC_helpers.sample_human_models_uniform([], n_particles)
        particles_orig = pf.Particles(particle_positions)
        particles_orig.update(human_model_constraints)

        for tag in filtered_human_traj_dict[domain][difficulty].keys():

            print('tag: {}'.format(tag))
            print('total number of trajectories: {}'.format(sum(filtered_count_dict[domain][difficulty][tag])))
            total_num_unique_traj = len(filtered_human_traj_dict[domain][difficulty][tag])
            for i, traj in enumerate(filtered_human_traj_dict[domain][difficulty][tag]):

                particles = copy.deepcopy(particles_orig)

                test_mdp = filtered_mdp_dict[domain][difficulty][tag][i]
                human_traj = filtered_human_traj_dict[domain][difficulty][tag][i]
                opt_traj = filtered_opt_traj_dict[domain][difficulty][tag][i]

                print('trajectory {}/{}'.format(i + 1, total_num_unique_traj))
                print('duplicate # of this trajectory: {}'.format(filtered_count_dict[domain][difficulty][tag][i]))
                # print('opt reward vs human reward: {} vs {}'.format(filtered_opt_reward_dict[domain][difficulty][tag][i], filtered_human_reward_dict[domain][difficulty][tag][i]))
                print('optimal?: {}'.format(filtered_opt_reward_dict[domain][difficulty][tag][i] == filtered_human_reward_dict[domain][difficulty][tag][i][0][0]))
                test_mdp.visualize_trajectory(human_traj)

                # print("Here is a diagnostic test for this unit")
                # human_traj, human_history = test_mdp.visualize_interaction(
                #     keys_map=params.keys_map)  # the latter is simply the gridworld locations of the agent

                human_feature_count = test_mdp.accumulate_reward_features(human_traj, discount=True)
                opt_feature_count = test_mdp.accumulate_reward_features(opt_traj, discount=True)

                # filter any features that the human got right from subsequent remedial demonstrations and tests
                running_variable_filter = (opt_feature_count - human_feature_count) == 0
                # however, always allow for differences in action features since essentially every demonstration will
                # convey information about the action weight when considering one or two step deviations (this may be mitigated if we used counterfactual reasoning)
                running_variable_filter[0][2] = 0

                if (human_feature_count == opt_feature_count).all():
                    print("You got the test right")
                    # constraint = opt_feature_count - human_feature_count
                    #
                    # particles.update([constraint])
                    # BEC_viz.visualize_pf_transition(constraint, particles_prev, particles, domain,
                    #                                 weights)
                else:
                    print("You got the test wrong. Here's the correct answer")

                    failed_BEC_constraint = opt_feature_count - human_feature_count
                    print("Failed BEC constraint: {}".format(failed_BEC_constraint))

                    particles.update([-failed_BEC_constraint])
                    if visualize_pf_transition:
                        BEC_viz.visualize_pf_transition([-failed_BEC_constraint], particles, domain, weights)

                    test_mdp.visualize_trajectory_comparison(opt_traj, human_traj)

                    print("Here is a remedial demonstration that might be helpful")

                    remedial_instruction, visited_env_traj_idxs = BEC.obtain_remedial_demonstrations(data_loc, pool,
                                                                                                     particles,
                                                                                                     n_human_models,
                                                                                                     failed_BEC_constraint,
                                                                                                     min_subset_constraints_record,
                                                                                                     env_record,
                                                                                                     traj_record,
                                                                                                     traj_features_record,
                                                                                                     [],
                                                                                                     visited_env_traj_idxs,
                                                                                                     running_variable_filter,
                                                                                                     mdp_features_record,
                                                                                                     consistent_state_count,
                                                                                                     weights,
                                                                                                     step_cost_flag, n_human_models_precomputed=n_human_models_precomputed)
                    remedial_mdp, remedial_traj, _, remedial_constraint, _ = remedial_instruction[0]
                    remedial_mdp.visualize_trajectory(remedial_traj)
                    # test_history.extend(remedial_instruction)

                    particles.update([remedial_constraint])
                    if visualize_pf_transition:
                        BEC_viz.visualize_pf_transition([remedial_constraint], particles, domain, weights)

                    with open('models/' + data_loc + '/remedial_instruction.pickle', 'wb') as f:
                        pickle.dump(remedial_instruction, f)

                    remedial_test_correct = False

                    print("Here is a remedial test to see if you've correctly learned the lesson")
                    while not remedial_test_correct:

                        remedial_test, visited_env_traj_idxs = BEC.obtain_remedial_demonstrations(data_loc, pool,
                                                                                                  particles,
                                                                                                  n_human_models,
                                                                                                  failed_BEC_constraint,
                                                                                                  min_subset_constraints_record,
                                                                                                  env_record,
                                                                                                  traj_record,
                                                                                                  traj_features_record,
                                                                                                  [],
                                                                                                  visited_env_traj_idxs,
                                                                                                  running_variable_filter,
                                                                                                  mdp_features_record,
                                                                                                  consistent_state_count,
                                                                                                  weights,
                                                                                                  step_cost_flag,
                                                                                                  type='testing', n_human_models_precomputed=n_human_models_precomputed)

                        remedial_mdp, remedial_traj, _, _, _ = remedial_test[0]
                        # test_history.extend(remedial_test)

                        human_traj, human_history = remedial_mdp.visualize_interaction(
                            keys_map=params.keys_map)  # the latter is simply the gridworld locations of the agent
                        # with open('models/' + data_loc + '/human_traj.pickle', 'wb') as f:
                        #     pickle.dump((human_traj, human_history), f)
                        # with open('models/' + data_loc + '/human_traj.pickle', 'rb') as f:
                        #     human_traj, human_history = pickle.load(f)

                        human_feature_count = remedial_mdp.accumulate_reward_features(human_traj, discount=True)
                        opt_feature_count = remedial_mdp.accumulate_reward_features(remedial_traj, discount=True)

                        if (human_feature_count == opt_feature_count).all():
                            print("You got the remedial test correct")
                            remedial_test_correct = True

                            particles.update([failed_BEC_constraint])
                            if visualize_pf_transition:
                                BEC_viz.visualize_pf_transition([failed_BEC_constraint], particles, domain, weights)

                        else:
                            failed_remedial_constraint = opt_feature_count - human_feature_count
                            print("You got the remedial test wrong. Here's the correct answer")
                            remedial_mdp.visualize_trajectory_comparison(remedial_traj, human_traj)

                            particles.update([-failed_remedial_constraint])
                            if visualize_pf_transition:
                                BEC_viz.visualize_pf_transition([failed_BEC_constraint], particles, domain, weights)
'''
'''
def contrast_PF_2_step_dev(domain, BEC_summary, visited_env_traj_idxs, particles_summary, pool, prior, n_particles, n_human_models, data_loc, weights, step_cost_flag, visualize_pf_transition=False):
    with open('filtered_human_responses.pickle', 'rb') as f:
        filtered_human_traj_dict, filtered_mdp_dict, filtered_count_dict, filtered_opt_reward_dict, filtered_human_reward_dict, filtered_opt_traj_dict = pickle.load(
            f)

    with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
        policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(
            f)

    # go through potentially duplicate trajectories and visualize them
    print('domain: {}'.format(domain))

    model_low = prior.copy()
    if domain == 'augmented_taxi2':
        model_low.extend([np.array([[1, 0, -4]]), np.array([[-1, 0, 2]])])
        model_medium = model_low.copy()
        model_medium.extend([np.array([[0, -1, -4]]), np.array([[0, 1, 2]])])
        model_medium = BEC_helpers.remove_redundant_constraints(model_medium, weights, step_cost_flag)
        model_high = model_medium.copy()
        model_high.extend([np.array([[1, 1, 0]])])
    elif domain == 'colored_tiles':
        model_low.extend([np.array([[1, 0, -8]]), np.array([[-1, 0, 6]])])
        model_medium = model_low.copy()
        model_medium.extend([np.array([[ 0, -1,  4]]), np.array([[ 0,  1, -6]])])
        model_high = model_medium.copy()
        model_high.extend([np.array([[1, -1, -2]])])
    elif domain == 'skateboard2':
        model_low.extend([np.array([[-6, 0, 1]]), np.array([[9, 0, -2]])])
        model_medium = model_low.copy()
        model_medium.extend([np.array([[ 0, -2,  1]]), np.array([[ 0,  5, -3]])])
        model_high = model_medium.copy()
        model_high.extend([np.array([[-6,  4, -1]]), np.array([[ 8,  1, -2]]), np.array([[ 5,  2, -2]])])

    try:
        with open('models/' + data_loc + '/' + 'PF_2-step_dev_comparison.pickle', 'rb') as f:
            overlap_counter, PF_best_idxs, VO_best_idxs, failed_constraints = pickle.load(f)
        pass
    except:
        overlap_counter = {'low': [], 'medium': [], 'high': []}
        PF_best_idxs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        VO_best_idxs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        failed_constraints = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for difficulty in filtered_human_traj_dict[domain].keys():
        # for difficulty in ['high']:
            if difficulty == 'low':
                human_model_constraints = model_low
            elif difficulty == 'medium':
                human_model_constraints = model_medium
            else:
                human_model_constraints = model_high
            print('difficulty: {}'.format(difficulty))

            # initialize a particle filter model of human
            particle_positions = BEC_helpers.sample_human_models_uniform([], n_particles)
            particles_orig = pf.Particles(particle_positions)
            particles_orig.update(human_model_constraints)

            # precompute the constraints for the potential demonstrations you can show for domain and difficulty
            try:
                with open('models/' + data_loc + '/' + 'PF_constraints_' + difficulty + '.pickle', 'rb') as f:
                    info_gains_record, min_env_constraints_record = pickle.load(f)

            except:
                # the human's incorrect response does not have a direct counterexample, and thus you need to use information gain to obtain the next example
                sample_human_models, model_weights = BEC_helpers.sample_human_models_pf(particles_orig, n_human_models)
                info_gains_record = []

                for model_idx, human_model in enumerate(sample_human_models):
                    print(colored('Model #: {}'.format(model_idx), 'red'))
                    print(colored('Model val: {}'.format(human_model), 'red'))

                    with open('models/' + data_loc + '/demo_gen_log.txt', 'a') as myfile:
                        myfile.write('Model #: {}\n'.format(model_idx))
                        myfile.write('Model val: {}\n'.format(human_model))

                    # based on the human's current model, obtain the information gain generated when comparing to the agent's
                    # optimal trajectories in each environment (human's corresponding optimal trajectories and constraints
                    # are saved for reference later)
                    print("Obtaining counterfactual information gains:")

                    cf_data_dir = 'models/' + data_loc + '/counterfactual_data_' + difficulty + '/model' + str(model_idx)
                    os.makedirs(cf_data_dir, exist_ok=True)

                    args = [(data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]), traj_record[i], None, human_model, step_cost_flag, difficulty, np.zeros((1, 3)), mdp_features_record[i], True) for i in range(len(traj_record))]
                    info_gain_envs = list(tqdm(pool.imap(BEC.compute_counterfactuals, args), total=len(args)))
                    info_gains_record.append(info_gain_envs)

                print("Combining the most limiting constraints across human models:")
                args = [(i, range(len(sample_human_models)), data_loc, difficulty, weights, step_cost_flag, np.zeros((1, 3)),
                         mdp_features_record[i],
                         traj_record[i], human_model_constraints, None, False, False) for
                        i in range(len(traj_record))]
                info_gains_record, min_env_constraints_record, n_diff_constraints_record, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = zip(
                    *pool.imap(BEC.combine_limiting_constraints_IG, tqdm(args)))

                with open('models/' + data_loc + '/' + 'PF_constraints_' + difficulty + '.pickle', 'wb') as f:
                    pickle.dump((info_gains_record, min_env_constraints_record), f)

            overlap_counter_per_difficulty = []

            for tag in filtered_human_traj_dict[domain][difficulty].keys():
            # for tag in ['2']:
                print('tag: {}'.format(tag))
                print('total number of trajectories: {}'.format(sum(filtered_count_dict[domain][difficulty][tag])))
                total_num_unique_traj = len(filtered_human_traj_dict[domain][difficulty][tag])
                for i, traj in enumerate(filtered_human_traj_dict[domain][difficulty][tag]):
                    # if i != 4:
                    #     continue

                    particles = copy.deepcopy(particles_orig)
                    particles_prev = copy.deepcopy(particles_orig)

                    test_mdp = filtered_mdp_dict[domain][difficulty][tag][i]
                    human_traj = filtered_human_traj_dict[domain][difficulty][tag][i]
                    opt_traj = filtered_opt_traj_dict[domain][difficulty][tag][i]

                    print('trajectory {}/{}'.format(i + 1, total_num_unique_traj))

                    human_feature_count = test_mdp.accumulate_reward_features(human_traj, discount=True)
                    opt_feature_count = test_mdp.accumulate_reward_features(opt_traj, discount=True)

                    # filter any features that the human got right from subsequent remedial demonstrations and tests
                    running_variable_filter = (opt_feature_count - human_feature_count) == 0
                    # however, always allow for differences in action features since essentially every demonstration will
                    # convey information about the action weight when considering one or two step deviations (this may be mitigated if we used counterfactual reasoning)
                    running_variable_filter[0][2] = 0

                    if (human_feature_count == opt_feature_count).all():
                        # print("You got the test right")
                        pass
                    else:
                        # print("You got the test wrong. Here's the correct answer")

                        failed_BEC_constraint = opt_feature_count - human_feature_count
                        # print("Failed BEC constraint: {}".format(failed_BEC_constraint))

                        particles.update([-failed_BEC_constraint])
                        if visualize_pf_transition:
                            BEC_viz.visualize_pf_transition([-failed_BEC_constraint], particles_prev, particles,
                                                            domain, weights)
                            particles_prev = copy.deepcopy(particles)

                        # test_mdp.visualize_trajectory_comparison(opt_traj, human_traj)
                        # print("Here is a remedial demonstration that might be helpful")

                        # count the demonstration overlap between 2-step dev/BEC and PF
                        # **Note that PF currently uses min_env_constraints_record, and BEC uses min_subset_constraints_record
                        # for pf
                        BEC_constraint_bookkeeping_pf = BEC_helpers.perform_BEC_constraint_bookkeeping(failed_BEC_constraint,
                                                                                                    min_env_constraints_record,
                                                                                                    visited_env_traj_idxs,
                                                                                                    traj_record,
                                                                                                    traj_features_record,
                                                                                                    mdp_features_record,
                                                                                                    variable_filter=running_variable_filter)

                        print('{} exact candidates for remedial demo/test'.format(len(BEC_constraint_bookkeeping_pf[0])))
                        if len(BEC_constraint_bookkeeping_pf[0]) > 0:
                            # the human's incorrect response can be corrected with a direct counterexample
                            best_env_idxs_pf, best_traj_idxs_pf = list(zip(*BEC_constraint_bookkeeping_pf[0]))

                        else:
                            nn_BEC_constraint_bookkeeping_pf, minimal_distances = BEC_helpers.perform_nn_BEC_constraint_bookkeeping(
                                failed_BEC_constraint,
                                min_env_constraints_record, visited_env_traj_idxs, traj_record, traj_features_record,
                                mdp_features_record, variable_filter=running_variable_filter)
                            print('{} approximate candidates for remedial demo/test'.format(
                                len(nn_BEC_constraint_bookkeeping_pf[0])))
                            if len(nn_BEC_constraint_bookkeeping_pf[0]) > 0:
                                # the human's incorrect response can be corrected with similar enough counterexample
                                best_env_idxs_pf, best_traj_idxs_pf = list(
                                    zip(*nn_BEC_constraint_bookkeeping_pf[0]))

                        # find the demonstration that minimizes PF information gain
                        best_info_gain = float('inf')
                        info_gains = {}

                        # obtain the demonstrations that will convey the lowest information gain (while still providing the desired information)
                        for j in range(len(best_env_idxs_pf)):
                            info_gain = particles.calc_info_gain(min_env_constraints_record[best_env_idxs_pf[j]][best_traj_idxs_pf[j]])
                            info_gains[(best_env_idxs_pf[j], best_traj_idxs_pf[j])] = (info_gain, min_env_constraints_record[best_env_idxs_pf[j]][best_traj_idxs_pf[j]])

                            if np.isclose(info_gain, best_info_gain):
                                best_infogains_env_traj_idxs.append((best_env_idxs_pf[j], best_traj_idxs_pf[j]))
                            elif info_gain < best_info_gain:
                                best_info_gain = info_gain
                                best_infogains_env_traj_idxs = [(best_env_idxs_pf[j], best_traj_idxs_pf[j])]


                        # for BEC
                        BEC_constraint_bookkeeping_BEC = BEC_helpers.perform_BEC_constraint_bookkeeping(failed_BEC_constraint,
                                                                                                    min_subset_constraints_record,
                                                                                                    visited_env_traj_idxs,
                                                                                                    traj_record,
                                                                                                    traj_features_record,
                                                                                                    mdp_features_record,
                                                                                                    variable_filter=running_variable_filter)

                        print('{} exact candidates for remedial demo/test'.format(len(BEC_constraint_bookkeeping_BEC[0])))
                        if len(BEC_constraint_bookkeeping_BEC[0]) > 0:
                            # the human's incorrect response can be corrected with a direct counterexample
                            best_env_idxs_BEC, best_traj_idxs_BEC = list(zip(*BEC_constraint_bookkeeping_BEC[0]))

                        else:
                            nn_BEC_constraint_bookkeeping_BEC, minimal_distances = BEC_helpers.perform_nn_BEC_constraint_bookkeeping(
                                failed_BEC_constraint,
                                min_subset_constraints_record, visited_env_traj_idxs, traj_record, traj_features_record,
                                mdp_features_record, variable_filter=running_variable_filter)
                            print('{} approximate candidates for remedial demo/test'.format(
                                len(nn_BEC_constraint_bookkeeping_BEC[0])))
                            if len(nn_BEC_constraint_bookkeeping_BEC[0]) > 0:
                                # the human's incorrect response can be corrected with similar enough counterexample
                                best_env_idxs_BEC, best_traj_idxs_BEC = list(
                                    zip(*nn_BEC_constraint_bookkeeping_BEC[0]))

                        # obtain the demonstrations that will convey the lowest information gain (while still providing the desired information)
                        smallest_BEC_area = float('inf')
                        for j in range(len(best_env_idxs_BEC)):

                            BEC_area = BEC_helpers.calc_solid_angles(
                                [min_subset_constraints_record[best_env_idxs_BEC[j]][best_traj_idxs_BEC[j]]])[0]
                            if np.isclose(BEC_area, smallest_BEC_area):
                                best_BEC_area_env_traj_idxs.append((best_env_idxs_BEC[j], best_traj_idxs_BEC[j]))
                            elif BEC_area < smallest_BEC_area:
                                smallest_BEC_area = BEC_area
                                best_BEC_area_env_traj_idxs = [(best_env_idxs_BEC[j], best_traj_idxs_BEC[j])]


                        best_env_idxs_BEC, best_traj_idxs_BEC = list(zip(*best_BEC_area_env_traj_idxs))
                        best_env_idxs_pf, best_traj_idxs_pf = list(zip(*best_infogains_env_traj_idxs))


                        # a) consider both information gain and visual optimization when counting demonstration overlap between 2-step dev/BEC and PF
                        best_env_idxs_BEC_vo_opt, best_traj_idxs_BEC_vo_opt = ps_helpers.optimize_visuals(data_loc, best_env_idxs_BEC,
                                                                                  best_traj_idxs_BEC, traj_record,
                                                                                  [], type=type, return_all_equiv=True)

                        # first optimize for PF information gain, then optimize for visuals
                        best_env_idxs_pf_vo_opt, best_traj_idxs_pf_vo_opt = ps_helpers.optimize_visuals(data_loc,
                                                                                                  best_env_idxs_pf,
                                                                                                  best_traj_idxs_pf,
                                                                                                  traj_record,
                                                                                                  [], type=type, return_all_equiv=True)

                        zipped_env_traj_idxs_BEC = list(zip(best_env_idxs_BEC_vo_opt, best_traj_idxs_BEC_vo_opt))
                        zipped_env_traj_idxs_pf = list(zip(best_env_idxs_pf_vo_opt, best_traj_idxs_pf_vo_opt))

                        # b) only consider information gain
                        # zipped_env_traj_idxs_BEC = list(zip(best_env_idxs_BEC, best_traj_idxs_BEC))
                        # zipped_env_traj_idxs_pf = list(zip(best_env_idxs_pf, best_traj_idxs_pf))


                        if len(set(zipped_env_traj_idxs_BEC) & set(zipped_env_traj_idxs_pf)) > 0:
                            overlap_counter_per_difficulty.append(True)
                        else:
                            overlap_counter_per_difficulty.append(False)

                        PF_best_idxs[difficulty][tag][i] = zipped_env_traj_idxs_pf
                        VO_best_idxs[difficulty][tag][i] = zipped_env_traj_idxs_BEC
                        failed_constraints[difficulty][tag][i] = failed_BEC_constraint

            overlap_counter[difficulty] = overlap_counter_per_difficulty

        with open('models/' + data_loc + '/' + 'PF_2-step_dev_comparison.pickle', 'wb') as f:
            pickle.dump((overlap_counter, PF_best_idxs, VO_best_idxs, failed_constraints), f)
def precompute_counterfactual_constraints(data_loc, mdp_parameters, weights, BEC_params, step_cost_flag, BEC_depth, n_human_models_precomputed):

    Precompute constraints generated by a diversity of potential human beliefs for future quick, real-time inference
    try:
        with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
            policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = pickle.load(f)
    except:
        # use policy BEC to extract constraints
        policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count = BEC.extract_constraints(data_loc, BEC_depth, step_cost_flag, pool, print_flag=True)
        with open('models/' + data_loc + '/base_constraints.pickle', 'wb') as f:
            pickle.dump((policy_constraints, min_subset_constraints_record, env_record, traj_record, traj_features_record, reward_record, mdp_features_record, consistent_state_count), f)

    precompute = False
    if not os.path.exists('models/' + data_loc + '/counterfactual_data_precomputed'):
        precompute = True
    elif len(os.listdir('models/' + data_loc + '/counterfactual_data_precomputed')) == 0:
        precompute = True

    if precompute:
        sample_human_models = BEC_helpers.sample_human_models_uniform([], n_human_models_precomputed)

        print("Precomputing counterfactual data for {} human models: ".format(n_human_models_precomputed))
        for model_idx, human_model in enumerate(sample_human_models):

            cf_data_dir = 'models/' + data_loc + '/counterfactual_data_precomputed/model' + str(model_idx)
            os.makedirs(cf_data_dir, exist_ok=True)

            print(colored('Model #: {}'.format(model_idx), 'red'))
            print(colored('Model val: {}'.format(human_model), 'red'))

            n_processed_envs = len(os.listdir(cf_data_dir))
            args = [
                (data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]),
                 traj_record[i], None, [], step_cost_flag, 'precomputed', np.array([[0, 0, 0]]),  mdp_features_record[i],
                 True) for i in range(n_processed_envs, len(traj_record))]
            _ = list(tqdm(pool.imap(BEC.compute_counterfactuals, args), total=len(args)))
'''

if __name__ == "__main__":
    #from numpy.random import seed
    #seed(0)

    #pool = Pool(min(params.n_cpu, 64))
    #os.makedirs('models/' + params.data_loc['base'], exist_ok=True)
    #os.makedirs('models/' + params.data_loc['BEC'], exist_ok=True)

    #with open('models/' + params.data_loc['BEC'] + '/params.pickle', 'wb') as f:
    #    pickle.dump(params, f)

    # potential pre-step computation of counterfactual constraints generated by potential human beliefs
    # precompute_counterfactual_constraints(params.data_loc['BEC'], params.mdp_parameters, params.weights['val'], params.BEC,
    #                          params.step_cost_flag, params.BEC['BEC_depth'], params.BEC['n_human_models_precomputed'])

    # a) generate an agent if you want to explore the Augmented Taxi MDP
    try:
        file = open('customizations.txt', mode = 'r')
        lines = file.readlines()
        lines = lines[0:7]
        file.close()
        new_list = []
        for line in lines:
            line = line.split(',')
            line = [i.strip() for i in line]
            line = [int(i) for i in line[1:]]
            new_list.append(line)
        
        #setting up agent
        for i in range(0, len(new_list[0]), 3):
            chunk = new_list[0][i:i+3] #currently only will create the agent as the last agent provided
            x = chunk[0]
            y = chunk[1]
            has_passenger = chunk[2]
            params.mdp_parameters["agent"] = dict(x=x,y=y,has_passenger=has_passenger)
            print(params.mdp_parameters["agent"])

        #setting up walls
        params.mdp_parameters["walls"] = []
        for i in range(0, len(new_list[1]), 2):
            chunk = new_list[1][i:i+2] #currently only will create the agent as the last agent provided
            x = chunk[0]
            y = chunk[1]
            params.mdp_parameters["walls"].append(dict(x=x,y=y))
        #print(params.mdp_parameters["walls"])

        #setting up passengers
        passengers = []
        for i in range(0, len(new_list[2]), 5):
            chunk = new_list[2][i:i+5] #currently only will create the agent as the last agent provided
            x = chunk[0]
            y = chunk[1]
            dest_x = chunk[2]
            dest_y = chunk[3]
            in_taxi = chunk[4]
            passengers.append(dict(x=x,y=y,dest_x=dest_x,dest_y=dest_y,in_taxi=in_taxi))
        #print(passengers)

        #setting up tolls:
        params.mdp_parameters["tolls"] = []
        for i in range(0, len(new_list[3]), 2):
            chunk = new_list[3][i:i+2] #currently only will create the agent as the last agent provided
            x = chunk[0]
            y = chunk[1]
            params.mdp_parameters["tolls"].append(dict(x=x,y=y))
        #print(params.mdp_parameters["tolls"])

        #setting up hotswap_station:
        params.mdp_parameters["hotswap_station"] = []
        for i in range(0, len(new_list[4]), 2):
            chunk = new_list[4][i:i+2] #currently only will create the agent as the last agent provided
            x = chunk[0]
            y = chunk[1]
            params.mdp_parameters["hotswap_station"].append(dict(x=x,y=y))
        #print(params.mdp_parameters["hotswap_station"])

        #setting up width:
        params.mdp_parameters["width"] = new_list[5][0]
        #print(params.mdp_parameters["width"])

        #setting up height:
        params.mdp_parameters["height"] = new_list[6][0]
        #print(params.mdp_parameters["height"])

    except:
        print("failed other")
        passengers = [{'x': 6, 'y': 5, 'dest_x': 1, 'dest_y': 1, 'in_taxi': 0},{'x': 5, 'y': 2, 'dest_x': 1, 'dest_y': 1, 'in_taxi': 0}, {'x': 6, 'y': 7, 'dest_x': 1, 'dest_y': 1, 'in_taxi': 0}]
    finally:   
        generate_agent(params.mdp_class, params.data_loc['base'], params.mdp_parameters, passengers=passengers, visualize=True)

    # b) obtain a BEC summary of the agent's policy
    '''
    BEC_summary, visited_env_traj_idxs, particles_summary = obtain_summary(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'],
                            params.step_cost_flag, params.BEC['summary_variant'], pool, params.BEC['n_train_demos'], params.BEC['BEC_depth'],
                            params.BEC['n_human_models'], params.BEC['n_particles'], params.prior, params.posterior, params.BEC['obj_func_proportion'])
    '''
    # c) run through the closed-loop teaching framework
    #simulate_teaching_loop(params.mdp_class, BEC_summary, visited_env_traj_idxs, particles_summary, pool, params.prior, params.BEC['n_particles'], params.BEC['n_human_models'], params.BEC['n_human_models_precomputed'], params.data_loc['BEC'], params.weights['val'], params.step_cost_flag)

    #n_human_models_real_time = 8 #never used

    # d) run remedial demonstration and test selection on previous participant responses from IROS
    # analyze_prev_study_tests(params.mdp_class, BEC_summary, visited_env_traj_idxs, particles_summary, pool, params.prior, params.BEC['n_particles'], n_human_models_real_time, params.BEC['n_human_models_precomputed'], params.data_loc['BEC'], params.weights['val'], params.step_cost_flag, visualize_pf_transition=True)

    # e) compare the remedial demonstration selection when using 2-step dev/BEC vs. PF (assuming 3 static humans models for low, medium, and high difficulties)
    # contrast_PF_2_step_dev(params.mdp_class, BEC_summary, visited_env_traj_idxs, particles_summary, pool, params.prior, params.BEC['n_particles'], n_human_models_real_time, params.data_loc['BEC'], params.weights['val'], params.step_cost_flag, visualize_pf_transition=False)

    # c) obtain test environments
    # obtain_test_environments(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'], params.BEC,
    #                          params.step_cost_flag, params.BEC['n_human_models'], params.prior, params.posterior, summary=BEC_summary, visualize_test_env=True, use_counterfactual=True)


    #pool.close()
    #pool.join()