#!/usr/bin/env python

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
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = '1.0'
mpl.rcParams['axes.labelsize'] = 'x-large'
mpl.rcParams['xtick.labelsize'] = 'large'


def generate_agent(mdp_class, data_loc, mdp_parameters, visualize=False):
    try:
        with open('models/' + data_loc + '/vi_agent.pickle', 'rb') as f:
            mdp_agent, vi_agent = pickle.load(f)
    except:
        cf_data_dir = 'models/' + data_loc
        os.makedirs(cf_data_dir, exist_ok=True)

        mdp_agent = make_mdp.make_custom_mdp(mdp_class, mdp_parameters)
        vi_agent = ValueIteration(mdp_agent, sample_rate=1)
        vi_agent.run_vi()

        with open('models/' + data_loc + '/vi_agent.pickle', 'wb') as f:
            pickle.dump((mdp_agent, vi_agent), f)

    # Visualize agent
    if visualize:
        fixed_agent = FixedPolicyAgent(vi_agent.policy)
        mdp_agent.visualize_agent(fixed_agent)
        mdp_agent.reset()  # reset the current state to the initial state
        mdp_agent.visualize_interaction()

def obtain_summary(mdp_class, data_loc, mdp_parameters, weights, step_cost_flag, summary_variant, pool, n_train_demos, n_human_models, n_particles, prior, posterior, obj_func_proportion, hardcode_envs=False):
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
            policy_constraints, min_subset_constraints_record, env_record, traj_record, reward_record, consistent_state_count = pickle.load(f)
    except:
        if hardcode_envs:
            # use demo BEC to extract constraints
            policy_constraints, min_subset_constraints_record, env_record, traj_record, reward_record, consistent_state_count = BEC.extract_constraints(data_loc, step_cost_flag, pool, vi_traj_triplets=vi_traj_triplets, print_flag=True)
        else:
            # use policy BEC to extract constraints
            policy_constraints, min_subset_constraints_record, env_record, traj_record, reward_record, consistent_state_count = BEC.extract_constraints(data_loc, step_cost_flag, pool, print_flag=True)
        with open('models/' + data_loc + '/base_constraints.pickle', 'wb') as f:
            pickle.dump((policy_constraints, min_subset_constraints_record, env_record, traj_record, reward_record, consistent_state_count), f)

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
        print(colored('entropy: {}'.format(particles.entropy), 'blue'))

        if summary_variant == 'particle_filter':
            BEC_summary, visited_env_traj_idxs, particles = BEC.obtain_summary_particle_filter(data_loc, particles, summary_variant, min_subset_constraints_record,
                                           min_BEC_constraints, env_record, traj_record, weights, step_cost_flag, pool,
                                           n_human_models, consistent_state_count)

        elif summary_variant == 'proposed' or summary_variant == 'counterfactual_only':
            BEC_summary, visited_env_traj_idxs = BEC.obtain_summary_counterfactual(data_loc, summary_variant, min_subset_constraints_record, min_BEC_constraints, env_record, traj_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count, n_train_demos=n_train_demos, prior=prior, obj_func_proportion=obj_func_proportion)
        elif summary_variant == 'feature_only' or summary_variant == 'baseline':
            BEC_summary = BEC.obtain_summary(data_loc, summary_variant, min_BEC_constraints, BEC_lengths_record, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag, n_train_demos=n_train_demos)
        else:
            raise AssertionError("Unknown summary variant.")

        if summary_variant != 'particle_filter':
            # update the particle filter model according to the generated summary
            for summary in BEC_summary:
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
    # constraints_record = prior
    # for summary in BEC_summary:
    #     print(summary[3])
    #     constraints_record.extend(summary[3])
    #     # constraints_record = summary[3]
    #
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
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
    #     # for constraints in [min_constraints]:
    #     #     BEC_viz.visualize_planes(constraints, fig=fig, ax=ax)
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
    #     if matplotlib.get_backend() == 'TkAgg':
    #         mng = plt.get_current_fig_manager()
    #         mng.resize(*mng.window.maxsize())
    #
    #     plt.show()

    # # particle filter visualization
    # from numpy.random import seed
    # seed(1)
    #
    # n_particles = 50
    # particle_positions = BEC_helpers.sample_human_models_uniform([], n_particles)
    # particles = pf.Particles(particle_positions)
    #
    #
    # constraints_running = prior
    #
    # print(pf.calc_info_gain(particles, prior))
    # particles = pf.update_particle_filter(particles, prior)
    #
    # # fig = plt.figure()
    # # ax = fig.gca(projection='3d')
    # # ax.set_facecolor('white')
    # # ax.xaxis.pane.fill = False
    # # ax.yaxis.pane.fill = False
    # # ax.zaxis.pane.fill = False
    # #
    # # ax.set_xlabel('x')
    # # ax.set_ylabel('y')
    # # ax.set_zlabel('z')
    # #
    # # pf.plot_particles(particles, fig=fig, ax=ax)
    # # BEC_viz.visualize_planes(constraints_running, fig=fig, ax=ax)
    # #
    # # # visualize spherical polygon
    # # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_running)
    # # poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    # # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)
    # #
    # # # visualize the ground truth constraint
    # # w = np.array([[-3, 3.5, -1]])  # toll, hotswap station, step cost
    # # w_normalized = w / np.linalg.norm(w[0, :], ord=2)
    # # ax.scatter(w_normalized[0, 0], w_normalized[0, 1], w_normalized[0, 2], marker='o', c='b', s=100)
    # #
    # # plt.show()
    #
    # for j, summary in enumerate(BEC_summary):
    #     print(j)
    #
    #     constraints = summary[3]
    #
    #     constraints_running.extend(constraints)
    #     constraints_running = BEC_helpers.remove_redundant_constraints(constraints_running, None, False)
    #
    #     # print(pf.calc_info_gain(particles, constraints))
    #
    #     particles = pf.update_particle_filter(particles, constraints)
    #
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     ax.set_facecolor('white')
    #     ax.xaxis.pane.fill = False
    #     ax.yaxis.pane.fill = False
    #     ax.zaxis.pane.fill = False
    #
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('z')
    #
    #     # pf.plot_particles(particles, fig=fig, ax=ax)
    #     # BEC_viz.visualize_planes(constraints_running, fig=fig, ax=ax)
    #     #
    #     # # visualize spherical polygon
    #     # ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_running)
    #     # poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation
    #     # BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False, alpha=0.75)
    #     #
    #     # # visualize the ground truth weight
    #     # w = np.array([[-3, 3.5, -1]])  # toll, hotswap station, step cost
    #     # w_normalized = w / np.linalg.norm(w[0, :], ord=2)
    #     # ax.scatter(w_normalized[0, 0], w_normalized[0, 1], w_normalized[0, 2], marker='o', c='b', s=100)
    #     #
    #     # plt.show()

    return BEC_summary, visited_env_traj_idxs, particles

def obtain_test_environments(mdp_class, data_loc, mdp_parameters, weights, BEC_params, step_cost_flag, n_human_models, prior, posterior, summary=None, use_counterfactual=True, visualize_test_env=False):
    '''
    Summary: Correlate the difficulty of a test environment with the generalized area of the BEC region obtain by the
    corresponding optimal demonstration. Return the desired number and difficulty of test environments (to be given
    to the human to test his understanding of the agent's policy).
    '''
    # use generalized area of the BEC region to select test environments
    try:
        with open('models/' + data_loc + '/test_environments.pickle', 'rb') as f:
            test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers = pickle.load(f)
    except:
        ps_helpers.obtain_env_policies(mdp_class, data_loc, np.expand_dims(weights, axis=0), mdp_parameters, pool)

        try:
            with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
                policy_constraints, min_subset_constraints_record, env_record, traj_record, reward_record, consistent_state_count = pickle.load(f)
        except:
            # use policy BEC to extract constraints
            policy_constraints, min_subset_constraints_record, env_record, traj_record, reward_record, consistent_state_count = BEC.extract_constraints(
                data_loc, step_cost_flag, pool, print_flag=True)
            with open('models/' + data_loc + '/base_constraints.pickle', 'wb') as f:
                pickle.dump((policy_constraints, min_subset_constraints_record, env_record, traj_record, reward_record,
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
                    pool.restart()
                    n_processed_envs = len(os.listdir(cf_data_dir))
                    args = [
                        (data_loc, model_idx, i, human_model, mp_helpers.lookup_env_filename(data_loc, env_record[i]),
                         traj_record[i], posterior, step_cost_flag, counterfactual_folder_idx, None,
                         True) for i in range(n_processed_envs, len(traj_record))]
                    _ = list(tqdm(pool.imap(BEC.compute_counterfactuals, args), total=len(args)))
                    pool.close()
                    pool.join()
                    pool.terminate()

            try:
                with open('models/' + data_loc + '/BEC_constraints_counterfactual.pickle', 'rb') as f:
                    min_subset_constraints_record_counterfactual = pickle.load(f)
            except:
                pool.restart()
                args = [(data_loc, i, min_subset_constraints_record[i], n_human_models, counterfactual_folder_idx, weights, step_cost_flag) for i in range(len(min_subset_constraints_record))]
                # combine the human counterfactual and one-step deviation constraints
                min_subset_constraints_record_counterfactual = list(tqdm(pool.imap(BEC_helpers.combine_counterfactual_constraints, args), total=len(args)))
                pool.close()
                pool.join()
                pool.terminate()

                with open('models/' + data_loc + '/BEC_constraints_counterfactual.pickle', 'wb') as f:
                    pickle.dump(min_subset_constraints_record_counterfactual, f)

            # take the overlap of the human posterior with BEC of suboptimal trajectories of one-step deviation
            pool.restart()
            args = [(i, n_human_models, min_subset_constraints, prior, posterior, data_loc, counterfactual_folder_idx, weights, traj_record[i], step_cost_flag, pool)
                    for i, min_subset_constraints in enumerate(min_subset_constraints_record_counterfactual)]

            print("Obtaining overlap in BEC area between posterior human model and potential test demonstrations: ")
            BEC_lengths_record_counterfactual = list(tqdm(pool.imap(BEC.overlap_demo_BEC_and_human_posterior, args), total=len(args)))
            pool.close()
            pool.join()
            pool.terminate()
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

def obtain_unit_tests(BEC_summary, visited_env_traj_idxs, particles, pool, n_human_models, data_loc, weights, step_cost_flag):
    summary_constraints = []
    for summary in BEC_summary:
        summary_constraints.extend(summary[3])

    min_constraints = BEC_helpers.remove_redundant_constraints(summary_constraints, weights, step_cost_flag)

    # todo: maybe pass in some of these objects later
    with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
        policy_constraints, min_subset_constraints_record, env_record, traj_record, reward_record, consistent_state_count = pickle.load(f)

    # preliminary_tests, visited_env_traj_idxs = BEC.obtain_preliminary_tests(data_loc, visited_env_traj_idxs, min_constraints, min_subset_constraints_record, traj_record)
    # with open('models/' + data_loc + '/preliminary_tests.pickle', 'wb') as f:
    #     pickle.dump((preliminary_tests, visited_env_traj_idxs), f)
    with open('models/' + data_loc + '/preliminary_tests.pickle', 'rb') as f:
        preliminary_tests, visited_env_traj_idxs = pickle.load(f)

    for test in preliminary_tests:
        test_mdp = test[0]
        opt_traj = test[1]
        test_constraint = test[-1]

        human_traj, human_history = test_mdp.visualize_interaction(keys_map=params.keys_map) # the latter is simply the gridworld locations of the agent
        # with open('models/' + data_loc + '/human_traj.pickle', 'wb') as f:
        #     pickle.dump((human_traj, human_history), f)
        # with open('models/' + data_loc + '/human_traj.pickle', 'rb') as f:
        #     human_traj, human_history = pickle.load(f)

        human_feature_count = test_mdp.accumulate_reward_features(human_traj, discount=True)
        opt_feature_count = test_mdp.accumulate_reward_features(opt_traj, discount=True)

        print(human_feature_count)
        print(opt_feature_count)

        if (human_feature_count == opt_feature_count).all():
            print("you got it right")
        else:
            print("you got it wrong")
            test_mdp.visualize_trajectory_comparison(opt_traj, human_traj)
            print("here is another example that might be helpful")
            remedial_instruction, visited_env_traj_idxs = BEC.obtain_remedial_demonstrations(data_loc, pool, particles, n_human_models, test_constraint, min_subset_constraints_record, traj_record, [[test_mdp, opt_traj]], visited_env_traj_idxs, step_cost_flag)
            remedial_mdp, remedial_traj, _, _ = remedial_instruction[0]
            remedial_mdp.visualize_trajectory(remedial_traj)

    BEC.visualize_summary(preliminary_tests, weights, step_cost_flag)

    return preliminary_tests, visited_env_traj_idxs

if __name__ == "__main__":
    pool = Pool(min(params.n_cpu, 60))
    os.makedirs('models/' + params.data_loc['base'], exist_ok=True)
    os.makedirs('models/' + params.data_loc['BEC'], exist_ok=True)

    with open('models/' + params.data_loc['BEC'] + '/params.pickle', 'wb') as f:
        pickle.dump(params, f)

    # a) generate an agent if you want to explore the Augmented Taxi MDP
    # generate_agent(params.mdp_class, params.data_loc['base'], params.mdp_parameters, visualize=True)

    # b) obtain a BEC summary of the agent's policy
    BEC_summary, visited_env_traj_idxs, particles = obtain_summary(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'],
                            params.step_cost_flag, params.BEC['summary_variant'], pool, params.BEC['n_train_demos'],
                            params.BEC['n_human_models'], params.BEC['n_particles'], params.prior, params.posterior, params.BEC['obj_func_proportion'])

    unit_tests = obtain_unit_tests(BEC_summary, visited_env_traj_idxs, particles, pool, params.BEC['n_human_models'], params.data_loc['BEC'], params.weights['val'], params.step_cost_flag)

    # c) obtain test environments
    # obtain_test_environments(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'], params.BEC,
    #                          params.step_cost_flag, params.BEC['n_human_models'], params.prior, params.posterior, summary=BEC_summary, visualize_test_env=True, use_counterfactual=True)