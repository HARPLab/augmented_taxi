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

def obtain_summary(mdp_class, data_loc, mdp_parameters, weights, step_cost_flag, summary_variant, pool, n_train_demos, n_human_models, prior, hardcode_envs=False):
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
            BEC_summary = pickle.load(f)
    except:
        # SCOT_summary = BEC.obtain_SCOT_summaries(data_loc, summary_variant, min_BEC_constraints, BEC_lengths_record, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag)

        BEC_summary = BEC.obtain_summary_counterfactual(data_loc, summary_variant, min_BEC_constraints, env_record, traj_record, weights, step_cost_flag, pool, n_human_models, consistent_state_count, n_train_demos=n_train_demos, prior=prior)

        if len(BEC_summary) > 0:
            with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
                pickle.dump(BEC_summary, f)

        # it = 0
        # while it < 1:
        #     BEC_summary = BEC.obtain_summary_counterfactual(data_loc, summary_variant, min_BEC_constraints, env_record, traj_record, weights, step_cost_flag, pool, n_train_demos=n_train_demos)
        #     if len(BEC_summary) > 0:
        #         with open('models/' + data_loc + '/BEC_summary' + str(it) + '_it.pickle', 'wb') as f:
        #             pickle.dump(BEC_summary, f)
        #     it += 1


    BEC.visualize_summary(BEC_summary, weights, step_cost_flag)

    # constraint  visualization
    constraints_record = priors
    for summary in BEC_summary:
        print(summary[2])
        constraints_record.extend(summary[2])

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        solid_angle = BEC_helpers.calc_solid_angles([constraints_record])[0]
        print(solid_angle)

        ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_record)
        poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation

        min_constraints = BEC_helpers.remove_redundant_constraints(constraints_record, weights, step_cost_flag)
        print(min_constraints)
        for constraints in [min_constraints]:
            BEC_viz.visualize_planes(constraints, fig=fig, ax=ax)

        # visualize spherical polygon
        BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False)

        ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='r', s=100)
        if mdp_class == 'augmented_taxi':
            ax.set_xlabel('X - Dropoff')
            ax.set_ylabel('Y - Toll')
        elif mdp_class == 'two_goal':
            ax.set_xlabel('X - Goal 1 (grey)')
            ax.set_ylabel('Y - Goal 2 (green)')
        else:
            ax.set_xlabel('X - Goal')
            ax.set_ylabel('Y - Skateboard')
        ax.set_zlabel('Z - Step Cost')

        if matplotlib.get_backend() == 'TkAgg':
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())

        plt.show()

    return BEC_summary

def obtain_test_environments(mdp_class, data_loc, mdp_parameters, weights, BEC_params, step_cost_flag, n_human_models, prior, posterior, summary=None, use_counterfactual=True, visualize_test_env=False):
    '''
    Summary: Correlate the difficulty of a test environment with the generalized area of the BEC region obtain by the
    corresponding optimal demonstration. Return the desired number and difficulty of test environments (to be given
    to the human to test his understanding of the agent's policy).
    '''
    # use generalized area of the BEC region to select test environments
    try:
        with open('models/' + data_loc + '/test_environments.pickle', 'rb') as f:
            # test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, test_human_counterfactual_trajs, test_overlap_pcts = pickle.load(f)
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
            # with open('models/' + data_loc + '/BEC_constraints_counterfactual.pickle', 'rb') as f:
            #     min_subset_constraints_record_counterfactual, BEC_lengths_record_counterfactual, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = pickle.load(f)
            with open('models/' + data_loc + '/BEC_constraints_counterfactual.pickle', 'rb') as f:
                min_subset_constraints_record_counterfactual, BEC_lengths_record_counterfactual, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = pickle.load(f)
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

            # pool.restart()
            # args = [(i, n_human_models, prior, posterior, data_loc, counterfactual_folder_idx, weights, traj_record[i], step_cost_flag, pool) for i in range(len(env_record))]
            #
            # print("Obtaining overlap in BEC area between posterior human model and potential test demonstrations: ")
            # min_subset_constraints_record_counterfactual, BEC_lengths_record_counterfactual, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = zip(*pool.imap(BEC.combine_limiting_constraints_BEC, tqdm(args)))
            # min_subset_constraints_record_counterfactual, BEC_lengths_record_counterfactual, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = zip(*pool.imap(BEC.average_constraints_BEC, tqdm(args)))
            # pool.close()
            # pool.join()
            # pool.terminate()
            #
            # with open('models/' + data_loc + '/BEC_constraints_counterfactual.pickle', 'wb') as f:
            #     pickle.dump((min_subset_constraints_record_counterfactual, BEC_lengths_record_counterfactual, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs), f)

            # add in constraints from human counterfactuals to constraints from one-step deviations
            min_subset_constraints_record_counterfactual = []
            for env_idx, min_subset_constraints in enumerate(min_subset_constraints_record):
                print(env_idx)
                constraints_env_across_models = []
                for model_idx in range(8):
                    with open('models/' + data_loc + '/counterfactual_data_' + str(counterfactual_folder_idx) + '/model' + str(
                            model_idx) + '/cf_data_env' + str(
                        env_idx).zfill(5) + '.pickle', 'rb') as f:
                        best_human_trajs_record_env, constraints_env, human_rewards_env = pickle.load(f)

                    # only consider the first best human trajectory (no need to consider partial trajectories)
                    constraints_env_across_models.append(constraints_env)

                # reorder such that each subarray is a comparison amongst the models
                constraints_env_across_models_per_traj = [list(itertools.chain.from_iterable(i)) for i in
                                                                      zip(*constraints_env_across_models)]


                new_min_subset_constraints = []
                for traj_idx, constraints_across_models in enumerate(constraints_env_across_models_per_traj):
                    joint_constraints = []
                    joint_constraints.extend(constraints_across_models)
                    joint_constraints.extend(min_subset_constraints[traj_idx])
                    joint_constraints = BEC_helpers.remove_redundant_constraints(joint_constraints, weights, step_cost_flag)
                    new_min_subset_constraints.append(joint_constraints)

                min_subset_constraints_record_counterfactual.append(new_min_subset_constraints)

            # take the overlap of the human posterior with BEC of suboptimal trajectories of one-step deviation
            pool.restart()
            args = [(i, n_human_models, min_subset_constraints, prior, posterior, data_loc, counterfactual_folder_idx, weights, traj_record[i], step_cost_flag, pool)
                    for i, min_subset_constraints in enumerate(min_subset_constraints_record_counterfactual)]

            print("Obtaining overlap in BEC area between posterior human model and potential test demonstrations: ")
            BEC_lengths_record_counterfactual, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs = zip(*pool.imap(BEC.overlap_demo_BEC_and_human_posterior, tqdm(args)))
            pool.close()
            pool.join()
            pool.terminate()
            with open('models/' + data_loc + '/BEC_constraints_counterfactual.pickle', 'wb') as f:
                pickle.dump((min_subset_constraints_record_counterfactual, BEC_lengths_record_counterfactual, overlap_in_opt_and_counterfactual_traj_avg, human_counterfactual_trajs), f)

        if use_counterfactual:
            test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers= \
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
        # BEC.visualize_test_envs(posterior, test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, test_human_counterfactual_trajs, test_overlap_pcts, weights, step_cost_flag)
        BEC.visualize_test_envs(posterior, test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, selected_env_traj_tracers, weights,
                                step_cost_flag)
    return test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints

if __name__ == "__main__":
    pool = Pool(min(params.n_cpu, 60))
    os.makedirs('models/' + params.data_loc['base'], exist_ok=True)
    os.makedirs('models/' + params.data_loc['BEC'], exist_ok=True)

    with open('models/' + params.data_loc['BEC'] + '/params.pickle', 'wb') as f:
        pickle.dump(params, f)

    # a) generate an agent if you want to explore the Augmented Taxi MDP
    # generate_agent(params.mdp_class, params.data_loc['base'], params.mdp_parameters, visualize=True)

    # b) obtain a BEC summary of the agent's policy
    BEC_summary = obtain_summary(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'],
                            params.step_cost_flag, params.BEC['summary_variant'], pool, params.BEC['n_train_demos'],
                            params.BEC['n_human_models'], params.prior)

    # c) obtain test environments
    obtain_test_environments(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'], params.BEC,
                             params.step_cost_flag, params.BEC['n_human_models'], params.prior, params.posterior, summary=BEC_summary, visualize_test_env=True, use_counterfactual=True)