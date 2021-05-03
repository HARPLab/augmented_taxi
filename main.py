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

def explore_BEC_solid_angle(mdp_class, data_loc, mdp_parameters, weights, step_cost_flag, summary_variant, pool, n_train_demos):
    hardcode_envs = False

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
            policy_constraints, min_subset_constraints_record, env_record, traj_record = pickle.load(f)
    except:
        if hardcode_envs:
            # use demo BEC to extract constraints
            policy_constraints, min_subset_constraints_record, env_record, traj_record = BEC.extract_constraints(data_loc, step_cost_flag, pool, vi_traj_triplets=vi_traj_triplets, print_flag=True)
        else:
            # use policy BEC to extract constraints
            policy_constraints, min_subset_constraints_record, env_record, traj_record = BEC.extract_constraints(data_loc, step_cost_flag, pool, print_flag=True)
        with open('models/' + data_loc + '/base_constraints.pickle', 'wb') as f:
            pickle.dump((policy_constraints, min_subset_constraints_record, env_record, traj_record), f)

    try:
        with open('models/' + data_loc + '/BEC_constraints.pickle', 'rb') as f:
            min_BEC_constraints, BEC_lengths_record = pickle.load(f)
    except:
        min_BEC_constraints, BEC_lengths_record = BEC.extract_BEC_constraints(policy_constraints, min_subset_constraints_record, env_record, weights, step_cost_flag, pool)

        with open('models/' + data_loc + '/BEC_constraints.pickle', 'wb') as f:
            pickle.dump((min_BEC_constraints, BEC_lengths_record), f)

    # for j, triplet in enumerate(vi_traj_triplets):
    #     triplet[1].mdp.visualize_trajectory(triplet[2])
    #     BEC.visualize_constraints(min_subset_constraints_record[j], weights, step_cost_flag)

    # for constraints in min_subset_constraints_record:
    #     BEC_viz.visualize_planes(constraints)

    constraints_record = [item for sublist in min_subset_constraints_record for item in sublist]

    try:
        with open('models/' + data_loc + '/BEC_summary.pickle', 'rb') as f:
            BEC_summary = pickle.load(f)
    except:
        # BEC_summary = BEC.obtain_SCOT_summaries(data_loc, summary_variant, min_BEC_constraints, BEC_lengths_record, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag)
        BEC_summary = BEC.obtain_summary_counterfactual(data_loc, summary_variant, min_BEC_constraints, env_record, traj_record, weights, step_cost_flag, pool, n_train_demos=n_train_demos)
        with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
            pickle.dump(BEC_summary, f)

    BEC.visualize_summary(BEC_summary, weights, step_cost_flag)

    # visualizations
    # for constraints_record in min_subset_constraints_record:
    constraints_record = []
    for summary in BEC_summary:
        print('new summary')
        print(summary[2])
        constraints_record.extend(summary[2])
        # constraints_record = summary[2]

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        solid_angle = BEC_helpers.calc_solid_angles([constraints_record])[0]
        print(solid_angle)

        ieqs = BEC_helpers.constraints_to_halfspace_matrix_sage(constraints_record)
        poly = Polyhedron.Polyhedron(ieqs=ieqs)  # automatically finds the minimal H-representation

        # visualize the BEC constraint planes
        hrep = np.array(poly.Hrepresentation())
        boundary_facet_idxs = np.where(hrep[:, 0] != 0)
        # first remove boundary constraint planes (those with a non-zero offset)
        min_constraints = np.delete(hrep, boundary_facet_idxs, axis=0)
        min_constraints = min_constraints[:, 1:]
        min_constraints = [constraint.reshape(1, -1) for constraint in min_constraints]

        for constraints in [min_constraints]:
            BEC_viz.visualize_planes(constraints, fig=fig, ax=ax)

        # visualize spherical polygon
        BEC_viz.visualize_spherical_polygon(poly, fig=fig, ax=ax, plot_ref_sphere=False)

        ax.scatter(weights[0, 0], weights[0, 1], weights[0, 2], marker='o', c='r', s=100)
        ax.set_xlabel('X - Dropoff')
        ax.set_ylabel('Y - Toll')
        ax.set_zlabel('Z - Step Cost')

        if matplotlib.get_backend() == 'TkAgg':
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())

        plt.show()

        BEC_viz.visualize_projection(constraints_record, weights, 'xy', xlabel='Dropoff', ylabel='Toll')
        BEC_viz.visualize_projection(constraints_record, weights, 'xz', xlabel='Dropoff', ylabel='Step Cost')
        BEC_viz.visualize_projection(constraints_record, weights, 'yz', xlabel='Toll', ylabel='Step Cost')

def obtain_BEC_summary(mdp_class, data_loc, mdp_parameters, weights, step_cost_flag, summary_variant, n_train_demos, pool, visualize_summary=False):
    try:
        with open('models/' + data_loc + '/BEC_summary.pickle', 'rb') as f:
            BEC_summary = pickle.load(f)
    except:
        ps_helpers.obtain_env_policies(mdp_class, data_loc, np.expand_dims(weights, axis=0), mdp_parameters, pool)

        try:
            with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
                policy_constraints, min_subset_constraints_record, env_record, traj_record = pickle.load(f)
        except:
            # use full policy to extract constraints
            policy_constraints, min_subset_constraints_record, env_record, traj_record = BEC.extract_constraints(data_loc, step_cost_flag, pool, print_flag=True)
            with open('models/' + data_loc + '/base_constraints.pickle', 'wb') as f:
                pickle.dump((policy_constraints, min_subset_constraints_record, env_record, traj_record), f)

        try:
            with open('models/' + data_loc + '/BEC_constraints.pickle', 'rb') as f:
                min_BEC_constraints, BEC_lengths_record = pickle.load(f)
        except:
            min_BEC_constraints, BEC_lengths_record = BEC.extract_BEC_constraints(policy_constraints, min_subset_constraints_record, weights, step_cost_flag)

            with open('models/' + data_loc + '/BEC_constraints.pickle', 'wb') as f:
                pickle.dump((min_BEC_constraints, BEC_lengths_record), f)

        try:
            with open('models/' + data_loc + '/BEC_summary.pickle', 'rb') as f:
                BEC_summary = pickle.load(f)
        except:
            # BEC_summary = BEC.obtain_summary(summary_variant, wt_vi_traj_candidates, min_BEC_constraints, BEC_lengths_record, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag, n_train_demos=n_train_demos)
            BEC_summary = BEC.obtain_summary_counterfactual(data_loc, summary_variant, min_BEC_constraints, env_record, traj_record, weights, step_cost_flag, pool, n_train_demos=n_train_demos)
            with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
                pickle.dump(BEC_summary, f)

    if visualize_summary:
        BEC.visualize_summary(BEC_summary, weights, step_cost_flag)

    return BEC_summary

# out of date
def obtain_test_environments(mdp_class, data_loc, mdp_parameters, weights, BEC_params, step_cost_flag, summary=None, visualize_test_env=False):
    '''
    Summary: Correlate the difficulty of a test environment with the generalized area of the BEC region obtain by the
    corresponding optimal demonstration. Return the desired number and difficulty of test environments (to be given
    to the human to test his understanding of the agent's policy).
    '''
    # use generalized area of the BEC region to select test environments
    try:
        with open('models/' + data_loc + '/test_environments.pickle', 'rb') as f:
            test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = pickle.load(f)

    except:
        wt_vi_traj_candidates = ps_helpers.obtain_env_policies(mdp_class, data_loc, np.expand_dims(weights, axis=0), mdp_parameters)

        try:
            with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
                policy_constraints, min_subset_constraints_record, env_record, traj_record = pickle.load(f)
        except:
            if params.BEC['summary_type'] == 'demo':
                # a) use optimal trajectories from starting states to extract constraints
                opt_trajs = []
                for wt_vi_traj_candidate in wt_vi_traj_candidates:
                    opt_trajs.append(wt_vi_traj_candidate[0][2])
                policy_constraints, min_subset_constraints_record, env_record, traj_record = BEC.extract_constraints(wt_vi_traj_candidates, weights, step_cost_flag, trajectories=opt_trajs, print_flag=True)
            else:
                # b) use full policy to extract constraints
                policy_constraints, min_subset_constraints_record, env_record, traj_record = BEC.extract_constraints(wt_vi_traj_candidates, weights, step_cost_flag, print_flag=True)
            with open('models/' + data_loc + '/base_constraints.pickle', 'wb') as f:
                pickle.dump((policy_constraints, min_subset_constraints_record, env_record, traj_record), f)

        test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = \
            ps_helpers.obtain_test_environments(wt_vi_traj_candidates, min_subset_constraints_record, env_record, traj_record, weights, BEC_params['n_test_demos'], BEC_params['test_difficulty'], step_cost_flag, summary, BEC_params['summary_type'])

        with open('models/' + data_loc + '/test_environments.pickle', 'wb') as f:
            pickle.dump((test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints), f)

    if visualize_test_env:
        BEC.visualize_test_envs(test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, weights, step_cost_flag)
    return test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints

if __name__ == "__main__":
    pool = Pool(min(params.n_cpu, 60))

    # a) generate an agent if you want to explore the Augmented Taxi MDP
    # generate_agent(params.mdp_class, params.data_loc['base'], params.mdp_parameters, visualize=True)

    # b) obtain a BEC summary of the agent's policy
    # BEC_summary = obtain_BEC_summary(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'],
    #                                               params.step_cost_flag, params.BEC['summary_variant'],
    #                                               params.BEC['n_train_demos'], pool, visualize_summary=True)
    # c) obtain test environments
    # obtain_test_environments(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'], params.BEC,
    #                          params.step_cost_flag, summary=BEC_summary, visualize_test_env=True)

    explore_BEC_solid_angle(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'], params.step_cost_flag, params.BEC['summary_variant'], pool, params.BEC['n_train_demos'])
