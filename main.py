#!/usr/bin/env python

# Python imports.
import sys
import dill as pickle
import numpy as np
import copy
from termcolor import colored

# Other imports.
sys.path.append("simple_rl")
import params
from simple_rl.agents import FixedPolicyAgent
from simple_rl.planning import ValueIteration
from simple_rl.utils import make_mdp
from policy_summarization import bayesian_IRL
from policy_summarization import policy_summarization_helpers as ps_helpers
from policy_summarization import BEC

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

def obtain_BIRL_summary(mdp_class, data_loc, mdp_parameters, BIRL_params, step_cost_flag, visualize_history_priors=False, visualize_summary=False):
    try:
        with open('models/' + data_loc + '/BIRL_summary_{}.pickle'.format(BIRL_params['eval_fn']), 'rb') as f:
            bayesian_IRL_summary, wt_candidates, history_priors = pickle.load(f)
    except:
        wt_candidates = ps_helpers.discretize_wt_candidates(data_loc, mdp_parameters['weights'], mdp_parameters['weights_lb'], mdp_parameters['weights_ub'],
                                                            step_cost_flag,
                                                            n_wt_partitions=BIRL_params['n_wt_partitions'],
                                                            iter_idx=BIRL_params['iter_idx'])
        wt_vi_traj_candidates = ps_helpers.obtain_env_policies(mdp_class, data_loc, wt_candidates, mdp_parameters, 'BIRL')

        bayesian_IRL_summary, wt_candidates, history_priors = bayesian_IRL.obtain_summary(
            BIRL_params['n_demonstrations'], mdp_parameters['weights'], wt_candidates, wt_vi_traj_candidates,
            BIRL_params['eval_fn'])

        with open('models/' + data_loc + '/BIRL_summary_{}.pickle'.format(BIRL_params['eval_fn']), 'wb') as f:
            pickle.dump((bayesian_IRL_summary, wt_candidates, history_priors), f)

    if visualize_history_priors or visualize_summary:
        bayesian_IRL.visualize_summary(bayesian_IRL_summary, wt_candidates, history_priors, visualize_summary=visualize_summary, visualize_history_priors=visualize_history_priors)

    return bayesian_IRL_summary, wt_candidates, history_priors

def obtain_BEC_summary(mdp_class, data_loc, mdp_parameters, weights, step_cost_flag, summary_type, summary_variant, n_train_demos, BEC_depth=1, visualize_summary=False):
    try:
        with open('models/' + data_loc + '/BEC_summary.pickle', 'rb') as f:
            BEC_summary = pickle.load(f)
    except:
        wt_vi_traj_candidates = ps_helpers.obtain_env_policies(mdp_class, data_loc, np.expand_dims(weights, axis=0),
                                                               mdp_parameters, 'ground_truth')
        try:
            with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
                min_subset_constraints_record, env_record, traj_record = pickle.load(f)
        except:
            if summary_type == 'demo':
                # a) use optimal trajectories from starting states to extract constraints
                opt_trajs = []
                for wt_vi_traj_candidate in wt_vi_traj_candidates:
                    opt_trajs.append(wt_vi_traj_candidate[0][2])
                min_subset_constraints_record, env_record, traj_record = BEC.extract_constraints(wt_vi_traj_candidates, weights, step_cost_flag, BEC_depth=BEC_depth, trajectories=opt_trajs, print_flag=True)
            else:
                # b) use full policy to extract constraints
                min_subset_constraints_record, env_record, traj_record = BEC.extract_constraints(wt_vi_traj_candidates, weights, step_cost_flag, print_flag=True)
            with open('models/' + data_loc + '/base_constraints.pickle', 'wb') as f:
                pickle.dump((min_subset_constraints_record, env_record, traj_record), f)

        try:
            with open('models/' + data_loc + '/BEC_constraints.pickle', 'rb') as f:
                min_BEC_constraints, BEC_lengths_record = pickle.load(f)
        except:
            min_BEC_constraints, BEC_lengths_record = BEC.extract_BEC_constraints(min_subset_constraints_record, weights, step_cost_flag)

            with open('models/' + data_loc + '/BEC_constraints.pickle', 'wb') as f:
                pickle.dump((min_BEC_constraints, BEC_lengths_record), f)

        try:
            with open('models/' + data_loc + '/BEC_summary.pickle', 'rb') as f:
                BEC_summary = pickle.load(f)
        except:
            BEC_summary = BEC.obtain_summary(summary_variant, wt_vi_traj_candidates, min_BEC_constraints, BEC_lengths_record, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag, n_train_demos=n_train_demos)
            with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
                pickle.dump(BEC_summary, f)

    if visualize_summary:
        BEC.visualize_summary(BEC_summary, weights, step_cost_flag)

    return BEC_summary

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
        wt_vi_traj_candidates = ps_helpers.obtain_env_policies(mdp_class, data_loc, np.expand_dims(weights, axis=0), mdp_parameters, 'ground_truth')

        try:
            with open('models/' + data_loc + '/base_constraints.pickle', 'rb') as f:
                min_subset_constraints_record, env_record, traj_record = pickle.load(f)
        except:
            if params.BEC['summary_type'] == 'demo':
                # a) use optimal trajectories from starting states to extract constraints
                opt_trajs = []
                for wt_vi_traj_candidate in wt_vi_traj_candidates:
                    opt_trajs.append(wt_vi_traj_candidate[0][2])
                min_subset_constraints_record, env_record, traj_record = BEC.extract_constraints(wt_vi_traj_candidates, weights, step_cost_flag, BEC_depth=BEC_params['depth'], trajectories=opt_trajs, print_flag=True)
            else:
                # b) use full policy to extract constraints
                min_subset_constraints_record, env_record, traj_record = BEC.extract_constraints(wt_vi_traj_candidates, weights, step_cost_flag, print_flag=True)
            with open('models/' + data_loc + '/base_constraints.pickle', 'wb') as f:
                pickle.dump((min_subset_constraints_record, env_record, traj_record), f)

        test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = \
            ps_helpers.obtain_test_environments(wt_vi_traj_candidates, min_subset_constraints_record, env_record, traj_record, weights, BEC_params['n_test_demos'], BEC_params['test_difficulty'], step_cost_flag, summary, BEC_params['summary_type'])

        with open('models/' + data_loc + '/test_environments.pickle', 'wb') as f:
            pickle.dump((test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints), f)

    if visualize_test_env:
        BEC.visualize_test_envs(test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints, weights, step_cost_flag)
    return test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints

if __name__ == "__main__":
    # a) generate an agent if you want to explore the Augmented Taxi MDP
    # generate_agent(params.mdp_class, params.data_loc['base'], params.mdp_parameters, visualize=True)

    # b) obtain a Bayesian IRL summary of the agent's policy
    # bayesian_IRL_summary, wt_candidates, history_priors = obtain_BIRL_summary(params.mdp_class, params.data_loc['BIRL'], params.mdp_parameters,
    #                                                                           params.BIRL, params.step_cost_flag,
    #                                                                           visualize_history_priors=False, visualize_summary=True)
    # c) obtain a BEC summary of the agent's policy
    BEC_summary = obtain_BEC_summary(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'],
                                                  params.step_cost_flag, params.BEC['summary_type'], params.BEC['summary_variant'],
                                                  params.BEC['n_train_demos'], BEC_depth=params.BEC['depth'], visualize_summary=True)
    # d) obtain test environments
    # obtain_test_environments(params.mdp_class, params.data_loc['BEC'], params.mdp_parameters, params.weights['val'], params.BEC,
    #                          params.step_cost_flag, summary=BEC_summary, visualize_test_env=True)
