#!/usr/bin/env python

# Python imports.
import sys
import dill as pickle
import numpy as np
from termcolor import colored

# Other imports.
sys.path.append("simple_rl")
import params
from simple_rl.agents import FixedPolicyAgent
from simple_rl.tasks import AugmentedTaxiOOMDP
from simple_rl.planning import ValueIteration
from policy_summarization import bayesian_IRL
from policy_summarization import policy_summarization_helpers as ps_helpers
from policy_summarization import BEC

def generate_agent(data_loc, aug_taxi, weights, visualize=False):
    try:
        with open('models/' + data_loc + '/vi_agent.pickle', 'rb') as f:
            mdp_agent, vi_agent = pickle.load(f)
    except:
        mdp_agent = AugmentedTaxiOOMDP(width=aug_taxi['width'], height=aug_taxi['height'], agent=aug_taxi['agent'], walls=aug_taxi['walls'],
                                       passengers=aug_taxi['passengers'], tolls=aug_taxi['tolls'], traffic=aug_taxi['traffic'],
                                       fuel_stations=aug_taxi['fuel_station'], gamma=aug_taxi['gamma'], weights=weights)
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

def obtain_BIRL_summary(data_loc, aug_taxi, BIRL_params, n_env, weights, step_cost_flag, visualize_history_priors=False, visualize_summary=False):
    try:
        with open('models/' + data_loc + '/BIRL_summary_{}.pickle'.format(BIRL_params['eval_fn']), 'rb') as f:
            bayesian_IRL_summary, wt_candidates, history_priors = pickle.load(f)
    except:
        wt_candidates = ps_helpers.discretize_wt_candidates(data_loc, weights['val'], weights['lb'], weights['ub'],
                                                            step_cost_flag,
                                                            n_wt_partitions=BIRL_params['n_wt_partitions'],
                                                            iter_idx=BIRL_params['iter_idx'])
        wt_vi_traj_candidates = ps_helpers.obtain_env_policies(data_loc, n_env, wt_candidates, aug_taxi, 'BIRL')

        bayesian_IRL_summary, wt_candidates, history_priors = bayesian_IRL.obtain_summary(
            BIRL_params['n_demonstrations'], weights['val'], wt_candidates, wt_vi_traj_candidates,
            BIRL_params['eval_fn'])

        with open('models/' + data_loc + '/BIRL_summary_{}.pickle'.format(BIRL_params['eval_fn']), 'wb') as f:
            pickle.dump((bayesian_IRL_summary, wt_candidates, history_priors), f)

    if visualize_history_priors or visualize_summary:
        bayesian_IRL.visualize_summary(bayesian_IRL_summary, wt_candidates, history_priors, visualize_summary=visualize_summary, visualize_history_priors=visualize_history_priors)

    return bayesian_IRL_summary, wt_candidates, history_priors

def obtain_BEC_summary(data_loc, aug_taxi, n_env, weights, step_cost_flag, summary_type, BEC_depth=1, visualize_constraints=False, visualize_summary=False):
    try:
        with open('models/' + data_loc + '/BEC_summary.pickle', 'rb') as f:
            BEC_summary = pickle.load(f)

        with open('models/' + data_loc + '/BEC_constraints.pickle', 'rb') as f:
            BEC_constraints, min_subset_constraints_record, env_record, traj_record = pickle.load(f)
    except:
        wt_vi_traj_candidates = ps_helpers.obtain_env_policies(data_loc, n_env, np.expand_dims(weights, axis=0),
                                                               aug_taxi, 'ground_truth')
        try:
            with open('models/' + data_loc + '/BEC_constraints.pickle', 'rb') as f:
                BEC_constraints, min_subset_constraints_record, env_record, traj_record = pickle.load(f)
        except:
            if summary_type == 'demo':
                # a) use optimal trajectories from starting states to extract constraints
                opt_trajs = []
                for wt_vi_traj_candidate in wt_vi_traj_candidates:
                    opt_trajs.append(wt_vi_traj_candidate[0][2])
                BEC_constraints, min_subset_constraints_record, env_record, traj_record = BEC.extract_constraints(wt_vi_traj_candidates, weights, step_cost_flag, BEC_depth=BEC_depth, trajectories=opt_trajs, print_flag=True)
            else:
                # b) use full policy to extract constraints
                BEC_constraints, min_subset_constraints_record, env_record, traj_record = BEC.extract_constraints(wt_vi_traj_candidates, weights, step_cost_flag, print_flag=True)
            with open('models/' + data_loc + '/BEC_constraints.pickle', 'wb') as f:
                pickle.dump((BEC_constraints, min_subset_constraints_record, env_record, traj_record), f)

        BEC_summary = BEC.obtain_summary(wt_vi_traj_candidates, BEC_constraints, min_subset_constraints_record, env_record, traj_record, weights, step_cost_flag, summary_type, BEC_depth)
        with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
            pickle.dump(BEC_summary, f)

    if visualize_constraints:
        BEC.visualize_constraints(BEC_constraints, weights, step_cost_flag)

    if visualize_summary:
        BEC.visualize_summary(BEC_summary, weights, step_cost_flag)

    return BEC_constraints, BEC_summary

def obtain_test_environments(data_loc, aug_taxi, weights, n_env, BEC_params, step_cost_flag, summary=None, visualize_test_env=False):
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
        wt_vi_traj_candidates = ps_helpers.obtain_env_policies(data_loc, n_env, np.expand_dims(weights, axis=0), aug_taxi, 'ground_truth')

        test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = \
            ps_helpers.obtain_test_environments(wt_vi_traj_candidates, weights, BEC_params['n_desired_test_env'], BEC_params['test_difficulty'], step_cost_flag, BEC_params['depth'], summary, BEC_params['summary_type'])

        with open('models/' + data_loc + '/test_environments.pickle', 'wb') as f:
            pickle.dump((test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints), f)

    if visualize_test_env:
        for j, test_wt_vi_traj_tuple in enumerate(test_wt_vi_traj_tuples):
            print(colored('Visualizing test environment {} with BEC length of {}'.format(j, test_BEC_lengths[j]),
                          'red'))

            BEC.visualize_constraints(test_BEC_constraints[j], weights, step_cost_flag)

            vi_candidate = test_wt_vi_traj_tuple[0][1]
            trajectory_candidate = test_wt_vi_traj_tuple[0][2]
            vi_candidate.mdp.visualize_trajectory(trajectory_candidate)

    return test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints

if __name__ == "__main__":
    # a) generate an agent if you want to explore the Augmented Taxi MDP
    # generate_agent(params.data_loc['base'], params.aug_taxi, params.weights['val'], visualize=True)

    # b) obtain a Bayesian IRL summary of the agent's policy
    # bayesian_IRL_summary, wt_candidates, history_priors = obtain_BIRL_summary(params.data_loc['BIRL'], params.aug_taxi,
    #                                                                           params.BIRL, params.n_env, params.weights,
    #                                                                           params.step_cost_flag,
    #                                                                           visualize_history_priors=True,
    #                                                                           visualize_summary=True)

    # c) obtain a BEC summary of the agent's policy
    constraints, BEC_summary = obtain_BEC_summary(params.data_loc['BEC'], params.aug_taxi, params.n_env,
                                                  params.weights['val'], params.step_cost_flag,
                                                  params.BEC['summary_type'], BEC_depth=params.BEC['depth'],
                                                  visualize_constraints=True, visualize_summary=True)
    BEC_length = BEC.calculate_BEC_length(constraints, params.weights['val'], params.step_cost_flag)

    # d) obtain test environments
    obtain_test_environments(params.data_loc['BEC'], params.aug_taxi, params.weights['val'], params.n_env, params.BEC,
                             params.step_cost_flag, summary=BEC_summary, visualize_test_env=True)
