#!/usr/bin/env python

# Python imports.
import sys
import dill as pickle
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import time
import itertools
import shutil

# Other imports.
sys.path.append("simple_rl")
from simple_rl.agents import FixedPolicyAgent
from simple_rl.tasks import AugmentedTaxiOOMDP
from simple_rl.planning import ValueIteration
from policy_summarization import bayesian_IRL
from policy_summarization import policy_summarization_helpers as ps_helpers
from simple_rl.utils import mdp_helpers
from policy_summarization import BEC

def generate_env(env_code):

    # first entry currently dictates where the passenger begins
    if env_code[0] == 0:
        requested_passenger = [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
    else:
        requested_passenger = [{"x": 2, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]

    # the last eight entries currently dictate the presence of tolls
    available_tolls = [{"x": 2, "y": 3, "fee": 1}, {"x": 3, "y": 3, "fee": 1}, {"x": 4, "y": 3, "fee": 1},
               {"x": 2, "y": 2, "fee": 1}, {"x": 3, "y": 2, "fee": 1}, {"x": 4, "y": 2, "fee": 1},
               {"x": 2, "y": 1, "fee": 1}, {"x": 3, "y": 1, "fee": 1}]

    requested_tolls = []

    offset = 1
    for x in range(offset, len(env_code)):
        entry = env_code[x]
        if entry:
            requested_tolls.append(available_tolls[x - offset])

    return requested_passenger, requested_tolls

def generate_agent(data_loc, agent_a, walls_a, traffic_a, fuel_station_a, passengers_a, tolls_a, gamma_a, width_a, height_a, weights, visualize=False):

    try:
        with open('models/' + data_loc + '/vi_agent.pickle', 'rb') as f:
            mdp_agent, vi_agent = pickle.load(f)
    except:
        mdp_agent = AugmentedTaxiOOMDP(width=width_a, height=height_a, agent=agent_a, walls=walls_a,
                                       passengers=passengers_a, tolls=tolls_a, traffic=traffic_a,
                                       fuel_stations=fuel_station_a, gamma=gamma_a, weights=weights)
        vi_agent = ValueIteration(mdp_agent, sample_rate=5)
        vi_agent.run_vi()

        with open('models/' + data_loc + '/vi_agent.pickle', 'wb') as f:
            pickle.dump((mdp_agent, vi_agent), f)

    # Visualize agent
    if visualize:
        fixed_agent = FixedPolicyAgent(vi_agent.policy)
        mdp_agent.visualize_agent(fixed_agent)
        # mdp.reset()  # reset the current state to the initial state
        # mdp.visualize_interaction()

def obtain_summary(data_loc, n_demonstrations, agent_a, walls_a, traffic_a, fuel_station_a, gamma_a, width_a, height_a, weights, eval_fn):
    # obtain uniformly the discretized reward weight candidates
    n_wt_partitions = 9
    try:
        with open('models/' + data_loc + '/wt_candidates.pickle', 'rb') as f:
            wt_uniform_sampling = pickle.load(f)
    except:
        wt_uniform_sampling = ps_helpers.discretize_wt_candidates(weights, weights_lb, weights_ub, n_wt_partitions)

        with open('models/' + data_loc + '/wt_candidates.pickle', 'wb') as f:
            pickle.dump(wt_uniform_sampling, f)

    # come up with an optimal policy for each of the candidates

    # generate codes that govern passenger's initial position and status of the eight tolls in the 4x3 environment
    env_codes = list(map(list, itertools.product([0, 1], repeat=9)))

    save_mark = 50
    try:
        with open('models/' + data_loc + '/wt_vi_traj_candidates.pickle', 'rb') as f:
            wt_vi_traj_candidates = pickle.load(f)

        if len(wt_vi_traj_candidates) == len(env_codes) and len(env_codes[-1]) == len(wt_uniform_sampling):
            # all environments and weights have been processed
            n_processed_envs = len(env_codes)
        else:
            # a portion of the environments and weights have been processed
            n_processed_envs = len(wt_vi_traj_candidates)
    except:
        wt_vi_traj_candidates = []
        n_processed_envs = 0

    # enumeration of all possible optimal policies from possible environments x weight candidates
    # if there are environments and weights yet to be processed
    if n_processed_envs < len(env_codes):
        for env_idx in range(n_processed_envs, len(env_codes)):
            env_code = env_codes[env_idx]
            passengers_a, tolls_a = generate_env(env_code)
            wt_counter = 0
            # a per-environment tuple of corresponding reward weight, optimal policy, and optimal trajectory
            wt_vi_traj_env = []
            for wt_candidate in wt_uniform_sampling:
                mdp_candidate = AugmentedTaxiOOMDP(width=width_a, height=height_a, agent=agent_a, walls=walls_a,
                                               passengers=passengers_a, tolls=tolls_a, traffic=traffic_a,
                                               fuel_stations=fuel_station_a, gamma=gamma_a, weights=wt_candidate)
                vi_candidate = ValueIteration(mdp_candidate, sample_rate=1)
                iterations, value_of_init_state = vi_candidate.run_vi()
                trajectory = mdp_helpers.rollout_policy(mdp_candidate, vi_candidate)
                wt_vi_traj_env.append((wt_candidate, vi_candidate, trajectory))

                wt_counter += 1
                print('wt_counter: {}, iterations: {}, init_val: {}, wt_candidate: {}'.format(wt_counter, iterations,
                                                                                       value_of_init_state,
                                                                                       wt_candidate))
            wt_vi_traj_candidates.append(wt_vi_traj_env)
            n_processed_envs += 1
            print('Finished analyzing environment {}'.format(n_processed_envs))

            if n_processed_envs % save_mark == 0:
                with open('models/' + data_loc + '/wt_vi_traj_candidates.pickle', 'wb') as f:
                    pickle.dump(wt_vi_traj_candidates, f)

                # make a backup in case the overwriting in the code above fails
                shutil.copy2('models/' + data_loc + '/wt_vi_traj_candidates.pickle', 'models/' + data_loc + '/wt_vi_traj_candidates_backup.pickle')

                print("Saved!")

        with open('models/' + data_loc + '/wt_vi_traj_candidates.pickle', 'wb') as f:
            pickle.dump(wt_vi_traj_candidates, f)

        # make a backup in case the overwriting in the code above fails
        shutil.copy2('models/' + data_loc + '/wt_vi_traj_candidates.pickle', 'models/' + data_loc + '/wt_vi_traj_candidates_backup.pickle')

    # compute the Bayesian IRL-based policy summary
    bayesian_IRL_summary, wt_uniform_sampling, history_priors = bayesian_IRL.obtain_summary(n_demonstrations, weights, wt_uniform_sampling, wt_vi_traj_candidates, eval_fn)

    with open('models/' + data_loc + '/BIRL_summary_{}.pickle'.format(eval_fn), 'wb') as f:
        pickle.dump((bayesian_IRL_summary, wt_uniform_sampling, history_priors), f)

if __name__ == "__main__":
    data_loc = '512_env_50_wts_uniform'
    eval_fn = 'approx_MP'
    n_demonstrations = 10

    # Augmented Taxi details
    agent_a = {"x": 4, "y": 1, "has_passenger": 0}
    walls_a = [{"x": 1, "y": 3, "fee": 1}, {"x": 1, "y": 2, "fee": 1}]
    traffic_a = [] # probability that you're stuck
    fuel_station_a = []
    gamma_a = 0.95
    width_a = 4
    height_a = 3

    passengers_a = [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
    tolls_a = [{"x": 3, "y": 1, "fee": 1}]

    # (on the goal with the passenger, on a toll). assume the L2 norm of the weights is equal 1. WLOG
    weights = np.array([[0.98893635, -0.14834045]])
    weights_lb = np.array([-1., -1.])
    weights_ub = np.array([1., 1.])

    # generate_agent(data_loc, agent_a, walls_a, traffic_a, fuel_station_a, passengers_a, tolls_a, gamma_a, width_a, height_a, weights, visualize=True)

    # obtain_summary(data_loc, n_demonstrations, agent_a, walls_a, traffic_a, fuel_station_a, gamma_a, width_a, height_a, weights, eval_fn)

    # with open('models/' + data_loc + '/BIRL_summary_{}.pickle'.format(eval_fn), 'rb') as f:
    #     bayesian_IRL_summary, wt_uniform_sampling, history_priors = pickle.load(f)
    # bayesian_IRL.visualize_summary(bayesian_IRL_summary, wt_uniform_sampling, history_priors, visualize_demos=True, visualize_history_priors=True)

    with open('models/' + data_loc + '/wt_vi_traj_candidates.pickle', 'rb') as f:
        wt_vi_traj_candidates = pickle.load(f)
    try:
        with open('models/' + data_loc + '/BEC_constraints.pickle'.format(eval_fn), 'rb') as f:
            constraints = pickle.load(f)
    except:
        constraints = BEC.extract_constraints(wt_vi_traj_candidates, visualize=True, gt_weight=weights)

        with open('models/' + data_loc + '/BEC_constraints.pickle'.format(eval_fn), 'wb') as f:
            pickle.dump(constraints, f)

    BEC.visualize_constraints(constraints, gt_weight=weights)

    opt_mdp_trajs = BEC.obtain_summary(wt_vi_traj_candidates, constraints)
    for mdp_traj in opt_mdp_trajs:
        mdp_traj[0].visualize_trajectory(mdp_traj[1])
    with open('models/' + data_loc + '/BEC_summary.pickle'.format(eval_fn), 'wb') as f:
        pickle.dump(opt_mdp_trajs, f)
