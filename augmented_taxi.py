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

def generate_agent(agent_a, walls_a, traffic_a, fuel_station_a, passengers_a, tolls_a, gamma_a, width_a, height_a, weights, vi_name, visualize=False):

    try:
        with open('models/vi_{}_agent.pickle'.format(vi_name), 'rb') as f:
            mdp_agent, vi_agent = pickle.load(f)
    except:
        mdp_agent = AugmentedTaxiOOMDP(width=width_a, height=height_a, agent=agent_a, walls=walls_a,
                                       passengers=passengers_a, tolls=tolls_a, traffic=traffic_a,
                                       fuel_stations=fuel_station_a, gamma=gamma_a, weights=weights)
        vi_agent = ValueIteration(mdp_agent, sample_rate=5)
        vi_agent.run_vi()

        with open('models/vi_{}_agent.pickle'.format(vi_name), 'wb') as f:
            pickle.dump((mdp_agent, vi_agent), f)

    # Visualize agent
    if visualize:
        fixed_agent = FixedPolicyAgent(vi_agent.policy)
        mdp_agent.visualize_agent(fixed_agent)
        # mdp.reset()  # reset the current state to the initial state
        # mdp.visualize_interaction()

def obtain_summary(agent_a, walls_a, traffic_a, fuel_station_a, gamma_a, width_a, height_a, weights, vi_name):
    # obtain uniformly the discretized reward weight candidates
    n_wt_partitions = 2
    try:
        with open('models/wt_candidates_{}.pickle'.format(vi_name), 'rb') as f:
            wt_uniform_sampling = pickle.load(f)
    except:
        wt_uniform_sampling = ps_helpers.discretize_wt_candidates(weights, weights_lb, weights_ub, n_wt_partitions)

        with open('models/wt_candidates_{}.pickle'.format(vi_name), 'wb') as f:
            pickle.dump(wt_uniform_sampling, f)

    # come up with an optimal policy for each of the candidates

    # generate codes that govern passenger's initial position and status of the eight tolls in the 4x3 environment
    env_codes = list(map(list, itertools.product([0, 1], repeat=9)))

    save_mark = 50
    try:
        with open('models/wt_vi_traj_candidates_{}.pickle'.format(vi_name), 'rb') as f:
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
                with open('models/wt_vi_traj_candidates_{}.pickle'.format(vi_name), 'wb') as f:
                    pickle.dump(wt_vi_traj_candidates, f)

                # make a backup in case the overwriting in the code above fails
                shutil.copy2('models/wt_vi_traj_candidates_{}.pickle'.format(vi_name), 'models/wt_vi_traj_candidates_backup_{}.pickle'.format(vi_name))

                print("Saved!")

        with open('models/wt_vi_traj_candidates_{}.pickle'.format(vi_name), 'wb') as f:
            pickle.dump(wt_vi_traj_candidates, f)

        # make a backup in case the overwriting in the code above fails
        shutil.copy2('models/wt_vi_traj_candidates_{}.pickle'.format(vi_name),
                     'models/wt_vi_traj_candidates_backup_{}.pickle'.format(vi_name))

    # compute the Bayesian IRL-based policy summary
    n_demonstrations = 10
    bayesian_IRL_summary, tracking_priors = bayesian_IRL.obtain_summary(n_demonstrations, weights, wt_uniform_sampling, wt_vi_traj_candidates, approximate=True, visualize=True)

    for policy_traj_tuple in bayesian_IRL_summary:
        mdp_demo = policy_traj_tuple[0].mdp
        mdp_demo.visualize_trajectory(policy_traj_tuple[1])

    with open('models/bayesian_IRL_demos_{}.pickle'.format(vi_name), 'wb') as f:
        pickle.dump((bayesian_IRL_summary, tracking_priors), f)


def replay_summary(vi_name):
    with open('models/bayesian_IRL_demos_{}.pickle'.format(vi_name), 'rb') as f:
        bayesian_IRL_summary, tracking_priors = pickle.load(f)

    for policy_traj_tuple in bayesian_IRL_summary:
        mdp_demo = policy_traj_tuple[0].mdp
        mdp_demo.visualize_trajectory(policy_traj_tuple[1])


if __name__ == "__main__":
    vi_name = 'feature-based'

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

    # (on the goal with the passenger, on a toll
    weights = np.array([[0.5, -0.075]])
    weights_lb = np.array([-3., -1.])
    weights_ub = np.array([3., 1.])

    generate_agent(agent_a, walls_a, traffic_a, fuel_station_a, passengers_a, tolls_a, gamma_a, width_a, height_a, weights, vi_name, visualize=True)
    obtain_summary(agent_a, walls_a, traffic_a, fuel_station_a, gamma_a, width_a, height_a, weights, vi_name)
    replay_summary(vi_name)
