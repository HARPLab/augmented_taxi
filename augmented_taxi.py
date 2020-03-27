#!/usr/bin/env python

# Python imports.
import sys
import dill as pickle
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import copy

# Other imports.
sys.path.append("simple_rl")
from simple_rl.agents import FixedPolicyAgent
from simple_rl.tasks import AugmentedTaxiOOMDP
from simple_rl.planning import ValueIteration
from policy_summarization import highlights
from policy_summarization import modified_highlights
from policy_summarization import bayesian_IRL
from policy_summarization import policy_summarization_helpers as ps_helpers

def main(open_plot=True):
    # Taxi initial state attributes..
    agent_a = {"x": 2, "y": 1, "has_passenger": 0}
    passengers_a = [{"x": 1, "y": 1, "dest_x": 2, "dest_y": 5, "in_taxi": 0}]
    walls_a = [{"x": 2, "y": 2}, {"x": 3, "y": 4}]
    tolls_a = [{"x": 1, "y": 2, "fee": 0.4}]
    traffic_a = [{"x": 2, "y": 4, "prob": 0.5}] # probability that you're stuck
    fuel_station_a = []
    gamma_a = 0.95
    width_a = 3
    height_a = 5

    # (on the goal with the passenger, on a toll, on a traffic cell)
    weights = np.array([[2, -0.4, -0.5]])
    weights_lb = np.array([-3., -1., -1.])
    weights_ub = np.array([3., 1., 1.])

    agent_h = copy.deepcopy(agent_a)
    passengers_h = copy.deepcopy(passengers_a)
    walls_h = copy.deepcopy(walls_a)
    tolls_h = []
    traffic_h = [] # probability that you're stuck
    fuel_station_h = []
    gamma_h = gamma_a
    width_h = width_a
    height_h = height_a

    vi_name = 'feature-based'
    try:
        with open('models/vi_{}_agent.pickle'.format(vi_name), 'rb') as f:
            mdp_agent, vi_agent = pickle.load(f)
    except:
        mdp_agent = AugmentedTaxiOOMDP(width=width_a, height=height_a, agent=agent_a, walls=walls_a,
                                       passengers=passengers_a, tolls=tolls_a, traffic=traffic_a,
                                       fuel_stations=fuel_station_a, gamma=gamma_a, weights=weights)
        vi_agent = ValueIteration(mdp_agent, sample_rate=20)
        vi_agent.run_vi()

        with open('models/vi_{}_agent.pickle'.format(vi_name), 'wb') as f:
            pickle.dump((mdp_agent, vi_agent), f)

    try:
        with open('models/vi_{}_human.pickle'.format(vi_name), 'rb') as f:
            mdp_human, vi_human = pickle.load(f)
    except:
        mdp_human = AugmentedTaxiOOMDP(width=width_h, height=height_h, agent=agent_h, walls=walls_h,
                                       passengers=passengers_h, tolls=tolls_h, traffic=traffic_h,
                                       fuel_stations=fuel_station_h, gamma=gamma_h)
        vi_human = ValueIteration(mdp_human, sample_rate=20)
        vi_human.run_vi()

        with open('models/vi_{}_human.pickle'.format(vi_name), 'wb') as f:
            pickle.dump((mdp_human, vi_human), f)

    # Visualize agents
    # fixed_agent = FixedPolicyAgent(vi_agent.policy)
    # fixed_human = FixedPolicyAgent(vi_human.policy)
    # mdp_agent.visualize_agent(fixed_agent)
    # mdp_human.visualize_agent(fixed_human)
    # mdp.reset()  # reset the current state to the initial state
    # mdp.visualize_interaction()

    # Bayesian IRL (see Enabling robots to communicate their objectives; Sandy Huang et al AURO 2019)
    n_demonstrations = 10
    n_wt_partitions = 3

    # obtain uniformly the discretized reward weight candidates
    try:
        with open('models/wt_candidates_{}.pickle'.format(vi_name), 'rb') as f:
            wt_uniform_sampling = pickle.load(f)
    except:
        wt_uniform_sampling = ps_helpers.discretize_wt_candidates(weights, weights_lb, weights_ub, n_wt_partitions)

        with open('models/wt_candidates_{}.pickle'.format(vi_name), 'wb') as f:
            pickle.dump(wt_uniform_sampling, f)

    # come up with an optimal policy for each of the candidates
    try:
        with open('models/wt_vi_candidates_{}.pickle'.format(vi_name), 'rb') as f:
            wt_vi_candidates = pickle.load(f)
    except:
        counter = 0
        wt_vi_candidates = []

        for wt_candidate in wt_uniform_sampling:
            mdp_candidate = AugmentedTaxiOOMDP(width=width_a, height=height_a, agent=agent_a, walls=walls_a, passengers=passengers_a,
                                               tolls=tolls_a, traffic=traffic_a, fuel_stations=fuel_station_a, gamma=gamma_a,
                                               weights=wt_candidate)
            vi_candidate = ValueIteration(mdp_candidate, sample_rate=5)
            iterations, value_of_init_state = vi_candidate.run_vi()
            wt_vi_candidates.append((wt_candidate, vi_candidate))
            counter += 1
            print('Counter: {}, iterations: {}, init_val: {}, wt_candidate: {}'.format(counter, iterations,
                                                                                       value_of_init_state,
                                                                                       wt_candidate))

        with open('models/wt_vi_candidates_{}.pickle'.format(vi_name), 'wb') as f:
            pickle.dump(wt_vi_candidates, f)

    # use obtain_summary() as a way to extract the full trajectory of the agent
    summary = highlights.obtain_summary(mdp_agent, vi_agent, max_summary_count=50, trajectory_length=1, n_simulations=1, interval_size=1, n_trailing_states=0)
    trajectories = []
    while not summary.empty():
        state_importance, _, trajectory, marked_state_importances = summary.get()
        trajectories.append(trajectory)

    # compute the Bayesian IRL-based policy summary
    bayesian_IRL_summary = bayesian_IRL.obtain_summary(mdp_agent, n_demonstrations, weights, wt_vi_candidates, trajectories, visualize=True)

    for trajectory in bayesian_IRL_summary:
        mdp_agent.visualize_trajectory(trajectory)


if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
