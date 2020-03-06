#!/usr/bin/env python

# Python imports.
import sys
import dill as pickle
import numpy as np

# Other imports.
sys.path.append("simple_rl")
from simple_rl.agents import QLearningAgent, RandomAgent, FixedPolicyAgent
from simple_rl.tasks import AugmentedTaxiOOMDP
from simple_rl.planning import ValueIteration
from policy_summarization import highlights
from policy_summarization import modified_highlights

def main(open_plot=True):
    # Taxi initial state attributes..
    max_fuel_capacity = 12
    agent = {"x": 2, "y": 1, "has_passenger": 0}
    passengers = [{"x": 1, "y": 1, "dest_x": 2, "dest_y": 5, "in_taxi": 0}]
    walls = [{"x": 2, "y": 2}, {"x": 3, "y": 4}]
    tolls = [{"x": 1, "y": 2, "fee": 0.4}]
    traffic = [{"x": 2, "y": 4, "prob": 0.5}] # probability that you're stuck
    fuel_station = []
    gamma = 0.95

    # Train
    mdp_agent = AugmentedTaxiOOMDP(width=3, height=5, agent=agent, walls=walls, passengers=passengers, tolls=tolls, traffic=traffic, fuel_stations=fuel_station, gamma=gamma)
    vi_agent = ValueIteration(mdp_agent, sample_rate=20)
    vi_agent.run_vi()

    agent = {"x": 2, "y": 1, "has_passenger": 0}
    passengers = [{"x": 1, "y": 1, "dest_x": 2, "dest_y": 5, "in_taxi": 0}]
    walls = [{"x": 2, "y": 2}, {"x": 3, "y": 4}]
    tolls = []
    traffic = [] # probability that you're stuck
    fuel_station = []
    gamma = 0.95

    mdp_human = AugmentedTaxiOOMDP(width=3, height=5, agent=agent, walls=walls, passengers=passengers, tolls=tolls, traffic=traffic, fuel_stations=fuel_station, gamma=gamma)
    vi_human = ValueIteration(mdp_human, sample_rate=20)
    vi_human.run_vi()

    vi_name = 'feature-based'
    with open('models/vi_{}_agent.pickle'.format(vi_name), 'wb') as f:
        pickle.dump((mdp_agent, vi_agent), f)
    with open('models/vi_{}_human.pickle'.format(vi_name), 'wb') as f:
        pickle.dump((mdp_human, vi_human), f)
    with open('models/vi_{}_agent.pickle'.format(vi_name), 'rb') as f:
        mdp_agent, vi_agent = pickle.load(f)
    with open('models/vi_{}_human.pickle'.format(vi_name), 'rb') as f:
        mdp_human, vi_human = pickle.load(f)

    # Visualize agents
    fixed_agent = FixedPolicyAgent(vi_agent.policy)
    fixed_human = FixedPolicyAgent(vi_human.policy)
    mdp_agent.visualize_agent(fixed_agent)
    mdp_human.visualize_agent(fixed_human)
    # mdp.reset()  # reset the current state to the initial state
    # mdp.visualize_interaction()

    # a) Extracting important states to convey by looping over all reachable states in an MDP
    # Compare the best and worst actions for all reachable state using agent's MDP and find the biggest differences
    q_diffs = modified_highlights.single_policy_extraction(mdp_agent.get_states(), mdp_agent, vi_agent, n_visualize=20)

    # Compare the best and worst actions for all reachable state using humans's MDP and find the biggest differences
    q_diffs = modified_highlights.single_policy_extraction(mdp_human.get_states(), mdp_human, vi_human, n_visualize=20)

    # Compare the best and worst actions between agent and human's MDP for all reachable states of agent's MDP and
    # find the biggest differences. note: the reachable states between the mdp of the agent and human may be different
    q_diffs = modified_highlights.double_policy_extraction(mdp_agent.get_states(), mdp_agent, vi_agent, vi_human, n_visualize=20)

    # Compare the Q-value difference between the human's Q-value for the human's optimal action and the agent's Q-value
    # for the human's optimal action, for all reachable states using human's MDP. note: the reachable states between the mdp of the agent and human may be different
    q_diffs = modified_highlights.double_policy_extraction(mdp_human.get_states(), mdp_agent, vi_agent, vi_human, n_visualize=20, action_conditioned=True)

    # b) Original HIGHLIGHTS summary of the agent's policy
    summary = highlights.obtain_summary(mdp_agent, vi_agent, max_summary_count=5, trajectory_length=1, n_simulations=1, interval_size=1, n_trailing_states=0)
    while not summary.empty():
        state_importance, _, trajectory, marked_state_importances = summary.get()
        mdp_agent.visualize_trajectory(trajectory, marked_state_importances=marked_state_importances)
        # # if you want to extract the critical state from the trajectory, use the code below
        # critical_state_loc = np.where(marked_state_importances != float('-inf'))[0][0]
        # critical_state = trajectory[critical_state_loc][0]

    # c) Extracting important states to convey by only considering states along an agent's path
    # Compare the best and worst actions for states on the path of agent's optimal policy and find the biggest differences
    critical_states = []
    # use obtain_summary() as a way to extract the full trajectory of the agent
    summary = highlights.obtain_summary(mdp_agent, vi_agent, max_summary_count=50, trajectory_length=1, n_simulations=1, interval_size=1, n_trailing_states=0)
    while not summary.empty():
        state_importance, _, trajectory, marked_state_importances = summary.get()
        # extract the critical states
        critical_state = trajectory[0][0]   # critical state location is always 0 since the full trajectory is extracted
        critical_states.append(critical_state)

    q_diffs = modified_highlights.double_policy_extraction(critical_states, mdp_agent, vi_agent, vi_human, n_visualize=20)

    # Compare the best and worst actions for states on the path of humans's optimal policy and find the biggest differences
    critical_states = []
    # use obtain_summary() as a way to extract the full trajectory of the agent
    summary = highlights.obtain_summary(mdp_human, vi_human, max_summary_count=50, trajectory_length=1, n_simulations=1, interval_size=1, n_trailing_states=0)
    while not summary.empty():
        state_importance, _, trajectory, marked_state_importances = summary.get()
        # extract the critical states
        critical_state = trajectory[0][0]
        critical_states.append(critical_state)

    q_diffs = modified_highlights.double_policy_extraction(critical_states, mdp_agent, vi_agent, vi_human, n_visualize=20)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
