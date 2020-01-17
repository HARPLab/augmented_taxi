#!/usr/bin/env python

# Python imports.
import sys
import dill as pickle

# Other imports.
sys.path.append("simple_rl")
from simple_rl.agents import QLearningAgent, RandomAgent, FixedPolicyAgent
from simple_rl.tasks import AugmentedTaxiOOMDP
from simple_rl.planning import ValueIteration
from simple_rl.run_experiments import run_agents_on_mdp, run_single_agent_on_mdp

def main(open_plot=True):
    # Taxi initial state attributes..
    max_fuel_capacity = 12
    agent = {"x": 2, "y": 1, "has_passenger": 0, "fuel": 5}
    passengers = [{"x": 1, "y": 1, "dest_x": 2, "dest_y": 5, "in_taxi": 0}]
    walls = [{"x": 2, "y": 2}, {"x": 3, "y": 4}]
    tolls = [{"x": 1, "y": 2, "fee": 0.05}]
    traffic = [{"x": 2, "y": 4, "prob": 0.95}, {"x": 1, "y": 4, "prob": 0.5}] # probability that you're stuck
    fuel_station = [{"x": 2, "y": 3, "max_fuel_capacity": max_fuel_capacity}]
    gamma = 0.95

    mdp = AugmentedTaxiOOMDP(width=3, height=5, agent=agent, walls=walls, passengers=passengers, tolls=tolls, traffic=traffic, fuel_stations=fuel_station, gamma=gamma)

    # Train
    value_iter = ValueIteration(mdp)
    value_iter.run_vi()

    vi_name = 'purple'
    with open('models/vi_{}.pickle'.format(vi_name), 'wb') as f:
        pickle.dump((mdp, value_iter), f)
    with open('models/vi_{}.pickle'.format(vi_name), 'rb') as f:
        mdp, value_iter = pickle.load(f)

    # Visualize agent
    fixed_agent = FixedPolicyAgent(value_iter.policy)
    mdp.visualize_agent(fixed_agent)
    mdp.reset()  # reset the current state to the initial state
    mdp.visualize_interaction()
    mdp.reset()
    mdp.visualize_value(value_iter)
    mdp.reset()
    mdp.visualize_policy(value_iter.policy)

    # Compare the best and worst actions for each state and find the biggest differences
    q_val_diffs = []
    for s in value_iter.get_states():
        max_q_val, best_action = value_iter._compute_max_qval_action_pair(s)
        min_q_val, worst_action = value_iter._compute_min_qval_action_pair(s)
        q_val_diffs.append([max_q_val - min_q_val, best_action, worst_action, s])
    q_val_diffs.sort(key=lambda x: x[0], reverse=True)

    # Visualize the top 50 states
    for state_number in range(50):
        print("Best action: {}".format(q_val_diffs[state_number][1]))
        print("Worst action: {}".format(q_val_diffs[state_number][2]))
        print("Q-val difference: {}".format(q_val_diffs[state_number][0]))
        mdp.visualize_state(q_val_diffs[state_number][3])

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
