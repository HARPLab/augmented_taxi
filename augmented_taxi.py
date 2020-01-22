#!/usr/bin/env python

# Python imports.
import sys
import dill as pickle

# Other imports.
sys.path.append("simple_rl")
from simple_rl.agents import QLearningAgent, RandomAgent, FixedPolicyAgent
from simple_rl.tasks import AugmentedTaxiOOMDP
from simple_rl.planning import ValueIteration
from policy_summarization import highlights

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
    # with open('models/vi_{}.pickle'.format(vi_name), 'rb') as f:
    #     mdp, value_iter = pickle.load(f)

    # Visualize agent
    # fixed_agent = FixedPolicyAgent(value_iter.policy)
    # mdp.visualize_agent(fixed_agent)
    # mdp.reset()  # reset the current state to the initial state
    # mdp.visualize_interaction()

    # (mdp, agent, max_summary_count=10, trajectory_length=5, n_simulations=10, interval_size=3, n_trailing_states=2):
    summary = highlights.obtain_summary(mdp, value_iter, max_summary_count=15, trajectory_length=5, n_simulations=1, interval_size=3, n_trailing_states=2)
    while not summary.empty():
        state_importance, _, trajectory, marked_state_importances = summary.get()
        mdp.visualize_trajectory(trajectory, marked_state_importances=marked_state_importances)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
