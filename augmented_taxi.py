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
    max_fuel_capacity = 15
    agent = {"x": 2, "y": 1, "has_passenger": 0, "fuel": 8}
    passengers = [{"x": 1, "y": 1, "dest_x": 2, "dest_y": 5, "in_taxi": 0}]
    walls = [{"x": 2, "y": 2}, {"x": 3, "y": 4}]
    tolls = [{"x": 1, "y": 2, "fee": 0.5}]
    traffic = [{"x": 2, "y": 4, "prob": 0.95}, {"x": 1, "y": 4, "prob": 0.5}] # probability that you're stuck
    fuel_station = [{"x": 2, "y": 3, "max_fuel_capacity": max_fuel_capacity}]
    gamma = 0.99

    mdp = AugmentedTaxiOOMDP(width=3, height=5, agent=agent, walls=walls, passengers=passengers, tolls=tolls, traffic=traffic, fuel_stations=fuel_station, gamma=gamma)

    # Train
    value_iter = ValueIteration(mdp)
    # value_iter.run_vi()
    # fixed_agent = FixedPolicyAgent(value_iter.policy)

    # Visualize agent
    # mdp.visualize_agent(fixed_agent)
    mdp.visualize_interaction()

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
