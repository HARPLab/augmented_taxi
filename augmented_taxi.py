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
    agent = {"x": 2, "y": 1, "has_passenger": 0}
    passengers = [{"x": 1, "y": 1, "dest_x": 2, "dest_y": 5, "in_taxi": 0}]
    walls = [{"x": 2, "y": 2}, {"x": 3, "y": 4}]
    tolls = [{"x": 1, "y": 2, "fee": 0.5}]
    traffic = [{"x": 2, "y": 4, "prob": 0.99}, {"x": 1, "y": 4, "prob": 0.5}] # probability that you're stuck
    gamma = 0.99

    mdp = AugmentedTaxiOOMDP(width=3, height=5, agent=agent, walls=walls, passengers=passengers, tolls=tolls, traffic=traffic, gamma=gamma)

    # Agents.
    ql_agent = QLearningAgent(actions=mdp.get_actions(), gamma=gamma)
    # rand_agent = RandomAgent(actions=mdp.get_actions())

    # Train
    # run_single_agent_on_mdp(ql_agent, mdp, episodes=500, steps=1000)
    value_iter = ValueIteration(mdp)
    value_iter.run_vi()
    fixed_agent = FixedPolicyAgent(value_iter.policy)

    # with open('models/ql_agent.pickle', 'wb') as f:
    #     pickle.dump(ql_agent, f)
    # with open('models/ql_agent.pickle', 'rb') as f:
    #     ql_agent = pickle.load(f)

    # Test on training domain
    # fixed_agent = FixedPolicyAgent(ql_agent.epsilon_greedy_q_policy)
    # fixed_agent = FixedPolicyAgent(vi.policy)
    # run_agents_on_mdp([fixed_agent, rand_agent], mdp, instances=10, episodes=1, steps=500, reset_at_terminal=False, open_plot=open_plot)

    # Visualize agent
    # mdp.visualize_agent(ql_agent)
    # mdp.visualize_agent(fixed_agent)

    # Test on testing domain
    agent = {"x": 2, "y": 1, "has_passenger": 0}
    passengers = [{"x": 1, "y": 3, "dest_x": 2, "dest_y": 5, "in_taxi": 0}]
    test_mdp = AugmentedTaxiOOMDP(width=3, height=5, agent=agent, walls=walls, passengers=passengers, tolls=tolls, traffic=traffic, gamma=gamma)
    test_mdp.visualize_agent(fixed_agent)


if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
