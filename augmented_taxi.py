#!/usr/bin/env python

# Python imports.
import sys

# Other imports.
sys.path.append("simple_rl")
from simple_rl.agents import QLearningAgent, RandomAgent, FixedPolicyAgent
from simple_rl.tasks import AugmentedTaxiOOMDP
from simple_rl.run_experiments import run_agents_on_mdp, run_single_agent_on_mdp

def main(open_plot=True):
    # Taxi initial state attributes..
    agent = {"x": 2, "y": 1, "has_passenger": 0}
    passengers = [{"x": 1, "y": 1, "dest_x": 2, "dest_y": 4, "in_taxi": 0}]
    walls = [{"x": 2, "y": 2}]
    tolls = [{"x": 1, "y": 2, "fee": 0.03}]
    gamma = 0.95
    mdp = AugmentedTaxiOOMDP(width=3, height=4, agent=agent, walls=walls, passengers=passengers, tolls=tolls, gamma=gamma)

    # Agents.
    ql_agent = QLearningAgent(actions=mdp.get_actions(), gamma=gamma)
    rand_agent = RandomAgent(actions=mdp.get_actions())

    # Train
    run_single_agent_on_mdp(ql_agent, mdp, episodes=500, steps=1000)

    # Test on training domain
    fixed_agent = FixedPolicyAgent(ql_agent.epsilon_greedy_q_policy)
    run_agents_on_mdp([fixed_agent, rand_agent], mdp, instances=10, episodes=1, steps=500, reset_at_terminal=False, open_plot=open_plot)
    viz = True
    if viz:
        mdp.visualize_agent(fixed_agent)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
