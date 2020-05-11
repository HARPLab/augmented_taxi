#!/usr/bin/env python

# Python imports.
import sys
import dill as pickle
import numpy as np

# Other imports.
sys.path.append("simple_rl")
from simple_rl.agents import FixedPolicyAgent
from simple_rl.tasks import AugmentedTaxiOOMDP
from simple_rl.planning import ValueIteration
from policy_summarization import bayesian_IRL
from policy_summarization import policy_summarization_helpers as ps_helpers
from policy_summarization import BEC

def generate_agent(data_loc, agent_a, walls_a, traffic_a, fuel_station_a, passengers_a, tolls_a, gamma_a, width_a, height_a, weights, visualize=False):

    try:
        with open('models/' + data_loc + '/vi_agent.pickle', 'rb') as f:
            mdp_agent, vi_agent = pickle.load(f)
    except:
        mdp_agent = AugmentedTaxiOOMDP(width=width_a, height=height_a, agent=agent_a, walls=walls_a,
                                       passengers=passengers_a, tolls=tolls_a, traffic=traffic_a,
                                       fuel_stations=fuel_station_a, gamma=gamma_a, weights=weights)
        vi_agent = ValueIteration(mdp_agent, sample_rate=1)
        vi_agent.run_vi()

        with open('models/' + data_loc + '/vi_agent.pickle', 'wb') as f:
            pickle.dump((mdp_agent, vi_agent), f)

    # Visualize agent
    if visualize:
        fixed_agent = FixedPolicyAgent(vi_agent.policy)
        mdp_agent.visualize_agent(fixed_agent)
        # mdp.reset()  # reset the current state to the initial state
        # mdp.visualize_interaction()

def obtain_BIRL_summary(data_loc, eval_fn, n_env, weights, weights_lb, weights_ub, n_wt_partitions, visualize_history_priors=False, visualize_summary=False):
    try:
        with open('models/' + data_loc + '/BIRL_summary_{}.pickle'.format(eval_fn), 'rb') as f:
            bayesian_IRL_summary, wt_candidates, history_priors = pickle.load(f)
    except:
        wt_candidates = ps_helpers.obtain_wt_candidates(data_loc, weights, weights_lb, weights_ub, n_wt_partitions=n_wt_partitions)
        wt_vi_traj_candidates = ps_helpers.obtain_env_policies(data_loc, n_env, wt_candidates, agent_a, walls_a, traffic_a, fuel_station_a, gamma_a, width_a, height_a)
        bayesian_IRL_summary, wt_candidates, history_priors = bayesian_IRL.obtain_summary(n_demonstrations, weights, wt_candidates, wt_vi_traj_candidates, eval_fn)

        with open('models/' + data_loc + '/BIRL_summary_{}.pickle'.format(eval_fn), 'wb') as f:
            pickle.dump((bayesian_IRL_summary, wt_candidates, history_priors), f)

    if visualize_history_priors or visualize_summary:
        bayesian_IRL.visualize_summary(bayesian_IRL_summary, wt_candidates, history_priors, visualize_summary=visualize_summary, visualize_history_priors=visualize_history_priors)

    return bayesian_IRL_summary, wt_candidates, history_priors

def obtain_BEC_summary(data_loc, n_env, weights, visualize_constraints=False, visualize_summary=False):
    try:
        with open('models/' + data_loc + '/BEC_summary.pickle', 'rb') as f:
            BEC_summary = pickle.load(f)

        with open('models/' + data_loc + '/BEC_constraints.pickle', 'rb') as f:
            constraints = pickle.load(f)
    except:
        wt_vi_traj_candidates = ps_helpers.obtain_env_policies(data_loc, n_env, np.expand_dims(weights, axis=0), agent_a, walls_a, traffic_a, fuel_station_a, gamma_a, width_a, height_a)
        try:
            with open('models/' + data_loc + '/BEC_constraints.pickle', 'rb') as f:
                constraints = pickle.load(f)
        except:
            constraints = BEC.extract_constraints(wt_vi_traj_candidates)
            with open('models/' + data_loc + '/BEC_constraints.pickle', 'wb') as f:
                pickle.dump(constraints, f)

        BEC_summary = BEC.obtain_summary(wt_vi_traj_candidates, constraints)
        with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
            pickle.dump(BEC_summary, f)

    if visualize_constraints:
        BEC.visualize_constraints(constraints, gt_weight=weights)

    if visualize_summary:
        BEC.visualize_summary(BEC_summary)

    return constraints, BEC_summary

if __name__ == "__main__":
    # Augmented Taxi details (note that I'm allowing functions below to directly access these taxi variables w/o passing them in)
    agent_a = {"x": 4, "y": 1, "has_passenger": 0}
    walls_a = [{"x": 1, "y": 3, "fee": 1}, {"x": 1, "y": 2, "fee": 1}]
    passengers_a = [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
    tolls_a = [{"x": 3, "y": 1, "fee": 1}]
    traffic_a = [] # probability that you're stuck
    fuel_station_a = []
    width_a = 4
    height_a = 3

    # (on the goal with the passenger, on a toll, step cost). assume the L2 norm of the weights is equal 1. WLOG
    weights_lb = np.array([-1., -1., -1.])
    weights_ub = np.array([1., 1., 1.])

    weights = np.array([[0.9908798, -0.13011553, -0.0350311]])
    gamma_a = 1.

    # Joint BIRL and BEC parameters
    n_env = 512                  # number of environments to consider
                                 # tip: select so that np.log(n_env) / np.log(2) yields an int for predictable behavior
                                 # see ps_helpers.obtain_env_policies()
    # BIRL parameters
    n_wt = 1                    # total number of weight candidates (including the ground truth)
                                 # tip: select n_wt to such that n_wt_partitions is an int to ensure that the exact
                                 # number of desired weight candidates is actually incorporated. see ps_helpers.obtain_wt_candidates()
                                 # also note that n_wt = n_wt_partitions ** weights.shape[1] + 1

    iter_idx = None              # weight dimension to discretize over. If None, discretize uniformly over all dimensions
    eval_fn = 'approx_MP'        # desired likelihood function for computing the posterior probability of weight candidates
    n_demonstrations = 10        # total number of demonstrations sought, in order of decreasing effectiveness

    if iter_idx == None:
        data_loc_BIRL = str(n_env) + '_env_' + str(n_wt) + '_wt_' + 'uniform'
        n_wt_partitions = (n_wt - 1) ** (1.0 / weights.shape[1])
    else:
        data_loc_BIRL = str(n_env) + '_env_' + str(n_wt) + '_wt_' + 'iter_idx_' + str(iter_idx)
        n_wt_partitions = n_wt - 1
    data_loc_BEC = str(n_env) + '_env_gt_wt'

    # a) generate an agent if you want to explore the Augmented Taxi MDP
    # generate_agent('base', agent_a, walls_a, traffic_a, fuel_station_a, passengers_a, tolls_a, gamma_a, width_a, height_a, weights, visualize=True)

    # b) obtain a Bayesian IRL summary of the agent's policy
    # bayesian_IRL_summary, wt_candidates, history_priors = obtain_BIRL_summary(data_loc_BIRL, eval_fn, n_env, weights, weights_lb, weights_ub, n_wt_partitions, visualize_history_priors=False, visualize_summary=True)

    # c) obtain a BEC summary of the agent's policy
    constraints, BEC_summary = obtain_BEC_summary(data_loc_BEC, n_env, weights, visualize_constraints=True, visualize_summary=True)
