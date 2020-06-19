#!/usr/bin/env python

# Python imports.
import sys
import dill as pickle
import numpy as np
from termcolor import colored

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
        # mdp_agent.visualize_interaction()

def obtain_BIRL_summary(data_loc, eval_fn, n_env, weights, weights_lb, weights_ub, n_wt_partitions, iter_idx, step_cost_flag, visualize_history_priors=False, visualize_summary=False):
    try:
        with open('models/' + data_loc + '/BIRL_summary_{}.pickle'.format(eval_fn), 'rb') as f:
            bayesian_IRL_summary, wt_candidates, history_priors = pickle.load(f)
    except:
        wt_candidates = ps_helpers.discretize_wt_candidates(data_loc, weights, weights_lb, weights_ub, step_cost_flag, n_wt_partitions=n_wt_partitions, iter_idx=iter_idx)
        wt_vi_traj_candidates = ps_helpers.obtain_env_policies(data_loc, n_env, wt_candidates, agent_a, walls_a, traffic_a, fuel_station_a, gamma_a, width_a, height_a, 'BIRL')
        bayesian_IRL_summary, wt_candidates, history_priors = bayesian_IRL.obtain_summary(n_demonstrations, weights, wt_candidates, wt_vi_traj_candidates, eval_fn)

        with open('models/' + data_loc + '/BIRL_summary_{}.pickle'.format(eval_fn), 'wb') as f:
            pickle.dump((bayesian_IRL_summary, wt_candidates, history_priors), f)

    if visualize_history_priors or visualize_summary:
        bayesian_IRL.visualize_summary(bayesian_IRL_summary, wt_candidates, history_priors, visualize_summary=visualize_summary, visualize_history_priors=visualize_history_priors)

    return bayesian_IRL_summary, wt_candidates, history_priors

def obtain_BEC_summary(data_loc, n_env, weights, step_cost_flag, summary_type, BEC_depth=1, visualize_constraints=False, visualize_summary=False):
    try:
        with open('models/' + data_loc + '/BEC_summary.pickle', 'rb') as f:
            BEC_summary = pickle.load(f)

        with open('models/' + data_loc + '/BEC_constraints.pickle', 'rb') as f:
            constraints = pickle.load(f)
    except:
        wt_vi_traj_candidates = ps_helpers.obtain_env_policies(data_loc, n_env, np.expand_dims(weights, axis=0), agent_a, walls_a, traffic_a, fuel_station_a, gamma_a, width_a, height_a, 'ground_truth')
        try:
            with open('models/' + data_loc + '/BEC_constraints.pickle', 'rb') as f:
                constraints = pickle.load(f)
        except:
            if summary_type == 'demo':
                # a) use optimal trajectories from starting states to extract constraints
                opt_trajs = []
                for wt_vi_traj_candidate in wt_vi_traj_candidates:
                    opt_trajs.append(wt_vi_traj_candidate[0][2])
                constraints = BEC.extract_constraints(wt_vi_traj_candidates, weights, step_cost_flag, BEC_depth=BEC_depth, trajectories=opt_trajs)
            else:
                # b) use full policy to extract constraints
                constraints = BEC.extract_constraints(wt_vi_traj_candidates, weights, step_cost_flag)
            with open('models/' + data_loc + '/BEC_constraints.pickle', 'wb') as f:
                pickle.dump(constraints, f)

        BEC_summary = BEC.obtain_summary(wt_vi_traj_candidates, constraints, weights, step_cost_flag, summary_type, BEC_depth)
        with open('models/' + data_loc + '/BEC_summary.pickle', 'wb') as f:
            pickle.dump(BEC_summary, f)

    if visualize_constraints:
        BEC.visualize_constraints(constraints, weights, step_cost_flag)

    if visualize_summary:
        BEC.visualize_summary(BEC_summary, weights, step_cost_flag)

    return constraints, BEC_summary

def obtain_test_environments(data_loc, weights, n_env, n_desired_test_env, difficulty, step_cost_flag, BEC_depth, summary=None, BEC_summary_type=None, visualize_test_env=False):
    '''
    Summary: Correlate the difficulty of a test environment with the generalized area of the BEC region obtain by the
    corresponding optimal demonstration. Return the desired number and difficulty of test environments (to be given
    to the human to test his understanding of the agent's policy).
    '''
    # use generalized area of the BEC region to select test environments
    try:
        with open('models/' + data_loc + '/test_environments.pickle', 'rb') as f:
            test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = pickle.load(f)

    except:
        wt_vi_traj_candidates = ps_helpers.obtain_env_policies(data_loc, n_env, np.expand_dims(weights, axis=0), agent_a,
                                                               walls_a, traffic_a, fuel_station_a, gamma_a, width_a,
                                                               height_a, 'ground_truth')

        test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = ps_helpers.obtain_test_environments(wt_vi_traj_candidates, weights, n_desired_test_env, difficulty, step_cost_flag, BEC_depth, summary, BEC_summary_type)

        with open('models/' + data_loc + '/test_environments.pickle', 'wb') as f:
            pickle.dump((test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints), f)

    if visualize_test_env:
        for j, test_wt_vi_traj_tuple in enumerate(test_wt_vi_traj_tuples):
            print(colored('Visualizing test environment {} with BEC length of {}'.format(j, test_BEC_lengths[j]),
                          'red'))

            BEC.visualize_constraints(test_BEC_constraints[j], weights, step_cost_flag)

            vi_candidate = test_wt_vi_traj_tuple[0][1]
            trajectory_candidate = test_wt_vi_traj_tuple[0][2]
            vi_candidate.mdp.visualize_trajectory(trajectory_candidate)

    return test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_lengths

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

    # reward weight parameters (on the goal with the passenger, on a toll, step cost).
    # assume the L1 norm of the weights is equal 1. WLOG
    weights_lb = np.array([-1., -1., -0.03076923])
    weights_ub = np.array([1., 1., -0.03076923])
    weights = np.array([[0.875, -0.09375, -0.03125]])

    gamma_a = 1.
    step_cost_flag = True   # indicates that the last weight element is a known step cost. code currently assumes a 2D
                            # weight vector if step_cost_flag = False, and a 3D weight vector if step_cost_flag = True

    # test environment selection parameters
    n_desired_test_env = 10      # number of desired test environments
    sample_radius = 0.75         # radius around the ground truth weight from which you will uniformly sample from
                                 # to obtain test weight candidates
    n_samples = 5                # number of test weight candidates to sample

    # Joint BIRL and BEC parameters
    n_env = 512                  # number of environments to consider
                                 # tip: select so that np.log(n_env) / np.log(2) yields an int for predictable behavior
                                 # see ps_helpers.obtain_env_policies()
    # BIRL parameters
    n_wt = 1                    # total number of weight candidates (including the ground truth)
                                 # tip: select n_wt to such that n_wt_partitions is an int to ensure that the exact
                                 # number of desired weight candidates is actually incorporated. see ps_helpers.discretize_wt_candidates()
                                 # also note that n_wt = n_wt_partitions ** (# of weights you're discretizing over) + 1

    iter_idx = None              # weight dimension to discretize over. If None, discretize uniformly over all dimensions
    eval_fn = 'approx_MP'        # desired likelihood function for computing the posterior probability of weight candidates
    n_demonstrations = 10        # total number of demonstrations sought, in order of decreasing effectiveness

    if iter_idx == None:
        data_loc_BIRL = str(n_env) + '_env/' + str(n_wt) + '_wt_' + 'uniform'
        if step_cost_flag:
            n_wt_partitions = int((n_wt - 1) ** (1.0 / (weights.shape[1] - 1)))
        else:
            n_wt_partitions = int((n_wt - 1) ** (1.0 / weights.shape[1]))
    else:
        data_loc_BIRL = str(n_env) + '_env/' + str(n_wt) + '_wt_' + 'iter_idx_' + str(iter_idx)
        n_wt_partitions = n_wt - 1
    data_loc = str(n_env) + '_env'

    BEC_summary_type = 'policy' # demo or policy: whether constratints are extraced from just the optimal demo from the
                                # starting state or from all possible states from the full policy
    BEC_depth = 1               # number of suboptimal actions to take before following the optimal policy to obtain the
                                # suboptimal trajectory (and the corresponding suboptimal expected feature counts)
                                # for computational feasibility, only considering depths > 1 for demo summary type
    BEC_test_difficulty = 'hard'

    # a) generate an agent if you want to explore the Augmented Taxi MDP
    # generate_agent('base', agent_a, walls_a, traffic_a, fuel_station_a, passengers_a, tolls_a, gamma_a, width_a, height_a, weights, visualize=True)

    # b) obtain a Bayesian IRL summary of the agent's policy
    # bayesian_IRL_summary, wt_candidates, history_priors = obtain_BIRL_summary(data_loc_BIRL, eval_fn, n_env, weights, weights_lb, weights_ub, n_wt_partitions, iter_idx, step_cost_flag, visualize_history_priors=False, visualize_summary=True)

    # c) obtain a BEC summary of the agent's policy
    constraints, BEC_summary = obtain_BEC_summary(data_loc, n_env, weights, step_cost_flag, BEC_summary_type, BEC_depth=BEC_depth, visualize_constraints=False, visualize_summary=False)
    BEC_length = BEC.calculate_BEC_length(constraints, weights, step_cost_flag)

    # d) obtain test environments
    obtain_test_environments(data_loc, weights, n_env, n_desired_test_env, BEC_test_difficulty, step_cost_flag, BEC_depth, summary=BEC_summary, BEC_summary_type=BEC_summary_type, visualize_test_env=False)
