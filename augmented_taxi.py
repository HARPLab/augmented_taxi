#!/usr/bin/env python

# Python imports.
import sys
import dill as pickle
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt

# Other imports.
sys.path.append("simple_rl")
from simple_rl.agents import QLearningAgent, RandomAgent, FixedPolicyAgent
from simple_rl.tasks import AugmentedTaxiOOMDP
from simple_rl.planning import ValueIteration
from policy_summarization import highlights
from policy_summarization import modified_highlights

def main(open_plot=True):
    # Taxi initial state attributes..
    agent_a = {"x": 2, "y": 1, "has_passenger": 0}
    passengers_a = [{"x": 1, "y": 1, "dest_x": 2, "dest_y": 5, "in_taxi": 0}]
    walls_a = [{"x": 2, "y": 2}, {"x": 3, "y": 4}]
    tolls_a = [{"x": 1, "y": 2, "fee": 0.4}]
    traffic_a = [{"x": 2, "y": 4, "prob": 0.5}] # probability that you're stuck
    fuel_station_a = []
    gamma_a = 0.95

    # (on the goal with the passenger, on a toll, on a traffic cell)
    weights = np.array([[2, -0.4, -0.5]])

    # Train
    mdp_agent = AugmentedTaxiOOMDP(width=3, height=5, agent=agent_a, walls=walls_a, passengers=passengers_a, tolls=tolls_a, traffic=traffic_a, fuel_stations=fuel_station_a, gamma=gamma_a, weights=weights)
    vi_agent = ValueIteration(mdp_agent, sample_rate=20)
    vi_agent.run_vi()

    agent_h = {"x": 2, "y": 1, "has_passenger": 0}
    passengers_h = [{"x": 1, "y": 1, "dest_x": 2, "dest_y": 5, "in_taxi": 0}]
    walls_h = [{"x": 2, "y": 2}, {"x": 3, "y": 4}]
    tolls_h = []
    traffic_h = [] # probability that you're stuck
    fuel_station_h = []
    gamma_h = 0.95

    # mdp_human = AugmentedTaxiOOMDP(width=3, height=5, agent=agent_h, walls=walls_h, passengers=passengers_h, tolls=tolls_h, traffic=traffic_h, fuel_stations=fuel_station_h, gamma=gamma_h)
    # vi_human = ValueIteration(mdp_human, sample_rate=20)
    # vi_human.run_vi()

    vi_name = 'feature-based'
    with open('models/vi_{}_agent.pickle'.format(vi_name), 'wb') as f:
        pickle.dump((mdp_agent, vi_agent), f)
    # with open('models/vi_{}_human.pickle'.format(vi_name), 'wb') as f:
    #     pickle.dump((mdp_human, vi_human), f)
    with open('models/vi_{}_agent.pickle'.format(vi_name), 'rb') as f:
        mdp_agent, vi_agent = pickle.load(f)
    # with open('models/vi_{}_human.pickle'.format(vi_name), 'rb') as f:
    #     mdp_human, vi_human = pickle.load(f)

    # Visualize agents
    # fixed_agent = FixedPolicyAgent(vi_agent.policy)
    # fixed_human = FixedPolicyAgent(vi_human.policy)
    # mdp_agent.visualize_agent(fixed_agent)
    # mdp_human.visualize_agent(fixed_human)
    # mdp.reset()  # reset the current state to the initial state
    # mdp.visualize_interaction()

    # Bayesian IRL (see Enabling robots to communicate their objectives; Sandy Huang et al AURO 2019)
    n_demonstrations = 10
    weights = np.array([[2, -0.4, -0.5]])
    weights_lb = np.array([-3., -1., -1.])
    weights_ub = np.array([3., 1., 1.])
    n_wt_partitions = 3

    mesh = np.array(np.meshgrid(np.linspace(weights_lb[0], weights_ub[0], n_wt_partitions),
                    np.linspace(weights_lb[1], weights_ub[1], n_wt_partitions),
                    np.linspace(weights_lb[2], weights_ub[2], n_wt_partitions)))
    wt_uniform_sampling = np.hstack((mesh[0].reshape(-1, 1), mesh[1].reshape(-1, 1), mesh[2].reshape(-1, 1)))
    wt_uniform_sampling = np.vstack((wt_uniform_sampling, weights))
    wt_uniform_sampling = wt_uniform_sampling.reshape(wt_uniform_sampling.shape[0], 1, wt_uniform_sampling.shape[1]) # for future dot products

    with open('models/wt_candidates_{}.pickle'.format(vi_name), 'wb') as f:
        pickle.dump(wt_uniform_sampling, f)
    with open('models/wt_candidates_{}.pickle'.format(vi_name), 'rb') as f:
        wt_uniform_sampling = pickle.load(f)

    counter = 0
    wt_vi_candidates = []

    # come up with an optimal policy for each of the candidates
    for wt_candidate in wt_uniform_sampling:
        mdp_candidate = AugmentedTaxiOOMDP(width=3, height=5, agent=agent_h, walls=walls_h, passengers=passengers_h, tolls=tolls_h,
                           traffic=traffic_h, fuel_stations=fuel_station_h, gamma=gamma_h, weights=wt_candidate)
        vi_candidate = ValueIteration(mdp_candidate, sample_rate=5)
        vi_candidate.run_vi()
        wt_vi_candidates.append((wt_candidate, vi_candidate))
        counter += 1
        print(counter)

    with open('models/wt_vi_candidates_{}.pickle'.format(vi_name), 'wb') as f:
        pickle.dump(wt_vi_candidates, f)

    with open('models/wt_vi_candidates_{}.pickle'.format(vi_name), 'rb') as f:
        wt_vi_candidates = pickle.load(f)

    # use obtain_summary() as a way to extract the full trajectory of the agent
    summary = highlights.obtain_summary(mdp_agent, vi_agent, max_summary_count=50, trajectory_length=1, n_simulations=1, interval_size=1, n_trailing_states=0)
    trajectories = []
    while not summary.empty():
        state_importance, _, trajectory, marked_state_importances = summary.get()
        trajectories.append(trajectory)

    priors = {}                # up-to-date priors on candidates
    debugging_priors = {}      # a history of updated priors for debugging

    # initialize the prior to be a uniform distribution
    for wt_vi_candidate_pair in wt_vi_candidates:
        wt_candidate = wt_vi_candidate_pair[0]
        priors[wt_candidate.tostring()] = 1. / len(wt_vi_candidates)
        debugging_priors[wt_candidate.tostring()] = [1. / len(wt_vi_candidates)]

    IRL_summary = []
    update_coeff = 0.01 # 10^-5 to 10^5 used by Huang et al. for approximate inference

    for j in range(n_demonstrations):
        cond_posteriors = np.zeros(len(trajectories))
        cond_trajectory_likelihoods_trajectories = []

        for k in range(len(trajectories)):
            Z = 0    # normalization factor
            trajectory = trajectories[k]
            cond_trajectory_likelihoods = {}

            # compute the normalization factor
            for wt_vi_candidate_pair in wt_vi_candidates:
                wt_candidate = wt_vi_candidate_pair[0]
                vi_candidate = wt_vi_candidate_pair[1]
                trajectory_candidate = []  # optimal trajectory for this weight candidate

                # obtain optimal trajectory for this weight candidate
                vi_candidate_policy = FixedPolicyAgent(vi_candidate.policy)
                for sas_prime in trajectory:
                    action = vi_candidate_policy.act(sas_prime[0], 0)
                    s_prime = vi_candidate.mdp._taxi_transition_func(sas_prime[0], action)

                    trajectory_candidate.append((sas_prime[0], action, s_prime))

                # should always be positive since the trajectory_candidate should be reward-optimal for wt_candidate
                reward_diff = (wt_candidate.dot(vi_candidate.mdp.accumulate_reward_features(trajectory_candidate).T) \
                               - wt_candidate.dot(mdp_agent.accumulate_reward_features(trajectory).T))[0][0]

                # a) exact inference IRL
                # if reward_diff >= 0:
                #     cond_trajectory_likelihood = 1
                # else:
                #     cond_trajectory_likelihood = 0

                # b) approximate inference IRL
                cond_trajectory_likelihood = np.exp(-update_coeff * reward_diff)

                cond_trajectory_likelihoods[wt_candidate.tostring()] = cond_trajectory_likelihood

                Z += cond_trajectory_likelihood * priors[wt_candidate.tostring()]

            cond_trajectory_likelihoods_trajectories.append(cond_trajectory_likelihoods)

            # this should always be zero, assuming that the agent and the human are using the same reward features
            reward_diff_star = (weights.dot(mdp_agent.accumulate_reward_features(trajectory).T) - \
                               weights.dot(mdp_agent.accumulate_reward_features(trajectory).T))[0][0]

            # a) exact inference IRL
            # if reward_diff_star >= 0:
            #     cond_trajectory_likelihood_star = 1
            # else:
            #     cond_trajectory_likelihood_star = 0

            # b) approximate inference IRL
            cond_trajectory_likelihood_star = np.exp(-update_coeff * reward_diff_star)

            # calculate what the new condition probability of the true weight vector would be given this demonstration
            cond_posteriors[k] = 1. / Z * cond_trajectory_likelihood_star * priors[weights.tostring()]

        # select the demonstration that maximally increases the conditional posterior probability of the true weight vector
        best_traj = np.argmax(cond_posteriors)
        print(colored('Best trajectory: {}'.format(best_traj), 'red'))
        IRL_summary.append(trajectories[best_traj])
        # remove this demonstration from further consideration
        trajectories.pop(best_traj)

        # update the prior distribution
        for wt_vi_candidate_pair in wt_vi_candidates:
            wt_candidate = wt_vi_candidate_pair[0]
            old_prior = priors[wt_candidate.tostring()]
            updated_prior = priors[wt_candidate.tostring()] * cond_trajectory_likelihoods_trajectories[best_traj][wt_candidate.tostring()]
            priors[wt_candidate.tostring()] = updated_prior
            debugging_priors[wt_candidate.tostring()].append(updated_prior)

            if np.array_equal(wt_candidate, weights[0]):
                print(colored('wt: {}, prior: {}, updated_prior: {}'.format(np.round(wt_candidate, 2), old_prior, updated_prior), 'red'))
            else:
                print('wt: {}, prior: {}, updated_prior: {}'.format(np.round(wt_candidate, 2), old_prior, updated_prior))

    # visualize the evolution of the prior distribution with each new demonstration
    for j in range(n_demonstrations):
        x = range(len(wt_vi_candidates))
        y = []
        for wt_vi_candidate_pair in wt_vi_candidates:
            wt_candidate = wt_vi_candidate_pair[0]
            y.append(debugging_priors[wt_candidate.tostring()][j])
        plt.plot(x, y)
        plt.xlabel('Candidate reward weight vectors')
        plt.ylabel('Probability of candidates')
        plt.show()

    for trajectory in IRL_summary:
        mdp_agent.visualize_trajectory(trajectory)


if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
