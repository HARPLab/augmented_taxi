import sys
import augmented_taxi
import params
import copy
sys.path.append("simple_rl")
from simple_rl.planning import ValueIteration
from simple_rl.utils import mdp_helpers

# For research integration with UMass Lowell

if __name__ == "__main__":
    # obtain BEC summary
    print("Visualizing demonstrations comprising BEC summary")
    constraints, BEC_summary = augmented_taxi.obtain_BEC_summary(params.data_loc['BEC'], params.aug_taxi, params.n_env,
                                                  params.weights['val'], params.step_cost_flag,
                                                  params.BEC['summary_type'], BEC_depth=params.BEC['depth'],
                                                  visualize_constraints=True, visualize_summary=True)

    # obtain test environment
    print("Visualizing agent's optimal demonstration in test environment")
    test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = augmented_taxi.obtain_test_environments(
        params.data_loc['BEC'], params.aug_taxi, params.weights['val'], params.n_env, params.BEC, params.step_cost_flag, summary=BEC_summary, visualize_test_env=True)

    agent_trajectory = test_wt_vi_traj_tuples[0][0][2]

    # simulate a human demonstrated trajectory
    mdp_human = copy.deepcopy(test_wt_vi_traj_tuples[0][0][1].mdp)
    mdp_human.weights = params.weights_human['val']
    vi_human = ValueIteration(mdp_human)

    iterations, value_of_init_state = vi_human.run_vi()
    human_trajectory = mdp_helpers.rollout_policy(mdp_human, vi_human)

    print("Visualizing simulated human demonstration in test environment")
    mdp_human.visualize_trajectory(human_trajectory)

