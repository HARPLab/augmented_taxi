import sys
import augmented_taxi
import params

sys.path.append("simple_rl")

# For research integration with UMass Lowell

def get_trajectories(human_action_callback, on_done):
    # obtain BEC summaries
    print("Visualizing demonstrations comprising BEC summary")
    constraints, BEC_summary = augmented_taxi.obtain_BEC_summary(params.data_loc['BEC'], params.aug_taxi, params.n_env,
                                                  params.weights['val'], params.step_cost_flag,
                                                  params.BEC['summary_type'], params.BEC['n_desired_summaries'], min_BEC_set_only=params.BEC['min_BEC_set_only'], BEC_depth=params.BEC['depth'],
                                                  visualize_constraints=False, visualize_summary=False)

    # obtain test environment(s)
    print("Visualizing agent's optimal demonstration in test environment")
    test_wt_vi_traj_tuples, test_BEC_lengths, test_BEC_constraints = augmented_taxi.obtain_test_environments(params.data_loc['BEC'], params.aug_taxi, params.weights['val'], params.n_env, params.BEC,
                             params.step_cost_flag, summary=BEC_summary, visualize_test_env=False)

    agent_trajectories = []
    human_trajectories = []

    # for each test environment
    for test_wt_vi_traj_tuple in test_wt_vi_traj_tuples:
        def on_done_():
            on_done(test_wt_vi_traj_tuple[2])

        # obtain human's prediction of the agent's trajectory
        human_trajectory = test_wt_vi_traj_tuple[1].mdp.visualize_interaction(human_action_callback, on_done_)
        human_trajectories.append(human_trajectory)

        # store the agent's actual trajectory
        agent_trajectories.append(test_wt_vi_traj_tuple[2])

    return agent_trajectories, human_trajectories

if __name__ == "__main__":
    get_trajectories()
