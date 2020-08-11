import sys
import augmented_taxi
import params

sys.path.append("simple_rl")

# For research integration with UMass Lowell

def get_trajectories(human_action_callback, on_done):
    # obtain BEC summaries
    print("Visualizing demonstrations comprising BEC summary")
    BEC_summary = augmented_taxi.obtain_BEC_summary(params.data_loc['BEC'], params.aug_taxi, params.n_env,
                                                  params.weights['val'], params.step_cost_flag,
                                                  params.BEC['summary_type'], params.BEC['n_train_demos'], BEC_depth=params.BEC['depth'],
                                                  visualize_summary=True)

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
        human_trajectory = test_wt_vi_traj_tuple[1].mdp.visualize_interaction(interaction_callback=human_action_callback, done_callback=on_done_, keys_map=params.keys_map)
        human_trajectories.append(human_trajectory)

        # store the agent's actual trajectory
        agent_trajectories.append(test_wt_vi_traj_tuple[2])

    return agent_trajectories, human_trajectories

if __name__ == "__main__":
    noop = lambda *args, **kwargs: None
    get_trajectories(noop, noop)
