import difflib

def normalize_trajectories(trajectory, actions, trajectory_counterfactual, actions_counterfactual):
    '''

    Args:
        trajectory: List of states (tuples) defining a trajectory
        actions: List of strings representing actions
        trajectory_counterfactual: List of states (tuples) defining a counterfactual trajectory
        actions_counterfactual: List of strings representing counterfactual actions

    Returns: Two lists of actions that are of the same length, such that a 'no-op' action is inserted in the shorter
    trajectory whenever it's fitting to wait for the other to catch up (i.e. as it waits on an anchor point, a state
    that's shared by the two trajectories)

    '''

    # subtract one since the code below was original created to work with trajectories of (state, action, next_state) tuples,
    # which will have one less element than a trajectory of simply individual states (i.e. state1, state2, state3, etc)
    len_traj = len(trajectory) - 1
    len_counter = len(trajectory_counterfactual) - 1

    anchor_points_wait = []
    matcher = difflib.SequenceMatcher(None, trajectory, trajectory_counterfactual, autojunk=False)
    matches = matcher.get_matching_blocks()

    for match in matches:
        # add states in overlap
        for i in range(match[2]):
            anchor_points_wait.append(trajectory[match[0] + i])

    # print(anchor_points_wait)

    anchor_points_wait.reverse()
    cur_anchor_point = anchor_points_wait.pop()

    step_traj_temp = 0
    step_counter_temp = 0

    normalized_actions = []
    normalized_actions_counterfactual = []

    while (step_traj_temp < len_traj or step_counter_temp < len_counter):
        state = trajectory[step_traj_temp]
        counter_state = trajectory_counterfactual[step_counter_temp]

        # wait
        if (state == cur_anchor_point and state != counter_state):
            step_traj_temp -= 1
            normalized_actions.append('no-op')
        else:
            normalized_actions.append(actions[step_traj_temp])

        if (counter_state == cur_anchor_point and state != counter_state):
            step_counter_temp -= 1
            normalized_actions_counterfactual.append('no-op')
        else:
            normalized_actions_counterfactual.append(actions_counterfactual[step_counter_temp])

        # consider anchor points one at a time
        if (state == counter_state) and (state == cur_anchor_point):
            if len(anchor_points_wait) > 0:
                cur_anchor_point = anchor_points_wait.pop()

        step_traj_temp += 1
        step_counter_temp += 1

    # print('Actions: {}'.format(normalized_actions))
    # print('C_Actions: {}'.format(normalized_actions_counterfactual))

    return normalized_actions, normalized_actions_counterfactual
