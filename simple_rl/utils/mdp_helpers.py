from termcolor import colored
'''
Args:
    mdp (MDP)
    agent (Agent)
    cur_state (State)
    cur_traj (list)
    cur_action_seq (list
    max depth (int)

Returns:
    list of trajectories (list of list of state, action, state tuples)

Summary:
    Return all possible, equally-rewarding roll outs of the agent's policy on the designated MDP
'''
def rollout_policy_recursive(mdp, agent, cur_state, trajs, cur_traj=[], cur_action_seq=[], max_depth=25, max_num_of_trajs=10):
    # cap the maximum number of recursive trajectories that'll be returned (can be unreasonable to enumerate all
    # equally rewarding trajectories sometimes)
    if len(trajs) > max_num_of_trajs:
        return

    if cur_state.is_terminal() or len(cur_traj) >= max_depth:
        trajs.append(cur_traj)
        return

    maxq_actions = agent.get_max_q_actions(cur_state)
    for action in maxq_actions:
        # print('cur action seq: {}'.format(cur_action_seq))
        # print('maxq actions: {}'.format(maxq_actions))
        # print('action: {}'.format(action))

        # mdp has a memory of its current state that needs to be adjusted accordingly
        mdp.set_curr_state(cur_state)

        # deepcopy of state occurs within transition function
        reward, next_state = mdp.execute_agent_action(action)
        next_traj = cur_traj.copy()
        next_traj.append((cur_state, action, next_state))
        next_action_seq = cur_action_seq.copy() # for debugging
        next_action_seq.append(action)

        rollout_policy_recursive(mdp, agent, next_state, trajs, cur_traj=next_traj, cur_action_seq=next_action_seq)

    return trajs

'''
Args:
    mdp (MDP)
    agent (Agent)
    cur_state (State)
    max depth (int)

Returns:
    trajectory (list of state, action, state tuples)

Summary:
    Roll out the agent's policy on the designated MDP and return the corresponding trajectory
'''
def rollout_policy(mdp, agent, cur_state=None, action_seq=None, max_depth=25, timeout=5):
    mdp.reset()
    depth = 0
    reward = 0
    trajectory = []
    timeout_counter = 0

    if cur_state == None:
        cur_state = mdp.get_init_state()
    else:
        # mdp has a memory of its current state that needs to be adjusted accordingly
        mdp.set_curr_state(cur_state)

    # execute the specified action first, if relevant
    if action_seq is not None:
        for idx in range(len(action_seq)):
            reward, next_state = mdp.execute_agent_action(action_seq[idx])
            trajectory.append((cur_state, action_seq[idx], next_state))

            # deepcopy occurs within transition function
            cur_state = next_state

            depth += 1

    while not cur_state.is_terminal() and depth < max_depth and timeout_counter <= timeout:
        action = agent.act(cur_state, reward)
        reward, next_state = mdp.execute_agent_action(action)
        trajectory.append((cur_state, action, next_state))

        if next_state == cur_state:
            timeout_counter += 1
        else:
            timeout_counter += 0

        # deepcopy occurs within transition function
        cur_state = next_state

        depth += 1

    return trajectory

