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
def rollout_policy(mdp, agent, cur_state=None, cur_action=None, max_depth=50, timeout=10):
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
    if cur_action is not None:
        reward, next_state = mdp.execute_agent_action(cur_action)
        trajectory.append((cur_state, cur_action, next_state))

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

