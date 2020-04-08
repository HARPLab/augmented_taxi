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
def rollout_policy(mdp, agent, cur_state=None, max_depth=100):
    mdp.reset()
    depth = 0
    reward = 0
    trajectory = []

    if cur_state == None:
        cur_state = mdp.get_init_state()
    else:
        # mdp has a memory of its current state that needs to be adjusted accordingly
        mdp.set_curr_state(cur_state)

    while not cur_state.is_terminal() and depth < max_depth:
        action = agent.act(cur_state, reward)
        reward, next_state = mdp.execute_agent_action(action)
        trajectory.append((cur_state, action, next_state))

        # deepcopy occurs within transition function
        cur_state = next_state

        depth += 1

    return trajectory

