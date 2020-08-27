''' Helper functions for executing actions in the Two Goal Problem '''

# Other imports.

def is_wall(mdp, x, y):
    '''
    Args:
        state (TwoGoalState)
        x (int) [agent's x]
        y (int) [agent's y]

    Returns:
        (bool): true iff the current loc of the agent is occupied by a wall.
    '''
    for wall in mdp.walls:
        if wall["x"] == x and wall["y"] == y:
            return True
    return False

def _is_wall_in_the_way(mdp, state, dx=0, dy=0):
    '''
    Args:
        state (TwoGoalState)
        dx (int) [optional]
        dy (int) [optional]

    Returns:
        (bool): true iff the new loc of the agent is occupied by a wall.
    '''
    for wall in mdp.walls:
        if wall["x"] == state.objects["agent"][0]["x"] + dx and \
            wall["y"] == state.objects["agent"][0]["y"] + dy:
            return True
    return False


def is_goal_state(mdp, state):
    '''
    Args:
        state (OOMDPState)

    Returns:
        (bool): True if the agent is at one of the goals
    '''

    # agent = state.get_first_obj_of_class("agent")

    for g in mdp.goals:
        if state.get_agent_x() == g.get_attribute("x") and state.get_agent_y() == g.get_attribute("y"):
            return True

    return False

def is_terminal_and_goal_state(mdp, state, exit_state):
    '''
    Args:
        state (OOMDPState)

    Returns:
        (bool): True if the agent is at one of the goals
    '''

    for g in mdp.goals:
        if state.get_agent_x() == g.get_attribute("x") and state.get_agent_y() == g.get_attribute("y"):
            return True, True

    if state == exit_state:
        return True, False

    return False, False