''' Helper functions for executing actions in the Skateboard Problem '''

def is_wall(mdp, x, y):
    '''
    Args:
        state (SkateboardState)
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
        state (SkateboardState)
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


def _move_skateboard_on_agent(state, dx=0, dy=0):
    '''
    Args:
        state (SkateboardState)
        x (int) [optional]
        y (int) [optional]

    Returns:
        (list of dict): List of new skateboard attributes.

    '''
    skateboard_attr_dict_ls = state.get_objects_of_class("skateboard")
    for i, skateboard in enumerate(skateboard_attr_dict_ls):
        if skateboard["on_agent"] == 1:
            skateboard_attr_dict_ls[i]["x"] += dx
            skateboard_attr_dict_ls[i]["y"] += dy

def is_terminal_and_goal_state(mdp, state):
    '''
    Args:
        state (OOMDPState)

    Returns:
        (bool): True if the agent is at one of the goals
    '''

    if state.get_agent_x() == mdp.goal["x"] and state.get_agent_y() == mdp.goal["y"]:
        return True, True

    return False, False
