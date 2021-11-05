''' Helper functions for executing actions in the Colored Tiles Problem '''

def is_wall(mdp, x, y):
    '''
    Args:
        state (ColoredTileState)
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
        state (ColoredTileState)
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

def _moved_off_of_A_tile(mdp, state, next_state):
    for tile in mdp.A_tiles:
        # if current state's agent x, y coincides with any x, y of the tolls
        if tile.attributes['x'] == state.get_agent_x() and tile.attributes['y'] == state.get_agent_y():
            # and if the next state's agent x, y doesn't coincide with this toll
            if tile.attributes['x'] != next_state.get_agent_x() or tile.attributes['y'] != next_state.get_agent_y():
                return True
    return False

def _moved_off_of_B_tile(mdp, state, next_state):
    for tile in mdp.B_tiles:
        # if current state's agent x, y coincides with any x, y of the tolls
        if tile.attributes['x'] == state.get_agent_x() and tile.attributes['y'] == state.get_agent_y():
            # and if the next state's agent x, y doesn't coincide with this toll
            if tile.attributes['x'] != next_state.get_agent_x() or tile.attributes['y'] != next_state.get_agent_y():
                return True
    return False


def is_terminal_and_goal_state(mdp, state, ref_exit_state):
    '''
    Args:
        state (OOMDPState)

    Returns:
        (bool): True if the agent is at one of the goals
    '''

    if state.get_agent_x() == mdp.goal["x"] and state.get_agent_y() == mdp.goal["y"]:
        return True, True

    if state.get_agent_x() == ref_exit_state.get_agent_x() and state.get_agent_y() == ref_exit_state.get_agent_y():
        return True, False

    return False, False

def is_goal_state(mdp, state):
    '''
    Args:
        state (OOMDPState)

    Returns:
        (bool): True if the agent is at one of the goals
    '''

    if state.get_agent_x() == mdp.goal["x"] and state.get_agent_y() == mdp.goal["y"]:
        return True
    else:
        return False
