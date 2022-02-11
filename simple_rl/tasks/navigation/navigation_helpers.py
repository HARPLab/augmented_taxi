''' Helper functions for executing actions in the car Problem '''

def is_wall(mdp, x, y):
    '''
    Args:
        state (carState)
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
        state (carState)
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

def _is_road_in_the_way(mdp, state, dx=0, dy=0):
    '''
    Args:
        state (carState)
        dx (int) [optional]
        dy (int) [optional]

    Returns:
        (bool): true iff the new loc of the agent is occupied by a wall.
    '''
    for road in mdp.roads:
        if road["x"] == state.objects["agent"][0]["x"] + dx and \
            road["y"] == state.objects["agent"][0]["y"] + dy:
            return True
    return False

def agent_on_road(mdp, state):
    '''
    Args:
        state (SkateboardState)
        x (int) [agent's x]
        y (int) [agent's y]

    Returns:
        (bool): true iff the current loc of the agent is on the road.
    '''
    for road in mdp.roads:
        if road["x"] == state.objects["agent"][0]["x"] and road["y"] ==  state.objects["agent"][0]["y"]:
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

def _move_car_on_agent(state, dx=0, dy=0):
    '''
    Args:
        state (SkateboardState)
        x (int) [optional]
        y (int) [optional]

    Returns:
        (list of dict): List of new skateboard attributes.

    '''
    car_attr_dict_ls = state.get_objects_of_class("car")
    for i, car in enumerate(car_attr_dict_ls):
        if car["on_agent"] == 1:
            car_attr_dict_ls[i]["x"] += dx
            car_attr_dict_ls[i]["y"] += dy


def _moved_off_of_grass(mdp, state, next_state):
    for grass in mdp.grass:
        # if current state's agent x, y coincides with any x, y of the grasss
        if grass.attributes['x'] == state.get_agent_x() and grass.attributes['y'] == state.get_agent_y():
            # and if the next state's agent x, y doesn't coincide with this grass
            if grass.attributes['x'] != next_state.get_agent_x() or grass.attributes['y'] != next_state.get_agent_y():
                return True
    return False

def _moved_off_of_gravel(mdp, state, next_state):
    for gravel in mdp.gravel:
        # if current state's agent x, y coincides with any x, y of the gravels
        if gravel.attributes['x'] == state.get_agent_x() and gravel.attributes['y'] == state.get_agent_y():
            # and if the next state's agent x, y doesn't coincide with this gravel
            if gravel.attributes['x'] != next_state.get_agent_x() or gravel.attributes['y'] != next_state.get_agent_y():
                return True
    return False

def _moved_off_of_road(mdp, state, next_state):
    for road in mdp.roads:
        # if current state's agent x, y coincides with any x, y of the roads
        if road.attributes['x'] == state.get_agent_x() and road.attributes['y'] == state.get_agent_y():
            # and if the next state's agent x, y doesn't coincide with this road
            if road.attributes['x'] != next_state.get_agent_x() or road.attributes['y'] != next_state.get_agent_y():
                return True
    return False


def _moved_off_of_hotswap_station(state, next_state):
    for station_idx, hotswap_station in enumerate(state.get_objects_of_class("hotswap_station")):
        # if current state's agent x, y coincides with any x, y of the hotswap stations
        if hotswap_station.attributes['x'] == state.get_agent_x() and hotswap_station.attributes['y'] == state.get_agent_y():
            # and if the next state's agent x, y doesn't coincide with this toll
            if hotswap_station.attributes['x'] != next_state.get_agent_x() or hotswap_station.attributes['y'] != next_state.get_agent_y():
                return True, station_idx
    return False, None


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
