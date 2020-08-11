''' Helper functions for executing actions in the Taxi Problem '''

# Other imports.
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject

def is_wall(mdp, x, y):
    '''
    Args:
        state (TaxiState)
        x (int) [agent's x]
        y (int) [agent's y]

    Returns:
        (bool): true iff the current loc of the agent is occupied by a wall.
    '''
    for wall in mdp.walls:
        if wall["x"] == x and wall["y"] == y:
            return True
    return False

def at_traffic(mdp, x, y):
    '''
    Args:
        state (TaxiState)
        x (int) [agent's x]
        y (int) [agent's y]

    Returns:
        (bool): true iff the current loc of the agent is a traffic cell.
        (float): probability of getting stuck at this traffic cell
    '''
    for traffic in mdp.traffic_cells:
        if traffic["x"] == x and traffic["y"] == y:
            return True, traffic["prob"]

    return False, 0.

def at_fuel_station(mdp, x, y):
    '''
    Args:
        state (TaxiState)
        x (int) [agent's x]
        y (int) [agent's y]

    Returns:
        (bool): true iff the current loc of the agent is a traffic cell.
        (int): fuel capacity to fill up to
    '''
    for fuel_station in mdp.fuel_stations:
        if fuel_station["x"] == x and fuel_station["y"] == y:
            return True, fuel_station["max_fuel_capacity"]

    return False, 0

def _is_wall_in_the_way(mdp, state, dx=0, dy=0):
    '''
    Args:
        state (TaxiState)
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


def _move_pass_in_taxi(state, dx=0, dy=0):
    '''
    Args:
        state (TaxiState)
        x (int) [optional]
        y (int) [optional]

    Returns:
        (list of dict): List of new passenger attributes.

    '''
    passenger_attr_dict_ls = state.get_objects_of_class("passenger")
    for i, passenger in enumerate(passenger_attr_dict_ls):
        if passenger["in_taxi"] == 1:
            passenger_attr_dict_ls[i]["x"] += dx
            passenger_attr_dict_ls[i]["y"] += dy

def _moved_into_toll(mdp, state, next_state):
    for toll in mdp.tolls:
        # if current state's agent x, y doesn't coincide with any x, y of the tolls
        if toll.attributes['x'] != state.get_agent_x() or toll.attributes['y'] != state.get_agent_y():
            # and if the next state's agent x, y moves into of the x, y of the toll
            if toll.attributes['x'] == next_state.get_agent_x() and toll.attributes['y'] == next_state.get_agent_y():
                return True, toll.attributes['fee']
    return False, 0

def is_taxi_terminal_state(state):
    '''
    Args:
        state (OOMDPState)

    Returns:
        (bool): True iff all passengers at at their destinations, not in the taxi.
    '''
    all_passengers_at_destination = True

    # check if all passengers are at destination
    for p in state.get_objects_of_class("passenger"):
        if p.get_attribute("in_taxi") == 1 or p.get_attribute("x") != p.get_attribute("dest_x") or \
            p.get_attribute("y") != p.get_attribute("dest_y"):
            all_passengers_at_destination = False

    if all_passengers_at_destination:
        return True
    else:
        # check fuel level if applicable
        if state.track_fuel():
            if state.objects["agent"][0]["fuel"] <= 0:
                return True
        return False

def is_taxi_terminal_and_goal_state(state):
    '''
    Args:
        state (OOMDPState)

    Returns:
        (bool): True iff all passengers are at their destinations, not in the taxi.
    '''
    all_passengers_at_destination = True

    # check if all passengers are at destination
    for p in state.get_objects_of_class("passenger"):
        if p.get_attribute("in_taxi") == 1 or p.get_attribute("x") != p.get_attribute("dest_x") or \
            p.get_attribute("y") != p.get_attribute("dest_y"):
            all_passengers_at_destination = False

    if all_passengers_at_destination:
        # is terminal and goal state
        return True, True
    else:
        # check fuel level if applicable
        if state.track_fuel():
            if state.objects["agent"][0]["fuel"] <= 0:
                # is terminal, but not goal state
                return True, False
        # is neither terminal nor goal state
        return False, False
