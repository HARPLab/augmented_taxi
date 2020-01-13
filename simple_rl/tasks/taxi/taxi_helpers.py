''' Helper functions for executing actions in the Taxi Problem '''

# Other imports.
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject

def is_wall(state, x, y):
    for wall in state.objects["wall"]:
        if wall["x"] == x and wall["y"] == y:
            return True
    return False

def at_traffic(state, x, y):
    for traffic in state.objects["traffic"]:
        if traffic["x"] == x and traffic["y"] == y:
            return True, traffic["prob"]

    return False, 0

def _is_wall_in_the_way(state, dx=0, dy=0):
    '''
    Args:
        state (TaxiState)
        dx (int) [optional]
        dy (int) [optional]

    Returns:
        (bool): true iff the new loc of the agent is occupied by a wall.
    '''
    for wall in state.objects["wall"]:
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

def _moved_off_of_toll(state, next_state):
    for toll in state.get_objects_of_class("toll"):
        # if current state's agent x, y coincides with any x, y of the tolls
        if toll.attributes['x'] == state.get_agent_x() and toll.attributes['y'] == state.get_agent_y():
            # and if the next state's agent x, y moved off of the x, y of the toll
            if toll.attributes['x'] != next_state.get_agent_x() or toll.attributes['y'] != next_state.get_agent_y():
                return True, toll.attributes['fee']
    return False, 0

def is_taxi_terminal_state(state):
    '''
    Args:
        state (OOMDPState)

    Returns:
        (bool): True iff all passengers at at their destinations, not in the taxi.
    '''
    for p in state.get_objects_of_class("passenger"):
        if p.get_attribute("in_taxi") == 1 or p.get_attribute("x") != p.get_attribute("dest_x") or \
            p.get_attribute("y") != p.get_attribute("dest_y"):
            return False
    return True
