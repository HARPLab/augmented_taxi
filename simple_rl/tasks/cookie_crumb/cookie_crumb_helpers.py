''' Helper functions for executing actions in the Cookie Crumb Problem '''

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


def is_terminal_and_goal_state(mdp, state):
    '''
    Args:
        state (OOMDPState)

    Returns:
        (bool): True if the agent is at one of the goals
    '''

    for g in mdp.goals:
        if state.get_agent_x() == g.get_attribute("x") and state.get_agent_y() == g.get_attribute("y"):
            return True, True

    return False, False
