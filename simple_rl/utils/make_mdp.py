'''
make_mdp.py

Utility for making MDP instances
'''

# Python imports.

# Other imports.
from simple_rl.tasks import AugmentedTaxiOOMDP, TwoGoalOOMDP

def make_custom_mdp(mdp_class, mdp_parameters):
    if mdp_class == 'augmented_taxi':
        mdp_candidate = AugmentedTaxiOOMDP(width=mdp_parameters['width'], height=mdp_parameters['height'], agent=mdp_parameters['agent'],
                                           walls=mdp_parameters['walls'], passengers=mdp_parameters['passengers'], tolls=mdp_parameters['tolls'],
                                           traffic=mdp_parameters['traffic'], fuel_stations=mdp_parameters['fuel_station'],
                                           gamma=mdp_parameters['gamma'], weights=mdp_parameters['weights'], env_code=mdp_parameters['env_code'])
    elif mdp_class == 'two_goal':
        mdp_candidate = TwoGoalOOMDP(width=mdp_parameters['width'], height=mdp_parameters['height'], agent=mdp_parameters['agent'],
                                           walls=mdp_parameters['walls'], goals=mdp_parameters['goals'],
                                           gamma=mdp_parameters['gamma'], weights=mdp_parameters['weights'], env_code=mdp_parameters['env_code'])

    else:
        raise Exception("Unknown MDP class.")

    return mdp_candidate


def make_mdp_obj(mdp_class, mdp_code):
    '''
    :param mdp_code: Vector representation of an augmented taxi environment (list of binary values)
    :return: Corresponding passenger and toll objects and code that specifically only concerns the environment (and not
    the initial state) of the MDP
    '''

    if mdp_class == 'augmented_taxi':
        requested_passenger = [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]

        # the last eight entries currently dictate the presence of tolls
        available_tolls = [{"x": 2, "y": 3, "fee": 1}, {"x": 3, "y": 3, "fee": 1}, {"x": 4, "y": 3, "fee": 1},
                   {"x": 2, "y": 2, "fee": 1}, {"x": 3, "y": 2, "fee": 1}, {"x": 4, "y": 2, "fee": 1},
                   {"x": 2, "y": 1, "fee": 1}, {"x": 3, "y": 1, "fee": 1}]

        requested_tolls = []

        # offset can facilitate additional MDP information present in the code before toll information
        offset = 0
        for x in range(offset, len(mdp_code)):
            entry = mdp_code[x]
            if entry:
                requested_tolls.append(available_tolls[x - offset])

        # note that what's considered mdp_code (potentially includes both initial state and environment info) and env_code
        # (only includes environment info) will always need to be manually defined
        return requested_passenger, requested_tolls, mdp_code
    elif mdp_class == 'two_goal':
        available_walls = [{'x': 1, 'y': 4}, {'x': 2, 'y': 4}, {'x': 3, 'y': 4}, {'x': 3, 'y': 2}, {'x': 4, 'y': 2},
                 {'x': 5, 'y': 3}]

        requested_walls = []

        for x in range(0, len(mdp_code)):
            entry = mdp_code[x]
            if entry:
                requested_walls.append(available_walls[x])

        return requested_walls, mdp_code
    else:
        raise Exception("Unknown MDP class.")

# hard-coded in order to evaluate hand-designed environments
def hardcode_mdp_obj(mdp_class, mdp_code):

    if mdp_class == 'augmented_taxi':
        # a) for resolving weight of toll
        if mdp_code == [0, 0]:
            requested_passenger = [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
            requested_tolls = [{"x": 3, "y": 1, "fee": 1}, {"x": 3, "y": 2, "fee": 1}]  # upperbound
        elif mdp_code == [0, 1]:
            requested_passenger = [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
            requested_tolls = [{"x": 3, "y": 1, "fee": 1}]                              # lowerbound
        # b) for resolving weight of dropping off passenger
        elif mdp_code == [1, 0]:
            requested_passenger = [{"x": 2, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
            requested_tolls = [{"x": 2, "y": 3, "fee": 1}, {"x": 3, "y": 3, "fee": 1}, {"x": 4, "y": 3, "fee": 1},
                       {"x": 2, "y": 2, "fee": 1}, {"x": 3, "y": 2, "fee": 1}, {"x": 2, "y": 1, "fee": 1},
                       {"x": 3, "y": 1, "fee": 1}]                              # lowerbound
        else:
            requested_passenger = [{"x": 2, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
            requested_tolls = [{"x": 2, "y": 3, "fee": 1}, {"x": 3, "y": 3, "fee": 1}, {"x": 4, "y": 3, "fee": 1},
                       {"x": 2, "y": 2, "fee": 1}, {"x": 3, "y": 2, "fee": 1}, {"x": 4, "y": 2, "fee": 1},
                       {"x": 2, "y": 1, "fee": 1}, {"x": 3, "y": 1, "fee": 1}]  # upperbound

        return requested_passenger, requested_tolls, mdp_code
    elif mdp_class == 'two_goal':
        if mdp_code == [0, 0]:
            walls = [{'x': 1, 'y': 4}, {'x': 2, 'y': 4}, {'x': 3, 'y': 4}, {'x': 3, 'y': 2}, {'x': 4, 'y': 2}, {'x': 5, 'y': 3}]
        elif mdp_code == [0, 1]:
            walls = [{'x': 1, 'y': 4}, {'x': 2, 'y': 4}, {'x': 3, 'y': 2}, {'x': 4, 'y': 2}, {'x': 5, 'y': 3}]
        elif mdp_code == [1, 0]:
            walls = [{'x': 1, 'y': 4}, {'x': 2, 'y': 4}, {'x': 3, 'y': 4}, {'x': 3, 'y': 2}, {'x': 5, 'y': 3}]
        else:
            walls = [{'x': 1, 'y': 4}, {'x': 2, 'y': 4}, {'x': 3, 'y': 4}, {'x': 4, 'y': 2}, {'x': 5, 'y': 3}]

        return walls, mdp_code
    else:
        raise Exception("Unknown MDP class.")
