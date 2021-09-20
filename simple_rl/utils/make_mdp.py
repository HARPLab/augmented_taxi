'''
make_mdp.py

Utility for making MDP instances
'''

# Python imports.

# Other imports.
from simple_rl.tasks import AugmentedTaxiOOMDP, TwoGoalOOMDP, SkateboardOOMDP, TaxiOOMDP, CookieCrumbOOMDP, AugmentedTaxi2OOMDP, TwoGoal2OOMDP

def make_custom_mdp(mdp_class, mdp_parameters):
    if mdp_class == 'augmented_taxi':
        mdp_candidate = AugmentedTaxiOOMDP(width=mdp_parameters['width'], height=mdp_parameters['height'], agent=mdp_parameters['agent'],
                                           walls=mdp_parameters['walls'], passengers=mdp_parameters['passengers'], tolls=mdp_parameters['tolls'],
                                           traffic=mdp_parameters['traffic'], fuel_stations=mdp_parameters['fuel_station'], gamma=mdp_parameters['gamma'],
                                           weights=mdp_parameters['weights'], env_code=mdp_parameters['env_code'], sample_rate=1)
    elif mdp_class == 'two_goal':
        mdp_candidate = TwoGoalOOMDP(width=mdp_parameters['width'], height=mdp_parameters['height'], agent=mdp_parameters['agent'],
                                           walls=mdp_parameters['walls'], goals=mdp_parameters['goals'], gamma=mdp_parameters['gamma'],
                                           weights=mdp_parameters['weights'], env_code=mdp_parameters['env_code'], sample_rate=1)
    elif mdp_class == 'skateboard':
        mdp_candidate = SkateboardOOMDP(width=mdp_parameters['width'], height=mdp_parameters['height'], agent=mdp_parameters['agent'],
                                           walls=mdp_parameters['walls'], goal=mdp_parameters['goal'], skateboard=mdp_parameters['skateboard'], gamma=mdp_parameters['gamma'],
                                           weights=mdp_parameters['weights'], env_code=mdp_parameters['env_code'], sample_rate=1)
    elif mdp_class == 'augmented_taxi2':
        mdp_candidate = AugmentedTaxi2OOMDP(width=mdp_parameters['width'], height=mdp_parameters['height'], agent=mdp_parameters['agent'],
                                           walls=mdp_parameters['walls'], passengers=mdp_parameters['passengers'], tolls=mdp_parameters['tolls'],
                                           traffic=mdp_parameters['traffic'], fuel_stations=mdp_parameters['fuel_station'], hotswap_stations=mdp_parameters['hotswap_station'], gamma=mdp_parameters['gamma'],
                                           weights=mdp_parameters['weights'], env_code=mdp_parameters['env_code'], sample_rate=1)
    elif mdp_class == 'two_goal2':
        mdp_candidate = TwoGoal2OOMDP(width=mdp_parameters['width'], height=mdp_parameters['height'], agent=mdp_parameters['agent'],
                                           walls=mdp_parameters['walls'], goals=mdp_parameters['goals'], gamma=mdp_parameters['gamma'],
                                           weights=mdp_parameters['weights'], env_code=mdp_parameters['env_code'], sample_rate=1)
    elif mdp_class == 'taxi':
        mdp_candidate = TaxiOOMDP(width=mdp_parameters['width'], height=mdp_parameters['height'], agent=mdp_parameters['agent'],
                                           walls=mdp_parameters['walls'], passengers=mdp_parameters['passengers'], gamma=mdp_parameters['gamma'], weights=mdp_parameters['weights'])
    elif mdp_class == 'cookie_crumb':
        mdp_candidate = CookieCrumbOOMDP(width=mdp_parameters['width'], height=mdp_parameters['height'], agent=mdp_parameters['agent'],
                                         walls=mdp_parameters['walls'], goals=mdp_parameters['goals'], crumbs=mdp_parameters['crumbs'], gamma=mdp_parameters['gamma'],
                                         weights=mdp_parameters['weights'], env_code=mdp_parameters['env_code'], sample_rate=1)
    else:
        raise Exception("Unknown MDP class.")

    return mdp_candidate


def make_mdp_obj(mdp_class, mdp_code, mdp_parameters):
    '''
    :param mdp_code: Vector representation of an augmented taxi environment (list of binary values)
    :return: Corresponding passenger and toll objects and code that specifically only concerns the environment (and not
    the initial state) of the MDP
    '''

    if mdp_class == 'augmented_taxi':
        requested_passenger = mdp_parameters['passengers']
        available_tolls = mdp_parameters['available_tolls']

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
    elif mdp_class == 'two_goal' or mdp_class == 'two_goal2':
        available_walls = mdp_parameters['available_walls']
        requested_walls = []

        for x in range(0, len(mdp_code)):
            entry = mdp_code[x]
            if entry:
                requested_walls.append(available_walls[x])

        return requested_walls, mdp_code
    elif mdp_class == 'skateboard':
        available_walls = mdp_parameters['available_walls']
        requested_walls = []

        for x in range(0, len(mdp_code)):
            entry = mdp_code[x]
            if entry:
                requested_walls.append(available_walls[x])

        # these are permanent walls
        requested_walls.extend([{'x': 5, 'y': 4}, {'x': 5, 'y': 3}, {'x': 5, 'y': 2}, {'x': 6, 'y': 3}, {'x': 6, 'y': 2}])

        return requested_walls, mdp_code
    elif mdp_class == 'augmented_taxi2':
        requested_passenger = mdp_parameters['passengers']
        available_tolls = mdp_parameters['available_tolls']
        available_hotswap_stations = mdp_parameters['available_hotswap_stations']

        requested_tolls = []

        # offset can facilitate additional MDP information present in the code before toll information
        offset = 1
        for x in range(offset, len(mdp_code)):
            entry = mdp_code[x]
            if entry:
                requested_tolls.append(available_tolls[x - offset])

        # currently only supporting one hotswap station
        requested_hotswap_stations = []
        if mdp_code[0]:
            requested_hotswap_stations.append(available_hotswap_stations[0])

        # note that what's considered mdp_code (potentially includes both initial state and environment info) and env_code
        # (only includes environment info) will always need to be manually defined
        return requested_passenger, requested_tolls, requested_hotswap_stations, mdp_code
    elif mdp_class == 'cookie_crumb':
        available_crumbs = mdp_parameters['available_crumbs']
        requested_crumbs = []

        for x in range(0, len(mdp_code)):
            entry = mdp_code[x]
            if entry:
                requested_crumbs.append(available_crumbs[x])

        return requested_crumbs, mdp_code
    else:
        raise Exception("Unknown MDP class.")

# hard-coded in order to evaluate hand-designed environments
def hardcode_mdp_obj(mdp_class, mdp_code):

    if mdp_class == 'augmented_taxi':
        # a) for resolving weight of dropping off passenger
        if mdp_code == [0, 0]:
            requested_passenger = [{"x": 4, "y": 2, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
            requested_tolls = []                                    # lowerbound
        elif mdp_code == [0, 1]:
            requested_passenger = [{"x": 4, "y": 3, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
            requested_tolls = []                                    # upperbound
        # b) for resolving weight of toll
        elif mdp_code == [1, 0]:
            requested_passenger = [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
            requested_tolls = [{"x": 3, "y": 1}]                    # lowerbound
        else:
            requested_passenger = [{"x": 4, "y": 1, "dest_x": 1, "dest_y": 1, "in_taxi": 0}]
            requested_tolls = [{"x": 3, "y": 1}, {"x": 3, "y": 2}]  # upperbound

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
    elif mdp_class == 'skateboard':
        if mdp_code == [0, 0]:
            walls = [{'x': 3, 'y': 4}, {'x': 3, 'y': 3}, {'x': 3, 'y': 2}]
        elif mdp_code == [0, 1]:
            walls = [{'x': 3, 'y': 4}, {'x': 3 , 'y': 3}]
        elif mdp_code == [1, 0]:
            walls = [{'x': 3, 'y': 4}]
        else:
            walls = []

        # these are permanent walls
        walls.extend([{'x': 5, 'y': 4}, {'x': 5, 'y': 3}, {'x': 5, 'y': 2}, {'x': 6, 'y': 3}, {'x': 6, 'y': 2}])

        return walls, mdp_code
    elif mdp_class == 'cookie_crumb':
        if mdp_code == [0, 0]:
            crumbs = [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 1, 'y': 4}, {'x': 2, 'y': 2}, {'x': 2, 'y': 3}, {'x': 2, 'y': 4}]
        elif mdp_code == [0, 1]:
            crumbs = [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 1, 'y': 4}, {'x': 2, 'y': 3}]
        elif mdp_code == [1, 0]:
            crumbs = [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}]
        else:
            crumbs = [{'x': 1, 'y': 2}, {'x': 1, 'y': 4}, {'x': 2, 'y': 2}, {'x': 2, 'y': 4}]

        return crumbs, mdp_code
    else:
        raise Exception("Unknown MDP class.")
