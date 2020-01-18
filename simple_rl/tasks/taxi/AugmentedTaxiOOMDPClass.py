'''
AugmentedTaxiMDPClass.py: Contains the AugmentedTaxiMDP class.

From:
    Dietterich, Thomas G. "Hierarchical reinforcement learning with the
    MAXQ value function decomposition." J. Artif. Intell. Res.(JAIR) 13
    (2000): 227-303.

Author: David Abel (cs.brown.edu/~dabel/)
'''

# Python imports.
from __future__ import print_function
import random
import copy

# Other imports.
from simple_rl.mdp.oomdp.OOMDPClass import OOMDP
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject
from simple_rl.tasks.taxi.AugmentedTaxiStateClass import AugmentedTaxiState
from simple_rl.tasks.taxi import taxi_helpers


class AugmentedTaxiOOMDP(OOMDP):
    ''' Class for a Taxi OO-MDP '''

    # Static constants.
    BASE_ACTIONS = ["up", "down", "left", "right", "pickup", "dropoff"]
    AUGMENTED_ACTIONS = ["up", "down", "left", "right", "pickup", "dropoff", "refuel"]
    ATTRIBUTES = ["x", "y", "has_passenger", "in_taxi", "dest_x", "dest_y"]
    CLASSES = ["agent", "wall", "passenger", "toll", "traffic", "fuel_station"]

    def __init__(self, width, height, agent, walls, passengers, tolls, traffic, fuel_stations, slip_prob=0, gamma=0.99):
        self.height = height
        self.width = width

        agent_obj = OOMDPObject(attributes=agent, name="agent")
        wall_objs = self._make_oomdp_objs_from_list_of_dict(walls, "wall")
        pass_objs = self._make_oomdp_objs_from_list_of_dict(passengers, "passenger")
        toll_objs = self._make_oomdp_objs_from_list_of_dict(tolls, "toll")
        traffic_objs = self._make_oomdp_objs_from_list_of_dict(traffic, "traffic")
        fuel_station_objs = self._make_oomdp_objs_from_list_of_dict(fuel_stations, "fuel_station")

        init_state = self._create_state(agent_obj, wall_objs, pass_objs, toll_objs, traffic_objs, fuel_station_objs)
        if init_state.track_fuel():
            OOMDP.__init__(self, AugmentedTaxiOOMDP.AUGMENTED_ACTIONS, self._taxi_transition_func, self._taxi_reward_func,
                           init_state=init_state, gamma=gamma)
        else:
            OOMDP.__init__(self, AugmentedTaxiOOMDP.BASE_ACTIONS, self._taxi_transition_func, self._taxi_reward_func,
                           init_state=init_state, gamma=gamma)
        self.slip_prob = slip_prob

    def _create_state(self, agent_oo_obj, walls, passengers, tolls, traffic, fuel_stations):
        '''
        Args:
            agent_oo_obj (OOMDPObjects)
            walls (list of OOMDPObject)
            passengers (list of OOMDPObject)
            tolls (list of OOMDPObject)
            traffic (list of OOMDPObject)
            fuel_stations (list of OOMDPObject)

        Returns:
            (OOMDP State)

        TODO: Make this more egneral and put it in OOMDPClass.
        '''

        objects = {c : [] for c in AugmentedTaxiOOMDP.CLASSES}

        objects["agent"].append(agent_oo_obj)

        # Make walls.
        for w in walls:
            objects["wall"].append(w)

        # Make passengers.
        for p in passengers:
            objects["passenger"].append(p)

        # Make tolls.
        for t in tolls:
            objects["toll"].append(t)

        # Make traffic cells.
        for t in traffic:
            objects["traffic"].append(t)

        # Make fuel stations.
        for f in fuel_stations:
            objects["fuel_station"].append(f)

        return AugmentedTaxiState(objects)

    def _taxi_reward_func(self, state, action, next_state=None):
        '''
        Args:
            state (OOMDP State)
            action (str)
            next_state (OOMDP State)

        Returns
            (float)
        '''
        _error_check(state, action)

        reward = 0

        [moved_off_of_toll, toll_fee] = taxi_helpers._moved_off_of_toll(state, next_state)
        if moved_off_of_toll:
            reward -= toll_fee

        # Stacked if statements for efficiency.
        if action == "dropoff":
            # If agent is dropping off.
            agent = state.get_first_obj_of_class("agent")

            # Check to see if all passengers at destination.
            if agent.get_attribute("has_passenger"):
                for p in state.get_objects_of_class("passenger"):
                    if p.get_attribute("x") != p.get_attribute("dest_x") or p.get_attribute("y") != p.get_attribute("dest_y"):
                        reward += 0 - self.step_cost
                        return reward
                reward += 1 - self.step_cost
                return reward
        reward += 0 - self.step_cost
        return reward

    def _taxi_transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        _error_check(state, action)

        # if there is a slip, prevent a navigation action from occurring
        stuck = False
        if self.slip_prob > random.random():
            stuck = True

        # if you're at a traffic cell, determine whether you're stuck or not with the corresponding traffic probability
        at_traffic, prob_traffic = taxi_helpers.at_traffic(state, state.get_agent_x(), state.get_agent_y())

        if at_traffic:
            if prob_traffic > random.random():
                stuck = True

        # decrement fuel if it exists
        if state.track_fuel():
            state.decrement_fuel()

        if action == "up" and state.get_agent_y() < self.height and not stuck:
            next_state = self.move_agent(state, dy=1)
        elif action == "down" and state.get_agent_y() > 1 and not stuck:
            next_state = self.move_agent(state, dy=-1)
        elif action == "right" and state.get_agent_x() < self.width and not stuck:
            next_state = self.move_agent(state, dx=1)
        elif action == "left" and state.get_agent_x() > 1 and not stuck:
            next_state = self.move_agent(state, dx=-1)
        elif action == "dropoff":
            next_state = self.agent_dropoff(state)
        elif action == "pickup":
            next_state = self.agent_pickup(state)
        elif action == "refuel":
            next_state = self.agent_refuel(state)
        else:
            next_state = state

        # Make terminal.
        is_terminal, is_goal = taxi_helpers.is_taxi_terminal_and_goal_state(next_state)
        if is_terminal:
            next_state.set_terminal(True)
        if is_goal:
            next_state.set_goal(True)

        # All OOMDP states must be updated.
        next_state.update()

        return next_state

    def __str__(self):
        return "taxi_h-" + str(self.height) + "_w-" + str(self.width)

    # Visualize the agent's policy. --> Press <spacebar> to advance the agent.
    def visualize_agent(self, agent, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_agent
        from .taxi_visualizer import _draw_state
        visualize_agent(self, agent, _draw_state, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale)

    # Press <1>, <2>, <3>, and so on to execute action 1, action 2, etc.
    def visualize_interaction(self, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_interaction
        from .taxi_visualizer import _draw_state
        visualize_interaction(self, _draw_state, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale)

    # Visualize the value of each of the grid cells. --> Color corresponds to higher value.
    # (Currently not very helpful - see first comment in taxi_visualizer.py)
    def visualize_value(self, agent=None, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_value
        from .taxi_visualizer import _draw_state
        visualize_value(self, _draw_state, agent, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale)

    # Visualize the optimal action for each of the grid cells
    # (Currently not very helpful - see first comment in taxi_visualizer.py)
    def visualize_policy(self, policy, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_policy
        from .taxi_visualizer import _draw_state

        action_char_dict = {
            "up": "^",       #u"\u2191",
            "down": "v",     #u"\u2193",
            "left": "<",     #u"\u2190",
            "right": ">",  # u"\u2192"
            "pickup": "pk",  # u"\u2192"
            "dropoff": "dp",  # u"\u2192"
            "refuel": "rf",  # u"\u2192"
        }
        visualize_policy(self, policy, _draw_state, action_char_dict, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale)

    def visualize_state(self, cur_state, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_state
        from .taxi_visualizer import _draw_state

        visualize_state(self, _draw_state, cur_state=cur_state, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale)


    # ----------------------------
    # -- Action Implementations --
    # ----------------------------

    def move_agent(self, state, dx=0, dy=0):
        '''
        Args:
            state (AugmentedTaxiState)
            dx (int) [optional]
            dy (int) [optional]

        Returns:
            (AugmentedTaxiState)
        '''

        if taxi_helpers._is_wall_in_the_way(state, dx=dx, dy=dy):
            # There's a wall in the way.
            return state

        next_state = copy.deepcopy(state)

        # Move Agent.
        agent_att = next_state.get_first_obj_of_class("agent").get_attributes()
        agent_att["x"] += dx
        agent_att["y"] += dy

        # Move passenger.
        taxi_helpers._move_pass_in_taxi(next_state, dx=dx, dy=dy)

        return next_state

    def agent_pickup(self, state):
        '''
        Args:
            state (AugmentedTaxiState)

        Returns:
            (AugmentedTaxiState)
        '''
        next_state = copy.deepcopy(state)

        agent = next_state.get_first_obj_of_class("agent")

        # update = False
        if agent.get_attribute("has_passenger") == 0:

            # If the agent does not have a passenger.
            for i, passenger in enumerate(next_state.get_objects_of_class("passenger")):
                if agent.get_attribute("x") == passenger.get_attribute("x") and agent.get_attribute("y") == passenger.get_attribute("y"):
                    # Pick up passenger at agent location.
                    agent.set_attribute("has_passenger", 1)
                    passenger.set_attribute("in_taxi", 1)

        return next_state

    def agent_dropoff(self, state):
        '''
        Args:
            state (AugmentedTaxiState)

        Returns:
            (AugmentedTaxiState)
        '''
        next_state = copy.deepcopy(state)

        # Get Agent, Walls, Passengers.
        agent = next_state.get_first_obj_of_class("agent")
        # agent = OOMDPObject(attributes=agent_att, name="agent")
        passengers = next_state.get_objects_of_class("passenger")

        if agent.get_attribute("has_passenger") == 1:
            # Update if the agent has a passenger.
            for i, passenger in enumerate(passengers):

                if passenger.get_attribute("in_taxi") == 1:
                    # Drop off the passenger.
                    passengers[i].set_attribute("in_taxi", 0)
                    agent.set_attribute("has_passenger", 0)

        return next_state

    def agent_refuel(self, state):
        '''
        Args:
            state (AugmentedTaxiState)

        Returns:
            (AugmentedTaxiState)
        '''
        next_state = copy.deepcopy(state)

        # Get Agent, Walls, Passengers.
        agent = next_state.get_first_obj_of_class("agent")

        at_fuel_station, max_fuel_capacity = taxi_helpers.at_fuel_station(state, state.get_agent_x(), state.get_agent_y())

        if at_fuel_station:
            agent["fuel"] = max_fuel_capacity

        return next_state

def _error_check(state, action):
    '''
    Args:
        state (State)
        action (str)

    Summary:
        Checks to make sure the received state and action are of the right type.
    '''

    if state.track_fuel():
        if action not in AugmentedTaxiOOMDP.AUGMENTED_ACTIONS:
            raise ValueError("Error: the action provided (" + str(action) + ") was invalid.")
    else:
        if action not in AugmentedTaxiOOMDP.BASE_ACTIONS:
            raise ValueError("Error: the action provided (" + str(action) + ") was invalid.")

    if not isinstance(state, AugmentedTaxiState):
        raise ValueError("Error: the given state (" + str(state) + ") was not of the correct class.")


def main():
    agent = {"x":1, "y":1, "has_passenger":0}
    passengers = [{"x":8, "y":4, "dest_x":2, "dest_y":2, "in_taxi":0}]
    taxi_world = AugmentedTaxiOOMDP(10, 10, agent=agent, walls=[], passengers=passengers)

if __name__ == "__main__":
    main()
