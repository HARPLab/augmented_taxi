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
import numpy as np

# Other imports.
from simple_rl.mdp.oomdp.OOMDPClass import OOMDP
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject
from simple_rl.tasks.taxi.AugmentedTaxiStateClass import AugmentedTaxiState
from simple_rl.tasks.taxi import taxi_helpers


class AugmentedTaxiOOMDP(OOMDP):
    ''' Class for a Taxi OO-MDP '''

    # Static constants.
    BASE_ACTIONS = ["up", "down", "left", "right", "pickup", "dropoff", "exit"]
    AUGMENTED_ACTIONS = ["up", "down", "left", "right", "pickup", "dropoff", "refuel", "exit"]
    ATTRIBUTES = ["x", "y", "has_passenger", "in_taxi", "dest_x", "dest_y"]
    CLASSES = ["agent", "wall", "passenger", "toll", "traffic", "fuel_station"]

    def __init__(self, width, height, agent, walls, passengers, tolls, traffic, fuel_stations, slip_prob=0, gamma=0.99, step_cost=0, weights=None, env_code=None):
        self.env_code = env_code
        self.height = height
        self.width = width
        if weights is not None:
            self.weights = weights
        else:
            # use true weighting over reward variables (on the goal with the passenger, on a toll, on a traffic cell)
            self.weights = np.array([[2, -0.4, -0.5]])

        # objects that belong in the state (changing)
        agent_obj = OOMDPObject(attributes=agent, name="agent")
        agent_exit = {"x": 100, "y": 100, "has_passenger": 0}  # grid world indices start at (1, 1) so (0, 0) is unreserved
        pass_objs = self._make_oomdp_objs_from_list_of_dict(passengers, "passenger")
        pass_objs_exit = self._make_oomdp_objs_from_list_of_dict(passengers, "passenger")

        # objects that belong to the MDP (static)
        wall_objs = self._make_oomdp_objs_from_list_of_dict(walls, "wall")
        toll_objs = self._make_oomdp_objs_from_list_of_dict(tolls, "toll")
        traffic_objs = self._make_oomdp_objs_from_list_of_dict(traffic, "traffic")
        fuel_station_objs = self._make_oomdp_objs_from_list_of_dict(fuel_stations, "fuel_station")
        self.tolls = toll_objs
        self.traffic_cells = traffic_objs
        self.walls = wall_objs
        self.fuel_stations = fuel_station_objs
        self.slip_prob = slip_prob

        init_state = self._create_state(agent_obj, pass_objs)
        self.exit_state = self._create_state(OOMDPObject(attributes=agent_exit, name="agent_exit"), pass_objs_exit)
        self.exit_state.set_terminal(True)
        self.exit_state.set_goal(False)
        if init_state.track_fuel():
            OOMDP.__init__(self, AugmentedTaxiOOMDP.AUGMENTED_ACTIONS, self._taxi_transition_func, self._taxi_reward_func,
                           init_state=init_state, gamma=gamma, step_cost=step_cost)
        else:
            OOMDP.__init__(self, AugmentedTaxiOOMDP.BASE_ACTIONS, self._taxi_transition_func, self._taxi_reward_func,
                           init_state=init_state, gamma=gamma, step_cost=step_cost)

    def _create_state(self, agent_oo_obj, passengers):
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

        # Make passengers.
        for p in passengers:
            objects["passenger"].append(p)

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

        # 1) MDP-based reward
        # reward = 0
        #
        # if len(self.tolls) != 0:
        #     [moved_into_toll, toll_fee] = taxi_helpers._moved_into_toll(self, state, next_state)
        #     if moved_into_toll:
        #         reward -= toll_fee
        #
        # # Stacked if statements for efficiency.
        # if action == "dropoff":
        #     # If agent is dropping off.
        #     agent = state.get_first_obj_of_class("agent")
        #
        #     # Check to see if all passengers at destination.
        #     if agent.get_attribute("has_passenger"):
        #         for p in state.get_objects_of_class("passenger"):
        #             if p.get_attribute("x") != p.get_attribute("dest_x") or p.get_attribute("y") != p.get_attribute("dest_y"):
        #                 reward += 0 - self.step_cost
        #                 return reward
        #         reward += 1 - self.step_cost
        #         passenger_flag = 1
        #         return reward
        # reward += 0 - self.step_cost
        # return reward

        # 2) feature-based reward
        return self.weights.dot(self.compute_reward_features(state, action, next_state).T)

    def compute_reward_features(self, state, action, next_state=None):
        '''
        Args:
            state (OOMDP State)
            action (str)
            next_state (OOMDP State)

        Returns
            array of reward features
        '''
        # reward features = [successfully dropped off passenger, moved off of toll]
        passenger_flag = 0
        toll_flag = 0
        # traffic_flag = 0
        step_cost_flag = 1

        if next_state == self.exit_state:
            step_cost_flag = 0

        if len(self.tolls) != 0:
            [moved_into_toll, toll_fee] = taxi_helpers._moved_into_toll(self, state, next_state)
            if moved_into_toll and not next_state == self.exit_state:
                toll_flag = 1

        # at_traffic, prob_traffic = taxi_helpers.at_traffic(self, state.get_agent_x(), state.get_agent_y())
        # if at_traffic:
        #     traffic_flag = 1

        # Stacked if statements for efficiency.
        if action == "dropoff":
            # If agent is dropping off.
            agent = state.get_first_obj_of_class("agent")

            # Check to see if all passengers at destination.
            if agent.get_attribute("has_passenger"):
                for p in state.get_objects_of_class("passenger"):
                    if p.get_attribute("x") != p.get_attribute("dest_x") or p.get_attribute("y") != p.get_attribute("dest_y"):
                        return np.array([[passenger_flag, toll_flag, step_cost_flag]])
                passenger_flag = 1
                return np.array([[passenger_flag, toll_flag, step_cost_flag]])

        return np.array([[passenger_flag, toll_flag, step_cost_flag]])

    def accumulate_reward_features(self, trajectory, discount=False):
        reward_features = np.zeros(self.weights.shape)

        # discount the accumulated reward features directly here as you're considering the entire trajectory and likely
        # won't be discounting per (s, a, s') tuple
        if discount:
            step = 0
            for sas in trajectory:
                reward_features += self.gamma ** step * self.compute_reward_features(sas[0], sas[1], sas[2])
                step += 1
        # but still provide the option to return undiscounted accumulated reward features as well
        else:
            for sas in trajectory:
                reward_features += self.compute_reward_features(sas[0], sas[1], sas[2])

        return reward_features


    def _taxi_transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        _error_check(state, action)

        state_is_terminal, state_is_goal = taxi_helpers.is_taxi_terminal_and_goal_state(state)
        if not state_is_terminal:
            # if there is a slip, prevent a navigation action from occurring
            stuck = False
            if self.slip_prob > random.random():
                stuck = True

            # if you're at a traffic cell, determine whether you're stuck or not with the corresponding traffic probability
            at_traffic, prob_traffic = taxi_helpers.at_traffic(self, state.get_agent_x(), state.get_agent_y())

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
            elif action == "exit":
                next_state = copy.deepcopy(self.exit_state)
            else:
                next_state = state

            # Make terminal.
            next_state_is_terminal, next_state_is_goal = taxi_helpers.is_taxi_terminal_and_goal_state(next_state)
            if next_state_is_terminal:
                next_state.set_terminal(True)
            if next_state_is_goal:
                next_state.set_goal(True)

            # All OOMDP states must be updated.
            next_state.update()
        else:
            next_state = state
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
        trajectory = visualize_interaction(self, _draw_state, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale)
        return trajectory

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

    def visualize_trajectory(self, trajectory, marked_state_importances=None, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_trajectory
        from .taxi_visualizer import _draw_state

        visualize_trajectory(self, trajectory, _draw_state, marked_state_importances=marked_state_importances, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale)

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

        if taxi_helpers._is_wall_in_the_way(self, state, dx=dx, dy=dy):
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

        at_fuel_station, max_fuel_capacity = taxi_helpers.at_fuel_station(self, state.get_agent_x(), state.get_agent_y())

        if at_fuel_station:
            agent["fuel"] = max_fuel_capacity

        return next_state

    def measure_env_complexity(self):
        # currently only measuring the number of tolls in the environment
        return len(self.tolls)

    def measure_visual_dissimilarity(self, start_state, other_mdp, other_start_state):
        start_state_weight = 2

        # measure the visual similarity between two MDPs through their start states and their tolls effectively
        dissimilarity = 0

        # start states
        dissimilarity += np.sum(np.abs(np.array([int(x) for x in str(hash(start_state))]) - np.array(
            [int(x) for x in str(hash(other_start_state))]))) * start_state_weight

        # tolls
        dissimilarity += np.sum(np.abs(np.array(self.env_code) - np.array(other_mdp.env_code)))

        return dissimilarity

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
