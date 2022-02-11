'''
SkateboardOOMDP.py: Contains the SkateboardOOMDP class.

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
from simple_rl.tasks.navigation.AugmentedNavigationStateClass import AugmentedNavigationState
from simple_rl.tasks.navigation import navigation_helpers

class AugmentedNavigationOODMP(OOMDP):
    ''' Class for a Skateboard OO-MDP '''

    # Static constants.
    ACTIONS = ["up", "down", "left", "right", "pickup", "dropoff"]
    ATTRIBUTES = ["x", "y", "has_skateboard", "has_car"]
    CLASSES = ["agent", "wall", "skateboard", "car", "hotswap_station"]

    def __init__(self, width, height, agent, walls, gravel, grass, roads, hotswap_stations, skateboard, cars, goal, slip_prob=0, gamma=0.99, step_cost=0, weights=None, env_code=None, sample_rate=5):
        self.env_code = env_code
        self.height = height
        self.width = width
        if weights is not None:
            self.weights = weights
        else:
            raise Exception("No reward weights specified.")

        # objects that belong in the state (changing)
        agent_obj = OOMDPObject(attributes=agent, name="agent")
        skateboard_objs = self._make_oomdp_objs_from_list_of_dict(skateboard, "skateboard")
        car_objs = self._make_oomdp_objs_from_list_of_dict(cars, "car")
        hotswap_station_objs = self._make_oomdp_objs_from_list_of_dict(hotswap_stations, "hotswap_station")

        # objects that belong to the MDP (static)
        wall_objs = self._make_oomdp_objs_from_list_of_dict(walls, "wall")
        road_objs = self._make_oomdp_objs_from_list_of_dict(roads, "road")
        gravel_objs = self._make_oomdp_objs_from_list_of_dict(gravel, "gravel")
        grass_objs = self._make_oomdp_objs_from_list_of_dict(grass, "grass")

        self.walls = wall_objs
        self.roads = road_objs
        self.gravel = gravel_objs
        self.grass = grass_objs

        self.goal = goal
        self.slip_prob = slip_prob

        init_state = self._create_state(agent_obj, skateboard_objs, car_objs, hotswap_station_objs)
        OOMDP.__init__(self, AugmentedNavigationOODMP.ACTIONS, self._navigation_transition_func, self._navigation_reward_func,
                       init_state=init_state, gamma=gamma, step_cost=step_cost, sample_rate=sample_rate)

    def _create_state(self, agent_oo_obj, skateboard, cars, hotswap_stations):
        '''
        Args:
            agent_oo_obj (OOMDPObjects)
            skateboard (list of OOMDPObject)

        Returns:
            (OOMDP State)

        TODO: Make this more egneral and put it in OOMDPClass.
        '''

        objects = {c : [] for c in AugmentedNavigationOODMP.CLASSES}

        objects["agent"].append(agent_oo_obj)

        # Make skateboard.
        for s in skateboard:
            objects["skateboard"].append(s)

        # Make car.
        for c in cars:
            objects["car"].append(c)

        # Make hotswap stations.
        for hs in hotswap_stations:
            objects["hotswap_station"].append(hs)

        return AugmentedNavigationState(objects)

    def _navigation_reward_func(self, state, action, next_state=None):
        '''
        Args:
            state (OOMDP State)
            action (str)
            next_state (OOMDP State)

        Returns
            (float)
        '''
        _error_check(state, action)

        # feature-based reward
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
        gravel_flag = 0
        grass_flag = 0
        road_flag = 0
        hotswap_flag = 0
        skateboard_flag = 0
        car_flag = 0
        base_step_cost_flag = 1

        agent = state.get_first_obj_of_class("agent")

        if len(self.grass) != 0:
            moved_off_of_grass = navigation_helpers._moved_off_of_grass(self, state, next_state)
            if moved_off_of_grass:
                grass_flag = 1

        if len(self.gravel) != 0:
            moved_off_of_gravel = navigation_helpers._moved_off_of_gravel(self, state, next_state)
            if moved_off_of_gravel:
                gravel_flag = 1

        if len(state.objects['hotswap_station']) != 0:
            moved_off_of_hotswap_station, _ = navigation_helpers._moved_off_of_hotswap_station(state, next_state)
            if moved_off_of_hotswap_station:
                hotswap_flag = 1

        if len(self.gravel) != 0:
            moved_off_of_road = navigation_helpers._moved_off_of_road(self, state, next_state)
            if moved_off_of_road:
                road_flag = 1

        # movement is penalized differently based on whether you have the skateboard or not
        if action == 'up' or action == 'down' or action == 'left' or action == 'right':
            if agent.get_attribute("has_skateboard") == 1:
                skateboard_flag = 1
            elif agent.get_attribute("has_car") == 1:
                car_flag = 1
        else:
            base_step_cost_flag = 1

        return np.array([[gravel_flag, grass_flag, road_flag, hotswap_flag, skateboard_flag, car_flag, base_step_cost_flag]])

    def accumulate_reward_features(self, trajectory, discount=False):
        reward_features = np.zeros(self.weights.shape, dtype='int')

        # discount the accumulated reward features directly here as you're considering the entire trajectory and likely
        # won't be discounting per (s, a, s') tuple
        if discount:
            step = 0
            for sas in trajectory:
                reward_features = reward_features + self.gamma ** step * self.compute_reward_features(sas[0], sas[1], sas[2])
                step += 1
        # but still provide the option to return undiscounted accumulated reward features as well
        else:
            for sas in trajectory:
                reward_features = reward_features + self.compute_reward_features(sas[0], sas[1], sas[2])

        return reward_features

    def _navigation_transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        _error_check(state, action)

        state_is_goal = navigation_helpers.is_goal_state(self, state)
        if not state_is_goal:
            # if there is a slip, prevent a navigation action from occurring
            stuck = False
            if self.slip_prob > random.random():
                stuck = True

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
            else:
                next_state = state

            # Make terminal.
            next_state_is_goal = navigation_helpers.is_goal_state(self, next_state)
            if next_state_is_goal:
                next_state.set_terminal(True)
                next_state.set_goal(True)

            # All OOMDP states must be updated.
            next_state.update()
        else:
            next_state = state
        return next_state

    def __str__(self):
        return "augmented_navigation_h-" + str(self.height) + "_w-" + str(self.width)

    # Visualize the agent's policy. --> Press <spacebar> to advance the agent.
    def visualize_agent(self, agent, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_agent
        from .navigation_visualizer import _draw_state
        visualize_agent(self, agent, _draw_state, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale, mdp_class='skateboard2')

    # Press <1>, <2>, <3>, and so on to execute action 1, action 2, etc.
    def visualize_interaction(self, interaction_callback=None, done_callback=None, keys_map=None, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_interaction
        from .navigation_visualizer import _draw_state
        trajectory = visualize_interaction(self, _draw_state, interaction_callback=interaction_callback, done_callback=done_callback, keys_map=keys_map, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale, mdp_class='skateboard2')
        return trajectory

    # Visualize the value of each of the grid cells. --> Color corresponds to higher value.
    def visualize_value(self, agent=None, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_value
        from .navigation_visualizer import _draw_state
        visualize_value(self, _draw_state, agent, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale)

    # Visualize the optimal action for each of the grid cells
    def visualize_policy(self, policy, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_policy
        from .navigation_visualizer import _draw_state

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
        from .navigation_visualizer import _draw_state

        visualize_state(self, _draw_state, cur_state=cur_state, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale)

    def visualize_trajectory(self, trajectory, marked_state_importances=None, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_trajectory
        from .navigation_visualizer import _draw_state

        visualize_trajectory(self, trajectory, _draw_state, marked_state_importances=marked_state_importances, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale, mdp_class='skateboard2')

    # ----------------------------
    # -- Action Implementations --
    # ----------------------------

    def move_agent(self, state, dx=0, dy=0):
        '''
        Args:
            state (AugmentedNavigationState)
            dx (int) [optional]
            dy (int) [optional]

        Returns:
            (AugmentedNavigationState)
        '''

        if navigation_helpers._is_wall_in_the_way(self, state, dx=dx, dy=dy):
            # There's a wall in the way.
            return state

        if navigation_helpers._is_road_in_the_way(self, state, dx=dx, dy=dy) and state.get_first_obj_of_class("agent").get_attribute("has_skateboard"):
            # Skateboard is not allowed on road
            return state

        if navigation_helpers._is_road_in_the_way(self, state, dx=dx, dy=dy) and state.get_first_obj_of_class("agent").get_attribute("has_car"):
            # Car is not allowed on road
            return state

        next_state = copy.deepcopy(state)

        # Move Agent.
        agent_att = next_state.get_first_obj_of_class("agent").get_attributes()
        agent_att["x"] += dx
        agent_att["y"] += dy

        # Move skateboard.
        navigation_helpers._move_skateboard_on_agent(next_state, dx=dx, dy=dy)
        # Move car.
        navigation_helpers._move_car_on_agent(next_state, dx=dx, dy=dy)

        return next_state

    def agent_pickup(self, state):
        '''
        Args:
            state (AugmentedNavigationState)

        Returns:
            (AugmentedNavigationState)
        '''
        next_state = copy.deepcopy(state)

        agent = next_state.get_first_obj_of_class("agent")

        if agent.get_attribute("has_skateboard") == 0:
            # If the agent does not have a skateboard.
            for i, skateboard in enumerate(next_state.get_objects_of_class("skateboard")):
                if agent.get_attribute("x") == skateboard.get_attribute("x") and agent.get_attribute("y") == skateboard.get_attribute("y"):
                    # Pick up skateboard at agent location.
                    agent.set_attribute("has_skateboard", 1)
                    skateboard.set_attribute("on_agent", 1)


        if agent.get_attribute("has_car") == 0:
            # If the agent does not have a car.
            for i, car in enumerate(next_state.get_objects_of_class("car")):
                if agent.get_attribute("x") == car.get_attribute("x") and agent.get_attribute("y") == car.get_attribute("y"):
                    # Pick up car at agent location.
                    agent.set_attribute("has_car", 1)
                    car.set_attribute("on_agent", 1)


        return next_state

    def agent_dropoff(self, state):
        '''
        Args:
            state (AugmentedNavigationState)

        Returns:
            (AugmentedNavigationState)
        '''
        next_state = copy.deepcopy(state)

        # Get Agent, Walls, skateboard.
        agent = next_state.get_first_obj_of_class("agent")
        # agent = OOMDPObject(attributes=agent_att, name="agent")
        skateboards = next_state.get_objects_of_class("skateboard")

        if agent.get_attribute("has_skateboard") == 1:
            # Update if the agent has a skateboard.
            for i, skateboard in enumerate(skateboards):

                if skateboard.get_attribute("on_agent") == 1:
                    # Drop off the skateboard.
                    skateboards[i].set_attribute("on_agent", 0)
                    agent.set_attribute("has_skateboard", 0)


        cars = next_state.get_objects_of_class("car")

        if agent.get_attribute("has_car") == 1:
            # Update if the agent has a car.
            for i, car in enumerate(cars):

                if car.get_attribute("on_agent") == 1:
                    # Drop off the car.
                    cars[i].set_attribute("on_agent", 0)
                    agent.set_attribute("has_car", 0)

        return next_state


    def measure_env_complexity(self):
        return len(self.gravel) + len(self.grass) + len(self.roads) + len(self.init_state.objects['hotswap_station']) + len(self.init_state.objects['skateboard']) + len(self.init_state.objects['skateboard'])

    def measure_visual_dissimilarity(self, start_state, other_mdp, other_start_state):
        # measure the visual similarity between two MDPs through their start states and their roads effectively
        dissimilarity = 0

        dissimilarity += np.sum(np.abs(hash(start_state) - hash(other_start_state)))

        # roads and skateboard
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

    if action not in AugmentedNavigationOODMP.ACTIONS:
        raise ValueError("Error: the action provided (" + str(action) + ") was invalid.")

    if not isinstance(state, AugmentedNavigationState):
        raise ValueError("Error: the given state (" + str(state) + ") was not of the correct class.")


def main():
    agent = {"x": 1, "y": 1, "has_skateboard": 0}
    skateboard = [{"x": 8, "y": 4, "on_agent": 0}]
    navigation_world = AugmentedNavigationOODMP(10, 10, agent=agent, walls=[], skateboard=skateboard)

if __name__ == "__main__":
    main()
