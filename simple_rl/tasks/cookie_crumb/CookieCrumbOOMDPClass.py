
'''
CookieCrumbOOMDP.py: Contains the CookieCrumbOOMDP class.
'''

# Python imports.
from __future__ import print_function
import random
import copy
import numpy as np

# Other imports.
from simple_rl.mdp.oomdp.OOMDPClass import OOMDP
from simple_rl.mdp.oomdp.OOMDPObjectClass import OOMDPObject
from simple_rl.tasks.cookie_crumb.CookieCrumbStateClass import CookieCrumbState
from simple_rl.tasks.cookie_crumb import cookie_crumb_helpers


class CookieCrumbOOMDP(OOMDP):
    ''' Class for a Cookie Crumb OO-MDP '''

    # Static constants.
    ACTIONS = ["up", "down", "left", "right"]
    CLASSES = ["agent", "crumb"]

    def __init__(self, width, height, agent, walls, goals, crumbs, slip_prob=0, gamma=0.99, step_cost=0, weights=None, env_code=None, sample_rate=5):
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
        crumb_objs = self._make_oomdp_objs_from_list_of_dict(crumbs, "crumb")

        # objects that belong to the MDP (static)
        wall_objs = self._make_oomdp_objs_from_list_of_dict(walls, "wall")
        self.walls = wall_objs
        goal_objs = self._make_oomdp_objs_from_list_of_dict(goals, "goal")
        self.goals = goal_objs
        self.slip_prob = slip_prob


        init_state = self._create_state(agent_obj, crumb_objs)
        OOMDP.__init__(self, CookieCrumbOOMDP.ACTIONS, self._cookie_crumb_transition_func, self._cookie_crumb_reward_func,
                       init_state=init_state, gamma=gamma, step_cost=step_cost, sample_rate=sample_rate)

    def _create_state(self, agent_oo_obj, crumbs):
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

        objects = {c : [] for c in CookieCrumbOOMDP.CLASSES}

        objects["agent"].append(agent_oo_obj)

        # Make crumbs.
        for c in crumbs:
            objects["crumb"].append(c)

        return CookieCrumbState(objects)

    def _cookie_crumb_reward_func(self, state, action, next_state=None):
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
        # reward features = [at goal, ate a cookie crumb, step cost]
        reward_features = []
        step_cost_flag = 1

        for g in self.goals:
            if next_state.get_agent_x() == g.get_attribute("x") and next_state.get_agent_y() == g.get_attribute("y"):
                reward_features.append(1)
            else:
                reward_features.append(0)

        if len(next_state.get_objects_of_class("crumb")) < len(state.get_objects_of_class("crumb")):
            reward_features.append(1)
        else:
            reward_features.append(0)

        reward_features.append(step_cost_flag)

        reward_features = np.array(reward_features).reshape(1, -1)
        return reward_features

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

    def _cookie_crumb_transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        _error_check(state, action)

        state_is_terminal, state_is_goal = cookie_crumb_helpers.is_terminal_and_goal_state(self, state)
        if not state_is_terminal:
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
            else:
                next_state = state

            # check if agent is on any one of the crumbs. if so, delete it
            crumbs = next_state.get_objects_of_class("crumb")
            for i, crumb in enumerate(crumbs):
                if crumb.attributes['x'] == next_state.get_agent_x() and crumb.attributes['y'] == next_state.get_agent_y():
                    # you can only move into one crumb at a time so you can safely break after
                    del crumbs[i]
                    break
            next_state.objects['crumb'] = crumbs

            # Make terminal.
            next_state_is_terminal, next_state_is_goal = cookie_crumb_helpers.is_terminal_and_goal_state(self, next_state)
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
        return "cookie_crumb_h-" + str(self.height) + "_w-" + str(self.width)

    # Visualize the agent's policy. --> Press <spacebar> to advance the agent.
    def visualize_agent(self, agent, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_agent
        from .cookie_crumb_visualizer import _draw_state
        visualize_agent(self, agent, _draw_state, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale)

    # Press <1>, <2>, <3>, and so on to execute action 1, action 2, etc.
    def visualize_interaction(self, interaction_callback=None, done_callback=None, keys_map=None, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_interaction
        from .cookie_crumb_visualizer import _draw_state
        trajectory = visualize_interaction(self, _draw_state, interaction_callback=interaction_callback, done_callback=done_callback, keys_map=keys_map, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale)
        return trajectory

    # Visualize the value of each of the grid cells. --> Color corresponds to higher value.
    # (Currently not very helpful - see first comment in cookie_crumb_visualizer.py)
    def visualize_value(self, agent=None, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_value
        from .cookie_crumb_visualizer import _draw_state
        visualize_value(self, _draw_state, agent, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale)

    # Visualize the optimal action for each of the grid cells
    # (Currently not very helpful - see first comment in cookie_crumb_visualizer.py)
    def visualize_policy(self, policy, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_policy
        from .cookie_crumb_visualizer import _draw_state

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
        from .cookie_crumb_visualizer import _draw_state

        visualize_state(self, _draw_state, cur_state=cur_state, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale)

    def visualize_trajectory(self, trajectory, marked_state_importances=None, width_scr_scale=180, height_scr_scale=180):
        from simple_rl.utils.mdp_visualizer import visualize_trajectory
        from .cookie_crumb_visualizer import _draw_state

        visualize_trajectory(self, trajectory, _draw_state, marked_state_importances=marked_state_importances, scr_width=self.width*width_scr_scale, scr_height=self.height*height_scr_scale)

    # ----------------------------
    # -- Action Implementations --
    # ----------------------------

    def move_agent(self, state, dx=0, dy=0):
        '''
        Args:
            state (CookieCrumbState)
            dx (int) [optional]
            dy (int) [optional]

        Returns:
            (CookieCrumbState)
        '''

        if cookie_crumb_helpers._is_wall_in_the_way(self, state, dx=dx, dy=dy):
            # There's a wall in the way.
            return state

        next_state = copy.deepcopy(state)

        # Move Agent.
        agent_att = next_state.get_first_obj_of_class("agent").get_attributes()
        agent_att["x"] += dx
        agent_att["y"] += dy

        return next_state

    def measure_env_complexity(self):
        # measures the starting number of crumbs in the environment
        return np.sum(self.env_code)

    def measure_visual_dissimilarity(self, start_state, other_mdp, other_start_state):
        start_state_weight = 2

        # measure the visual similarity between two MDPs through their start states and their tolls effectively
        dissimilarity = 0

        # starting location of the agent
        # dissimilarity += np.sum(np.abs(np.array([int(x) for x in str(hash(start_state))]) - np.array(
        #     [int(x) for x in str(hash(other_start_state))]))) * start_state_weight

        dissimilarity += abs(start_state.get_agent_x() - other_start_state.get_agent_x()) + abs(start_state.get_agent_y() - other_start_state.get_agent_y()) * start_state_weight

        # crumbs
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

    if action not in CookieCrumbOOMDP.ACTIONS:
        raise ValueError("Error: the action provided (" + str(action) + ") was invalid.")

    if not isinstance(state, CookieCrumbState):
        raise ValueError("Error: the given state (" + str(state) + ") was not of the correct class.")


def main():
    agent = {"x":1, "y":1}
    cookie_crumb_world = CookieCrumbOOMDP(10, 10, agent=agent, walls=[])

if __name__ == "__main__":
    main()
