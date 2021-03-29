''' MDPClass.py: Contains the MDP Class. '''

# Python imports.
import copy
import sys
if sys.version_info[0] < 3:
	import Queue as queue
else:
	import queue


class MDP(object):
    ''' Abstract class for a Markov Decision Process. '''

    def __init__(self, actions, transition_func, reward_func, init_state, gamma=0.99, step_cost=0, sample_rate=5):
        self.actions = actions
        self.transition_func = transition_func
        self.reward_func = reward_func
        self.gamma = gamma
        self.init_state = copy.deepcopy(init_state)
        self.cur_state = init_state
        self.step_cost = step_cost
        self.states = set([])
        self.reachability_done = False
        self.sample_rate = sample_rate

    # ---------------
    # -- Accessors --
    # ---------------

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = {}
        param_dict["gamma"] = self.gamma
        param_dict["step_cost"] = self.step_cost

        return param_dict

    def get_init_state(self):
        return self.init_state

    def get_curr_state(self):
        return self.cur_state

    def get_actions(self):
        return self.actions

    def get_gamma(self):
        return self.gamma

    def get_reward_func(self):
        return self.reward_func

    def get_transition_func(self):
        return self.transition_func

    def get_num_state_feats(self):
        return self.init_state.get_num_feats()

    def get_slip_prob(self):
        pass

    def get_name(self):
        return str(self)

    def get_num_states(self):
        if not self.reachability_done:
            self._compute_reachable_state_space()
        return len(self.states)

    def get_states(self):
        if self.reachability_done:
            return list(self.states)
        else:
            self._compute_reachable_state_space()
            return list(self.states)

    # --------------
    # -- Mutators --
    # --------------

    def set_gamma(self, new_gamma):
        self.gamma = new_gamma

    def set_step_cost(self, new_step_cost):
        self.step_cost = new_step_cost

    def set_slip_prob(self, slip_prob):
        pass

    def set_init_state(self, new_init_state):
        self.init_state = copy.deepcopy(new_init_state)

    def set_curr_state(self, cur_state):
        self.cur_state = cur_state

    # ----------
    # -- Core --
    # ----------

    def _compute_reachable_state_space(self):
        '''
        Summary:
            Starting with @self.start_state, determines all reachable states
            and stores them in self.states.
        '''

        if self.reachability_done:
            return

        state_queue = queue.Queue()
        state_queue.put(self.init_state)
        self.states.add(self.init_state)

        while not state_queue.empty():
            s = state_queue.get()
            for a in self.actions:
                for samples in range(self.sample_rate): # Take @sample_rate samples to estimate E[V]
                    next_state = self.transition_func(copy.deepcopy(s), a)

                    if next_state not in self.states and not next_state.is_terminal():
                        self.states.add(next_state)
                        state_queue.put(next_state)

        self.reachability_done = True

    def execute_agent_action(self, action):
        '''
        Args:
            action (str)

        Returns:
            (tuple: <float,State>): reward, State

        Summary:
            Core method of all of simple_rl. Facilitates interaction
            between the MDP and an agent.
        '''
        next_state = self.transition_func(self.cur_state, action)
        reward = self.reward_func(self.cur_state, action, next_state)
        self.cur_state = next_state

        return reward, next_state

    def reset(self):
        self.cur_state = copy.deepcopy(self.init_state)

    def end_of_instance(self):
        pass
