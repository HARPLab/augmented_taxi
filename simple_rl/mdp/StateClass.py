# Python imports
import numpy as np

''' StateClass.py: Contains the State Class. '''

class State(object):
    ''' Abstract State class '''

    def __init__(self, data=[], is_terminal=False, is_goal=False):
        self.data = data
        self._is_terminal = is_terminal
        self._is_goal = is_goal

    def features(self):
        '''
        Summary
            Used by function approximators to represent the state.
            Override this method in State subclasses to have functiona
            approximators use a different set of features.
        Returns:
            (iterable)
        '''
        return np.array(self.data).flatten()

    def get_data(self):
        return self.data

    def get_num_feats(self):
        return len(self.features())

    def is_terminal(self):
        return self._is_terminal

    def set_terminal(self, is_term=True):
        self._is_terminal = is_term

    def is_goal(self):
        return self._is_goal

    def set_goal(self, is_goal=True):
        self._is_goal = is_goal

    def __hash__(self):
        if type(self.data).__module__ == np.__name__:
            # Numpy arrays
            return hash(str(self.data))
        elif self.data.__hash__ is None:
            return hash(tuple(self.data))
        else:
            return hash(self.data)

    def __str__(self):
        return "s." + str(self.data)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.data == other.data
        return False

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
