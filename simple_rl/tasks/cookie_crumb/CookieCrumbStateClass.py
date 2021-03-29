''' CookieCrumbStateClass.py: Contains the CookieCrumb class. '''

# Other imports
from simple_rl.mdp.oomdp.OOMDPStateClass import OOMDPState

class CookieCrumbState(OOMDPState):
    ''' Class for CookieCrumb World States '''

    def __init__(self, objects):
        OOMDPState.__init__(self, objects=objects)

    def get_agent_x(self):
        return self.objects["agent"][0]["x"]

    def get_agent_y(self):
        return self.objects["agent"][0]["y"]

    def __hash__(self):

        state_hash = str(self.get_agent_x()) + str(self.get_agent_y())

        for c in self.objects["crumb"]:
            state_hash += str(c["x"]) + str(c["y"])

        return int(state_hash)

    def __eq__(self, other_state):
        return hash(self) == hash(other_state)
