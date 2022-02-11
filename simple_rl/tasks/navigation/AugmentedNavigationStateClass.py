''' SkateboardStateClass.py: Contains the SkateboardState class. '''

# Other imports
from simple_rl.mdp.oomdp.OOMDPStateClass import OOMDPState

class AugmentedNavigationState(OOMDPState):
    ''' Class for Skateboard World States '''

    def __init__(self, objects):
        OOMDPState.__init__(self, objects=objects)

    def get_agent_x(self):
        return self.objects["agent"][0]["x"]

    def get_agent_y(self):
        return self.objects["agent"][0]["y"]

    def is_an_exit_state(self, ref_exit_state):
        if self.get_agent_x() == ref_exit_state.get_agent_x() and self.get_agent_y() == ref_exit_state.get_agent_y():
            return True
        else:
            return False


    def __hash__(self):

        state_hash = str(self.get_agent_x()) + str(self.get_agent_y())

        # currently assuming that there is only one of each object and locations are all single digits
        try:
            s = self.objects["skateboard"][0]
            state_hash += str(s["x"]) + str(s["y"]) + str(s["on_agent"])
        except:
            state_hash += '000'

        try:
            p = self.objects["car"][0]
            state_hash += str(p["x"]) + str(p["y"]) + str(p["on_agent"])
        except:
            state_hash += '000'

        try:
            hs = self.objects["hotswap_station"][0]
            state_hash = str(hs["x"]) + str(hs["y"])
        except:
            state_hash += '00'

        return int(state_hash)

    def __eq__(self, other_skateboard_state):
        return hash(self) == hash(other_skateboard_state)
