''' SkateboardStateClass.py: Contains the SkateboardState class. '''

# Other imports
from simple_rl.mdp.oomdp.OOMDPStateClass import OOMDPState

class SkateboardState(OOMDPState):
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

        state_hash = str(self.get_agent_x()) + str(self.get_agent_y()) + "00"

        for s in self.objects["skateboard"]:
            state_hash += str(s["x"]) + str(s["y"]) + str(s["on_agent"])

        return int(state_hash)

    def __eq__(self, other_skateboard_state):
        return hash(self) == hash(other_skateboard_state)
