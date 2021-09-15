''' AugmentedTaxiStateClass.py: Contains the AugmentedTaxiState class. '''

# Other imports
from simple_rl.mdp.oomdp.OOMDPStateClass import OOMDPState

class AugmentedTaxiState(OOMDPState):
    ''' Class for Taxi World States '''

    def __init__(self, objects):
        OOMDPState.__init__(self, objects=objects)

    def get_agent_x(self):
        return self.objects["agent"][0]["x"]

    def get_agent_y(self):
        return self.objects["agent"][0]["y"]

    def get_fuel(self):
        return self.objects["agent"][0]["fuel"]

    def track_fuel(self):
        try:
            if self.objects["agent"][0]["fuel"] is not None:
                return True
            else:
                return False
        except:
            return False

    def decrement_fuel(self):
        self.objects["agent"][0]["fuel"] -= 1

    def is_an_exit_state(self, ref_exit_state):
        if self.get_agent_x() == ref_exit_state.get_agent_x() and self.get_agent_y() == ref_exit_state.get_agent_y():
            return True
        else:
            return False

    # selectively return attributes of the state to print out
    def abbr_str(self):
        return "Fuel: " + str(self.objects["agent"][0]["fuel"])

    def __hash__(self):

        state_hash = str(self.get_agent_x()) + str(self.get_agent_y())

        for p in self.objects["passenger"]:
            state_hash += str(p["x"]) + str(p["y"]) + str(p["in_taxi"])

        if "hotswap_station" in self.objects.keys():
            for hs in self.objects["hotswap_station"]:
                state_hash += str(hs["x"]) + str(hs["y"])

        if self.track_fuel():
            state_hash += str(self.get_fuel())

        return int(state_hash)

    def __eq__(self, other_taxi_state):
        return hash(self) == hash(other_taxi_state)
