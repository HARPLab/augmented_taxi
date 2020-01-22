from queue import PriorityQueue
from datetime import datetime
import copy

def computeHighlightsStateImportance(agent, state):
    '''
    Args:
        agent (Agent)
        state (State)

    Returns:
        difference between max and min q_val (float), best_action (str)

    Summary:
        Return a collection of informative trajectories that summarize the agent's policy
    '''
    max_q_val, best_action = agent._compute_max_qval_action_pair(state)
    min_q_val, worst_action = agent._compute_min_qval_action_pair(state)
    return max_q_val - min_q_val, best_action

def mark_critical_state(location, state_importances):
    '''
    Args:
        location (int)
        state_importances (list of floats)

    Returns:
        state_importance (float), state_importances (list of floats)

    Summary:
        Returns importance value of the critical state indexed by location and a list of other state importances blacked out by -inf
    '''
    for x in range(len(state_importances)):
        # location indexes from the back of the array
        if x != location:
            state_importances[-(x+1)] = float('-inf')
        else:
            state_importance = state_importances[-(x+1)]
    return state_importance, state_importances

def obtain_summary(mdp, agent, max_summary_count=10, trajectory_length=5, n_simulations=10, interval_size=3, n_trailing_states=2):
    '''
    Args:
        mdp (MDP)
        agent (Agent)
        max_summary_count (int, max number of trajectories in summary)
        trajectory_length (int, max length of a trajectory)
        n_simulations (int)
        interval_size (int, number of states to wait before considering another critical state)
        n_trailing_states (int, number of states to track after a critical state for a trajectory)

    Returns:
        summary (PriorityQueue of (importance, timestamp, trajectory))

    Summary:
        Return a collection of informative trajectories that summarize the agent's policy
    '''

    # error check
    if interval_size < 1:
        raise ValueError("Interval size should be at least 1.")
    if (trajectory_length <= n_trailing_states) or (n_trailing_states < 0):
        raise ValueError("Number of trailing states should be between greater than or equal to 0, and less than "
                         "the trajectory length.")

    simulation = 0
    summary = PriorityQueue(maxsize=max_summary_count)

    # while there are more simulations to run
    while simulation < n_simulations:
        # initialize the simulation
        mdp.reset()
        cur_state = mdp.get_init_state() # todo
        trajectory = []                     # a list of consecutive states
        state_importances = []              # corresponding state_importances of states in the trajectory
        trailing_count_up = []              # track number of states passed since tracking a particular state
        interval_count_down = [1]           # track number of states passed since tracking a particular state
                                            # (seeded with a one for the first pass through the code)
        reward = 0
        tracking_trajectory = False

        while not cur_state.is_terminal(): # todo
            action = agent.act(cur_state, reward)
            reward, next_state = mdp.execute_agent_action(action)

            if len(trajectory) == trajectory_length:
                trajectory.pop(0)
            trajectory.append((cur_state, action))

            # housekeeping
            for x in range(len(trailing_count_up)):
                trailing_count_up[x] += 1
            for x in range(len(interval_count_down)):
                interval_count_down[x] -= 1

            # compute importance of this state
            state_importance, best_action = computeHighlightsStateImportance(agent, cur_state) # todo
            # state_importance += 1

            if len(state_importances) == trajectory_length:
                state_importances.pop(0)
            # state importances of all states currently in trajectory
            state_importances.append(state_importance)

            # if summary is incomplete or this state is superior, and a sufficient number of states have passed
            # since the last critical state, count this state as critical
            # (note: can peak with summary.queue[0][0] since the smallest element of a heap is always the root)
            if (summary.qsize() < max_summary_count or state_importance > summary.queue[0][0]) and (interval_count_down[0] == 0):
                if summary.qsize() == max_summary_count:
                    summary.get()
                tracking_trajectory = True

                # housekeeping
                trailing_count_up.append(0)
                interval_count_down.pop(0)
                interval_count_down.append(interval_size)

            if len(trailing_count_up) > 0:
                # if a sufficient number of trailing states have been observed, return the trajectory with the state
                # importance of the critical state marked (both as a singular value - marked_state_importance,
                # and in the context of the larger trajectory - marked_state_importances)
                if trailing_count_up[0] == n_trailing_states:
                    marked_state_importance, marked_state_importances = mark_critical_state(trailing_count_up[0],
                                                                        copy.deepcopy(state_importances))
                    # also store the time as a unique tiebreaker in the case of equal state importances
                    summary.put(
                        (marked_state_importance, str(datetime.now()), copy.deepcopy(trajectory), copy.deepcopy(marked_state_importances)))
                    trailing_count_up.pop(0)
                    tracking_trajectory = False

            cur_state = copy.deepcopy(next_state) # todo

        # return any portion of a trajectory being tracked that was cut off due to the end of an episode
        if tracking_trajectory:
            marked_state_importance, marked_state_importances = mark_critical_state(trailing_count_up[0],
                                                                 copy.deepcopy(state_importances))
            # include preceding states and include as many of the desired number of trailing states as possible
            summary.put((marked_state_importance, str(datetime.now()), copy.deepcopy(trajectory[-(trajectory_length - n_trailing_states + trailing_count_up[0]):]),
                         copy.deepcopy(marked_state_importances[-(trajectory_length - n_trailing_states + trailing_count_up[0]):])))
        simulation += 1

    return summary