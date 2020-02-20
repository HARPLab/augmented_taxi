'''
This module contains variants of the HIGHLIGHTS algorithm (see highlights.py) for selecting important states to convey
for summarizing a policy.

Accordingly, all variants are a different take on comparing Q-values of the best and worst action (e.g. only comparing
within the agent's policy, comparing between the agent and human's policy, etc.
'''

def single_policy_extraction(states, mdp, agent, n_visualize=0):
    '''
    Args:
        states (list of States)
        mdp (MDP)
        agent (Agent)
        n_visualize (int)

    Returns:
        q_diffs (list of [q-val difference, best action, worst action, state])

    Summary:
        Extract important summarization states to convey by considering the Q-value of actions within one agent's policy.
        Consider all reachable states in the given MDP.
    '''

    q_diffs = []
    for s in states:
        max_q_val, best_action = agent._compute_max_qval_action_pair(s)
        min_q_val, worst_action = agent._compute_min_qval_action_pair(s)
        q_diffs.append([max_q_val - min_q_val, best_action, worst_action, s])

    q_diffs.sort(key=lambda x: x[0], reverse=True)

    # Visualize the top n states
    if n_visualize > 0:
        for state_number in range(n_visualize):
            print("Number: {}".format(state_number))
            print("Best action: {}".format(q_diffs[state_number][1]))
            print("Worst action: {}".format(q_diffs[state_number][2]))
            print("Q-val difference: {}".format(q_diffs[state_number][0]))
            mdp.visualize_state(q_diffs[state_number][3])

    return q_diffs


def double_policy_extraction(states, mdp, agent_1, agent_2, n_visualize=0, action_conditioned=False):
    '''
    Args:
        states (list of States)
        mdp (MDP)
        agent (Agent)
        n_visualize (int)

    Returns:
        q_diffs [[q-val difference, best action, worst action, state], ...]

    Summary:
        Extract important summarization states to convey by considering the Q-value of actions across the policies of
        two agents. Consider all reachable states in the given MDP.
    '''

    q_diffs = []
    for s in states:
        # compare the two agents' Q-values of the human's best action
        if action_conditioned:
            max_q_val_agent, best_action_agent = agent_1._compute_max_qval_action_pair(s)
            min_q_val_agent, worst_action_agent = agent_1._compute_min_qval_action_pair(s)
            max_q_val_human, best_action_human = agent_2._compute_max_qval_action_pair(s)
            min_q_val_human, worst_action_human = agent_2._compute_min_qval_action_pair(s)

            q_val_agent_human_action = agent_1.get_q_value(s, best_action_human)

            if best_action_human != best_action_agent:
                q_val_diff = abs(q_val_agent_human_action - max_q_val_human)
            else:
                q_val_diff = float('-inf')

            q_diffs.append([q_val_diff, best_action_agent, worst_action_agent, best_action_human, worst_action_human, s])
        # compare the two agents' Q-values of their respective best actions
        else:
            max_q_val_agent, best_action_agent = agent_1._compute_max_qval_action_pair(s)
            min_q_val_agent, worst_action_agent = agent_1._compute_min_qval_action_pair(s)
            max_q_val_human, best_action_human = agent_2._compute_max_qval_action_pair(s)
            min_q_val_human, worst_action_human = agent_2._compute_min_qval_action_pair(s)
            q_diffs.append([abs((max_q_val_agent - min_q_val_agent) - (max_q_val_human - min_q_val_human)),
                                      best_action_agent, worst_action_agent, best_action_human, worst_action_human, s])

    q_diffs.sort(key=lambda x: x[0], reverse=True)

    if n_visualize > 0:
        # Visualize the top n states or the total length of q_diffs, whichever is shorter
        for state_number in range(min(len(q_diffs), n_visualize)):
            print("Number: {}".format(state_number))
            print("Best action agent: {}".format(q_diffs[state_number][1]))
            print("Worst action agent: {}".format(q_diffs[state_number][2]))
            print("Best action human: {}".format(q_diffs[state_number][3]))
            print("Worst action human: {}".format(q_diffs[state_number][4]))
            print("Q-val difference: {}".format(q_diffs[state_number][0]))
            mdp.visualize_state(q_diffs[state_number][5])