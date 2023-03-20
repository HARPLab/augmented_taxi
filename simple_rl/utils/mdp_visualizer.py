# Python imports.
from __future__ import print_function
import sys
import time
try:
    import pygame
    from pygame.locals import *
    pygame.init()
    title_font = pygame.font.SysFont("CMU Serif", 48)
except ImportError:
    print("Error: pygame not installed (needed for visuals).")
    exit()

def val_to_color(val, good_col=(169, 193, 249), bad_col=(249, 193, 169)):
    '''
    Args:
        val (float)
        good_col (tuple)
        bad_col (tuple)

    Returns:
        (tuple)

    Summary:
        Smoothly interpolates between @good_col and @bad_col. That is,
        if @val is 1, we get good_col, if it's 0.5, we get a color
        halfway between the two, and so on.
    '''
    # Make sure val is in the appropriate range.
    val = max(min(1.0, val), -1.0)

    if val > 0:
        # Show positive as interpolated between white (0) and good_cal (1.0)
        result = tuple([255 * (1 - val) + (col * val) for col in good_col])
    else:
        # Show negative as interpolated between white (0) and bad_col (-1.0)
        result = tuple([255 * (1 - abs(val)) + (col * abs(val)) for col in bad_col])

    return result

def _draw_title_text(mdp, screen):
    '''
    Args:
        mdp (simple_rl.MDP)
        screen (pygame.Surface)

    Summary:
        Draws the name of the MDP to the top of the screen.
    '''
    scr_width, scr_height = screen.get_width(), screen.get_height()
    title_text = title_font.render(str(mdp), True, (46, 49, 49))
    screen.blit(title_text, (scr_width / 2.0 - len(str(mdp))*6, scr_width / 20.0))

def _draw_agent_text(agent, screen):
    '''
    Args:
        agent (simple_rl.Agent)
        screen (pygame.Surface)

    Summary:
        Draws the name of the agent to the bottom right of the screen.
    '''
    scr_width, scr_height = screen.get_width(), screen.get_height()
    formatted_agent_text = "agent: " + str(agent)
    agent_text_point = (3*scr_width / 4.0 - len(formatted_agent_text)*6, 18*scr_height / 20.0)
    agent_text = title_font.render(formatted_agent_text, True, (46, 49, 49))
    screen.blit(agent_text, agent_text_point)

def _draw_lower_right_text(text, screen):
    '''
    Args:
        text (str)
        screen (pygame.Surface)

    Summary:
        Draws the desired text to the bottom right of the screen
    '''
    scr_width, scr_height = screen.get_width(), screen.get_height()
    text_point = (scr_width / 2.0 + len(text)*6, 18*scr_height / 20.0)
    pygame.draw.rect(screen, (255,255,255), (text_point[0] - 20, text_point[1]) + (200,40))
    state_text = title_font.render(text, True, (46, 49, 49))
    screen.blit(state_text, text_point)

def _draw_lower_left_text(state, screen, score=-1):
    '''
    Args:
        state (simple_rl.State)
        screen (pygame.Surface)
        score (int)

    Summary:
        Draws the name of the current state to the bottom left of the screen.
    '''
    scr_width, scr_height = screen.get_width(), screen.get_height()
    # Clear.
    formatted_state_text = str(state) if score == -1 else score
    if len(formatted_state_text) > 20:
        # See if state has an abbreviated version of state information
        try:
            formatted_state_text = state.abbr_str()
        # State text is too long, ignore.
        except:
            return
    state_text_point = (scr_width / 4.0 - len(formatted_state_text)*7, 18*scr_height / 20.0)
    pygame.draw.rect(screen, (255,255,255), (state_text_point[0] - 20, state_text_point[1]) + (200,40))
    state_text = title_font.render(formatted_state_text, True, (46, 49, 49))
    screen.blit(state_text, state_text_point)

def _draw_terminal_text(mdp_class, cur_state, scr_width, scr_height, title_font):
    if mdp_class == 'augmented_taxi':
        if cur_state.is_goal():
            goal_text = "Decided to deliver the circle!"
        else:
            goal_text = "Decided to exit!"
    elif mdp_class == 'two_goal':
        if cur_state.is_goal():
            goal_text = "Decided to go to one of the rings!"
        else:
            goal_text = "Decided to exit!"
    elif mdp_class == 'skateboard':
        if cur_state.is_goal():
            goal_text = "Decided to go to the square!"
        else:
            goal_text = "Decided to exit!"
    elif mdp_class == 'cookie_crumb':
        if cur_state.is_goal():
            goal_text = "Decided to go to the square!"
        else:
            goal_text = "Decided to exit!"
    elif mdp_class == 'taxi':
        if cur_state.is_goal():
            goal_text = "Decided to deliver the hexagon!"
        else:
            goal_text = "Decided to exit!"
    else:
        if cur_state.is_goal():
            # Done! Agent found goal.
            goal_text = "Game completed!"
        else:
            # Done! Agent failed.
            goal_text = "Fail!"

    goal_text_rendered = title_font.render(goal_text, True, (75, 75, 75))
    goal_text_point = scr_width / 2.0 - (len(goal_text) * 8), 18 * scr_height / 20.0

    return goal_text_rendered, goal_text_point

def _draw_polygon_alpha(surface, color, points):
    lx, ly = zip(*points)
    min_x, min_y, max_x, max_y = min(lx), min(ly), max(lx), max(ly)
    target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.polygon(shape_surf, color, [(x - min_x, y - min_y) for x, y in points])
    surface.blit(shape_surf, target_rect)
    return target_rect

def _draw_circle_alpha(surface, color, center, radius):
    target_rect = pygame.Rect(center, (0, 0)).inflate((radius * 2, radius * 2))
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.circle(shape_surf, color, (radius, radius), radius)
    surface.blit(shape_surf, target_rect)

def visualize_state(mdp, draw_state, cur_state=None, scr_width=720, scr_height=720):
    '''
    Args:
        mdp (MDP)
        draw_state (lambda)
        cur_state (State)
        scr_width (int)
        scr_height (int)

    Summary:

    '''
    screen = pygame.display.set_mode((scr_width, scr_height))

    # Setup and draw initial state.
    cur_state = mdp.get_init_state() if cur_state is None else cur_state

    dynamic_shapes, _ = _vis_init(screen, mdp, draw_state, cur_state, value=True)
    draw_state(screen, mdp, cur_state, show_value=False, draw_statics=True)
    _draw_lower_left_text(cur_state, screen)
    pygame.display.flip()
    while True:
        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.display.quit()
                return

        time.sleep(0.1)


def visualize_policy(mdp, policy, draw_state, action_char_dict, cur_state=None, scr_width=720, scr_height=720):
    '''
    Args:
        mdp (MDP)
        policy (lambda: S --> A)
        draw_state (lambda)
        action_char_dict (dict):
            Key: action
            Val: str
        cur_state (State)
        scr_width (int)
        scr_height (int)

    Summary:

    '''
    screen = pygame.display.set_mode((scr_width, scr_height))

    # Setup and draw initial state.
    cur_state = mdp.get_init_state() if cur_state is None else cur_state

    dynamic_shapes, _ = _vis_init(screen, mdp, draw_state, cur_state, value=True)
    draw_state(screen, mdp, cur_state, policy=policy, action_char_dict=action_char_dict, show_value=False, draw_statics=True)
    pygame.display.flip()
    while True:
        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.display.quit()
                return

        time.sleep(0.1)

def visualize_value(mdp, draw_state, agent=None, cur_state=None, scr_width=720, scr_height=720):
    '''
    Args:
        mdp (MDP)
        draw_state (State)
        scr_width (int)
        scr_height (int)

    Summary:
        Draws the MDP with values labeled on states.
    '''

    screen = pygame.display.set_mode((scr_width, scr_height))

    # Setup and draw initial state.
    cur_state = mdp.get_init_state() if cur_state is None else cur_state

    dynamic_shapes, _ = _vis_init(screen, mdp, draw_state, cur_state, value=True)
    draw_state(screen, mdp, cur_state, agent=agent, show_value=True, draw_statics=True)
    pygame.display.flip()

    while True:
        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.display.quit()
                return

        time.sleep(0.1)

    print("Press ESC to quit")
    while True:
        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.display.quit()
                return

def visualize_learning(mdp, agent, draw_state, cur_state=None, scr_width=720, scr_height=720, delay=0, num_ep=None, num_steps=None):
    '''
    Args:
        mdp (MDP)
        agent (Agent)
        draw_state (lambda: State --> pygame.Rect)
        cur_state (State)
        scr_width (int)
        scr_height (int)
        delay (float): seconds to add in between actions.

    Summary:
        Creates a *live* 2d visual of the agent's
        interactions with the MDP, showing the agent's
        estimated values of each state.
    '''
    screen = pygame.display.set_mode((scr_width, scr_height))

    # Setup and draw initial state.
    cur_state = mdp.get_init_state() if cur_state is None else cur_state
    reward = 0
    rpl = 0
    score = 0
    default_goal_x, default_goal_y = mdp.width, mdp.height
    dynamic_shapes, _ = _vis_init(screen, mdp, draw_state, cur_state, agent, score=score)

    pygame.display.update()
    done = False

    if not num_ep:
        # Main loop.
        while not done:
            # Check for key presses.
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    # Quit.
                    pygame.display.quit()
                    return
                elif event.type == KEYDOWN and event.key == K_r:
                    score = 0
                    agent.reset()
                    mdp.goal_locs = [(default_goal_x, default_goal_y)]
                    mdp.reset()

                elif event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    x, y = pos[0], pos[1]
                    width_buffer = scr_width / 10.0
                    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
                    cell_x, cell_y = convert_x_y_to_grid_cell(x, y, scr_width, scr_height, mdp.width, mdp.height)

                    if event.button == 1:
                        # Left clicked a cell, move the goal.
                        mdp.goal_locs = [(cell_x, cell_y)]
                        mdp.reset()
                    elif event.button == 3:
                        # Right clicked a cell, move the lava location.
                        if (cell_x, cell_y) in mdp.lava_locs:
                            mdp.lava_locs.remove((cell_x, cell_y))
                        else:
                            mdp.lava_locs += [(cell_x, cell_y)]


            # Move agent.
            action = agent.act(cur_state, reward)

            if cur_state.is_terminal():
                score += 1
                cur_state = mdp.get_init_state()
                mdp.reset()
                agent.end_of_episode()
                dynamic_shapes, _ = _vis_init(screen, mdp, draw_state, cur_state, agent, score=score)


            reward, cur_state = mdp.execute_agent_action(action)
            dynamic_shapes, _ = draw_state(screen, mdp, cur_state, agent=agent, show_value=True, draw_statics=True)

            score += int(reward)

            pygame.display.update()

            time.sleep(delay)

    else:
        # Main loop.
        i = 0
        while i < num_ep:
            j = 0
            while j < num_steps:
                # Check for key presses.
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        # Quit.
                        pygame.display.quit()
                        return
                    elif event.type == KEYDOWN and event.key == K_r:
                        score = 0
                        agent.reset()
                        mdp.goal_locs = [(default_goal_x, default_goal_y)]
                        mdp.reset()

                # Move agent.
                action = agent.act(cur_state, reward)
                reward, cur_state = mdp.execute_agent_action(action)
                dynamic_shapes, _ = draw_state(screen, mdp, cur_state, agent=agent, show_value=True, draw_statics=True)

                score = round(rpl)
                rpl += reward

                pygame.display.update()

                time.sleep(delay)
                j+=1

                if cur_state.is_terminal():
                    cur_state = mdp.get_init_state()
                    mdp.reset()
                    dynamic_shapes, _ = _vis_init(screen, mdp, draw_state, cur_state, agent, score=score)
                    break

            i+=1
            cur_state = mdp.get_init_state()
            mdp.reset()
            dynamic_shapes, _ = _vis_init(screen, mdp, draw_state, cur_state, agent, score=score)

    pygame.display.flip()

    print("Press ESC to quit")
    while True:
        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.display.quit()
                return

def visualize_trajectory(mdp, trajectory, draw_state, marked_state_importances=None, cur_state=None, scr_width=720, scr_height=720, mdp_class=None, counterfactual_traj=None, delay=0.1):
    '''
    Args:
        mdp (MDP)
        trajectory (list of states and actions)
        draw_state (lambda: State --> pygame.Rect)
        marked_state_importances (list of state importances)
        cur_state (State)
        scr_width (int)
        scr_height (int)

    Summary:
        Visualizes the sequence of states and actions stored in the trajectory.
    '''
    screen = pygame.display.set_mode((scr_width, scr_height))
    cur_state = trajectory[0][0]

    # Setup and draw initial state.
    dynamic_shapes, agent_history = _vis_init(screen, mdp, draw_state, cur_state, counterfactual_traj=counterfactual_traj)
    pygame.event.clear()
    step = 0

    if marked_state_importances is not None:
        # indicate if this is the critical state by displaying its state importance value
        if marked_state_importances[step] != float('-inf'):
            _draw_lower_right_text('SI: {}'.format(round(marked_state_importances[step], 3)), screen)

    while True and step != len(trajectory):

        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.display.quit()
                return
            if event.type == KEYDOWN and event.key == K_SPACE:
                # clear the old shapes
                for shape in dynamic_shapes:
                    pygame.draw.rect(screen, (255,255,255), shape)

                action = trajectory[step][1]
                cur_state = trajectory[step][2]

                if marked_state_importances is not None:
                    # indicate if this is the critical state by displaying its state importance value
                    if marked_state_importances[step] != float('-inf'):
                        _draw_lower_right_text('SI: {}'.format(round(marked_state_importances[step], 3)), screen)
                    else:
                        # clear the text
                        _draw_lower_right_text('       ', screen)

                dynamic_shapes, agent_history = draw_state(screen, mdp, cur_state, agent_history=agent_history, counterfactual_traj=counterfactual_traj)
                # print("A: " + str(action))

                # Update state text.
                _draw_lower_left_text(cur_state, screen)

                step += 1

        pygame.display.flip()

        time.sleep(delay)

    if cur_state.is_terminal():
        goal_text_rendered, goal_text_point = _draw_terminal_text(mdp_class, cur_state, scr_width, scr_height, title_font)
        screen.blit(goal_text_rendered, goal_text_point)

    pygame.display.flip()

    print("Press ESC to quit")
    while True:
        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.display.quit()
                return

        time.sleep(delay)


def visualize_trajectory_comparison(mdp, trajectory, trajectory_counterfactual, draw_state, marked_state_importances=None, cur_state=None, scr_width=720, scr_height=720, mdp_class=None, delay=0.1):
    '''
    Args:
        mdp (MDP)
        trajectory (list of states and actions)
        draw_state (lambda: State --> pygame.Rect)
        marked_state_importances (list of state importances)
        cur_state (State)
        scr_width (int)
        scr_height (int)

    Summary:
        Visualizes the sequence of states and actions stored in the trajectory.
    '''
    screen = pygame.display.set_mode((scr_width, scr_height))
    cur_state = trajectory[0][0]
    cur_state_counterfactual = trajectory_counterfactual[0][0]

    # Setup and draw initial state.
    dynamic_shapes, agent_history = _vis_init(screen, mdp, draw_state, cur_state, offset_direction=1)
    dynamic_shapes_counterfactual, agent_history_counterfactual = draw_state(screen, mdp, cur_state_counterfactual, draw_statics=False, agent_history=[], offset_direction=-1, alpha=150)

    pygame.event.clear()
    step = 0

    while True and (step < len(trajectory) or step < len(trajectory_counterfactual)):

        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.display.quit()
                return
            if event.type == KEYDOWN and event.key == K_SPACE:
                # clear the old dynamic shapes
                for shape in dynamic_shapes:
                    pygame.draw.rect(screen, (255, 255, 255), shape)
                for shape in dynamic_shapes_counterfactual:
                    pygame.draw.rect(screen, (255, 255, 255), shape)

                if step < len(trajectory):
                    cur_state = trajectory[step][2]
                else:
                    cur_state = trajectory[-1][2]

                dynamic_shapes, agent_history = draw_state(screen, mdp, cur_state,
                                                           agent_history=agent_history, offset_direction=1, visualize_history=False)

                if step < len(trajectory_counterfactual):
                    cur_state = trajectory_counterfactual[step][2]
                else:
                    cur_state = trajectory_counterfactual[-1][2]

                dynamic_shapes_counterfactual, agent_history_counterfactual = draw_state(screen, mdp, cur_state,agent_history=agent_history_counterfactual,
                                                                                    draw_statics=False, offset_direction=-1, alpha=150, visualize_history=False)

                step += 1

        pygame.display.flip()

        time.sleep(delay)

    if cur_state.is_terminal():
        goal_text_rendered, goal_text_point = _draw_terminal_text(mdp_class, cur_state, scr_width, scr_height, title_font)
        screen.blit(goal_text_rendered, goal_text_point)

    pygame.display.flip()

    print("Press ESC to quit")
    while True:
        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.display.quit()
                return

        time.sleep(delay)
def visualize_agent(mdp, agent, draw_state, cur_state=None, scr_width=720, scr_height=720, mdp_class=None):
    '''
    Args:
        mdp (MDP)
        agent (Agent)
        draw_state (lambda: State --> pygame.Rect)
        cur_state (State)
        scr_width (int)
        scr_height (int)

    Summary:
        Creates a 2d visual of the agent's interactions with the MDP.
    '''
    screen = pygame.display.set_mode((scr_width, scr_height))

    # Setup and draw initial state.
    if cur_state is not None:
        mdp.set_curr_state(cur_state)
    else:
        cur_state = mdp.get_init_state()
        mdp.reset()
    reward = 0
    cumulative_reward = 0
    step = 0
    gamma = mdp.gamma
    dynamic_shapes, _ = _vis_init(screen, mdp, draw_state, cur_state, agent)
    pygame.event.clear()

    done = False
    while not done:

        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.display.quit()
                return
            if event.type == KEYDOWN and event.key == K_SPACE:
                # clear the old shapes
                for shape in dynamic_shapes:
                    pygame.draw.rect(screen, (255,255,255), shape)

                # Move agent.
                action = agent.act(cur_state, reward)
                print("A: " + str(action))
                reward, cur_state = mdp.execute_agent_action(action)
                cumulative_reward += reward * gamma ** step
                dynamic_shapes, _ = draw_state(screen, mdp, cur_state)

                # Update state text.
                _draw_lower_left_text(cur_state, screen)

                step += 1

        if cur_state.is_terminal():
            goal_text_rendered, goal_text_point = _draw_terminal_text(mdp_class, cur_state, scr_width, scr_height, title_font)
            screen.blit(goal_text_rendered, goal_text_point)
            done = True
            print('Cumulative reward: {}'.format(cumulative_reward))

        pygame.display.flip()

    print("Press ESC to quit")
    while True:
        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.display.quit()
                return

def interaction_reset(mdp, cur_state, screen, draw_state):
    # Setup and draw initial state.
    cur_state = mdp.get_init_state() if cur_state is None else cur_state
    mdp.set_curr_state(cur_state)
    dynamic_shapes, agent_history = _vis_init(screen, mdp, draw_state, cur_state)
    pygame.event.clear()
    cumulative_reward = 0
    step = 0

    return mdp, cur_state, dynamic_shapes, agent_history, cumulative_reward, step

def visualize_interaction(mdp, draw_state, cur_state=None, interaction_callback=None, done_callback=None, keys_map=None, scr_width=720, scr_height=720, mdp_class=None):
    '''
    Args:
        mdp (MDP)
        interaction_callback (lambda: string)
        draw_state (lambda: State --> pygame.Rect)
        cur_state (State)
        scr_width (int)
        scr_height (int)

    Summary:
        Creates a 2d visual of the agent's interactions with the MDP.
    '''
    screen = pygame.display.set_mode((scr_width, scr_height))

    gamma = mdp.gamma
    actions = mdp.get_actions()
    mdp, cur_state, dynamic_shapes, agent_history, cumulative_reward, step = interaction_reset(mdp, cur_state, screen, draw_state)

    if keys_map is None:
        keys = [K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9, K_0]
        keys = keys[:len(actions) + 2]
    else:
        keys = []
        for key in keys_map:
            keys.append(eval(key))
        keys = keys[:len(actions) + 2]

    trajectory = []

    done = False
    while not done:

        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.display.quit()
                return trajectory, agent_history
            if event.type == KEYDOWN and event.key in keys:
                if event.key == eval('K_r'):
                    # 'r' == reset
                    mdp, cur_state, dynamic_shapes, agent_history, cumulative_reward, step = interaction_reset(mdp, None, screen, draw_state)
                    current_reward = 0
                    trajectory = []
                    continue
                elif event.key == eval('K_u'):
                    # 'u' == undo
                    if len(trajectory) > 0:
                        # clear the old shapes
                        for shape in dynamic_shapes:
                            pygame.draw.rect(screen, (255, 255, 255), shape)

                        prev_sequence = trajectory.pop()
                        agent_history = agent_history[:-2] # remove two items since draw_state will add the current state back in
                        cur_state = prev_sequence[0]
                        mdp.set_curr_state(cur_state)
                        cumulative_reward -= current_reward
                        dynamic_shapes, agent_history = draw_state(screen, mdp, cur_state, agent_history=agent_history)
                        continue
                    else:
                        continue

                # clear the old shapes
                for shape in dynamic_shapes:
                    pygame.draw.rect(screen, (255,255,255), shape)

                prev_state = cur_state
                action = actions[keys.index(event.key)]
                reward, cur_state = mdp.execute_agent_action(action=action)
                current_reward = reward * gamma ** step
                cumulative_reward += current_reward
                dynamic_shapes, agent_history = draw_state(screen, mdp, cur_state, agent_history=agent_history)
                trajectory.append((prev_state, action, cur_state))
                if interaction_callback is not None:
                    interaction_callback(action)

                # Update state text.
                _draw_lower_left_text(cur_state, screen)

                step += 1
        if cur_state.is_terminal():
            goal_text_rendered, goal_text_point = _draw_terminal_text(mdp_class, cur_state, scr_width, scr_height, title_font)
            screen.blit(goal_text_rendered, goal_text_point)
            done = True
            if done_callback is not None:
                done_callback()
            print(cumulative_reward)
        pygame.display.flip()

    print("Press ESC to quit")
    while True:
        # Check for key presses.
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                # Quit.
                pygame.display.quit()
                return trajectory, agent_history

def _vis_init(screen, mdp, draw_state, cur_state, agent=None, value=False, score=-1, counterfactual_traj=None, offset_direction=0):
    # Pygame setup.
    pygame.init()
    screen.fill((255, 255, 255))
    pygame.display.update()
    done = False

    if score != -1:
        _draw_lower_left_text("Score: " + str(score), screen)
    else:
        _draw_lower_left_text(cur_state, screen)

    dynamic_shapes, agent_history = draw_state(screen, mdp, cur_state, agent=agent, draw_statics=True, agent_history=[], counterfactual_traj=counterfactual_traj, offset_direction=offset_direction)

    return dynamic_shapes, agent_history

def convert_x_y_to_grid_cell(x, y, scr_width, scr_height, mdp_width, mdp_height):
    '''
    Args:
        x (int)
        y (int)
        scr_width (int)
        scr_height (int)
        num
    '''
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.

    lower_left_x, lower_left_y = x - width_buffer, scr_height - y - height_buffer

    cell_width = (scr_width - width_buffer * 2) / mdp_width
    cell_height = (scr_height - height_buffer * 2) / mdp_height

    cell_x, cell_y = int(lower_left_x / cell_width) + 1, int(lower_left_y / cell_height) + 1

    return cell_x, cell_y
