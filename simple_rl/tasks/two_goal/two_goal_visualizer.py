# Python imports.
from __future__ import print_function
from collections import defaultdict
try:
    import pygame
    import pygame.gfxdraw
    title_font = pygame.font.SysFont("CMU Serif", 48)
except ImportError:
    print("Warning: pygame not installed (needed for visuals).")

# Other imports.
import math
from simple_rl.utils.chart_utils import color_ls
from simple_rl.planning import ValueIteration
from simple_rl.utils import mdp_visualizer as mdpv
from simple_rl.tasks.two_goal import two_goal_helpers

def _draw_state(screen,
                two_goal_oomdp,
                state,
                policy=None,
                action_char_dict={},
                show_value=False,
                agent=None,
                draw_statics=True,
                agent_shape=None,
                agent_history=[]):
    '''
    Args:
        screen (pygame.Surface)
        two_goal_oomdp (TwoGoalOOMDP)
        state (State)
        agent_shape (pygame.rect)

    Returns:
        (pygame.Shape)
    '''
    # Make value dict.
    val_text_dict = defaultdict(lambda: defaultdict(float))
    if show_value:
        if agent is not None:
            if agent.name == 'Q-learning':
                # Use agent value estimates.
                for s in agent.q_func.keys():
                    val_text_dict[s.get_agent_x()][s.get_agent_y()] = agent.get_value(s)
            # slightly abusing the distinction between agents and planning modules...
            else:
                for s in two_goal_oomdp.get_states():
                    val_text_dict[s.get_agent_x()][s.get_agent_y()] = agent.get_value(s)
        else:
            # Use Value Iteration to compute value.
            vi = ValueIteration(two_goal_oomdp, sample_rate=10)
            vi.run_vi()
            for s in vi.get_states():
                val_text_dict[s.get_agent_x()][s.get_agent_y()] = vi.get_value(s)

    # Make policy dict.
    policy_dict = defaultdict(lambda : defaultdict(str))
    if policy:
        for s in two_goal_oomdp.get_states():
            policy_dict[s.get_agent_x()][s.get_agent_y()] = policy(s)

    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer * 2) / two_goal_oomdp.width
    cell_height = (scr_height - height_buffer * 2) / two_goal_oomdp.height
    objects = state.get_objects()
    agent_x, agent_y = objects["agent"][0]["x"], objects["agent"][0]["y"]
    font_size = int(min(cell_width, cell_height) / 4.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size)
    cc_font = pygame.font.SysFont("Courier", font_size * 2 + 2)

    if agent_shape is not None:
        # Clear the old shape.
        pygame.draw.rect(screen, (255,255,255), agent_shape)

    # Statics
    if draw_statics:
        # Draw walls.
        for w in two_goal_oomdp.walls:
            w_x, w_y = w["x"], w["y"]
            top_left_point = width_buffer + cell_width * (w_x - 1) + 5, height_buffer + cell_height * (
                    two_goal_oomdp.height - w_y) + 5
            pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width - 10, cell_height - 10), 0)

    # Draw the two goals
    col_idxs = [-2, 0]
    for i, g in enumerate(two_goal_oomdp.goals):
        dest_x, dest_y = g["x"], g["y"]
        top_left_point = int(width_buffer + cell_width*(dest_x - 1) + 75), int(height_buffer + cell_height*(two_goal_oomdp.height - dest_y) + 65)
        dest_col = (int(max(color_ls[col_idxs[i]][0]-30, 0)), int(max(color_ls[col_idxs[i]][1]-30, 0)), int(max(color_ls[col_idxs[i]][2]-30, 0)))
        center = top_left_point + (cell_width / 2, cell_height / 2)
        radius = 45
        iterations = 150
        for i in range(iterations):
            ang = i * 3.14159 * 2 / iterations
            dx = int(math.cos(ang) * radius)
            dy = int(math.sin(ang) * radius)
            x = center[0] + dx
            y = center[1] + dy
            pygame.draw.circle(screen, dest_col, (x, y), 5)

    # Draw history of past agent locations if applicable
    if len(agent_history) > 0:
        for i, position in enumerate(agent_history):
            if i == 0:
                top_left_point = int(width_buffer + cell_width * (position[0] - 0.5)), int(
                    height_buffer + cell_height * (two_goal_oomdp.height - position[1] + 0.5))
                pygame.draw.circle(screen, (103, 115, 135), top_left_point, int(min(cell_width, cell_height) / 15))
                top_left_point_rect = int(width_buffer + cell_width * (position[0] - 0.5) - cell_width/8), int(
                    height_buffer + cell_height * (two_goal_oomdp.height - position[1] + 0.5) - 2)
                pygame.draw.rect(screen, (103, 115, 135), top_left_point_rect + (cell_width / 4, cell_height / 20), 0)
            else:
                top_left_point = int(width_buffer + cell_width * (position[0] - 0.5)), int(
                    height_buffer + cell_height * (two_goal_oomdp.height - position[1] + 0.5))
                pygame.draw.circle(screen, (103, 115, 135), top_left_point, int(min(cell_width, cell_height) / 15))
    agent_history.append((agent_x, agent_y))


    # Draw new agent.
    top_left_point = width_buffer + cell_width * (agent_x - 1), height_buffer + cell_height * (
                two_goal_oomdp.height - agent_y)
    agent_center = int(top_left_point[0] + cell_width / 2.0), int(top_left_point[1] + cell_height / 2.0)
    agent_shape = _draw_agent(agent_center, screen, base_size=min(cell_width, cell_height) / 2.5 - 4)

    if draw_statics:
        # For each row:
        for i in range(two_goal_oomdp.width):
            # For each column:
            for j in range(two_goal_oomdp.height):
                top_left_point = width_buffer + cell_width*i, height_buffer + cell_height*j
                r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

                # Show value of states.
                if show_value and not two_goal_helpers.is_wall(two_goal_oomdp, i + 1, two_goal_oomdp.height - j):
                    # Draw the value.
                    val = val_text_dict[i + 1][two_goal_oomdp.height - j]
                    color = mdpv.val_to_color(val)
                    pygame.draw.rect(screen, color, top_left_point + (cell_width, cell_height), 0)
                    value_text = reg_font.render(str(round(val, 2)), True, (46, 49, 49))
                    text_center_point = int(top_left_point[0] + cell_width / 2.0 - 10), int(
                        top_left_point[1] + cell_height / 3.0)
                    screen.blit(value_text, text_center_point)

                # Show optimal action to take in each grid cell.
                if policy and not two_goal_helpers.is_wall(two_goal_oomdp, i + 1, two_goal_oomdp.height - j):
                    a = policy_dict[i+1][two_goal_oomdp.height - j]
                    if a not in action_char_dict:
                        text_a = a
                    else:
                        text_a = action_char_dict[a]
                    text_center_point = int(top_left_point[0] + cell_width/2.0 - 10), int(top_left_point[1] + cell_height/3.0)
                    text_rendered_a = cc_font.render(text_a, True, (46, 49, 49))
                    screen.blit(text_rendered_a, text_center_point)

    pygame.display.flip()

    return agent_shape, agent_history

def _draw_agent(center_point, screen, base_size=30):
    '''
    Args:
        center_point (tuple): (x,y)
        screen (pygame.Surface)

    Returns:
        (pygame.rect)
    '''
    tri_bot_left = center_point[0] - base_size, center_point[1] + base_size
    tri_bot_right = center_point[0] + base_size, center_point[1] + base_size
    tri_top = center_point[0], center_point[1] - base_size
    tri = [tri_bot_left, tri_top, tri_bot_right]
    tri_color = (98, 140, 190)

    return pygame.draw.polygon(screen, tri_color, tri)
