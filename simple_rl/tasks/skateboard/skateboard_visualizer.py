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
from simple_rl.utils.chart_utils import color_ls
from simple_rl.planning import ValueIteration
from simple_rl.utils import mdp_visualizer as mdpv
from simple_rl.tasks.skateboard import skateboard_helpers
import math

def _draw_state(screen,
                skateboard_oomdp,
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
        skateboard_oomdp (SkateboardOOMDP)
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
                for s in skateboard_oomdp.get_states():
                    val_text_dict[s.get_agent_x()][s.get_agent_y()] = agent.get_value(s)
        else:
            # Use Value Iteration to compute value.
            vi = ValueIteration(skateboard_oomdp, sample_rate=10)
            vi.run_vi()
            for s in vi.get_states():
                val_text_dict[s.get_agent_x()][s.get_agent_y()] = vi.get_value(s)

    # Make policy dict.
    policy_dict = defaultdict(lambda : defaultdict(str))
    if policy:
        for s in skateboard_oomdp.get_states():
            policy_dict[s.get_agent_x()][s.get_agent_y()] = policy(s)

    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer * 2) / skateboard_oomdp.width
    cell_height = (scr_height - height_buffer * 2) / skateboard_oomdp.height
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
        for w in skateboard_oomdp.walls:
            w_x, w_y = w["x"], w["y"]
            top_left_point = width_buffer + cell_width * (w_x - 1) + 5, height_buffer + cell_height * (
                    skateboard_oomdp.height - w_y) + 5
            pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width - 10, cell_height - 10), 0)

        # Draw paths.
        for p in skateboard_oomdp.paths:
            p_x, p_y = p["x"], p["y"]
            top_left_point = width_buffer + cell_width * (p_x - 1) + 5, height_buffer + cell_height * (
                    skateboard_oomdp.height - p_y) + 5
            # Clear the space and redraw with correct transparency (instead of simply adding a new layer which would
            # affect the transparency
            pygame.draw.rect(screen, (255, 255, 255), top_left_point + (cell_width - 10, cell_height - 10), 0)
            pygame.gfxdraw.box(screen, top_left_point + (cell_width - 10, cell_height - 10), (220, 187, 252))

    # Draw the destination.
    dest_x, dest_y = skateboard_oomdp.goal["x"], skateboard_oomdp.goal["y"]
    top_left_point = int(width_buffer + cell_width * (dest_x - 1) + 37), int(
        height_buffer + cell_height * (skateboard_oomdp.height - dest_y) + 34)
    dest_col = (int(max(color_ls[-2][0]-30, 0)), int(max(color_ls[-2][1]-30, 0)), int(max(color_ls[-2][2]-30, 0)))
    pygame.draw.rect(screen, dest_col, top_left_point + (cell_width / 2, cell_height / 2))

    # Draw history of past agent locations if applicable
    if len(agent_history) > 0:
        for i, position in enumerate(agent_history):
            if i == 0:
                top_left_point = int(width_buffer + cell_width * (position[0] - 0.5)), int(
                    height_buffer + cell_height * (skateboard_oomdp.height - position[1] + 0.5))
                pygame.draw.circle(screen, (103, 115, 135), top_left_point, int(min(cell_width, cell_height) / 15))
                top_left_point_rect = int(width_buffer + cell_width * (position[0] - 0.5) - cell_width/8), int(
                    height_buffer + cell_height * (skateboard_oomdp.height - position[1] + 0.5) - 2)
                pygame.draw.rect(screen, (103, 115, 135), top_left_point_rect + (cell_width / 4, cell_height / 20), 0)
            else:
                top_left_point = int(width_buffer + cell_width * (position[0] - 0.5)), int(
                    height_buffer + cell_height * (skateboard_oomdp.height - position[1] + 0.5))
                pygame.draw.circle(screen, (103, 115, 135), top_left_point, int(min(cell_width, cell_height) / 15))
    agent_history.append((agent_x, agent_y))

    # Draw new agent.
    top_left_point = width_buffer + cell_width * (agent_x - 1), height_buffer + cell_height * (
                skateboard_oomdp.height - agent_y)
    agent_center = int(top_left_point[0] + cell_width / 2.0), int(top_left_point[1] + cell_height / 2.0)
    agent_shape = _draw_agent(agent_center, screen, base_size=min(cell_width, cell_height) / 2.5 - 4)

    # Draw the skateboards.
    for i, p in enumerate(objects["skateboard"]):
        # skateboard
        pass_x, pass_y = p["x"], p["y"]
        agent_size = int(min(cell_width, cell_height) / 5.0)
        if p["on_agent"]:
            top_left_point = int(width_buffer + cell_width * (pass_x - 1) + agent_size + 10), int(
                height_buffer + cell_height * (skateboard_oomdp.height - pass_y) + agent_size + 75)
        else:
            top_left_point = int(width_buffer + cell_width * (pass_x - 1) + agent_size + 10), int(
                height_buffer + cell_height * (skateboard_oomdp.height - pass_y) + agent_size + 38)
        dest_col = (max(color_ls[-i-3][0]-30, 0), max(color_ls[-i-3][1]-30, 0), max(color_ls[-i-3][2]-30, 0))
        pygame.draw.rect(screen, dest_col, top_left_point + (cell_width / 2, cell_height / 10), 0)

    if draw_statics:
        # For each row:
        for i in range(skateboard_oomdp.width):
            # For each column:
            for j in range(skateboard_oomdp.height):
                top_left_point = width_buffer + cell_width*i, height_buffer + cell_height*j
                r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

                # Show value of states.
                if show_value and not skateboard_helpers.is_wall(skateboard_oomdp, i + 1, skateboard_oomdp.height - j):
                    # Draw the value.
                    val = val_text_dict[i + 1][skateboard_oomdp.height - j]
                    color = mdpv.val_to_color(val)
                    pygame.draw.rect(screen, color, top_left_point + (cell_width, cell_height), 0)
                    value_text = reg_font.render(str(round(val, 2)), True, (46, 49, 49))
                    text_center_point = int(top_left_point[0] + cell_width / 2.0 - 10), int(
                        top_left_point[1] + cell_height / 3.0)
                    screen.blit(value_text, text_center_point)

                # Show optimal action to take in each grid cell.
                if policy and not skateboard_helpers.is_wall(skateboard_oomdp, i + 1, skateboard_oomdp.height - j):
                    a = policy_dict[i+1][skateboard_oomdp.height - j]
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
