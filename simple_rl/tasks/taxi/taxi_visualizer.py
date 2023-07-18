# Python imports.
from __future__ import print_function
from collections import defaultdict
try:
    import pygame
    import pygame.gfxdraw
    title_font = pygame.font.SysFont("CMU Serif", 48)
    import math
except ImportError:
    print("Warning: pygame not installed (needed for visuals).")

# Other imports.
from simple_rl.utils.chart_utils import color_ls
from simple_rl.planning import ValueIteration
from simple_rl.utils import mdp_visualizer as mdpv
from simple_rl.tasks.taxi import taxi_helpers

def _draw_augmented_state(screen,
                taxi_oomdp,
                state,
                policy=None,
                action_char_dict={},
                show_value=False,
                agent=None,
                draw_statics=True,
                agent_history=[],
                counterfactual_traj=None,
                alpha=255, offset_direction=0, visualize_history=True, augmented_inputs=False):
    '''
    Args:
        screen (pygame.Surface)
        taxi_oomdp (TaxiOOMDP)
        state (State)
        agent_shape (pygame.rect)

    Returns:
        (pygame.Shape)
    '''
    # There are multiple potential states for each grid cell (e.g. state also depends on whether the taxi currently has
    # the passenger or not), but the value and the policy for each cell is simply given by the most recent state
    # returned by get_states(). Began trying to display at least two values and optimal actions for each cell (depending
    # on the onboarding status of the passenger), but quickly realized that it gets very complicated as the MDP gets
    # more complicated (e.g. state also depends on the location of the passenger).
    # Displaying multiple values and optimal actions and will require either handcrafting the pipeline or
    # investing a lot of time into making the pipeline customizable and robust. Leaving incomplete attempt below as
    # commented out code.

    # Make value dict.
    val_text_dict = defaultdict(lambda: defaultdict(float))
    # val_text_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    if show_value:
        if agent is not None:
            if agent.name == 'Q-learning':
                # Use agent value estimates.
                for s in agent.q_func.keys():
                    val_text_dict[s.get_agent_x()][s.get_agent_y()] = agent.get_value(s)
                    # val_text_dict[s.get_agent_x()][s.get_agent_y()][
                    #     s.get_first_obj_of_class("passenger")["in_taxi"]] += agent.get_value(s)
            # slightly abusing the distinction between agents and planning modules...
            else:
                for s in taxi_oomdp.get_states():
                    val_text_dict[s.get_agent_x()][s.get_agent_y()] = agent.get_value(s)
                    # val_text_dict[s.get_agent_x()][s.get_agent_y()][
                    #     s.get_first_obj_of_class("passenger")["in_taxi"]] += agent.get_value(s)
        else:
            # Use Value Iteration to compute value.
            vi = ValueIteration(taxi_oomdp, sample_rate=10)
            vi.run_vi()
            for s in vi.get_states():
                val_text_dict[s.get_agent_x()][s.get_agent_y()] = vi.get_value(s)
                # val_text_dict[s.get_agent_x()][s.get_agent_y()][
                #     s.get_first_obj_of_class("passenger")["in_taxi"]] += vi.get_value(s)

    # Make policy dict.
    policy_dict = defaultdict(lambda : defaultdict(str))
    # policy_dict = defaultdict(lambda: defaultdict(lambda : defaultdict(str)))
    if policy:
        for s in taxi_oomdp.get_states():
            policy_dict[s.get_agent_x()][s.get_agent_y()] = policy(s)
            # if policy_dict[s.get_agent_x()][s.get_agent_y()][s.get_first_obj_of_class("passenger")["in_taxi"]] != '':
            #     policy_dict[s.get_agent_x()][s.get_agent_y()][s.get_first_obj_of_class("passenger")["in_taxi"]] = policy(s)

    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    if augmented_inputs:
        width_buffer_left = scr_width / 20.0
        witdth_buffer_right = scr_width / 5.0
    else:
        width_buffer_left = scr_width / 10.0
        witdth_buffer_right = scr_width / 10.0
    color_light = (170,170,170)
    color_dark = (100, 100, 100)
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer_left - witdth_buffer_right) / taxi_oomdp.width
    cell_height = (scr_height - height_buffer * 2) / taxi_oomdp.height
    objects = state.get_objects()
    agent_x, agent_y = objects["agent"][0]["x"], objects["agent"][0]["y"]
    font_size = int(min(cell_width, cell_height) / 4.0)
    reg_font = pygame.font.SysFont("CMU Serif", font_size)
    cc_font = pygame.font.SysFont("Courier", font_size * 2 + 2)
    aug_text = reg_font.render('Passenger 1', True, (255,255,255))

    # for visualizing two agents/paths at once
    offset_magnitude = cell_width / 8.0
    if offset_direction != 0:
        offset_counterfactual = offset_magnitude * offset_direction
    else:
        offset_counterfactual = 0

    # for clearing dynamic shapes (e.g. agent)
    dynamic_shapes_list = []

    # Statics
    if draw_statics:
        #draw buttons: (and later another agent)
        if augmented_inputs:
            top_left_point = scr_width - ((3.0*witdth_buffer_right)/4.0), scr_height/2 -20
            pygame.draw.rect(screen, color_light, top_left_point, top_left_point + (witdth_buffer_right/2.0,40))
            screen.blit(aug_text, (top_left_point + (10,25)))
        # Draw walls.
        for w in taxi_oomdp.walls:
            w_x, w_y = w["x"], w["y"]
            top_left_point = width_buffer_left + cell_width * (w_x - 1) + 5, height_buffer + cell_height * (
                    taxi_oomdp.height - w_y) + 5
            pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width - 10, cell_height - 10), 0)

        # Draw tolls.
        for t in taxi_oomdp.tolls:
            t_x, t_y = t["x"], t["y"]
            top_left_point = width_buffer_left + cell_width * (t_x - 1) + 5, height_buffer + cell_height * (
                    taxi_oomdp.height - t_y) + 5
            # Clear the space and redraw with correct transparency (instead of simply adding a new layer which would
            # affect the transparency
            pygame.draw.rect(screen, (255, 255, 255), top_left_point + (cell_width - 10, cell_height - 10), 0)
            pygame.gfxdraw.box(screen, top_left_point + (cell_width - 10, cell_height - 10), (224, 230, 67))

        # Draw traffic cells.
        for t in taxi_oomdp.traffic_cells:
            t_x, t_y = t["x"], t["y"]
            top_left_point = width_buffer_left + cell_width * (t_x - 1) + 5, height_buffer + cell_height * (
                    taxi_oomdp.height - t_y) + 5
            # Clear the space and redraw with correct transparency (instead of simply adding a new layer which would
            # affect the transparency
            pygame.draw.rect(screen, (255, 255, 255), top_left_point + (cell_width - 10, cell_height - 10), 0)
            pygame.gfxdraw.box(screen, top_left_point + (cell_width - 10, cell_height - 10), (58, 28, 232))

        # Draw fuel stations.
        for f in taxi_oomdp.fuel_stations:
            f_x, f_y = f["x"], f["y"]
            top_left_point = width_buffer_left + cell_width * (f_x - 1) + 5, height_buffer + cell_height * (
                    taxi_oomdp.height - f_y) + 5
            pygame.draw.rect(screen, (144, 0, 255), top_left_point + (cell_width - 10, cell_height - 10), 0)

        # Draw the destination.
        for i, p in enumerate(objects["passenger"]):
            # Dest.
            dest_x, dest_y = p["dest_x"], p["dest_y"]
            top_left_point = int(width_buffer_left + cell_width*(dest_x - 1) + 27), int(height_buffer + cell_height*(taxi_oomdp.height - dest_y) + 14)
            dest_col = (int(max(color_ls[-i-1][0]-30, 0)), int(max(color_ls[-i-1][1]-30, 0)), int(max(color_ls[-i-1][2]-30, 0)))
            pygame.draw.rect(screen, dest_col, top_left_point + (cell_width / 6, cell_height / 6), 0)

    # Draw hotswap stations.
    if "hotswap_station" in objects.keys():
        for f in objects["hotswap_station"]:
            x, y = f["x"], f["y"]
            top_left_point = int(width_buffer_left + cell_width * (x - 1) + 70 + offset_counterfactual), int(
                height_buffer + cell_height * (taxi_oomdp.height - y) + 65)
            dest_col = (
            int(max(color_ls[0][0] - 30, 0)), int(max(color_ls[0][1] - 30, 0)), int(max(color_ls[0][2] - 30, 0)), alpha)

            n, r = 6, cell_width / 8
            x, y = top_left_point[0], top_left_point[1]
            color = dest_col
            hotswap_station_shape = mdpv._draw_polygon_alpha(screen, color, [
                (x + r * math.cos(2 * math.pi * i / n), y + r * math.sin(2 * math.pi * i / n))
                for i in range(n)
            ])
            dynamic_shapes_list.append(hotswap_station_shape)

    # Draw history of past agent locations if applicable
    if len(agent_history) > 0 and visualize_history:
        for i, position in enumerate(agent_history):
            if i == 0:
                top_left_point = int(width_buffer_left + cell_width * (position[0] - 0.5)), int(
                    height_buffer + cell_height * (taxi_oomdp.height - position[1] + 0.5))
                pygame.draw.circle(screen, (103, 115, 135), top_left_point, int(min(cell_width, cell_height) / 15))
                top_left_point_rect = int(width_buffer_left + cell_width * (position[0] - 0.5) - cell_width/8), int(
                    height_buffer + cell_height * (taxi_oomdp.height - position[1] + 0.5) - 2)
                pygame.draw.rect(screen, (103, 115, 135), top_left_point_rect + (cell_width / 4, cell_height / 20), 0)
            else:
                top_left_point = int(width_buffer_left + cell_width * (position[0] - 0.5)), int(
                    height_buffer + cell_height * (taxi_oomdp.height - position[1] + 0.5))
                pygame.draw.circle(screen, (103, 115, 135), top_left_point, int(min(cell_width, cell_height) / 15))

    # Draw history of past counterfactual agent locations if applicable
    if counterfactual_traj is not None:
        for i, position in enumerate(counterfactual_traj):
            if i == 0:
                top_left_point = int(width_buffer_left + cell_width * (position[0] - 0.5)), int(
                    height_buffer + cell_height * (taxi_oomdp.height - position[1] + 0.5))
                pygame.draw.circle(screen, (255, 0, 0), top_left_point, int(min(cell_width, cell_height) / 15))
                top_left_point_rect = int(width_buffer_left + cell_width * (position[0] - 0.5) - cell_width/8), int(
                    height_buffer + cell_height * (taxi_oomdp.height - position[1] + 0.5) - 2)
                pygame.draw.rect(screen, (255, 0, 0), top_left_point_rect + (cell_width / 4, cell_height / 20), 0)
            else:
                top_left_point = int(width_buffer_left + cell_width * (position[0] - 0.5)), int(
                    height_buffer + cell_height * (taxi_oomdp.height - position[1] + 0.5))
                pygame.draw.circle(screen, (255, 0, 0), top_left_point, int(min(cell_width, cell_height) / 15))


    # Draw new agent.
    top_left_point = width_buffer_left + cell_width * (agent_x - 1), height_buffer + cell_height * (
                taxi_oomdp.height - agent_y)
    agent_center = int(top_left_point[0] + cell_width / 2.0 + offset_counterfactual), int(top_left_point[1] + cell_height / 2.0)
    agent_shape = _draw_agent(agent_center, screen, base_size=min(cell_width, cell_height) / 2.5 - 4, alpha=alpha)
    agent_history.append((agent_x, agent_y))

    # Draw the passengers.
    for i, p in enumerate(objects["passenger"]):
        # Passenger
        pass_x, pass_y = p["x"], p["y"]
        taxi_size = int(min(cell_width, cell_height) / 9.0)
        if p["in_taxi"]:
            top_left_point = int(width_buffer_left + cell_width * (pass_x - 1) + taxi_size + 58 + offset_counterfactual), int(
                height_buffer + cell_height * (taxi_oomdp.height - pass_y) + taxi_size + 16)
        else:
            top_left_point = int(width_buffer_left + cell_width * (pass_x - 1) + taxi_size + 26 + offset_counterfactual), int(
                height_buffer + cell_height * (taxi_oomdp.height - pass_y) + taxi_size + 38)
        dest_col = (max(color_ls[-i-1][0]-30, 0), max(color_ls[-i-1][1]-30, 0), max(color_ls[-i-1][2]-30, 0), alpha)
        passenger_shape = mdpv._draw_circle_alpha(screen, dest_col, top_left_point, taxi_size)
        dynamic_shapes_list.append(passenger_shape)

    if draw_statics:
        # For each row:
        for i in range(taxi_oomdp.width):
            # For each column:
            for j in range(taxi_oomdp.height):
                top_left_point = width_buffer_left + cell_width*i, height_buffer + cell_height*j
                r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

                # Show value of states.
                if show_value and not taxi_helpers.is_wall(taxi_oomdp, i + 1, taxi_oomdp.height - j):
                    # Draw the value.
                    val = val_text_dict[i + 1][taxi_oomdp.height - j]
                    color = mdpv.val_to_color(val)
                    pygame.draw.rect(screen, color, top_left_point + (cell_width, cell_height), 0)
                    value_text = reg_font.render(str(round(val, 2)), True, (46, 49, 49))
                    text_center_point = int(top_left_point[0] + cell_width / 2.0 - 10), int(
                        top_left_point[1] + cell_height / 3.0)
                    screen.blit(value_text, text_center_point)

                    # Draw the value depending on the status of the passenger (incomplete)
                    # val_1 = val_text_dict[s.get_agent_x()][s.get_agent_y()][0]  # passenger not in taxi
                    # val_2 = val_text_dict[s.get_agent_x()][s.get_agent_y()][1]  # passenger not in taxi
                    # color = mdpv.val_to_color((val_1 + val_2) / 2.)
                    # pygame.draw.rect(screen, color, top_left_point + (cell_width, cell_height), 0)
                    #
                    # value_text = reg_font.render(str(round(val_1, 2)), True, (46, 49, 49))
                    # text_center_point = int(top_left_point[0] + cell_width / 2.0 - 10), int(
                    #     top_left_point[1] + cell_height / 1.5)
                    # screen.blit(value_text, text_center_point)
                    #
                    # value_text = reg_font.render(str(round(val_2, 2)), True, (46, 49, 49))
                    # text_center_point = int(top_left_point[0] + cell_width / 2.0 - 10), int(
                    #     top_left_point[1] + cell_height / 4.5)
                    # screen.blit(value_text, text_center_point)

                # Show optimal action to take in each grid cell.
                if policy and not taxi_helpers.is_wall(taxi_oomdp, i + 1, taxi_oomdp.height - j):
                    a = policy_dict[i+1][taxi_oomdp.height - j]
                    if a not in action_char_dict:
                        text_a = a
                    else:
                        text_a = action_char_dict[a]
                    text_center_point = int(top_left_point[0] + cell_width/2.0 - 10), int(top_left_point[1] + cell_height/3.0)
                    text_rendered_a = cc_font.render(text_a, True, (46, 49, 49))
                    screen.blit(text_rendered_a, text_center_point)

                    # Draw the policy depending on the status of the passenger (incomplete)
                    # a_1 = policy_dict[i + 1][taxi_oomdp.height - j][0]
                    # a_2 = policy_dict[i + 1][taxi_oomdp.height - j][0]
                    # if a_1 not in action_char_dict: text_a_1 = a_1
                    # else: text_a_1 = action_char_dict[a_1]
                    # if a_2 not in action_char_dict: text_a_2 = a_2
                    # else: text_a_2 = action_char_dict[a_2]
                    # text_center_point = int(top_left_point[0] + cell_width/2.0 - 10), int(top_left_point[1] + cell_height/1.5)
                    # text_rendered_a = cc_font.render(text_a_1, True, (46, 49, 49))
                    # screen.blit(text_rendered_a, text_center_point)
                    # text_center_point = int(top_left_point[0] + cell_width / 2.0 - 10), int(
                    #     top_left_point[1] + cell_height / 4.5)
                    # text_rendered_a = cc_font.render(text_a_2, True, (46, 49, 49))
                    # screen.blit(text_rendered_a, text_center_point)

    pygame.display.flip()

    dynamic_shapes_list.append(agent_shape)

    return dynamic_shapes_list, agent_history

def _draw_state(screen,
                taxi_oomdp,
                state,
                policy=None,
                action_char_dict={},
                show_value=False,
                agent=None,
                draw_statics=True,
                agent_shape=None):
    '''
    Args:
        screen (pygame.Surface)
        taxi_oomdp (TaxiOOMDP)
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
                for s in taxi_oomdp.get_states():
                    val_text_dict[s.get_agent_x()][s.get_agent_y()] = agent.get_value(s)
        else:
            # Use Value Iteration to compute value.
            vi = ValueIteration(taxi_oomdp, sample_rate=10)
            vi.run_vi()
            for s in vi.get_states():
                val_text_dict[s.get_agent_x()][s.get_agent_y()] = vi.get_value(s)

    # Make policy dict.
    policy_dict = defaultdict(lambda : defaultdict(str))
    if policy:
        for s in taxi_oomdp.get_states():
            policy_dict[s.get_agent_x()][s.get_agent_y()] = policy(s)

    # Prep some dimensions to make drawing easier.
    scr_width, scr_height = screen.get_width(), screen.get_height()
    width_buffer = scr_width / 10.0
    height_buffer = 30 + (scr_height / 10.0) # Add 30 for title.
    cell_width = (scr_width - width_buffer * 2) / taxi_oomdp.width
    cell_height = (scr_height - height_buffer * 2) / taxi_oomdp.height
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
        for w in taxi_oomdp.walls:
            w_x, w_y = w["x"], w["y"]
            top_left_point = width_buffer + cell_width * (w_x - 1) + 5, height_buffer + cell_height * (
                    taxi_oomdp.height - w_y) + 5
            pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width - 10, cell_height - 10), 0)

    # Draw the destination.
    for i, p in enumerate(objects["passenger"]):
        # Dest.
        dest_x, dest_y = p["dest_x"], p["dest_y"]
        top_left_point = int(width_buffer + cell_width*(dest_x - 1) + 38), int(height_buffer + cell_height*(taxi_oomdp.height - dest_y) + 18)

        passenger_size = cell_width / 11
        # purple
        dest_col = (188, 30, 230)

        n, r = 6, passenger_size
        x, y = top_left_point[0], top_left_point[1]
        color = dest_col
        pygame.draw.polygon(screen, color, [
            (x + r * math.cos(2 * math.pi * i / n), y + r * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ])

    # Draw new agent.
    top_left_point = width_buffer + cell_width * (agent_x - 1), height_buffer + cell_height * (
                taxi_oomdp.height - agent_y)
    agent_center = int(top_left_point[0] + cell_width / 2.0), int(top_left_point[1] + cell_height / 2.0)
    agent_shape = _draw_agent(agent_center, screen, base_size=min(cell_width, cell_height) / 2.5 - 4)

    for i, p in enumerate(objects["passenger"]):
        # Dest.
        x, y = p["x"], p["y"]
        passenger_size = cell_width / 11
        if p["in_taxi"]:
            top_left_point = int(width_buffer + cell_width * (x - 1) + passenger_size + 58), int(
                height_buffer + cell_height * (taxi_oomdp.height - y) + passenger_size + 15)
        else:
            top_left_point = int(width_buffer + cell_width * (x - 1) + passenger_size + 25), int(
                height_buffer + cell_height * (taxi_oomdp.height - y) + passenger_size + 34)

        # light green
        dest_col = (59, 189, 23)

        n, r = 6, passenger_size
        x, y = top_left_point[0], top_left_point[1]
        color = dest_col
        pygame.draw.polygon(screen, color, [
            (x + r * math.cos(2 * math.pi * i / n), y + r * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ])

    if draw_statics:
        # For each row:
        for i in range(taxi_oomdp.width):
            # For each column:
            for j in range(taxi_oomdp.height):
                top_left_point = width_buffer + cell_width*i, height_buffer + cell_height*j
                r = pygame.draw.rect(screen, (46, 49, 49), top_left_point + (cell_width, cell_height), 3)

                # Show value of states.
                if show_value and not taxi_helpers.is_wall(taxi_oomdp, i + 1, taxi_oomdp.height - j):
                    # Draw the value.
                    val = val_text_dict[i + 1][taxi_oomdp.height - j]
                    color = mdpv.val_to_color(val)
                    pygame.draw.rect(screen, color, top_left_point + (cell_width, cell_height), 0)
                    value_text = reg_font.render(str(round(val, 2)), True, (46, 49, 49))
                    text_center_point = int(top_left_point[0] + cell_width / 2.0 - 10), int(
                        top_left_point[1] + cell_height / 3.0)
                    screen.blit(value_text, text_center_point)

                # Show optimal action to take in each grid cell.
                if policy and not taxi_helpers.is_wall(taxi_oomdp, i + 1, taxi_oomdp.height - j):
                    a = policy_dict[i+1][taxi_oomdp.height - j]
                    if a not in action_char_dict:
                        text_a = a
                    else:
                        text_a = action_char_dict[a]
                    text_center_point = int(top_left_point[0] + cell_width/2.0 - 10), int(top_left_point[1] + cell_height/3.0)
                    text_rendered_a = cc_font.render(text_a, True, (46, 49, 49))
                    screen.blit(text_rendered_a, text_center_point)

    pygame.display.flip()

    return agent_shape

def _draw_agent(center_point, screen, base_size=30, alpha=255):
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
    tri_color = (98, 140, 190, alpha)
    if alpha < 255:
        return mdpv._draw_polygon_alpha(screen, tri_color, tri)
    else:
        return pygame.draw.polygon(screen, tri_color, tri)