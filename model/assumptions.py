from typing import List

N_ASSUMPTIONS = 26


class Assumptions:
    def __init__(self, camera_updated: int, expected_colors: int, expected_camera_res: int, low_camera_distortion: int,
                 low_camera_noise: int, one_charger: int, charger_visible: int, obstacle_visible: int, one_robot: int,
                 robot_visible: int, charger_bigger: int, consistent_map: int, expected_robot_size: int,
                 path_exists: int, stationary_world: int, uniform_cost: int, desirable_path: int,
                 expected_spin_speed: int, robot_moves_straight: int, expected_robot_speed: int, actuators_engaged: int,
                 expected_robot_movements: int, approaching_goal: int, performance_acceptable: int,
                 mud_visible: int, expected_mud_delay: int) -> None:
        self.camera_updated = camera_updated
        self.expected_colors = expected_colors
        self.expected_camera_res = expected_camera_res
        self.low_camera_distortion = low_camera_distortion
        self.low_camera_noise = low_camera_noise
        self.one_charger = one_charger
        self.charger_visible = charger_visible
        self.obstacle_visible = obstacle_visible
        self.one_robot = one_robot
        self.robot_visible = robot_visible
        self.charger_bigger = charger_bigger
        self.consistent_map = consistent_map
        self.expected_robot_size = expected_robot_size
        self.path_exists = path_exists
        self.stationary_world = stationary_world
        self.uniform_cost = uniform_cost
        self.desirable_path = desirable_path
        self.expected_spin_speed = expected_spin_speed
        self.robot_moves_straight = robot_moves_straight
        self.expected_robot_speed = expected_robot_speed
        self.actuators_engaged = actuators_engaged
        self.expected_robot_movements = expected_robot_movements
        self.approaching_goal = approaching_goal
        self.performance_acceptable = performance_acceptable
        self.mud_visible = mud_visible
        self.expected_mud_delay = expected_mud_delay


class AssumptionsHelper(object):
    @staticmethod
    def create_from_list(vals: List[int]) -> Assumptions:
        if len(vals) != N_ASSUMPTIONS:
            raise Exception(f'Invalid number of assumptions values given: {len(vals)}')

        return Assumptions(*vals)
