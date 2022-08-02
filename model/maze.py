from model.charger import Charger
from model.mud import Mud
from model.obstacle import Obstacle
from model.point import Point
from model.robot import Robot
from typing import Dict, List

MAIN_ROBOT_ID = 0


class Maze:
    def __init__(self, charger: Charger, obstacles: List[Obstacle], robots: Dict[int, Robot],
                 mud_patches: List[Mud]) -> None:
        self.charger = charger
        self.obstacles = obstacles
        self.robots = robots
        self.mud_patches = mud_patches

    def hit_obstacle(self, location: Point, robot_id: int) -> int:
        robot_in_question = self.robots.get(robot_id, None)

        if robot_in_question is None:
            raise Exception(f'Could not find robot with ID {robot_id}')

        r = robot_in_question.radius()

        for obstacle in self.obstacles:
            collided = obstacle.collision(location, r)

            if collided:
                return 1

        return 0

    def hit_other_robot(self, location: Point, robot_id: int) -> int:
        robot_in_question = self.robots.get(robot_id, None)

        if robot_in_question is None:
            raise Exception(f'Could not find robot with ID {robot_id}')

        r = robot_in_question.radius()

        for robot in self.robots.values():
            if robot.id != robot_id:
                collided = robot.collision(location, r)

                if collided:
                    return 1

        return 0

    def in_mud(self, location: Point, robot_id: int) -> int:
        robot_in_question = self.robots.get(robot_id, None)

        if robot_in_question is None:
            raise Exception(f'Could not find robot with ID {robot_id}')

        r = robot_in_question.radius()

        for mud in self.mud_patches:
            in_curr_mud_patch = mud.in_mud(location, r)

            if in_curr_mud_patch:
                return 1

        return 0
