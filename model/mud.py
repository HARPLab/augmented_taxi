from model.point import Point


class Mud:
    def __init__(self, id: int, location: Point, height: float, width: float, delay: float, r: float, g: float,
                 b: float) -> None:
        self.id = id
        self.location = location
        self.height = height
        self.width = width
        self.delay = delay
        self.r = r
        self.g = g
        self.b = b

    def in_mud(self, location: Point, r: float) -> bool:
        x_min, x_max = self.location.x - (self.width / 2), self.location.x + (self.width / 2)
        y_min, y_max = self.location.y - (self.height / 2), self.location.y + (self.height / 2)

        return x_min <= location.x <= x_max and y_min <= location.y <= y_max

    def get_gui_color(self):
        return self.r * 255, self.g * 255, self.b * 255
