from model.point import Point


class Robot:
    def __init__(self, id: int, location: Point, height: float, width: float, r: float, g: float,
                 b: float) -> None:
        self.id = id
        self.location = location
        self.height = height
        self.width = width
        self.r = r
        self.g = g
        self.b = b

    def radius(self):
        return self.width / 2

    def collision(self, location: Point, r: float) -> bool:
        dist = ((location.x - self.location.x) ** 2 + (location.y - self.location.y) ** 2) ** 0.5

        return True if dist <= r + (self.width / 2) else False

    def get_gui_color(self):
        return self.r * 255, self.g * 255, self.b * 255
