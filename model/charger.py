from model.point import Point


class Charger:
    def __init__(self, id: int, location: Point, height: float, width: float, r: float, g: float, b: float) -> None:
        self.id = id
        self.location = location
        self.height = height
        self.width = width
        self.r = r
        self.g = g
        self.b = b

    def get_gui_color(self):
        return self.r * 255, self.g * 255, self.b * 255
