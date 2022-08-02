from model.point import Point


class Obstacle:
    def __init__(self, id: int, start_point: Point, end_point: Point) -> None:
        self.id = id
        self.start_point = start_point
        self.end_point = end_point

    def collision(self, location: Point, r: float) -> bool:
        sx, sy, ex, ey = self.start_point.x, self.start_point.y, self.end_point.x, self.end_point.y
        x, y = location.x, location.y

        if sx == ex:
            if abs(x - sx) > r:
                return False

            elif abs(x - sx) == r:
                if ((y > sy) and (y > ey)) or ((y < sy) and (y < ey)):
                    return False

                else:
                    return True

            else:
                answer = r * r - ((sx - x) * (sx - x))
                y1 = y - (answer ** 0.5)
                y2 = y + (answer ** 0.5)

                if (((y1 > sy) and (y1 > ey)) or ((y1 < sy) and (y1 < ey))) and (((y2 > sy) and (y2 > ey)) or
                                                                                 ((y2 < sy) and (y2 < ey))):
                    return False

                else:
                    return True

        else:
            m = (ey - sy) / (ex - sx)
            b = sy - (m * sx)
            qa = 1 + m * m
            qb = (2 * (b - y) * m) - 2 * x
            qc = x * x + (b - y) * (b - y) - r * r

            answer = (qb * qb) - 4 * qa * qc

            if answer < 0:
                return False

            elif answer == 0:
                root_x = -qb / (2 * qa)

                if ((root_x > sx) and (root_x > ex)) or ((root_x < sx) and (root_x < ex)):
                    return False

                else:
                    return True

            else:
                root_x1 = (-qb - (answer ** 0.5)) / (2 * qa)
                root_x2 = (-qb + (answer ** 0.5)) / (2 * qa)

                if (((root_x1 > sx) and (root_x1 > ex)) or ((root_x1 < sx) and (root_x1 < ex))) \
                        and (((root_x2 > sx) and (root_x2 > ex)) or ((root_x2 < sx) and (root_x2 < ex))):
                    return False

                else:
                    return True
