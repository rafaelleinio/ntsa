class Solver:
    def __init__(self, t0, y0, h, dydt):
        self.t0 = t0
        self.y0 = y0
        self.h = h
        self.dydt = dydt

    def calculate_next_y(self, t, y):
        raise NotImplementedError

    def calculate_y_series(self, n):
        points = [self.t0 + i * self.h for i in range(n)]
        y_series = [self.y0]
        current_y = self.y0

        for t in points:
            next_y = self.calculate_next_y(t, current_y)
            y_series.append(next_y)
            current_y = next_y

        return [{"t": t, "y": y} for t, y in zip(points, y_series)]
