from ntsa.algorithms.solver import Solver


class Heun(Solver):
    """Heun's algorithm method."""

    def __init__(self, t0, y0, h, dydt):
        super().__init__(t0, y0, h, dydt)

    def calculate_predictor(self, t, y):
        return y + self.h * self.dydt(t, y)

    def calculate_next_y(self, t, y):
        # predictor
        p = self.calculate_predictor(t, y)
        # corrector
        return y + (self.h / 2) * (self.dydt(t, y) + self.dydt(t + self.h, p))
