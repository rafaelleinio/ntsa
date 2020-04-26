from ntsa.algorithms.solver import Solver


class Euler(Solver):
    """Classical Euler method ODE Solver."""

    def __init__(self, t0, y0, h, dydt):
        super().__init__(t0, y0, h, dydt)

    def calculate_next_y(self, t, y):
        return y + self.h * (self.dydt(t, y))
