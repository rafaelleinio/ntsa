from ntsa.algorithms.solver import Solver


class RungeKuttaFourthOrder(Solver):
    """Fourth Order Runge Kutta method."""

    def __init__(self, t0, y0, h, dydt):
        super().__init__(t0, y0, h, dydt)

    def calculate_next_y(self, t, y):
        k1 = self.dydt(t, y)
        k2 = self.dydt(t + 1 / 2 * self.h, y + 1 / 2 * self.h * k1)
        k3 = self.dydt(t + 1 / 2 * self.h, y + 1 / 2 * self.h * k2)
        k4 = self.dydt(t + self.h, y + self.h * k3)
        return y + (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
