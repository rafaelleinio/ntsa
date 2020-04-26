from ntsa.algorithms.solver import Solver


class RungeKuttaFifthOrder(Solver):
    """Fifth Order Runge Kutta method."""

    def __init__(self, t0, y0, h, dydt):
        super().__init__(t0, y0, h, dydt)

    def calculate_next_y(self, t, y):
        k1 = self.dydt(t, y)
        k2 = self.dydt(t + 1 / 4 * self.h, y + 1 / 4 * self.h * k1)
        k3 = self.dydt(
            t + 1 / 4 * self.h, y + 1 / 8 * self.h * k1 + 1 / 8 * self.h * k2
        )
        k4 = self.dydt(t + 1 / 2 * self.h, y - 1 / 2 * self.h * k2 + self.h * k3)
        k5 = self.dydt(
            t + 3 / 4 * self.h, y + 3 / 16 * self.h * k1 + 9 / 16 * self.h * k4
        )
        k6 = self.dydt(
            t + self.h,
            y
            - 3 / 7 * self.h * k1
            + 2 / 7 * self.h * k2
            + 12 / 7 * self.h * k3
            - 12 / 7 * self.h * k4
            + 8 / 7 * self.h * k5,
        )
        return y + self.h / 90 * (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6)
