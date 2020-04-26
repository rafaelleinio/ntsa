from ntsa.algorithms.solver import Solver


class RungeKuttaSecondOrder(Solver):
    """Second Order Runge Kutta method."""

    def __init__(self, t0, y0, h, dydt, a):
        super().__init__(t0, y0, h, dydt)
        a_options = [1 / 2, 0, 1 / 3]
        if a not in a_options:
            raise ValueError(
                "a need to be one of the following values: {}".format(a_options)
            )
        self.a = a

    def calculate_next_y(self, t, y):
        k1 = self.dydt(t, y)

        if self.a == 1 / 2:
            k2 = self.dydt(t + self.h, y + self.h * k1)
            return y + (self.h / 2) * (k1 + k2)

        if self.a == 0:
            k2 = self.dydt(t + (1 / 2) * self.h, y + (1 / 2) * self.h * k1)
            return y + self.h * k2

        if self.a == 1 / 3:
            k2 = self.dydt(t + (3 / 4) * self.h, y + (3 / 4) * self.h * k1)
            return y + (self.h / 3) * (k1 + 2 * k2)
