from ntsa.algorithms.solver import Solver


class RungeKuttaFehlberg(Solver):
    """Runge Kutta Fehlberg method (RKF45)."""

    def __init__(self, t0, y0, h, dydt):
        super().__init__(t0, y0, h, dydt)

    def calculate_next_y(self, t, y):
        k1 = self.dydt(t, y)
        k2 = self.dydt(t + 1 / 5 * self.h, y + 1 / 5 * self.h * k1)
        k3 = self.dydt(
            t + 3 / 10 * self.h, y + 3 / 40 * self.h * k1 + 9 / 40 * self.h * k2
        )
        k4 = self.dydt(
            t + 3 / 5 * self.h,
            y + 3 / 10 * self.h * k1 - 9 / 10 * self.h * k2 + 6 / 5 * self.h * k3,
        )
        k5 = self.dydt(
            t + self.h,
            y
            - 11 / 54 * self.h * k1
            + 5 / 2 * self.h * k2
            - 70 / 27 * self.h * k3
            + 35 / 27 * self.h * k4,
        )
        k6 = self.dydt(
            t + 7 / 8 * self.h,
            y
            + 1631 / 55296 * self.h * k1
            + 175 / 512 * self.h * k2
            + 575 / 13824 * self.h * k3
            + 44275 / 110592 * self.h * k4
            + 253 / 4096 * self.h * k5,
        )
        return y + self.h * (
            2825 / 27648 * k1
            + 18575 / 48384 * k3
            + 13525 / 55296 * k4
            + 277 / 14336 * k5
            + 1 / 4 * k6
        )
