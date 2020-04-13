from ntsa.algorithms.solver import Solver


class RungeKutta(Solver):
    """4th Order Classical Runge Kutta ODE Solver."""

    def __init__(self, t0, y0, h, dydt):
        super().__init__(t0, y0, h, dydt)

    def _get_initial_params(self, target_t):
        h = self.h
        n = int(abs(target_t - self.t0) / h)
        t = self.t0
        y = self.y0
        dydt = self.dydt
        return dydt, t, y, h, n

    def calculate_y(self, target_t):
        dydt, t, y, h, n = self._get_initial_params(target_t)
        for i in range(n):
            k1 = h * dydt(t, y)
            k2 = h * dydt(t + h / 2, y + k1 / 2)
            k3 = h * dydt(t + h / 2, y + k2 / 2)
            k4 = h * dydt(t + h, y + k3)

            # update next value of y
            y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

            # update next value of t
            t = t + h

        return y

    def calculate_y_series(self, start_t, step, n):
        points = [start_t + x * step for x in range(n)]
        return [{"t": t, "y": self.calculate_y(t)} for t in points]
