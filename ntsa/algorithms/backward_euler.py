from sympy import Eq, Symbol, solve

from ntsa.algorithms.solver import Solver


class BackwardEuler(Solver):
    """Backward Euler implicit method ODE Solver."""

    def __init__(self, t0, y0, h, dydt):
        super().__init__(t0, y0, h, dydt)
        self.y_i = Symbol("y")

    def calculate_next_y(self, t, y):
        # backward euler formula
        eq = Eq(y + self.dydt(t + self.h, self.y_i) * self.h, self.y_i)

        # solving linear equation of one variable
        next_y = solve(eq).pop()

        return next_y

    def calculate_y_series(self, n):
        points = [self.t0 + t * self.h for t in range(n)]
        y_series = [self.y0]
        current_y = self.y0

        for t in points:
            next_y = self.calculate_next_y(t, current_y)
            y_series.append(next_y)
            current_y = next_y

        return [{"t": t, "y": y} for t, y in zip(points, y_series)]
