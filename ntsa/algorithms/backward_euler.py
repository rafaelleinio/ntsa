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
