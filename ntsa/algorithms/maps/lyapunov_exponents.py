"""Based on the implementation found [here](https://github.com/cbnfreitas/lyapunov_exponent_map_and_ode)"""

from numpy import *
from numpy.linalg import *

from ntsa.algorithms.maps.map import Map


class LyapunovExponents:
    def __init__(self, map: Map, tolerance: float = 0.00001, max_iterations: int = 1000):
        self.map = map
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def calculate_from_initial_conditions(self, initial_conditions: ndarray):
        # setup variables
        n = len(initial_conditions)
        x = initial_conditions
        w = eye(n)
        h = zeros(n)
        l = -1

        # Numerical Lyapunov exponents calculation
        for i in range(0, self.max_iterations):
            x_next, w_next = self.map.f(x), self.map.df(x, w)
            w_next = self._orthogonalize_columns(w_next)

            h_next = h + self._log_of_the_norm_of_the_columns(w_next)
            l_next = h_next / (i + 1)

            if norm(l_next - l) < self.tolerance:
                return sort(l_next)

            h = h_next
            x = x_next
            w = self._normalize_columns(w_next)
            l = l_next

        # Max iter reach without a Solution in desired tolerance
        raise ValueError(
            "Lyapunov Exponents calculation did not converge."
            f" Initial conditions = {initial_conditions}"
            f", Tolerance = {self.tolerance}"
            f", Max Iterations = {self.max_iterations}"
        )

    def calculate_over_set_of_initial_conditions(self, set: ndarray):
        solutions = array(
            [
                self.calculate_from_initial_conditions(initial_conditions)
                for initial_conditions in set
            ]
        )
        return apply_along_axis(lambda v: mean(v), 0, solutions)

    def _orthogonalize_columns(self, a):
        q, r = qr(a)
        return q @ diag(r.diagonal())

    def _normalize_columns(self, a):
        return apply_along_axis(lambda v: v / norm(v), 0, a)

    def _log_of_the_norm_of_the_columns(self, a):
        return apply_along_axis(lambda v: log(norm(v)), 0, a)
