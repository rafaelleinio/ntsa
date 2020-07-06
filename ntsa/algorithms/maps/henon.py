"""Based on the implementation found [here](https://github.com/cbnfreitas/lyapunov_exponent_map_and_ode)"""

from numpy import array

from ntsa.algorithms.maps.map import Map


class Henon(Map):
    def __init__(self, a=1.4, b=0.3):
        self.a, self.b = a, b

    def f(self, xy):
        x, y = xy
        return array([self.a - x ** 2 + self.b * y, x])

    def df(self, xy, w):
        x, y = xy
        jacobian_matrix = array([[-2 * x, self.b], [1, 0]])
        return jacobian_matrix @ w
