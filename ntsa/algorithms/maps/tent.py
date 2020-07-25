from numpy import array

from ntsa.algorithms.maps.map import Map


class Tent(Map):
    def __init__(self, u: float = 1.5):
        self.u = u

    def f(self, x):
        x, = x
        if 0 <= x <= 1 / 2:
            return array([self.u * x])
        elif 1 / 2 < x <= 1:
            return array([self.u * (1 - x)])
        else:
            raise ValueError("x not between 0 and 1")

    def df(self, x, w):
        x, = x
        if 0 <= x <= 1 / 2:
            jacobian_matrix = array([[self.u]])
            return jacobian_matrix @ w
        elif 1 / 2 < x <= 1:
            jacobian_matrix = array([[-self.u]])
            return jacobian_matrix @ w
        else:
            raise ValueError("y not between 0 and 1")
