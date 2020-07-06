from numpy import array

from ntsa.algorithms.maps.map import Map


class Baker(Map):
    def __init__(self):
        pass

    def f(self, xy):
        x, y = xy

        if 0 <= y <= 1 / 2:
            return array([1 / 3 * x, 2 * y])
        elif 1 / 2 < y <= 1:
            return array([1 / 3 * x + 2 / 3, 2 * y - 1])
        else:
            raise ValueError("y not between 0 and 1")

    def df(self, xy, w):
        x, y = xy

        if 0 <= y <= 1 / 2:
            jacobian_matrix = array([[1 / 3, 0], [0, 2]])
            return jacobian_matrix @ w
        elif 1 / 2 < y <= 1:
            jacobian_matrix = array([[1 / 3, 0], [0, 2]])
            return jacobian_matrix @ w
        else:
            raise ValueError("y not between 0 and 1")
