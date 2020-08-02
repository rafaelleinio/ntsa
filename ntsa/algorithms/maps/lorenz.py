"""Based on the implementation found [here](https://github.com/cbnfreitas/lyapunov_exponent_map_and_ode/blob/master/lorenz_map.py)"""

from scipy import integrate
from numpy import *


class Lorenz:
    def __init__(self, sigma=10, rho=28, beta=8 / 3, h0=0.01):
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.h0 = h0

    @staticmethod
    def pack_variables(xyz, w):
        return concatenate((xyz, reshape(w, 9)), axis=0)

    @staticmethod
    def unpack_variables(xyzw):
        return xyzw[0:3], reshape(xyzw[3::], (3, 3))

    def dot_xyz(self, x, y, z):
        return array(
            [self.sigma * (-x + y), x * (self.rho - z) - y, x * y - self.beta * z]
        )

    def variational_equation(self, xyzw, t=None):
        xyz, w = self.unpack_variables(xyzw)
        x, y, z = xyz

        dot_xyz = self.dot_xyz(x, y, z)
        dot_w = (
            array(
                [
                    [-self.sigma, self.sigma, 0],
                    [self.rho - z, -1, -x],
                    [y, x, -self.beta],
                ]
            )
            @ w
        )

        return self.pack_variables(dot_xyz, dot_w)

    def f_df(self, xyz, w):
        xyzw = self.pack_variables(xyz, w)
        next_xyzw = integrate.odeint(
            self.variational_equation, xyzw, array([0, 1]), h0=self.h0
        )
        return self.unpack_variables(next_xyzw[1])

    def f(self, xyz):
        w = eye(3)
        return self.f_df(xyz, w)[0]

    def df(self, xyz, w):
        return self.f_df(xyz, w)[1]
