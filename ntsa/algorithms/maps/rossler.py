from scipy import integrate
from numpy import *


class Rossler:
    def __init__(self, a=0.1, b=0.1, c=14, h0=0.01):
        self.a = a
        self.b = b
        self.c = c
        self.h0 = h0

    @staticmethod
    def pack_variables(xyz, w):
        return concatenate((xyz, reshape(w, 9)), axis=0)

    @staticmethod
    def unpack_variables(xyzw):
        return xyzw[0:3], reshape(xyzw[3::], (3, 3))

    def dot_xyz(self, x, y, z):
        return array([-y - z, x + self.a * y, self.b + (x - self.c) * z])

    def variational_equation(self, xyzw, t=None):
        xyz, w = self.unpack_variables(xyzw)
        x, y, z = xyz

        dot_xyz = self.dot_xyz(x, y, z)
        dot_w = array([[0, -1, -1], [1, self.a, 0], [z, 0, self.c + x]]) @ w

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
