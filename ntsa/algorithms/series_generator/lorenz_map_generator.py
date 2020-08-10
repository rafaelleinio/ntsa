import numpy as np

from ntsa.algorithms.series_generator.series_generator import SeriesGenerator
from ntsa.algorithms.maps import Lorenz

def generate_orbit_xyz(initial_condition, map_, dt, steps):
    xs = np.empty(steps + 1)
    ys = np.empty(steps + 1)
    zs = np.empty(steps + 1)
    xs[0], ys[0], zs[0] = initial_condition

    for i in range(steps):
        x_dot, y_dot, z_dot = map_.dot_xyz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    return xs, ys, zs

class LorenzMapMapGenerator(SeriesGenerator):
    def __init__(self, start_point=(0, 2, 0), sigma=10, rho=28, beta=8 / 3, dt=0.01):
        self.start_point = start_point
        self.map = Lorenz(sigma, rho, beta)
        self.dt = dt

    def generate_series(self, n=1000):
        return generate_orbit_xyz(
            initial_condition=self.start_point,
            map_=self.map,
            dt=self.dt,
            steps=n
        )
