from ntsa.algorithms.series_generator.series_generator import SeriesGenerator

from math import sqrt
from scipy.stats import norm
import numpy as np


class BrownianMotionGenerator(SeriesGenerator):
    def __init__(self, start_point=0, delta=2, dt=0.02):
        self.start_point = start_point
        self.delta = delta
        self.dt = dt

    def generate_series(self, n=1000):
        start_point = np.asarray(self.start_point)
        r = norm.rvs(size=start_point.shape + (n,), scale=self.delta * sqrt(self.dt))
        xs = np.empty(r.shape)
        np.cumsum(r, axis=-1, out=xs)
        xs += np.expand_dims(start_point, axis=-1)
        return xs
