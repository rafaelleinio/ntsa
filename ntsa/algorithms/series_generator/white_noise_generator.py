import numpy as np

from ntsa.algorithms.series_generator.series_generator import SeriesGenerator


class WhiteNoiseGenerator(SeriesGenerator):
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def generate_series(self, n=1000):
        return np.random.normal(self.mu, self.sigma, size=n)
