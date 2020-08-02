import numpy as np

from ntsa.algorithms.series_generator.series_generator import SeriesGenerator


class PeriodicOscilatorGenerator(SeriesGenerator):
    def __init__(self, mode="sin", start=0, step=0.1):
        self.mode = mode
        self.start = start
        self.step = step

    def generate_series(self, n=1000):
        xs = np.arange(self.start, self.start + n * self.step, self.step)
        if self.mode == "sin":
            return np.sin(xs)
        if self.mode == "cos":
            return np.cos(xs)
        if self.mode == "tan":
            return np.tan(xs)
