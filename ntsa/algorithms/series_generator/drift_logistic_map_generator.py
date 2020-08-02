from ntsa.algorithms.series_generator.series_generator import SeriesGenerator


class DriftLogisticMapGenerator(SeriesGenerator):
    def __init__(self, start_point=0, r=4, c=0.01):
        self.start_point = start_point
        self.c = c
        self.map = lambda x: (r * x * (1 - x))

    def generate_series(self, n=1000):
        xs = [self.start_point]
        for i in range(n):
            xs.append(self.map(xs[i]))
        drifted_xs = [(x + self.c * i) for i, x in enumerate(xs)]
        return drifted_xs
