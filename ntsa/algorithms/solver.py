class Solver:
    def __init__(self, t0, y0, h, dydt):
        self.t0 = t0
        self.y0 = y0
        self.h = h
        self.dydt = dydt
