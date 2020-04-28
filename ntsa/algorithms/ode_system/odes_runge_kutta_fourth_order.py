import numpy as np
from typing import List, Callable


class ODESRungeKuttaFourthOrder:
    """Fourth Order Runge Kutta method solver for system of ODEs."""

    def __init__(
        self,
        h: float,
        dydt_equations: List[Callable],
        initial_t: float,
        initial_ys: List[float],
    ):
        self.h = h
        self.dydt_equations = dydt_equations
        self.initial_t = initial_t
        self.initial_ys = np.array(initial_ys)

    def calculate_next_y(self, t: float, ys: List[float]):
        k1 = np.array([dydt(t, *ys) for dydt in self.dydt_equations])
        k2 = np.array(
            [
                dydt(t + 1 / 2 * self.h, *(ys + 1 / 2 * self.h * k1))
                for dydt in self.dydt_equations
            ]
        )
        k3 = np.array(
            [
                dydt(t + 1 / 2 * self.h, *(ys + 1 / 2 * self.h * k2))
                for dydt in self.dydt_equations
            ]
        )
        k4 = np.array(
            [dydt(t + self.h, *(ys + self.h * k3)) for dydt in self.dydt_equations]
        )
        return ys + (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def calculate_y_series(self, n: int):
        points = [self.initial_t + i * self.h for i in range(n)]
        ys_series = [self.initial_ys]
        current_ys = self.initial_ys

        for t in points:
            next_ys = self.calculate_next_y(t, current_ys)
            ys_series.append(next_ys)
            current_ys = next_ys

        return [
            {**{"t": t}, **{"y{}".format(i): y for i, y in enumerate(ys)}}
            for t, ys in zip(points, ys_series)
        ]
