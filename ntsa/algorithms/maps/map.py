from abc import ABC, abstractmethod


class Map(ABC):
    """Abstract class for implementation the of Maps."""

    @abstractmethod
    def f(self, xs):
        pass

    @abstractmethod
    def df(self, xs, w):
        pass
