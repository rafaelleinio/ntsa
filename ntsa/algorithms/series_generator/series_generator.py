from abc import ABC, abstractmethod


class SeriesGenerator(ABC):
    @abstractmethod
    def generate_series(self, n=1000):
        pass
