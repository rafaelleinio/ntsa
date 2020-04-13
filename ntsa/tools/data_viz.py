import matplotlib.pyplot as plt
import pandas as pd


class DataViz:
    def __init(self):
        plt.figure(figsize=(20, 20))

    def plot_data(self, data, x="t", y="y", label=None):
        ax = plt.gca()
        df = pd.DataFrame(data).astype(float)
        df.plot(kind="line", x=x, y=y, ax=ax, label=label)

    def show(self):
        plt.show()
