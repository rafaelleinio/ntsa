import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from IPython.display import display
import pandas as pd


class DataViz:
    def __init__():
        plt.figure(figsize=(20, 20))

    @staticmethod
    def plot_data(data, x="t", y="y", label=None):
        ax = plt.gca()
        df = pd.DataFrame(data).astype(float)
        df.plot(kind="line", x=x, y=y, ax=ax, label=label)

    @staticmethod
    def show():
        plt.show()

    @staticmethod
    def display_image(image):
        display(image)

    @staticmethod
    def create_animation(images, cmap="gray", figsize=(20, 20)):
        frames = len(images)
        fig = plt.figure(figsize=figsize)
        im = plt.imshow(images[0], cmap=cmap)
        plt.axis("off")
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)

        def update(i):
            img = images[i]
            im.set_data(img)
            return im

        return animation.FuncAnimation(fig, update, frames=frames, repeat=True)

    @staticmethod
    def display_animation(animation):
        display(HTML(animation.to_html5_video()))

    @staticmethod
    def display_recurrence_plot(xs, threshold):
        threshold_map = lambda x, y: 1 if abs(x - y) >= threshold else 0
        matrix = [[threshold_map(x, y) for y in reversed(xs)] for x in xs]
        plt.imshow(matrix, cmap="gray")
        plt.show()
