import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from IPython.display import display
import pandas as pd


class DataViz:
    def __init__(self):
        plt.figure(figsize=(20, 20))

    def plot_data(self, data, x="t", y="y", label=None):
        ax = plt.gca()
        df = pd.DataFrame(data).astype(float)
        df.plot(kind="line", x=x, y=y, ax=ax, label=label)

    def show(self):
        plt.show()

    def display_image(self, image):
        display(image)

    def create_animation(self, images, cmap="gray", figsize=(20, 20)):
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

    def display_animation(self, animation):
        display(HTML(animation.to_html5_video()))
