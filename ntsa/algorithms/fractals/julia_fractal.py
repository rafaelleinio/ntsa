"""Based on https://www.geeksforgeeks.org/julia-fractal-python/"""
from PIL import Image


class JuliaFractal:
    def __init__(
        self, center_x=-0.7, center_y=0.27015, slide_x=0.0, slide_y=0.0, max_iter=255
    ):
        self.center_x = center_x
        self.center_y = center_y
        self.slide_x = slide_x
        self.slide_y = slide_y
        self.max_iter = max_iter

    def make_fractal(self, width, height, zoom=1):
        bitmap = Image.new("L", (width, height), "white")
        pix = bitmap.load()  # image pixel access handler

        for x in range(width):
            for y in range(height):
                zx = (1.5 * (x - width / 2) / (0.5 * zoom * width)) + self.slide_x
                zy = (1.0 * (y - height / 2) / (0.5 * zoom * height)) + self.slide_y
                i = self.max_iter
                while zx * zx + zy * zy < 4 and i > 1:
                    zx, zy = (
                        zx * zx - zy * zy + self.center_x,
                        2 * zx * zy + self.center_y,
                    )
                    i -= 1
                pix[x, y] = i

        return bitmap
