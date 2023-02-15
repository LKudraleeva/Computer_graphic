from PIL import Image
import numpy as np


def correct_points(x0: int, y0: int, x1: int, y1: int, steep=False):
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    return x0, x1, y0, y1, steep


class Picture:

    def __init__(self, h: int = 256, w: int = 256, color: bool = False):
        self.h = h
        self.w = w
        self.color = color
        if color:
            self.image_array = np.zeros((h, w, 3), dtype='uint8')
        else:
            self.image_array = np.zeros((h, w), dtype='uint8')

    def clear(self):
        if self.color:
            self.set_color([0, 0, 0])
        else:
            self.set_color([0])

    def save(self, filename):
        im = Image.fromarray(self.image_array)
        im.save(filename)

    def set_pixel(self, x, y, color):
        self.image_array[int(-y)][int(-x)] = color

    def set_color(self, colour_array=None):
        if colour_array is None:
            colour_array = [0]
        self.image_array[...] = colour_array

    def make_gradient(self):
        for i in range(self.w):
            for j in range(self.h):
                pix = (i + j) % 256
                self.image_array[i][j] = [pix, pix, pix]

    def line_1(self, x0: int, y0: int, x1: int, y1: int, color):
        t = 0.0
        delta = 0.01
        while t < 1.0:
            x = x0 * (1. - t) + x1 * t
            y = y0 * (1. - t) + y1 * t
            self.set_pixel(x, y, color)
            t += delta

    def line_2(self, x0: int, y0: int, x1: int, y1: int, color):
        x = x0
        while x <= x1:
            t = (x - x0) / (x1 - x0)
            y = y0 * (1. - t) + y1 * t
            self.set_pixel(x, y, color)
            x += 1

    def line_3(self, x0: int, y0: int, x1: int, y1: int, color):
        steep = False
        x = x0
        x0, x1, y0, y1, steep = correct_points(x0, y0, x1, y1, steep)

        while x <= x1:
            t = (x - x0) / (x1 - x0)
            y = y0 * (1.0 - t) + y1 * t
            if steep:
                self.set_pixel(x, y, color)
            else:
                self.set_pixel(y, x, color)
            x += 1

    def line_4(self, x0: int, y0: int, x1: int, y1: int, color):
        steep = False
        x0, x1, y0, y1, steep = correct_points(x0, y0, x1, y1, steep)

        d_x = x1 - x0
        d_y = y1 - y0
        d_error = abs(d_y / float(d_x))
        error = 0.
        y = y0

        for x in range(int(x0), int(x1) + 1):
            if steep:
                self.set_pixel(y, x, color)
            else:
                self.set_pixel(x, y, color)
            error += d_error
            if error > 0.5:
                if y1 > y0:
                    y += 1
                else:
                    y -= 1
                error -= 1.

    @staticmethod
    def make_star(f, color):
        for i in range(13):
            a = 2 * np.pi * i / 13
            f(100, 100, 100 + 95 * np.cos(a), 100 + 95 * np.sin(a), color)


class RenderPicture:

    def __init__(self):
        self.vertex = None
        self.polygon = None
        self.vertex_picture = None
        self.poly_picture = None

    def read_obj(self, filename):
        vertex = []
        polygon = []
        f = open(filename, 'r')
        lines = f.read()
        for line in lines.split('\n'):
            try:
                v, x, y, z = line.split(" ")
            except:
                continue
            if v == 'v':
                vertex.append([float(x), float(y)])
            if v == 'f':
                polygon.append([int(x.split("/")[0]) - 1, int(y.split("/")[0]) - 1, int(z.split("/")[0]) - 1])
        f.close()
        self.vertex = vertex
        self.polygon = polygon

    def draw_vertex(self, height, weight, k, b):
        picture = Picture(height, weight)
        for v in self.vertex:
            picture.set_pixel(k * v[0] + b, k * v[1] + b, color=255)
        self.vertex_picture = picture

    def draw_polygon(self, height, weight, k=4000, b=500):
        picture = Picture(height, weight)
        for p in self.polygon:
            for i in range(-1, 2):
                picture.line_4(self.vertex[p[i]][0] * k + b, self.vertex[p[i]][1] * k + b,
                               self.vertex[p[i+1]][0] * k + b, self.vertex[p[i+1]][1] * k + b, color=255)
        self.poly_picture = picture