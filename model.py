import random

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


def get_barycentric_coordinates(x: int, y: int, x0: float, y0: float, x1: float, y1: float, x2: float,
                                y2: float):
    lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
    lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
    lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    return np.array([lambda0, lambda1, lambda2])


def normalized_dot(normal, vect_l):
    return np.dot(normal, vect_l) / (np.linalg.norm(normal) * np.linalg.norm(vect_l))


def search_minmax(x0, x1, x2, y0, y1, y2):
    x_min = min(x0, x1, x2)
    if x_min < 0:
        x_min = 0
    y_min = min(y0, y1, y2)
    if y_min < 0:
        y_min = 0
    x_max = max(x0, x1, x2)
    if x_max < 0:
        x_max = 0
    y_max = max(y0, y1, y2)
    if y_max < 0:
        y_max = 0

    return x_min, y_min, x_max, y_max


class Picture:

    def __init__(self, h: int = 256, w: int = 256, color: bool = False):
        self.h = h
        self.w = w
        self.color = color
        self.z_buffer = np.zeros((h, w))

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
        self.image_array[int(y)][int(x)] = color

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
                vertex.append([float(x), float(y), float(z)])
            if v == 'f':
                polygon.append([int(x.split("/")[0]) - 1, int(y.split("/")[0]) - 1, int(z.split("/")[0]) - 1])
        f.close()
        self.vertex = vertex
        self.polygon = polygon

    def draw_vertex(self, height, weight, k, b):
        picture = Picture(height, weight)
        for v in self.vertex:
            picture.set_pixel(k * v[0] + b, -k * v[1] + b, color=255)
        self.vertex_picture = picture

    def draw_polygon(self, height, weight, k=4000, b=500):
        picture = Picture(height, weight)
        for p in self.polygon:
            for i in range(-1, 2):
                picture.line_4(self.vertex[p[i]][0] * k + b, -self.vertex[p[i]][1] * k + b,
                               self.vertex[p[i + 1]][0] * k + b, -self.vertex[p[i + 1]][1] * k + b, color=255)
        self.poly_picture = picture

    def draw_triangle(self, height, weight, color: bool = False, k=4000, b=500):
        picture = Picture(height, weight, color)

        vect_l = [0, 0, 1]

        for p in self.polygon:
            # for tasks 10, 11
            triangle_color_rgb = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            triangle_color_gray = random.randint(0, 255)

            x0 = k * self.vertex[p[0]][0] + b
            y0 = -k * self.vertex[p[0]][1] + b
            x1 = k * self.vertex[p[1]][0] + b
            y1 = -k * self.vertex[p[1]][1] + b
            x2 = k * self.vertex[p[2]][0] + b
            y2 = -k * self.vertex[p[2]][1] + b

            z0 = k * self.vertex[p[0]][2] + b
            z1 = k * self.vertex[p[1]][2] + b
            z2 = k * self.vertex[p[2]][2] + b

            normal = [(y1 - y0) * (z1 - z2) - (y1 - y2) * (z1 - z0),
                      (z1 - z0) * (x1 - x2) - (x1 - x0) * (z1 - z2),
                      (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)]

            x_min, y_min, x_max, y_max = search_minmax(x0, x1, x2, y0, y1, y2)

            norm_dot = normalized_dot(normal, vect_l)

            if norm_dot < 0:
                for x in range(int(x_min), int(x_max) + 1):
                    for y in range(int(y_min), int(y_max) + 1):

                        lambdas = get_barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)
                        summ = np.sum(lambdas)
                        if summ == 1:
                            if np.all(lambdas >= 0):

                                # z-buffer
                                new_z = lambdas[0] * z0 + lambdas[1] * z1 + lambdas[2] * z2

                                if picture.h > x >= 0 and picture.w > y >= 0:
                                    if new_z > picture.z_buffer[x][y]:
                                        picture.z_buffer[x][y] = new_z
                                        if not color:
                                            picture.set_pixel(x, y, triangle_color_gray)
                                        else:
                                            picture.set_pixel(x, y, [255 * norm_dot, 0, 0])

                self.vertex_picture = picture
