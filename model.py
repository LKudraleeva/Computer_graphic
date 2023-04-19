from PIL import Image
from functions import *


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

    def __init__(self, filename):
        self.vertex, self.polygon, self.normal, self.normal_to_polygon, self.texture, self.vt = read_obj(filename)

    def draw_vertex(self, height, weight, k, b):
        picture = Picture(height, weight)
        for v in self.vertex:
            picture.set_pixel(k * v[0] + b, -k * v[1] + b, color=255)
        return picture

    def draw_polygon(self, height, weight, k=4000, b=500):
        picture = Picture(height, weight)
        for p in self.polygon:
            for i in range(-1, 2):
                picture.line_4(self.vertex[p[i]][0] * k + b, -self.vertex[p[i]][1] * k + b,
                               self.vertex[p[i + 1]][0] * k + b, -self.vertex[p[i + 1]][1] * k + b, color=255)
        return picture

    def draw_texture(self, height, weight, file_texture, color: bool = False):

        texture_image = Image.open(file_texture, 'r')
        text_width, text_height = texture_image.size
        texture_image = np.asarray(texture_image)

        picture = Picture(height, weight, color)

        for p, n, t in zip(self.polygon, self.normal_to_polygon, self.texture):
            u0, v0 = self.vt[t[0]]
            u1, v1 = self.vt[t[1]]
            u2, v2 = self.vt[t[2]]

            u = np.array([u0, u1, u2])
            v = np.array([v0, v1, v2])

            new_vertexes = projective_transformation(self.get_vertexes(p))
            x0, y0, z0 = new_vertexes[0]
            x1, y1, z1 = new_vertexes[1]
            x2, y2, z2 = new_vertexes[2]
            x_min, y_min, x_max, y_max = search_minmax(x0, x1, x2, y0, y1, y2)

            for x, y in [(x, y) for x in range(int(x_min), int(x_max) + 1) for y in range(int(y_min), int(y_max) + 1)]:
                barycentric = get_barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)
                if np.all(barycentric > 0):
                    z = barycentric[0] * z0 + barycentric[1] * z1 + barycentric[2] * z2
                    if 0 <= x < picture.w and 0 <= y < picture.h:
                        if z > picture.z_buffer[x, y]:
                            idx0, idx1 = np.sum(text_width * (barycentric * u)), np.sum(text_width * (barycentric * v))
                            picture.z_buffer[x, y] = z
                            picture.set_pixel(x, y, texture_image[int(np.round(idx0))][int(np.round(idx1))])
        return picture

    def draw_guro(self, height, weight, color: bool = False):
        picture = Picture(height, weight, color)

        for p, n in zip(self.polygon, self.normal_to_polygon):
            new_vertexes = projective_transformation(self.get_vertexes(p))
            l_norm = self.get_coefficients(n)

            draw_with_z_buffer(new_vertexes, picture, l_norm)
        return picture

    def get_coefficients(self, indexes):
        vec = [0, 0, 1]
        return [normalized_dot(self.normal[idx], vec) for idx in indexes]

    def get_vertexes(self, indexes):
        return [self.vertex[idx] for idx in indexes]
