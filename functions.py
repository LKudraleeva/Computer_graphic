import numpy as np


def read_obj(filename):
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
    return vertex, polygon


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
