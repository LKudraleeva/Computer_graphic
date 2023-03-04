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


def projective_transformation(vertex):
    matrix_k = np.array([[5000, 0, 500],
                         [0, 5000, 500],
                         [0, 0, 1]])
    t = np.array([0.050, -0.045, 0.5])
    s = vertex + t
    new_vertex = np.dot(matrix_k, vertex + t)

    return new_vertex[0], new_vertex[1], new_vertex[2]


def rotation_matrix():
    alpha = 0 / 180 * np.pi
    betta = 0 / 180 * np.pi
    gamma = -28.5 / 180 * np.pi

    r_1 = np.array([[1, 0, 0],
                    [0, np.cos(alpha), np.sin(alpha)],
                    [0, -1 * np.sin(alpha), np.cos(alpha)]])

    r_2 = np.array([[np.cos(betta), 0, np.sin(betta)],
                    [0, 1, 0],
                    [-1 * np.sin(betta), 0, np.cos(betta)]])

    r_3 = np.array([[np.cos(gamma), np.sin(gamma), 0],
                    [-1 * np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])
    matrix_r = np.dot(np.dot(r_1, r_2), r_3)

    return matrix_r
