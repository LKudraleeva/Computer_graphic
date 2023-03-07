import numpy as np


def read_obj(filename):
    vertex = []
    polygon = []
    normal_to_polygon = []
    normal = []
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
            normal_to_polygon.append([int(x.split("/")[2]) - 1, int(y.split("/")[2]) - 1, int(z.split("/")[2]) - 1])
        if v == 'vn':
            normal.append([float(x), float(y), float(z)])
    f.close()
    return vertex, polygon, normal, normal_to_polygon


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


def projective_transformation(vertexes, rot=None):
    if rot is None:
        rot = [0, 0, 0]

    matrix = rotation_matrix(rot[0], rot[1], rot[2])

    matrix_k = np.array([[-4000, 0, 500],
                         [0, -4000, 500],
                         [0, 0, 1]])
    t = np.array([0, 0, 0.9])

    new_vertexes = []
    for v in vertexes:
        if rot:
            new_vertexes.append(np.dot(matrix, np.dot(matrix_k, v + t)))
        else:
            new_vertexes.append(np.dot(matrix_k, v + t))
    return new_vertexes


def rotation_matrix(alpha, betta, gamma):
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


def get_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    return [(y1 - y0) * (z1 - z2) - (y1 - y2) * (z1 - z0),
            (z1 - z0) * (x1 - x2) - (x1 - x0) * (z1 - z2),
            (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)]


def draw_with_z_buffer(vertex_list, picture, l_norm):
    x0, y0, z0 = vertex_list[0]
    x1, y1, z1 = vertex_list[1]
    x2, y2, z2 = vertex_list[2]

    x_min, y_min, x_max, y_max = search_minmax(x0, x1, x2, y0, y1, y2)

    norm_dot = normalized_dot(get_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2), [0, 0, 1])

    if norm_dot < 0:

        for x, y in [(x, y) for x in range(int(x_min), int(x_max) + 1) for y in range(int(y_min), int(y_max) + 1)]:
            lambdas = get_barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)

            if np.sum(lambdas) == 1 and np.all(lambdas >= 0):
                new_z = lambdas[0] * z0 + lambdas[1] * z1 + lambdas[2] * z2
                if new_z > picture.z_buffer[x][y]:
                    picture.z_buffer[x][y] = new_z
                    picture.set_pixel(x, y, [255 * (lambdas[0] * l_norm[0] + lambdas[1] * l_norm[1] + lambdas[2] *
                                                    l_norm[2]), 0, 0])
