from model import Picture, RenderPicture


def task_1():
    # черное изображение
    black_picture = Picture()
    black_picture.save('result/task 1.1.jpg')

    # белое изображение
    white_picture = Picture()
    white_picture.set_color([255])
    white_picture.save('result/task 1.2.jpg')

    # красное изображение
    red_picture = Picture(color=True)
    red_picture.set_color([255, 0, 0])
    red_picture.save('result/task 1.3.jpg')

    # градиент
    picture = Picture(color=True)
    picture.make_gradient()
    picture.save('result/task 1.4.jpg')


def task_2():
    pic = Picture(200, 200, color=True)
    color = [255, 255, 255]

    pic.make_star(pic.line_1, color)
    pic.save('result/task 2.1.jpg')
    pic.clear()

    pic.make_star(pic.line_2, color)
    pic.save('result/task 2.2.jpg')
    pic.clear()

    pic.make_star(pic.line_3, color)
    pic.save('result/task 2.3.jpg')
    pic.clear()

    pic.make_star(pic.line_4, color)
    pic.save('result/task 2.4.jpg')
    pic.clear()


def task_3_6():
    file_name = 'model_1.obj'
    height, weight = 1000, 1000
    k_values = [50, 100, 500, 4000]
    b = 500

    im = RenderPicture(file_name)
    # 4
    for i, k in enumerate(k_values):
        im.draw_vertex(height, weight, k, b).save('result/task 4.' + str(i + 1) + '.jpg')
    # 6
    im.draw_polygon(height, weight).save('result/task 6.jpg')


def task_9_11():
    file_name = 'model_1.obj'
    height, weight = 1000, 1000

    im = RenderPicture(file_name)

    # 10
    # im.draw_triangle(height, weight)
    # im.vertex_picture.save('result/task 10.jpg')

    # 11
    # im.draw_triangle(height, weight, True)
    # im.vertex_picture.save('result/task 11.jpg')


def task_12_14():
    file_name = 'model_1.obj'
    height, weight = 1000, 1000

    im = RenderPicture(file_name)

    im.draw_triangle(height, weight, True).save('result/task 14.jpg')


def task_15():
    file_name = 'model_1.obj'
    height, weight = 1000, 1000

    im = RenderPicture(file_name)

    im.draw_triangle(height, weight, True).save('result/task 15.jpg')


if __name__ == '__main__':
    # task_1()
    # task_2()
    # task_3_6()
    # task_9_11()
    # task_12_14()
    task_15()
