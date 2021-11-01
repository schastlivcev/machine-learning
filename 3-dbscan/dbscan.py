import pygame
import numpy as np


def dist(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def dbscan(points, eps, min_points):
    labels = [0] * len(points)
    cluster_id = 0
    for i in range(0, len(points)):
        if not (labels[i] == 0):
            continue
        neighbours = find_neighbours(points, i, eps)
        if len(neighbours) < min_points:
            labels[i] = -1
        else:
            cluster_id += 1
            create_cluster(points, labels, i, neighbours, cluster_id, eps, min_points)
    return labels


def find_neighbours(points, cent_point_id, eps):
    neighbours = []
    for point in range(0, len(points)):
        if dist(points[cent_point_id], points[point]) < eps:
            neighbours.append(point)
    return neighbours


def create_cluster(points, labels, cent_point_id, cent_point_neighbours, cluster_id, eps, min_points):
    labels[cent_point_id] = cluster_id
    i = 0
    while i < len(cent_point_neighbours):
        point = cent_point_neighbours[i]
        if labels[point] == -1:
            labels[point] = cluster_id
        elif labels[point] == 0:
            labels[point] = cluster_id
            point_neighbours = find_neighbours(points, point, eps)
            if len(point_neighbours) >= min_points:
                cent_point_neighbours = cent_point_neighbours + point_neighbours
        i += 1


def draw_circles(points, cluster_ids):
    for point, cluster_id in zip(points, cluster_ids):
        width = 0
        if cluster_id == -1:
            width = empty_width
        pygame.draw.circle(screen, colors[cluster_id], point, circle_radius, width)


if __name__ == '__main__':
    # dbscan params
    eps = 30
    min_points = 3

    # pygame params
    window_width = 1280
    window_height = 720
    fps = 60
    circle_radius = 7
    empty_width = 2

    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("DBSCAN")
    clock = pygame.time.Clock()
    white = (255, 255, 255)
    screen.fill(white)
    colors = []
    for i in range(1, 255):
        colors.append(tuple(np.random.choice(range(256), size=3)))
    grey = (123, 123, 123)
    colors.append(grey)

    points = []
    q = False
    while not q:
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                q = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                q = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                points = []
                screen.fill(white)
            if pygame.mouse.get_pressed()[0]:
                points.append(pygame.mouse.get_pos())
                pygame.draw.circle(screen, colors[-1], points[-1], circle_radius, empty_width)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                cluster_ids = dbscan(np.array(points), eps, min_points)
                screen.fill(white)
                draw_circles(points, cluster_ids)
        pygame.display.update()
