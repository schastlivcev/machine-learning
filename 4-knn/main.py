import pygame
import numpy as np
from sklearn.datasets import make_blobs

from knn import KNN

clusters_num = 8
initial_points_num = clusters_num * 10

font_size = 24
circle_radius = 7

window_width = 1280
window_height = 720
fps = 30

pygame.init()
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("KNN")
clock = pygame.time.Clock()
white = (255, 255, 255)
gray = (125, 125, 125)
colors = []


def reload_colors():
    global colors
    colors = []
    for i in range(1, 255):
        colors.append(tuple(np.random.choice(range(256), size=3)))
    colors.append(gray)


reload_colors()
screen.fill(white)
knn = KNN()


class Scene:

    def __init__(self):
        self.k = None
        self.curr_cluster = 0
        self.curr_point = None
        self.clusters = []
        self.points = []
        self.still_training = True
        self.init_clusters()

    def init_clusters(self):
        points, clusters = make_blobs(n_samples=initial_points_num, centers=clusters_num,
                                      cluster_std=30, center_box=(100, window_height - 100))
        self.points = list(map(lambda x: [x[0], x[1]], points))
        self.clusters = list(map(lambda x: x + 1, clusters))

    def add_point(self, cluster):
        if self.curr_point is not None:
            self.points.append(self.curr_point)
            self.clusters.append(cluster)
            self.curr_point = None

    @staticmethod
    def draw_circles(points_arr, clusters):
        for point, cluster in zip(points_arr, clusters):
            pygame.draw.circle(screen, colors[int(cluster)], point, circle_radius)

    def draw_description(self, clusters):
        font = pygame.font.SysFont('Consolas', font_size, True)
        if self.still_training:
            surf = font.render(f'Num keys to choose drawing color. [Q] to start KNN, [R] to restart, [Esc] to exit.',
                               False, colors[-1])
            screen.blit(surf, (3, 0))
        else:
            surf = font.render(f'KNN started. RMB to calculate optimal k, LMB to use calculated k. [R] to restart, '
                               f'[Esc] to exit.', False, colors[-1])
            screen.blit(surf, (3, 0))

        for i in range(1, clusters + 1):
            surf = font.render(f'[{i}] ', False, colors[i])
            screen.blit(surf, (2 * font_size * (i - 1), font_size))

        if self.k is not None:
            pygame.draw.rect(screen, white, (2 * font_size * clusters, font_size, font_size * 16, font_size))
            surf = font.render(f'Optimal k = {self.k}', False, colors[-1])
            screen.blit(surf, (2 * font_size * clusters, font_size))

    def start(self):
        self.curr_point = None
        self.still_training = False
        screen.fill(white)

    def restart(self):
        self.curr_cluster = 0
        self.curr_point = None
        self.points = []
        self.k = None
        knn.optimal_k = None
        self.still_training = True
        self.init_clusters()
        reload_colors()
        screen.fill(white)

    def run(self):
        q = False

        while not q:
            clock.tick(fps)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    q = True

                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    self.start()

                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    self.restart()

                if event.type == pygame.KEYDOWN and '1' <= event.unicode <= str(clusters_num):
                    self.curr_cluster = int(event.unicode)

                left = pygame.mouse.get_pressed()[0]
                right = pygame.mouse.get_pressed()[2]
                if left or right:
                    self.curr_point = pygame.mouse.get_pos()
                    if self.still_training:
                        if self.curr_cluster != 0:
                            knn.fit(self.points, self.clusters)
                            pred, optimal_k = knn.predict_cluster([self.curr_point], True)
                            self.k = optimal_k
                            self.add_point(self.curr_cluster)
                    else:
                        knn.fit(self.points, self.clusters)
                        pred, optimal_k = knn.predict_cluster([self.curr_point], right)
                        self.k = optimal_k
                        self.points.append(self.curr_point)
                        self.clusters += pred
                        self.curr_point = []

            self.draw_description(clusters_num)
            pygame.display.update()

            if len(self.points) > 0:
                self.draw_circles(points_arr=self.points, clusters=self.clusters)


scene = Scene()
scene.run()
