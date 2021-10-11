import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_random_points_csv(n, file):
    x = np.random.randint(0, 100, n)
    y = np.random.randint(0, 100, n)
    pd.DataFrame({'x': x, 'y': y}).to_csv(file, index=False)


def read_points_csv(file):
    return pd.read_csv(file)


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def init_centroids(points, k):
    x_center = points['x'].mean()
    y_center = points['y'].mean()
    R = dist(x_center, y_center, points['x'][0], points['y'][0])
    for i in range(len(points)):
        R = max(R, dist(x_center, y_center, points['x'][i], points['y'][i]))
    x_c, y_c = [], []
    for i in range(k):
        x_c.append(x_center + R * np.cos(2 * np.pi * i / k))
        y_c.append(y_center + R * np.sin(2 * np.pi * i / k))
    return [x_c, y_c]


def probability_matrix(points, centroids, m):
    matrix = []
    for i in range(len(centroids[0])):
        matrix.append([])
    for i in range(len(points['x'])):
        dist_sum = 0
        for j in range(len(centroids[0])):
            dist_sum += dist(points['x'][i], points['y'][i], centroids[0][j], centroids[1][j]) ** (2/(1-m))
        for j in range(len(centroids[0])):
            prob = (dist(points['x'][i], points['y'][i], centroids[0][j], centroids[1][j]) ** (2/(1-m)) ) / dist_sum
            matrix[j].append(prob)
    return matrix


def recalculate_centroids(points, probability_matrix, centroids, m):
    x_c, y_c = [], []
    for i in range(len(centroids[0])):
        prob_sum, x_sum, y_sum = 0, 0, 0
        for j in range(len(points['x'])):
            prob = probability_matrix[i][j] ** m
            prob_sum += prob
            x_sum += points['x'][j] * prob
            y_sum += points['y'][j] * prob
        x_c.append(x_sum / prob_sum)
        y_c.append(y_sum / prob_sum)
    return [x_c, y_c]


def probability_matrix_equal(prev_probabilities, probabilities, epsilon):
    for i in range(len(prev_probabilities)):
        for j in range(len(prev_probabilities[i])):
            if abs(prev_probabilities[i][j] - probabilities[i][j]) > epsilon:
                return False
    return True


def cmeans(points, k, m, e):
    centroids = init_centroids(points, k)
    prev_matrix = probability_matrix(points, centroids, m)
    centroids = recalculate_centroids(points, prev_matrix, centroids, m)
    matrix = probability_matrix(points, centroids, m)
    step = 0
    while not probability_matrix_equal(prev_matrix, matrix, e):
        draw(points, k, matrix, centroids, 'step ' + str(step))
        step += 1
        centroids = recalculate_centroids(points, matrix, centroids, m)
        prev_matrix = matrix
        matrix = probability_matrix(points, centroids, m)
    return matrix, centroids


def show_matrix(points, probability_matrix):
    for i in range(len(probability_matrix[0])):
        print('(%2d, %2d)' % (points['x'][i], points['y'][i]), end=' ')
        for j in range(len(probability_matrix)):
            print('%.2f' % probability_matrix[j][i], end=' ')
        print()


def draw(points, k, probability_matrix, centroids, title):
    colors = plt.get_cmap('gist_rainbow')
    nearest_centroid_ids = [-1] * len(points['x'])
    for i in range(len(probability_matrix[0])):
        max_probability = 0
        for j in range(len(probability_matrix)):
            if(probability_matrix[j][i] > max_probability):
                max_probability = probability_matrix[j][i]
                nearest_centroid_ids[i] = j

    for i in range(len(points['x'])):
        plt.scatter(points['x'][i], points['y'][i], color=colors(nearest_centroid_ids[i] / k))
        plt.text(points['x'][i], points['y'][i], '%.2f' % probability_matrix[nearest_centroid_ids[i]][i])

    for i in range(len(centroids[0])):
        plt.scatter(centroids[0][i], centroids[1][i], marker='x', linewidths=2, color=colors(i / k))
    plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 5})

    file = 'dataset.csv'
    n_points = 200
    # generate_random_points_csv(n_points, file)

    k = 4
    m = 2
    e = 0.01

    points = read_points_csv(file)
    prob_matrix, cents = cmeans(points, k, m, e)
    show_matrix(points, prob_matrix)
    draw(points, k, prob_matrix, cents, 'final')
