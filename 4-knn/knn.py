import numpy as np
from collections import Counter
from operator import itemgetter


class KNN:
    def __init__(self):
        self.optimal_k = None
        self.train_points = []
        self.train_clusters = []

    def fit(self, x_train, y_train):
        self.train_points = x_train
        self.train_clusters = y_train

    @staticmethod
    def dist(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_distances(self, point):
        dists = []
        for i in range(len(self.train_points)):
            dists.append((self.train_clusters[i], self.dist(self.train_points[i], point)))
        dists.sort(key=lambda x: x[1])
        return dists

    @staticmethod
    def get_k_neighbours(k, distances):
        neighbours = []
        for i in range(k):
            neighbours.append(distances[i][0])
        return neighbours

    def predict_cluster(self, points, optimal):
        predictions = []
        for point in points:
            if not optimal and self.optimal_k is not None:
                distances = self.calculate_distances(point)
                neighbours = self.get_k_neighbours(self.optimal_k, distances)
                labels = [neighbour for neighbour in neighbours]
                prediction = max(labels, key=labels.count)
                predictions.append(prediction)
            else:
                ks = []
                distances = self.calculate_distances(point)
                max_k = int(np.ceil(np.sqrt(len(self.train_points))))
                for i in range(3, max_k):
                    neighbours = self.get_k_neighbours(i, distances)
                    counter = Counter(neighbours)
                    probabilities = [(count[0], count[1] / len(neighbours) * 100.0) for count in counter.most_common()]
                    ks.append((probabilities[0][0], probabilities[0][1], i))
                prediction = max(ks, key=itemgetter(1))
                self.optimal_k = prediction[2]
                predictions.append(prediction[0])
        return predictions, self.optimal_k
