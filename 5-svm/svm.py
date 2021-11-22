import numpy as np
from sklearn import svm


class SVM:
    def __init__(self, c=1.0):
        self.c = c  # alpha
        self.model = svm.SVC(kernel='linear', C=self.c)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def calculate_lines(self):
        w = self.model.coef_[0]  # w consists of 2 elements
        b = self.model.intercept_[0]  # b consists of 1 element
        x_points = np.linspace(0, 1024)  # generating x-points from -1 to 1
        y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
        main_line_arr = []
        for x, y in zip(x_points, y_points):
            main_line_arr.append([x, y])

        w_hat = self.model.coef_[0] / (np.sqrt(np.sum(self.model.coef_[0] ** 2)))  # Step 3 (margin):
        margin = 1 / np.sqrt(np.sum(self.model.coef_[0] ** 2))  # Step 4 (calculate points of the margin lines):
        decision_boundary_points = np.array(list(zip(x_points, y_points)))
        points_of_line_above = decision_boundary_points + w_hat * margin
        points_of_line_below = decision_boundary_points - w_hat * margin
        return main_line_arr, points_of_line_below, points_of_line_above
