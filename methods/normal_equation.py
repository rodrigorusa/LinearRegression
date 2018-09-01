import numpy as np

from methods.general import CostCalculus


class NormalEquation:
    @staticmethod
    def normal_equation(x, y):
        # Define x0 = 1
        ones = np.ones((x.shape[0], 1))
        x = np.concatenate((ones, x), axis=1)

        coefficients = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

        print('coefficients:', coefficients)
        print('minimum cost:', CostCalculus.compute_error(coefficients, x, y))

        return coefficients