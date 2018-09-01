import numpy as np

from methods.general import CostCalculus


class LinearRegressor:
    @staticmethod
    def linear_regressor(x, y, iterations, learning_rate):

        # Define x0 = 1
        ones = np.ones((x.shape[0], 1))
        x = np.concatenate((ones, x), axis=1)

        # Data dimensions
        n = x.shape[1]
        m = x.shape[0]

        # Set random coefficients values [0,1) to start
        coefficients = np.random.rand(n)
        coefficients = np.array([coefficients, ] * m)

        tmp_coefficients = np.zeros(n)
        error = np.zeros(iterations)

        iter = 0
        while iter < iterations:
            # Cost function
            h = CostCalculus.cost(coefficients, x)

            for j in range(0, n):
                tmp = (h - y) * x[:, j]
                sum = np.sum(tmp)
                tmp_coefficients[j] = sum / m

            # Update coefficients
            coefficients[0, :] = coefficients[0, :] - learning_rate * tmp_coefficients
            coefficients = np.array([coefficients[0, :], ] * m)

            # Compute Error
            error[iter] = CostCalculus.compute_error(coefficients, x, y)

            print('iteration:', iter, ', Error:', error[iter])

            if iter >= 1:
                if abs(error[iter - 1] - error[iter]) <= 0.0001:
                    break

            iter = iter + 1

        print('coefficients:', coefficients[0, :])
        print('minimum cost:', error[iter - 1])
        print('learning rate:', learning_rate)

        return coefficients[0, :], error, iter - 1