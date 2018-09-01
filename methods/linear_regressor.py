import numpy as np

from methods.general import CostCalculus


class LinearRegressor:
    @staticmethod
    def regressor(train_x, train_y, val_x, val_y, max_iterations, learning_rate, tolerance):

        # Define x0 = 1
        ones = np.ones((train_x.shape[0], 1))
        train_x = np.concatenate((ones, train_x), axis=1)
        val_x = np.concatenate((ones[0:val_x.shape[0]], val_x), axis=1)

        # Data dimensions
        n = train_x.shape[1]
        m = train_x.shape[0]

        # Set random parameters values [0,1) to start
        params = np.random.rand(n)
        params = np.array([params, ] * m)

        # Temporary parameters
        tmp_params = np.zeros(n)

        # Array error per iteration
        train_error = np.zeros(max_iterations + 1)
        val_error = np.zeros(max_iterations + 1)

        # Do process
        i = 1
        while i <= max_iterations:
            # Compute model h_theta(x)
            h = CostCalculus.cost(params, train_x)

            # For each variable Xn
            for j in range(0, n):
                tmp = (h - train_y) * train_x[:, j]
                tmp_params[j] = np.sum(tmp) / m

            # Update coefficients
            params[0, :] = params[0, :] - learning_rate * tmp_params
            params = np.array([params[0, :], ] * m)

            # Compute Error
            train_error[i] = CostCalculus.compute_error(params, train_x, train_y)

            # Validation Error
            val_error[i] = CostCalculus.compute_error(params[0:val_x.shape[0], :], val_x, val_y)

            print('Iteration:', i, ', ( Training Error:', train_error[i], ', Validation Error:', val_error[i]), ')'

            if i >= 2:
                if abs(train_error[i-1] - train_error[i]) <= tolerance:
                    break

            i = i + 1

        return params[0, :], train_error, val_error, i-1
