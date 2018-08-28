import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Linear Regression.')
parser.add_argument('-training', dest='training_path')


def dummy_coding(dataset):
    print('dummy_coding')

    cut = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    clarity = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

    for i in range(0, dataset.shape[0]):
        dataset.iloc[i, 1] = cut.index(dataset.iloc[i, 1])
        dataset.iloc[i, 2] = color.index(dataset.iloc[i, 2])
        dataset.iloc[i, 3] = clarity.index(dataset.iloc[i, 3])


def normalize(dataset):
    print('normalize')


def pre_processing(dataset):
    print('pre-processing')

    # Coding categorical/nominal variables
    dummy_coding(dataset)

    # Normalize data set
    normalize(dataset)


def cost(coefficients, variables):
    ret = 0
    for i in range(0, coefficients.shape[0]):
        ret = ret + coefficients[i]*variables[i]
    return ret


def compute_error(coefficients, x, y):
    m = x.shape[0]
    sum = 0
    for i in range(0, m):
        h = cost(coefficients, x.iloc[i, :])
        sum = sum + (h - y[i])*(h - y[i])
    return sum/(2.0*m)


def linear_regressor(x, y, iterations, learning_rate):
    n = x.shape[1]
    m = x.shape[0]

    # Set random coefficients values [0,1)
    coefficients = np.random.rand(n)
    tmp_coefficients = np.zeros(n)

    iter = 0
    while iter <= iterations:
        for j in range(0, n):
            sum = 0
            for i in range(0, m):
                h = cost(coefficients, x.iloc[i, :])
                sum = sum + (h - y[i])*x.iloc[i, j]

            tmp_coefficients[j] = sum/m

        # Update coefficients
        for j in range(0, n):
            coefficients[j] = coefficients[j] - learning_rate*tmp_coefficients[j]

        # Compute Error
        error = compute_error(coefficients, x, y)

        print('iteration:', iter, ', Error:', error)

        iter = iter + 1

    print('coefficients:', coefficients)
    print('minimum cost:', cost(coefficients))



def main():
    args = parser.parse_args()

    # Load training set
    training_set = pd.read_csv(args.training_path)

    print('Training set dimensions:', training_set.shape)

    # Split data set in variables(x) and target(y)
    training_set_x = training_set.iloc[:, 0:training_set.shape[1] - 1]
    training_set_y = training_set.iloc[:, -1]

    # Data pre-processing
    pre_processing(training_set_x)

    # Gradient descent
    linear_regressor(training_set_x, training_set_y, 10, 0.0001)


if __name__ == '__main__':
    main()