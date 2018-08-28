import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Linear Regression.')
parser.add_argument('-training', dest='training_path')


def dummy_coding(dataset):
    print('dummy_coding')

    # Categorical variables
    cut = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    clarity = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

    for i in range(0, dataset.shape[0]):
        dataset.iloc[i, 1] = cut.index(dataset.iloc[i, 1])
        dataset.iloc[i, 2] = color.index(dataset.iloc[i, 2])
        dataset.iloc[i, 3] = clarity.index(dataset.iloc[i, 3])


def normalize(dataset):
    print('normalize')

    mean = np.mean(dataset.values, axis=0)
    std = np.std(dataset.values, axis=0)

    m = dataset.shape[0]
    mean = np.array([mean, ] * m)
    std = np.array([std, ] * m)

    norm = (dataset.values - mean)/std

    return norm


def pre_processing(dataset):
    print('pre-processing')

    # Coding categorical/nominal variables
    dummy_coding(dataset)

    # Normalize data set
    norm = normalize(dataset)

    return norm


def cost(coefficients, variables):
    h = coefficients*variables

    return np.sum(h, axis=1)


def compute_error(coefficients, x, y):
    m = x.shape[0]
    h = cost(coefficients, x)
    tmp = (h - y)*(h - y)

    return np.sum(tmp)/(2.0*m)


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
        h = cost(coefficients, x)

        for j in range(0, n):
            tmp = (h - y)*x[:, j]
            sum = np.sum(tmp)
            tmp_coefficients[j] = sum/m

        # Update coefficients
        coefficients[0, :] = coefficients[0, :] - learning_rate*tmp_coefficients
        coefficients = np.array([coefficients[0, :], ] * m)

        # Compute Error
        error[iter] = compute_error(coefficients, x, y)

        print('iteration:', iter, ', Error:', error[iter])

        if iter >= 1:
            if abs(error[iter - 1] - error[iter]) <= 0.0001:
                break

        iter = iter + 1

    print('coefficients:', coefficients[0, :])
    print('minimum cost:', error[iter-1])

    return coefficients[0, :], error, iter-1


def main():
    args = parser.parse_args()

    # Load training set
    df_train = pd.read_csv(args.training_path)

    # Split training data in training(80%) and validation(20%)
    validation_set = df_train.sample(frac=0.2, random_state=1)
    training_set = df_train.drop(validation_set.index)

    print('Training set dimensions:', training_set.shape)
    print('Validation set dimensions:', validation_set.shape)

    # Split training set in variables(x) and target(y)
    training_set_x = training_set.iloc[:, 0:training_set.shape[1] - 1]
    training_set_y = training_set.iloc[:, -1]

    # Split validation set in variables(x) and target(y)
    #validation_set_x = validation_set.iloc[:, 0:validation_set.shape[1] - 1]
    #validation_set_y = validation_set.iloc[:, -1]

    # Data pre-processing
    training_set_x = pre_processing(training_set_x)

    # Gradient descent
    coefficients, error, iter_stop = linear_regressor(training_set_x, training_set_y.values, 1000, 0.1)

    plt.plot(error[0:iter_stop])
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.show()
    plt.savefig('plot.png')

if __name__ == '__main__':
    main()