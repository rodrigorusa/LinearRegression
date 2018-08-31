import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Parameters
FRAC_VALIDATION = 0.2
MAX_ITERATIONS = 1000
LEARNING_RATE = 0.01
TOLERANCE = 0.000001
VARIABLES_DEGREES = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

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


def normalize(dataset, mean=None, std=None):
    print('normalize')

    # Compute mean and standard deviation
    if mean is None:
        mean = np.mean(dataset.values, axis=0)
    if std is None:
        std = np.std(dataset.values, axis=0)

    m = dataset.shape[0]
    mean = np.array([mean, ] * m)
    std = np.array([std, ] * m)

    # Normalization
    norm = (dataset.values - mean)/std

    return norm, mean[0, :], std[0, :]


def pre_processing(dataset, training_mean=None, training_std=None):
    print('pre-processing')

    # Coding categorical/nominal variables
    dummy_coding(dataset)

    # Normalize data set
    norm, mean, std = normalize(dataset, training_mean, training_std)

    return norm, mean, std


def data_visualization(x, y):
    print('data visualization')

    titles = ['Carat', 'Cut', 'Color', 'Clarity', 'X', 'Y', 'Z', 'Depth', 'Table']
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    k = 0
    for i in range(0, 3):
        for j in range(0, 3):
            axs[i, j].scatter(x[:, k], y)
            axs[i, j].set_title(titles[k])
            axs[i, j].set(ylabel='Price')
            k = k + 1
    plt.tight_layout()
    plt.show()
    fig.savefig('data_visualization.png')


def compute_model(parameters, x, x_degrees):
    h = parameters*np.power(x, x_degrees)

    return np.sum(h, axis=1)


def compute_error(parameters, x, x_degrees, y):
    m = x.shape[0]
    h = compute_model(parameters, x, x_degrees)
    tmp = (h - y)*(h - y)

    return np.sum(tmp)/(2.0*m)


def linear_regressor(x, y, x_val, y_val, var_degrees, max_iterations, learning_rate, tolerance):

    # Define x0 = 1
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate((ones, x), axis=1)
    x_val = np.concatenate((ones[0:x_val.shape[0]], x_val), axis=1)

    # Data dimensions
    n = x.shape[1]
    m = x.shape[0]

    # Set random parameters values [0,1) to start
    parameters = np.random.rand(n)
    parameters = np.array([parameters, ] * m)

    # Temporary coefficients
    tmp_parameters = np.zeros(n)

    # Replicate exponents to fast processing
    var_degrees = np.array([var_degrees, ] * m)

    # Array error per iteration
    training_error = np.zeros(max_iterations+1)
    validation_error = np.zeros(max_iterations + 1)

    # Do process
    iter = 1
    while iter <= max_iterations:
        # Compute model h_theta(x)
        h = compute_model(parameters, x, var_degrees)

        # For each variable Xn
        for j in range(0, n):
            tmp = (h - y)*np.power(x[:, j], var_degrees[:, j])
            sum = np.sum(tmp)
            tmp_parameters[j] = sum/m

        # Update coefficients
        parameters[0, :] = parameters[0, :] - learning_rate*tmp_parameters
        parameters = np.array([parameters[0, :], ] * m)

        # Compute Error
        training_error[iter] = compute_error(parameters, x, var_degrees, y)

        print('iteration:', iter, ', Error:', training_error[iter])

        # Validation Error
        validation_error[iter] = compute_error(parameters[0:x_val.shape[0], :], x_val,
                                               var_degrees[0:x_val.shape[0], :], y_val)

        if iter >= 2:
            if abs(training_error[iter - 1] - training_error[iter]) <= tolerance:
                break

        iter = iter + 1

    return parameters[0, :], training_error, validation_error, iter-1


def main():
    args = parser.parse_args()

    # Load training set
    df_train = pd.read_csv(args.training_path)

    # Split training data in training and validation
    validation_set = df_train.sample(frac=FRAC_VALIDATION, random_state=1)
    training_set = df_train.drop(validation_set.index)

    print('Training set dimensions (', (1-FRAC_VALIDATION)*100.0, '% ):', training_set.shape)
    print('Validation set dimensions (', FRAC_VALIDATION*100.0, '% ):', validation_set.shape)

    # Split training set in variables(x) and target(y)
    training_set_x = training_set.iloc[:, 0:training_set.shape[1] - 1]
    training_set_y = training_set.iloc[:, -1]

    # Split validation set in variables(x) and target(y)
    validation_set_x = validation_set.iloc[:, 0:validation_set.shape[1] - 1]
    validation_set_y = validation_set.iloc[:, -1]

    # Data pre-processing
    training_set_x, training_mean, training_std = pre_processing(training_set_x)
    validation_set_x, validation_mean, validation_std = pre_processing(validation_set_x, training_mean, training_std)

    # Data visualization
    data_visualization(training_set_x, training_set_y.values)

    # Gradient descent
    parameters, training_error, validation_error, iter_stop = linear_regressor(training_set_x,
                                                                               training_set_y.values,
                                                                               validation_set_x,
                                                                               validation_set_y.values,
                                                                               VARIABLES_DEGREES,
                                                                               MAX_ITERATIONS,
                                                                               LEARNING_RATE,
                                                                               TOLERANCE)

    print('\nGradient Descent: ')
    print('Number of iterations: ', iter_stop)
    print('Coefficients (model): \n', parameters)
    print('Training Mean squared error: %.2f' % training_error[iter_stop])
    print('Validation Mean squared error: %.2f' % validation_error[iter_stop])

    # Plot error
    fig = plt.figure()
    plt.plot(training_error[1:iter_stop], '-r', label='Training error')
    plt.plot(validation_error[1:iter_stop], '-b', label='Validation error')
    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()
    fig.savefig('plot.png')

    # Scikit SGD Linear Regressor
    regr = linear_model.SGDRegressor(max_iter=100, eta0=0.01)

    var_degrees = np.array([VARIABLES_DEGREES, ] * training_set_x.shape[0])
    for j in range(0, training_set_x.shape[1]):
        training_set_x[:, j] = np.power(training_set_x[:, j], var_degrees[:, j+1])

    # Fit model
    regr.fit(training_set_x, training_set_y.values)

    # Predict
    training_set_y_pred = regr.predict(training_set_x)
    validation_set_y_pred = regr.predict(validation_set_x)

    print('\nSGDRegressor Scikit Learn:')
    print('Coefficients (model): \n', regr.coef_)
    print('Intercept: \n', regr.intercept_)
    print('Training Mean squared error: %.2f' % mean_squared_error(training_set_y_pred, training_set_y.values))
    print('Validation Mean squared error: %.2f' % mean_squared_error(validation_set_y_pred, validation_set_y.values))

if __name__ == '__main__':
    main()