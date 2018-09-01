import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

from methods.linear_regressor import LinearRegressor
from methods.normal_equation import NormalEquation
from methods.scikit_regressor import ScikitRegressor
from reader.csv_reader import DiamondCsvReader

parser = argparse.ArgumentParser(description='Linear Regression.')
parser.add_argument('-training', dest='training_path')

FRAC_VALIDATION = 0.2
POW_ARRAY = [1, 1, 1, 1, 1, 1, 1, 1, 1]


def normalize(df, mean=None, std=None):

    # Compute mean and standard deviation
    if mean is None:
        mean = np.mean(df.values, axis=0)
    if std is None:
        std = np.std(df.values, axis=0)

    m = df.shape[0]
    mean = np.array([mean, ] * m)
    std = np.array([std, ] * m)

    # Normalization
    norm = (df.values - mean)/std

    return norm, mean[0, :], std[0, :]


def data_visualization(x, y, titles):

    n = 3
    m = math.ceil(x.shape[1]/n)

    fig, axs = plt.subplots(m, n, figsize=(15, 10))
    k = 0
    for i in range(0, m):
        for j in range(0, n):
            axs[i, j].scatter(x[:, k], y)
            axs[i, j].set_title(titles[k])
            axs[i, j].set(ylabel='Price')
            k = k + 1
            if k == x.shape[1]:
                break
    plt.tight_layout()
    plt.show()
    fig.savefig('data_visualization.png')


def data_correlation(df):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(data=corr, annot=True, ax=ax)
    plt.show()
    fig.savefig('data_correlation.png')


def init_dataset(args):

    # Load training set
    df_train = DiamondCsvReader.get_data_frame(args.training_path)

    # Split training data in training and validation
    validation_set = df_train.sample(frac=FRAC_VALIDATION, random_state=1)
    training_set = df_train.drop(validation_set.index)

    print('Training set dimensions (', (1 - FRAC_VALIDATION) * 100.0, '% ):', training_set.shape)
    print('Validation set dimensions (', FRAC_VALIDATION * 100.0, '% ):', validation_set.shape)

    # Split training set in variables(x) and target(y)
    training_set_x = training_set.iloc[:, 0:training_set.shape[1] - 1]
    training_set_y = training_set.iloc[:, -1]

    # Split validation set in variables(x) and target(y)
    validation_set_x = validation_set.iloc[:, 0:validation_set.shape[1] - 1]
    validation_set_y = validation_set.iloc[:, -1]

    # Data pre-processing
    training_set_x, training_mean, training_std = normalize(training_set_x)
    validation_set_x, _, _ = normalize(validation_set_x, training_mean, training_std)

    # Data visualization
    titles = ['Carat', 'Cut', 'Color', 'Clarity', 'X', 'Y', 'Z', 'Depth', 'Table']
    data_visualization(training_set_x, training_set_y.values, titles)

    # Data correlation
    data_correlation(training_set)

    # Apply pow array
    pow_array = np.array([POW_ARRAY, ] * training_set_x.shape[0])
    for j in range(0, training_set_x.shape[1]):
        training_set_x[:, j] = np.power(training_set_x[:, j], pow_array[:, j])
        validation_set_x[:, j] = np.power(validation_set_x[:, j], pow_array[0:validation_set_x.shape[0], j])

    return training_set_x, training_set_y, validation_set_x, validation_set_y


def gradient_descent(train_set_x, train_set_y, val_set_x, val_set_y):
    val = input('Set maximum iterations (default: 1000): ')
    max_iterations = 1000
    if val != '':
        max_iterations = int(val)
    val = input('Set learning rate (default: 0.01): ')
    learning_rate = 0.01
    if val != '':
        learning_rate = float(val)
    val = input('Set tolerance (default: 0.000001): ')
    tolerance = 0.000001
    if val != '':
        tolerance = float(val)

    # Gradient descent
    params, train_error, val_error, iter_stop = LinearRegressor.regressor(train_set_x, train_set_y.values, val_set_x,
                                                                          val_set_y.values, max_iterations,
                                                                          learning_rate, tolerance)

    print('\nGradient Descent: ')
    print('Number of iterations: ', iter_stop)
    print('Coefficients (model): \n', params)
    print('Training Mean squared error: %.2f' % train_error[iter_stop])
    print('Validation Mean squared error: %.2f' % val_error[iter_stop])

    # Plot error
    fig = plt.figure()
    plt.plot(train_error[1:iter_stop], '-r', label='Training error')
    plt.plot(val_error[1:iter_stop], '-b', label='Validation error')
    plt.legend(loc='upper right')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()
    fig.savefig('error_plot.png')


def scikit_regressor(train_set_x, train_set_y, val_set_x, val_set_y):
    val = input('Set maximum iterations (default: 1000): ')
    max_iterations = 1000
    if val != '':
        max_iterations = int(val)
    val = input('Set learning rate (default: 0.01): ')
    learning_rate = 0.01
    if val != '':
        learning_rate = float(val)
    val = input('Set tolerance (default: 0.000001): ')
    tolerance = 0.000001
    if val != '':
        tolerance = float(val)

    model, train_error, val_error = ScikitRegressor.regressor(train_set_x, train_set_y.values, val_set_x,
                                                              val_set_y.values, max_iterations, learning_rate, tolerance)

    print('\nSGDRegressor Scikit Learn:')
    print('Coefficients (model): \n', model.coef_)
    print('Intercept: \n', model.intercept_)
    print('Training Mean squared error: %.2f' % train_error)
    print('Training R^2 score: ', model.score(train_set_x, train_set_y.values))
    print('Validation Mean squared error: %.2f' % val_error)
    print('Validation R^2 score: ', model.score(val_set_x, val_set_y.values))


def normal_equation(train_set_x, train_set_y, val_set_x, val_set_y):

    params, train_error, val_error = NormalEquation.normal_equation(train_set_x, train_set_y.values, val_set_x,
                                                                    val_set_y.values)

    print('\nNormal Equation:')
    print('Coefficients (model): \n', params)
    print('Training Mean squared error: %.2f' % train_error)
    print('Validation Mean squared error: %.2f' % val_error)


def main():
    args = parser.parse_args()

    training_set_x, training_set_y, validation_set_x, validation_set_y = init_dataset(args)

    print('Choose your method:')
    print('1 - Linear Regression with Gradient Descent')
    print('2 - Linear Regression with Scikit SGDRegressor')
    print('3 - Normal Equation')
    print('Anyone - Exit')

    opt = int(input('Option: ')) or 0

    if opt == 1:
        gradient_descent(training_set_x, training_set_y, validation_set_x, validation_set_y)
    elif opt == 2:
        scikit_regressor(training_set_x, training_set_y, validation_set_x, validation_set_y)
    elif opt == 3:
        normal_equation(training_set_x, training_set_y, validation_set_x, validation_set_y)


if __name__ == '__main__':
    main()