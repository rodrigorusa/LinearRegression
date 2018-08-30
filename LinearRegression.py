import argparse

import matplotlib.pyplot as plt
import numpy as np

from methods.linear_regressor import LinearRegressor
from methods.normal_equation import NormalEquation
from methods.scikit_regressor import ScikitRegressor
from reader.csv_reader import DiamondCsvReader

parser = argparse.ArgumentParser(description='Linear Regression.')
parser.add_argument('-training', dest='training_path')


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

    # Normalize data set
    norm = normalize(dataset)

    return norm

def init_dataset(args):
    # Load training set
    df_train = DiamondCsvReader.getDataFrame(args.training_path)

    # Split training data in training(80%) and validation(20%)
    validation_set = df_train.sample(frac=0.2, random_state=1)
    training_set = df_train.drop(validation_set.index)

    print('Training set dimensions:', training_set.shape)
    print('Validation set dimensions:', validation_set.shape)

    # Split training set in variables(x) and target(y)
    training_set_x = training_set.iloc[:, 0:training_set.shape[1] - 1]
    training_set_y = training_set.iloc[:, -1]

    # Data pre-processing
    training_set_x = pre_processing(training_set_x)

    return training_set_x, training_set_y, validation_set

def gradient_descent(training_set_x, training_set_y):
    iterations = int(input('Set iterations: ')) or 1000
    learning_rate = float(input('Set learning rate: ')) or 0.1

    # Gradient descent
    coefficients, error, iter_stop = \
        LinearRegressor.linear_regressor(training_set_x, training_set_y.values, iterations, learning_rate)

    plt.plot(error[0:iter_stop])
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.show()
    plt.savefig('plot.png')

def scikit_regressor(training_set_x, training_set_y):
    iterations = int(input('Set iterations: ')) or 1000
    learning_rate = float(input('Set learning rate: ')) or 0.1

    coefficients = ScikitRegressor.scikit_regressor(training_set_x, training_set_y, iterations, learning_rate)


def normal_equation(training_set_x, training_set_y):
    coefficients = NormalEquation.normal_equation(training_set_x, training_set_y.values)

    training_set_y.values.sort()

    plt.plot(training_set_y.values)
    plt.xlabel('Id')
    plt.ylabel('Price of Diamond')
    plt.show()
    plt.savefig('plot.png')

def main():
    args = parser.parse_args()

    training_set_x, training_set_y, validation_set = init_dataset(args)

    print('Choose your method:')
    print('1 - Linear Regression with Gradient Descent')
    print('2 - Linear Regression with Scikit SGDRegressor')
    print('3 - Normal Equation')
    print('Anyone - Exit')

    opt=int(input('Opt:')) or 0

    if opt == 1:
        gradient_descent(training_set_x, training_set_y)
    elif opt == 2:
        scikit_regressor(training_set_x, training_set_y)
    elif opt == 3:
        normal_equation(training_set_x, training_set_y)

if __name__ == '__main__':
    main()
