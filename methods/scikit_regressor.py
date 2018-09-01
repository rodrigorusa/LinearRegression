from sklearn import linear_model

from methods.general import CostCalculus


class ScikitRegressor:
    @staticmethod
    def scikit_regressor(x, y, iterations, learning_rate):
        model = linear_model.SGDRegressor(max_iter=iterations, eta0=learning_rate)
        model.fit(x, y)

        coefficients = model.coef_

        print('coefficients:', coefficients)
        print('cost function: ', CostCalculus.compute_error(coefficients, x, y))
        print('R^2: ', model.score(x, y))

        return model