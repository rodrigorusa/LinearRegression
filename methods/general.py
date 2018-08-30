import numpy as np

class CostCalculus:
    @staticmethod
    def cost(coefficients, variables):
        h = coefficients*variables

        return np.sum(h, axis=1)

    @staticmethod
    def compute_error(coefficients, x, y):
        m = x.shape[0]
        h = CostCalculus.cost(coefficients, x)
        tmp = (h - y)*(h - y)

        return np.sum(tmp)/(2.0*m)