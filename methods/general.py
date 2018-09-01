import numpy as np


class CostCalculus:
    @staticmethod
    def cost(params, variables):
        h = params*variables

        return np.sum(h, axis=1)

    @staticmethod
    def compute_error(params, x, y):
        m = x.shape[0]
        h = CostCalculus.cost(params, x)
        tmp = (h - y)*(h - y)

        return np.sum(tmp)/(2.0*m)
