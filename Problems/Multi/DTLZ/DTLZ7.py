import numpy as np
from Problems.PROBLEM import PROBLEM


class DTLZ7(PROBLEM):
    def __init__(self, num_dec=None, num_obj=None, lower=0, upper=1):
        problem_type = PROBLEM.REAL
        if num_obj is None:
            num_obj = 3
        if num_dec is None:
            num_dec = num_obj + 19
        super().__init__(problem_type, num_dec, num_obj, lower, upper)

    def _cal_objs(self, X):
        M = self.num_obj
        PopObj = np.zeros((X.shape[0], M))
        g = 1 + 9 * np.mean(X[:, M - 1:], axis=1)
        PopObj[:, :M - 1] = X[:, :M - 1]
        term1 = PopObj[:, :M - 1] / (1 + np.tile(g, (M - 1, 1)).T)
        term2 = 1 + np.sin(3 * np.pi * PopObj[:, :M - 1])
        PopObj[:, M - 1] = (1 + g) * (M - np.sum(term1 * term2, axis=1))
        return PopObj
