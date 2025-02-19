import numpy as np
from Problems.PROBLEM import PROBLEM


class Ackley(PROBLEM):
    def __init__(self, num_dec=None, lower=-32, upper=32):
        problem_type = PROBLEM.REAL
        num_obj = 1
        if num_dec is None: num_dec = 30
        super().__init__(problem_type, num_dec, num_obj, lower, upper)

    def _cal_objs(self, X):
        objs = -20 * np.exp(-0.2 * np.sqrt(np.sum(X ** 2, -1) / self.num_dec)) - np.exp(
            np.sum(np.cos(2 * np.pi * X), -1) / self.num_dec) + 20 + np.e
        return objs
