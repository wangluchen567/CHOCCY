import numpy as np
from Problems.PROBLEM import PROBLEM


class Square(PROBLEM):
    def __init__(self, num_dec=30, lower=-10, upper=10):
        problem_type = PROBLEM.REAL
        num_obj = 1
        super().__init__(problem_type, num_dec, num_obj, lower, upper)

    def _cal_objs(self, X):
        objs = np.sum(X**2, axis=-1)
        return objs
