import numpy as np
from Problems.PROBLEM import PROBLEM
from Algorithms.Utility.Utils import get_uniform_vectors

class DTLZ1(PROBLEM):
    def __init__(self, num_dec=None, num_obj=None, lower=0, upper=1):
        problem_type = PROBLEM.REAL
        if num_obj is None:
            num_obj = 3
        if num_dec is None:
            num_dec = num_obj + 4
        super().__init__(problem_type, num_dec, num_obj, lower, upper)

    def _cal_objs(self, X):
        D = self.num_dec
        M = self.num_obj
        g = 100 * (D - M + 1 + np.sum(
            (X[:, M - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (X[:, M - 1:] - 0.5)), axis=1))
        objs = 0.5 * np.tile(1 + g, (M, 1)).T * np.fliplr(
            np.cumprod(np.hstack((np.ones((X.shape[0], 1)), X[:, :M - 1])), axis=1)) * np.hstack(
            (np.ones((X.shape[0], 1)), 1 - X[:, M - 2::-1]))
        return objs

    def get_optimum(self, N=1000):
        """获取理论最优目标值"""
        optimums = get_uniform_vectors(N, self.num_obj) / 2
        return optimums

    def get_pareto_front(self, N=1000):
        """获取帕累托最优前沿"""
        return self.get_optimum(N)