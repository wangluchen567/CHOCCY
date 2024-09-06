import numpy as np
from Problems.PROBLEM import PROBLEM


class ZDT6(PROBLEM):
    def __init__(self, num_dec=None, lower=0, upper=1):
        problem_type = 0
        num_obj = 2
        if num_dec is None:
            num_dec = 10
        super().__init__(problem_type, num_dec, num_obj, lower, upper)

    def cal_objs(self, X):
        X = np.array(X)
        n = X.shape[1]
        g = np.sum(X[:, 1:], axis=1)
        g = 1 + 9 * np.power(g / (n - 1), 0.25)
        f1 = 1 - np.exp(-4 * X[:, 0]) * np.power(np.sin(6 * np.pi * X[:, 0]), 6)
        f2 = g * (1 - np.square(f1 / g))
        objs = np.column_stack((f1, f2))
        return objs

    def get_optimum(self, N=1000):
        """获取理论最优目标值"""
        min_f1 = 0.280775
        optimums = np.zeros((N, 2))
        optimums[:, 0] = np.linspace(min_f1, 1, N)
        optimums[:, 1] = 1 - optimums[:, 0] ** 2
        return optimums

    def get_pareto_front(self, N=1000):
        """获取帕累托最优前沿"""
        return self.get_optimum(N)
