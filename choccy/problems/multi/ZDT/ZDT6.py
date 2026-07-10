# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from ...problem import Problem


class ZDT6(Problem):
    def __init__(self, n_vars: int = 10):
        """
        ZDT6

        References:
            Comparison of multiobjective evolutionary algorithms: Empirical results,
            E. Zitzler, K. Deb, and L. Thiele
        :param n_vars: 决策变量个数
        """
        super().__init__(self.REAL, n_vars, n_objs=2, l_bounds=0.0, u_bounds=1.0)

    def calc_objs_mat(self, xs: np.ndarray):
        g = np.sum(xs[:, 1:], axis=1)
        g = 1 + 9 * np.power(g / (xs.shape[1] - 1), 0.25)
        f1 = 1 - np.exp(-4 * xs[:, 0]) * np.power(np.sin(6 * np.pi * xs[:, 0]), 6)
        f2 = g * (1 - np.square(f1 / g))
        objs = np.column_stack((f1, f2))
        return objs

    def get_optimums(self):
        """获取理论最优目标值"""
        optimums = np.zeros((self.n_samples, 2))
        optimums[:, 0] = np.linspace(0.280775318815370, 1, self.n_samples)
        optimums[:, 1] = 1 - optimums[:, 0] ** 2
        return optimums

    def get_pareto_front(self):
        """获取帕累托最优前沿(以绘图)"""
        return self.optimums
