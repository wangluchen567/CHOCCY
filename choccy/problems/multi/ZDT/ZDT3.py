# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from ...problem import Problem
from ....utilities.commons import fast_nd_sort


class ZDT3(Problem):
    def __init__(self, n_vars: int = 30):
        """
        ZDT3

        References:
            Comparison of multiobjective evolutionary algorithms: Empirical results,
            E. Zitzler, K. Deb, and L. Thiele
        :param n_vars: 决策变量个数
        """
        super().__init__(self.REAL, n_vars, n_objs=2, l_bounds=0.0, u_bounds=1.0)

    def calc_objs_mat(self, xs: np.ndarray):
        g = 1 + 9 * np.sum(xs[:, 1:], axis=1) / (self.n_vars - 1)
        f1 = xs[:, 0]
        f1_div_g = f1 / g
        f2 = g * (1 - np.sqrt(f1_div_g) - f1_div_g * np.sin(10 * np.pi * f1))
        objs = np.column_stack((f1, f2))
        return objs

    def get_optimums(self):
        """获取理论最优目标值"""
        optimums = np.zeros((self.n_optimums, 2))
        optimums[:, 0] = np.linspace(0, 1, self.n_optimums)
        optimums[:, 1] = 1 - optimums[:, 0] ** 0.5 - optimums[:, 0] * np.sin(10 * np.pi * optimums[:, 0])
        fronts, ranks = fast_nd_sort(optimums)
        return optimums[fronts[0]]

    def get_pareto_front(self):
        """获取帕累托最优前沿(以绘图)"""
        optimums = np.zeros((self.n_pareto, 2))
        optimums[:, 0] = np.linspace(0, 1, self.n_pareto)
        optimums[:, 1] = 1 - optimums[:, 0] ** 0.5 - optimums[:, 0] * np.sin(10 * np.pi * optimums[:, 0])
        fronts, ranks = fast_nd_sort(optimums)
        optimums[ranks > 1] = np.nan
        return optimums
