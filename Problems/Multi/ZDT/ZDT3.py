"""
Copyright (c) 2024 LuChen Wang
CHOCCY is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
import numpy as np
from Problems import PROBLEM
from Algorithms.Utility.Utils import fast_nd_sort


class ZDT3(PROBLEM):
    def __init__(self, num_dec=30):
        """
        ZDT3

        References:
            Comparison of multiobjective evolutionary algorithms: Empirical results,
            E. Zitzler, K. Deb, and L. Thiele
        :param num_dec: 决策变量个数
        """
        super().__init__(PROBLEM.REAL, num_dec, num_obj=2, lower=0, upper=1)

    def _cal_objs(self, X):
        X = np.array(X)
        g = 1 + 9 * np.sum(X[:, 1:], axis=1) / (self.num_dec - 1)
        f1 = X[:, 0]
        f1_div_g = f1 / g
        f2 = g * (1 - np.sqrt(f1_div_g) - f1_div_g * np.sin(10 * np.pi * f1))
        objs = np.column_stack((f1, f2))
        return objs

    def get_optimum(self, N=1000):
        """获取理论最优目标值"""
        optimums = np.zeros((N, 2))
        optimums[:, 0] = np.linspace(0, 1, N)
        optimums[:, 1] = 1 - optimums[:, 0] ** 0.5 - optimums[:, 0] * np.sin(10 * np.pi * optimums[:, 0])
        fronts, ranks = fast_nd_sort(optimums)
        return optimums[fronts[0]]

    def get_pareto_front(self, N=1000):
        """获取帕累托最优前沿(以绘图)"""
        optimums = np.zeros((N, 2))
        optimums[:, 0] = np.linspace(0, 1, N)
        optimums[:, 1] = 1 - optimums[:, 0] ** 0.5 - optimums[:, 0] * np.sin(10 * np.pi * optimums[:, 0])
        fronts, ranks = fast_nd_sort(optimums)
        optimums[ranks > 1] = np.nan
        return optimums
