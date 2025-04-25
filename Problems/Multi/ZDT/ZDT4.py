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


class ZDT4(PROBLEM):
    def __init__(self, num_dec=10):
        """
        ZDT4

        References:
            Comparison of multiobjective evolutionary algorithms: Empirical results,
            E. Zitzler, K. Deb, and L. Thiele
        :param num_dec: 决策变量个数
        """
        num_obj = 2
        lower = np.zeros(num_dec) - 5
        lower[0] = 0
        upper = np.zeros(num_dec) + 5
        upper[0] = 1
        super().__init__(PROBLEM.REAL, num_dec, num_obj, lower, upper)

    def _cal_objs(self, X):
        X = np.array(X)
        g = np.sum(X[:, 1:] ** 2 - 10 * np.cos(4 * np.pi * X[:, 1:]), axis=1)
        g = g + 1 + 10 * (self.num_dec - 1)
        f1 = X[:, 0]
        f2 = g * (1 - np.sqrt(f1 / g))
        objs = np.column_stack((f1, f2))
        return objs

    def get_optimum(self, N=1000):
        """获取理论最优目标值"""
        optimums = np.zeros((N, 2))
        optimums[:, 0] = np.linspace(0, 1, N)
        optimums[:, 1] = 1 - optimums[:, 0] ** 0.5
        return optimums

    def get_pareto_front(self, N=1000):
        """获取帕累托最优前沿(以绘图)"""
        return self.get_optimum(N)
