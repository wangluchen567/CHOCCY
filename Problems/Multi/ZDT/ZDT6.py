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


class ZDT6(PROBLEM):
    def __init__(self, num_dec=10):
        """
        ZDT6

        References:
            Comparison of multiobjective evolutionary algorithms: Empirical results,
            E. Zitzler, K. Deb, and L. Thiele
        Code References:
            PlatEMO(https://github.com/BIMK/PlatEMO)
        :param num_dec: 决策变量个数
        """
        super().__init__(PROBLEM.REAL, num_dec, num_obj=2, lower=0, upper=1)

    def _cal_objs(self, X):
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
        """获取帕累托最优前沿(以绘图)"""
        return self.get_optimum(N)
