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


class DTLZ5(PROBLEM):
    def __init__(self, num_dec=None, num_obj=3):
        """
        DTLZ5

        References:
            Scalable test problems for evolutionary multiobjective optimization,
            K. Deb, L. Thiele, M. Laumanns, and E. Zitzler
        Code References:
            PlatEMO(https://github.com/BIMK/PlatEMO)
        :param num_dec: 决策变量个数
        :param num_obj: 优化目标个数
        """
        if num_dec is None:
            num_dec = num_obj + 9
        super().__init__(PROBLEM.REAL, num_dec, num_obj, lower=0, upper=1)

    def _cal_objs(self, X):
        M = self.num_obj
        g = np.sum((X[:, M - 1:] - 0.5) ** 2, axis=1)
        Temp = np.tile(g, (M - 2, 1)).T
        X[:, 1:M - 1] = (1 + 2 * Temp * X[:, 1:M - 1]) / (2 + 2 * Temp)
        objs = (np.tile(1 + g, (M, 1)).T *
                  np.fliplr(np.cumprod(np.hstack((np.ones((g.shape[0], 1), dtype=float),
                                                  np.cos(X[:, :M - 1] * np.pi / 2))), axis=1)) *
                  np.hstack((np.ones((g.shape[0], 1), dtype=float),
                             np.sin(X[:, M - 2::-1] * np.pi / 2))))

        return objs

    def get_optimum(self, N=1000):
        """获取理论最优目标值"""
        M = self.num_obj
        optimums = np.linspace([0, 1], [1, 0], N)  # shape (N, 2)
        optimums = optimums / np.linalg.norm(optimums, axis=1, keepdims=True)  # row-wise normalization
        first_col_repeated = np.repeat(optimums[:, [0]], M - 2, axis=1)  # shape (N, M-2)
        optimums = np.hstack([first_col_repeated, optimums])  # shape (N, M)
        exponents = np.concatenate([np.array([M - 2]), np.arange(M - 2, -1, -1)])
        scaling_factors = (1 / np.sqrt(2)) ** exponents
        optimums = optimums * scaling_factors
        return optimums

    def get_pareto_front(self, N=1000):
        """获取帕累托最优前沿(以绘图)"""
        optimums = self.get_optimum(N)
        # 添加微小扰动，以方便以三角绘图
        optimums[:, 0] += np.random.normal(0, 1e-6, size=optimums.shape[0])
        optimums[:, 1] += np.random.normal(0, 1e-6, size=optimums.shape[0])
        return optimums
