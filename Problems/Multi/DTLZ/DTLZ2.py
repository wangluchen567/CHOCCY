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
from Algorithms.Utility.SupportUtils import get_uniform_vectors

class DTLZ2(PROBLEM):
    def __init__(self, num_dec=None, num_obj=3):
        """
        DTLZ2

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
        objs = (np.tile(1 + g, (M, 1)).T *
                  np.fliplr(np.cumprod(np.hstack((np.ones((g.shape[0], 1), dtype=float),
                                                  np.cos(X[:, :M - 1] * np.pi / 2))), axis=1)) *
                  np.hstack((np.ones((g.shape[0], 1), dtype=float),
                             np.sin(X[:, M - 2::-1] * np.pi / 2))))
        return objs

    def get_optimum(self, N=1000):
        """获取理论最优目标值"""
        optimums = get_uniform_vectors(N, self.num_obj)
        optimums = optimums / np.sqrt(np.sum(optimums**2, axis=1, keepdims=True))
        return optimums

    def get_pareto_front(self, N=1000):
        """获取帕累托最优前沿(以绘图)"""
        return self.get_optimum(N)