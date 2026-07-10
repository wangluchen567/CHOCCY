# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ...problem import Problem
from ....utilities.commons import generate_uniform_weights


class DTLZ1(Problem):
    def __init__(self, n_vars: Optional[int] = None, n_objs: int = 3):
        """
        DTLZ1

        References:
            Scalable test problems for evolutionary multiobjective optimization,
            K. Deb, L. Thiele, M. Laumanns, and E. Zitzler
        Code References:
            PlatEMO(https://github.com/BIMK/PlatEMO)
        :param n_vars: 决策变量个数
        :param n_objs: 优化目标个数
        """
        if n_vars is None:
            n_vars = n_objs + 4
        super().__init__(self.REAL, n_vars, n_objs, l_bounds=0.0, u_bounds=1.0)

    def calc_objs_mat(self, xs: np.ndarray):
        m = self.n_objs
        g = 100 * (self.n_vars - m + 1 + np.sum(
            (xs[:, m - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (xs[:, m - 1:] - 0.5)), axis=1))
        objs = 0.5 * np.tile(1 + g, (m, 1)).T * np.fliplr(
            np.cumprod(np.hstack((np.ones((xs.shape[0], 1)), xs[:, :m - 1])), axis=1)) * np.hstack(
            (np.ones((xs.shape[0], 1)), 1 - xs[:, m - 2::-1]))
        return objs

    def get_optimums(self):
        """获取理论最优目标值"""
        return generate_uniform_weights(self.n_samples, self.n_objs) / 2

    def get_pareto_front(self):
        """获取帕累托最优前沿(以绘图)"""
        if self.n_objs == 2:
            return self.optimums
        elif self.n_objs == 3:
            alpha = np.linspace(0, 1, 16).reshape(-1, 1)
            return [
                (alpha @ alpha.T) / 2,
                (alpha @ (1 - alpha.T)) / 2,
                ((1 - alpha) @ np.ones((1, 16))) / 2
            ]
        else:
            return None
