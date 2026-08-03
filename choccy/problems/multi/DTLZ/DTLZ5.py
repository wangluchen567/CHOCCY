# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ...problem import Problem


class DTLZ5(Problem):
    def __init__(self, n_vars: Optional[int] = None, n_objs: int = 3):
        """
        DTLZ5

        References:
            Scalable test problems for evolutionary multiobjective optimization,
            K. Deb, L. Thiele, M. Laumanns, and E. Zitzler
        Code References:
            PlatEMO(https://github.com/BIMK/PlatEMO)
        :param n_vars: 决策变量个数
        :param n_objs: 优化目标个数
        """
        if n_vars is None:
            n_vars = n_objs + 9
        super().__init__(self.REAL, n_vars, n_objs, l_bounds=0.0, u_bounds=1.0)

    def calc_objs_mat(self, xs: np.ndarray):
        ms = self.n_objs
        g = np.sum((xs[:, ms - 1:] - 0.5) ** 2, axis=1)
        Temp = np.tile(g, (ms - 2, 1)).T
        xs[:, 1:ms - 1] = (1 + 2 * Temp * xs[:, 1:ms - 1]) / (2 + 2 * Temp)
        objs = (np.tile(1 + g, (ms, 1)).T *
                np.fliplr(np.cumprod(np.hstack((np.ones((g.shape[0], 1), dtype=float),
                                                np.cos(xs[:, :ms - 1] * np.pi / 2))), axis=1)) *
                np.hstack((np.ones((g.shape[0], 1), dtype=float),
                           np.sin(xs[:, ms - 2::-1] * np.pi / 2))))
        return objs

    def get_optimums(self):
        """获取理论最优目标值"""
        ms = self.n_objs
        optimums = np.linspace([0, 1], [1, 0], self.n_samples)  # shape (N, 2)
        optimums = optimums / np.linalg.norm(optimums, axis=1, keepdims=True)  # row-wise normalization
        first_col_repeated = np.repeat(optimums[:, [0]], ms - 2, axis=1)  # shape (N, M-2)
        optimums = np.hstack([first_col_repeated, optimums])  # shape (N, M)
        exponents = np.concatenate([np.array([ms - 2]), np.arange(ms - 2, -1, -1)])
        scaling_factors = (1 / np.sqrt(2)) ** exponents
        optimums = optimums * scaling_factors
        return optimums

    def get_pareto_front(self):
        """获取帕累托最优前沿(以绘图)"""
        if self.n_objs <= 3:
            return self.optimums
        else:
            return None
