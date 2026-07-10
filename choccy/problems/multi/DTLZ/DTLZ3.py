# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ...problem import Problem
from ....utilities.commons import generate_uniform_weights


class DTLZ3(Problem):
    def __init__(self, n_vars: Optional[int] = None, n_objs: int = 3):
        """
        DTLZ3

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
        g = 100 * (self.n_vars - ms + 1 +
                   np.sum((xs[:, ms - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (xs[:, ms - 1:] - 0.5)), axis=1))
        objs = (np.tile(1 + g, (ms, 1)).T *
                np.fliplr(np.cumprod(np.hstack((np.ones((xs.shape[0], 1), dtype=float),
                                                np.cos(xs[:, :ms - 1] * np.pi / 2))), axis=1)) *
                np.hstack((np.ones((xs.shape[0], 1), dtype=float),
                           np.sin(xs[:, ms - 2::-1] * np.pi / 2))))
        return objs

    def get_optimums(self):
        """获取理论最优目标值"""
        optimums = generate_uniform_weights(self.n_samples, self.n_objs)
        optimums = optimums / np.sqrt(np.sum(optimums ** 2, axis=1, keepdims=True))
        return optimums

    def get_pareto_front(self):
        """获取帕累托最优前沿(以绘图)"""
        if self.n_objs == 2:
            return self.optimums
        elif self.n_objs == 3:
            theta = np.linspace(0, np.pi / 2, 16).reshape(-1, 1)
            return [
                np.sin(theta) @ np.cos(theta.T),
                np.sin(theta) @ np.sin(theta.T),
                np.cos(theta) @ np.ones((1, 16))
            ]
        else:
            return None
