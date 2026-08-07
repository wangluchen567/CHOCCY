# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from ...problem import Problem


class MOP3(Problem):
    def __init__(self, l_bounds: float = -3.14, u_bounds: float = 3.14):
        """
        MOP3 (Poloni)

        References:
            Multi-objective evolutionary algorithm test suites,
            CAC Coello, GB Lamont, DAV Veldhuizen
        :param l_bounds: 决策变量下界
        :param u_bounds: 决策变量上界
        """
        super().__init__(self.REAL, n_vars=2, n_objs=2, l_bounds=l_bounds, u_bounds=u_bounds)

    def calc_objs_mat(self, xs: np.ndarray):
        """计算目标值矩阵"""
        x, y = xs[:, 0], xs[:, 1]
        a1 = 0.5 * np.sin(1) - 2 * np.cos(1) + 1 * np.sin(2) - 1.5 * np.cos(2)
        a2 = 1.5 * np.sin(1) - 1 * np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)
        b1 = 0.5 * np.sin(x) - 2 * np.cos(x) + 1 * np.sin(y) - 1.5 * np.cos(y)
        b2 = 1.5 * np.sin(x) - 1 * np.cos(x) + 2 * np.sin(y) - 0.5 * np.cos(y)
        f1 = -(1 + (a1 - b1) ** 2 + (a2 - b2) ** 2)
        f2 = -((x + 3) ** 2 + (y + 1) ** 2)
        objs = np.column_stack((f1, f2))
        return objs
