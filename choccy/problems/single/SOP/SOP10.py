# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from ...problem import Problem


class SOP10(Problem):

    def __init__(self, n_vars=30, l_bounds=-32.768, u_bounds=32.768):
        """
        SOP10: Ackley's Function

        References: Evolutionary programming made faster,
        X. Yao, Y. Liu, and G. Lin
        :param n_vars: 决策变量个数
        :param l_bounds: 决策变量下界
        :param u_bounds: 决策变量上界
        """
        super().__init__(self.REAL, n_vars, n_objs=1, l_bounds=l_bounds, u_bounds=u_bounds)

    def calc_objs_mat(self, xs: np.ndarray):
        objs = -20 * np.exp(-0.2 * np.sqrt(np.sum(xs ** 2, axis=1) / self.n_vars)) - np.exp(
            np.sum(np.cos(2 * np.pi * xs), axis=1) / self.n_vars) + 20 + np.e
        return objs

    def get_optimums(self):
        return 0.0
