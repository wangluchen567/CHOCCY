# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from ...problem import Problem


class SOP9(Problem):
    def __init__(self, n_vars=30, l_bounds=-5.12, u_bounds=5.12):
        """
        SOP9: Generalized Rastrigin's Function

        References: Evolutionary programming made faster,
        X. Yao, Y. Liu, and G. Lin
        :param n_vars: 决策变量个数
        :param l_bounds: 决策变量下界
        :param u_bounds: 决策变量上界
        """
        super().__init__(self.REAL, n_vars, n_objs=1, l_bounds=l_bounds, u_bounds=u_bounds)

    def calc_objs_mat(self, xs: np.ndarray):
        objs = np.sum(xs ** 2 - 10 * np.cos(2 * np.pi * xs) + 10, axis=1)
        return objs

    def get_optimums(self):
        return 0.0
