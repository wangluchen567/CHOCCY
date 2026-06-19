# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from ...problem import Problem


class MIP_Rastrigin(Problem):
    def __init__(self, n_real=10, n_int=15, l_bounds=-5.12, u_bounds=5.12):
        """
        Mixed Integer Generalized Rastrigin's Function

        References: Evolutionary programming made faster,
        X. Yao, Y. Liu, and G. Lin
        :param n_vars: 决策变量个数
        :param l_bounds: 决策变量下界
        :param u_bounds: 决策变量上界
        """
        n_vars = n_real + n_int
        var_types = np.concatenate([np.zeros(n_real) + self.REAL, np.zeros(n_int) + self.INT])
        super().__init__(var_types, n_vars, n_objs=1, l_bounds=l_bounds, u_bounds=u_bounds)

    def calc_objs_mat(self, xs: np.ndarray):
        objs = np.sum(xs ** 2 - 10 * np.cos(2 * np.pi * xs) + 10, axis=1)
        return objs

    def get_optimums(self):
        return 0.0
