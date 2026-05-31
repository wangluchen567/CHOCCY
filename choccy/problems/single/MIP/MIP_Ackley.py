# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from ...problem import Problem


class MIP_Ackley(Problem):

    def __init__(self, n_real=15, n_int=15, l_bounds=-32.768, u_bounds=32.768):
        """
        Mixed Integer Ackley's Function

        :param n_real: 决策变量中实数变量的个数
        :param n_int: 决策变量中整数变量的个数
        :param l_bounds: 决策变量下界
        :param u_bounds: 决策变量上界
        """
        n_vars = n_real + n_int
        var_types = np.concatenate([np.zeros(n_real), np.ones(n_int)])
        super().__init__(var_types, n_vars, n_objs=1, l_bounds=l_bounds, u_bounds=u_bounds)

    def calc_objs_mat(self, xs: np.ndarray):
        objs = -20 * np.exp(-0.2 * np.sqrt(np.sum(xs ** 2, axis=1) / self.n_vars)) - np.exp(
            np.sum(np.cos(2 * np.pi * xs), axis=1) / self.n_vars) + 20 + np.e
        return objs

    def get_optimums(self):
        return 0.0
