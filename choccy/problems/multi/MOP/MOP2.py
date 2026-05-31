# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from ...problem import Problem


class MOP2(Problem):
    def __init__(self, n_vars=3, l_bounds=-4.0, u_bounds=4.0):
        """
        MOP2

        References: Multi-objective evolutionary algorithm test suites,
        DA Van Veldhuizen, GB Lamont
        :param n_vars: 决策变量个数
        :param l_bounds: 决策变量下界
        :param u_bounds: 决策变量上界
        """
        super().__init__(self.REAL, n_vars=n_vars, n_objs=2, l_bounds=l_bounds, u_bounds=u_bounds)

    def calc_objs_mat(self, xs: np.ndarray):
        f1 = 1 - np.exp(-np.sum((xs - 1 / np.sqrt(self.n_vars)) ** 2, axis=1))
        f2 = 1 - np.exp(-np.sum((xs + 1 / np.sqrt(self.n_vars)) ** 2, axis=1))
        objs = np.column_stack((f1, f2))
        return objs
