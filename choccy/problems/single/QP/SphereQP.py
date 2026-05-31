# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from ...problem import Problem


class SphereQP(Problem):
    def __init__(self, n_vars=10, l_bounds=0, u_bounds=10):
        """
        最简单的不等式约束 QP

        min   sum(x_i^2)
        s.t.  sum(x_i) >= 1
              x_i >= 0

        :param n_vars: 决策变量个数
        """
        super().__init__(self.REAL, n_vars, n_objs=1, l_bounds=l_bounds, u_bounds=u_bounds)

    def calc_objs_mat(self, xs: np.ndarray):
        return np.sum(xs ** 2, axis=1)

    def calc_cons_mat(self, xs: np.ndarray):
        return 1 - np.sum(xs, axis=1)  # sum(x) >= 1

    def get_optimums(self):
        return 1 / self.n_vars
