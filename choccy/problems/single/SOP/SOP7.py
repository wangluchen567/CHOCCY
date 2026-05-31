# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from ...problem import Problem


class SOP7(Problem):
    def __init__(self, n_vars=30, l_bounds=-1.28, u_bounds=1.28):
        """
        SOP7: Quartic Function with Noise

        References: Evolutionary programming made faster,
        X. Yao, Y. Liu, and G. Lin
        :param n_vars: 决策变量个数
        :param l_bounds: 决策变量下界
        :param u_bounds: 决策变量上界
        """
        super().__init__(self.REAL, n_vars, n_objs=1, l_bounds=l_bounds, u_bounds=u_bounds)

    def calc_objs_mat(self, xs: np.ndarray):
        objs = (np.sum(np.arange(1, self.n_vars + 1)[np.newaxis, :].repeat(len(xs), axis=0) * xs ** 4, axis=1) +
                np.random.rand(xs.shape[0]))
        return objs

    def get_optimums(self):
        return 0.0
