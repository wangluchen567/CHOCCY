# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0
import numpy as np
from ...problem import Problem


class MOP1(Problem):
    def __init__(self, l_bounds: float = -1000.0, u_bounds: float = 1000.0):
        """
        MOP1 (Schaffer 1)

        References:
            Multi-objective evolutionary algorithm test suites,
            CAC Coello, GB Lamont, DAV Veldhuizen
        :param l_bounds: 决策变量下界
        :param u_bounds: 决策变量上界
        """
        super().__init__(self.REAL, n_vars=1, n_objs=2, l_bounds=l_bounds, u_bounds=u_bounds)

    def calc_objs_mat(self, xs: np.ndarray):
        f1 = xs ** 2
        f2 = (xs - 2) ** 2
        objs = np.column_stack((f1, f2))
        return objs

    def get_optimums(self):
        """获取理论最优目标值"""
        optimums = np.zeros((self.n_samples, 2))
        optimums[:, 0] = np.linspace(0, 4, self.n_samples)
        optimums[:, 1] = (np.sqrt(optimums[:, 0]) - 2) ** 2
        return optimums

    def get_pareto_front(self):
        """获取帕累托最优前沿(以绘图)"""
        return self.optimums
