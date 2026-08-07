# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ...problem import Problem


class DTLZ9(Problem):
    def __init__(self, n_vars: Optional[int] = None, n_objs: int = 2):
        """
        DTLZ9

        References:
            Scalable test problems for evolutionary multiobjective optimization,
            K. Deb, L. Thiele, M. Laumanns, and E. Zitzler
        Code References:
            PlatEMO(https://github.com/BIMK/PlatEMO)
        :param n_vars: 决策变量个数
        :param n_objs: 优化目标个数
        """
        if n_vars is None:
            n_vars = 10 * n_objs
            # 确保决策变量个数是 n_objs 的倍数
        self.n_vars_per_obj = int(np.ceil(n_vars / n_objs))
        n_vars = self.n_vars_per_obj * n_objs
        super().__init__(self.REAL, n_vars, n_objs, l_bounds=0.0, u_bounds=1.0)
        self.n_vars_per_obj = n_vars // n_objs

    def calc_objs_mat(self, xs: np.ndarray):
        ms = self.n_objs
        n_per = self.n_vars_per_obj
        xs = np.clip(xs, self.l_bounds, self.u_bounds)
        xs_ = xs ** 0.1
        objs = np.zeros((xs.shape[0], ms))
        for m in range(ms):
            start_idx = m * n_per
            end_idx = (m + 1) * n_per
            objs[:, m] = np.sum(xs_[:, start_idx:end_idx], axis=1)
        return objs

    def calc_cons_mat(self, xs: np.ndarray):
        ms = self.n_objs
        objs = self.calc_objs_mat(xs)
        cons = 1 - np.tile(objs[:, ms - 1:] ** 2, (1, ms - 1)) - objs[:, :ms - 1] ** 2
        return cons

    def get_optimums(self) -> np.ndarray:
        """获取理论最优目标值"""
        ms = self.n_objs
        temp = np.linspace(0, 1, self.n_samples).reshape(-1, 1)
        optimums = np.column_stack((
            np.tile(np.cos(0.5 * np.pi * temp), (1, ms - 1)),
            np.sin(0.5 * np.pi * temp)
        ))
        return optimums
