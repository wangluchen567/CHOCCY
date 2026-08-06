# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ...problem import Problem
from ....utilities.commons import generate_uniform_weights


class DTLZ8(Problem):
    def __init__(self, n_vars: Optional[int] = None, n_objs: int = 3):
        """
        DTLZ8

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
        objs = np.zeros((xs.shape[0], ms))
        for m in range(ms):
            start_idx = m * n_per
            end_idx = (m + 1) * n_per
            objs[:, m] = np.mean(xs[:, start_idx:end_idx], axis=1)
        return objs

    def calc_cons_mat(self, xs: np.ndarray):
        ms = self.n_objs
        objs = self.calc_objs_mat(xs)
        cons = np.zeros((xs.shape[0], ms))
        cons[:, :ms - 1] = 1 - np.tile(objs[:, ms - 1:], (1, ms - 1)) - 4 * objs[:, :ms - 1]
        if ms == 2:
            cons[:, ms - 1] = 0
        else:
            sorted_objs = np.sort(objs[:, :ms - 1], axis=1)
            cons[:, ms - 1] = 1 - 2 * objs[:, ms - 1] - np.sum(sorted_objs[:, :2], axis=1)
        return cons

    def get_optimums(self) -> np.ndarray:
        """获取理论最优目标值"""
        ms = self.n_objs

        # 两目标：生成线性前沿 f1 + f2 = 1/4
        if ms == 2:
            temp = np.linspace(0, 1, self.n_samples).reshape(-1, 1)
            return np.column_stack(((1 - temp) / 4, temp))

        # 高维问题
        n_points = int(np.ceil(self.n_samples / (ms - 1)))
        temp = generate_uniform_weights(n_points, 3, method='nbi')
        temp[:, 2] = temp[:, 2] / 2

        mask = (temp[:, 0] >= (1 - temp[:, 2]) / 4) & \
               (temp[:, 0] <= temp[:, 1]) & \
               (temp[:, 2] <= 1 / 3)
        temp = temp[mask, :]

        if len(temp) == 0:
            return np.empty((0, ms))
        
        opt_list = []
        for i in range(len(temp)):
            for col_idx in range(ms - 1):
                row = [temp[i, 1]] * (ms - 1) + [temp[i, 2]]
                row[col_idx] = temp[i, 0]
                opt_list.append(row)
        optimums = np.array(opt_list)

        # 补充点
        unique_last_col = np.sort(np.unique(optimums[:, ms - 1]))
        if len(unique_last_col) >= 2:
            gap = np.min(np.diff(unique_last_col))
        else:
            gap = 0.01

        temp2 = np.arange(1 / 3, 1 + gap / 2, gap)
        temp2 = temp2[temp2 <= 1.0 + 1e-10]
        if len(temp2) == 0 or temp2[-1] < 0.999:
            temp2 = np.append(temp2, 1.0)
        temp2 = temp2.reshape(-1, 1)

        if len(temp2) > 0:
            extra_points = np.column_stack((
                np.tile((1 - temp2) / 4, (1, ms - 1)),
                temp2
            ))
            optimums = np.vstack((optimums, extra_points))

        return np.unique(optimums, axis=0)