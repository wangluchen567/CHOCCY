# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from ...problem import Problem


class ZDT5(Problem):
    def __init__(self, n_vars: int = 80):
        """
        ZDT5

        References:
            Comparison of multiobjective evolutionary algorithms: Empirical results,
            E. Zitzler, K. Deb, and L. Thiele
        :param n_vars: 决策变量个数
        """
        n_vars = int(np.ceil(max(n_vars - 30, 1) / 5) * 5 + 30)
        super().__init__(self.BIN, n_vars, n_objs=2, l_bounds=0.0, u_bounds=1.0)

    def calc_objs_mat(self, xs: np.ndarray):
        # 计算u向量
        u_temp = np.zeros((xs.shape[0], 1 + (xs.shape[1] - 30) // 5))
        # 第一列是前30列的和
        u_temp[:, 0] = np.sum(xs[:, :30], axis=1)
        # 计算剩余列的u
        for i in range(1, u_temp.shape[1]):
            start_col = (i - 1) * 5 + 30
            end_col = start_col + 5
            u_temp[:, i] = np.sum(xs[:, start_col:end_col], axis=1)
        # 计算 v向量
        v_temp = np.zeros_like(u_temp)
        v_temp[u_temp < 5] = 2 + u_temp[u_temp < 5]
        v_temp[u_temp == 5] = 1
        # 初始化目标值
        objs = np.zeros((len(xs), 2))
        objs[:, 0] = 1 + u_temp[:, 0]  # 第一列目标值
        g = np.sum(v_temp[:, 1:], axis=1)  # g 是 v 的第二列到最后一列的和
        h = 1 / objs[:, 0]  # h 是第一列目标值的倒数
        objs[:, 1] = g * h  # 第二列目标值
        return objs

    def get_optimums(self):
        """获取理论最优目标值"""
        optimums = np.zeros((31, 2))
        optimums[:, 0] = np.arange(1, 32)
        optimums[:, 1] = (self.n_vars - 30) / 5 / optimums[:, 0]
        return optimums

    def get_pareto_front(self):
        """获取帕累托最优前沿(以绘图)"""
        return self.get_optimums()
