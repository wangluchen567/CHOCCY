# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ...problem import Problem
from ....utilities.commons import fast_nd_sort
from ....utilities.commons import generate_uniform_weights


class DTLZ7(Problem):
    def __init__(self, n_vars: Optional[int] = None, n_objs: int = 3):
        """
        DTLZ7

        References:
            Scalable test problems for evolutionary multiobjective optimization,
            K. Deb, L. Thiele, M. Laumanns, and E. Zitzler
        Code References:
            PlatEMO(https://github.com/BIMK/PlatEMO)
        :param n_vars: 决策变量个数
        :param n_objs: 优化目标个数
        """
        if n_vars is None:
            n_vars = n_objs + 19
        super().__init__(self.REAL, n_vars, n_objs, l_bounds=0.0, u_bounds=1.0)

    def calc_objs_mat(self, xs: np.ndarray):
        ms = self.n_objs
        g = 1 + 9 * np.mean(xs[:, ms - 1:], axis=1)
        objs = np.zeros((xs.shape[0], ms))
        objs[:, :ms - 1] = xs[:, :ms - 1]
        temp_sum = np.sum(
            objs[:, :ms - 1] / (1 + g.reshape(-1, 1)) * (1 + np.sin(3 * np.pi * objs[:, :ms - 1])),
            axis=1
        )
        objs[:, ms - 1] = (1 + g) * (ms - temp_sum)
        return objs

    def get_optimums(self):
        """获取理论最优目标值"""
        ms = self.n_objs
        # 定义分段线性映射的区间边界
        interval = np.array([0, 0.251412, 0.631627, 0.859401])
        # 计算中位数分割点（用于将 [0,1] 映射到两个不同的区间）
        median = (interval[1] - interval[0]) / (
                interval[3] - interval[2] + interval[1] - interval[0]
        )
        # 在 (M-1) 维超立方体中生成均匀网格点
        xs = generate_uniform_weights(self.n_samples, ms - 1, method='grid')
        # 数值裁剪，避免浮点误差导致 xs 略超出 [0,1] 范围
        xs = np.clip(xs, 0, 1)
        # 分段线性映射
        # 左半部分映射到 [interval[0], interval[1]]
        xs[xs <= median] = xs[xs <= median] * (interval[1] - interval[0]) / median + interval[0]
        # 右半部分映射到 [interval[2], interval[3]]
        xs[xs > median] = (xs[xs > median] - median) * (interval[3] - interval[2]) / (1 - median) + interval[2]
        # 获得最优解集
        optimums = np.column_stack((
            xs, 2 * (ms - np.sum(xs / 2 * (1 + np.sin(3 * np.pi * xs)), axis=1, keepdims=True))
        ))
        return optimums

    def get_pareto_front(self):
        """获取帕累托最优前沿(以绘图)"""
        if self.n_objs == 2:
            x = np.linspace(0, 1, 100).reshape(-1, 1)
            y = np.asarray(2 * (2 - x / 2 * (1 + np.sin(3 * np.pi * x))))
            points = np.column_stack((x, y))
            fronts, ranks = fast_nd_sort(points)
            x[ranks > 1] = np.nan
            y[ranks > 1] = np.nan
            return np.column_stack((x, y))
        elif self.n_objs == 3:
            x_grid, y_grid = np.meshgrid(np.linspace(0, 1, 30),
                                         np.linspace(0, 1, 30))
            z_grid = np.asarray(2 * (3 - x_grid / 2 * (1 + np.sin(3 * np.pi * x_grid)) -
                                y_grid / 2 * (1 + np.sin(3 * np.pi * y_grid))))
            points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
            fronts, ranks = fast_nd_sort(points)
            z_grid[ranks.reshape(z_grid.shape) > 1] = np.nan
            return [x_grid, y_grid, z_grid]
        else:
            return None
