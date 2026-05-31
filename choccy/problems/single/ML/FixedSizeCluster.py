# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from ...problem import Problem
from ....core import warn_once
from ....utilities.visualization import Frame


class FixedSizeCluster(Problem):
    def __init__(self, n_vars: int = 120, n_types: int = 3, seed: int = 42):
        """
        固定标签数量大小的聚类问题
        :param n_vars: 决策变量个数
        :param n_types: 标签种类个数
        :param seed: 随机种子
        """
        # 继承并初始化父类参数
        super().__init__(Problem.FIX, n_vars, n_objs=1, l_bounds=1, u_bounds=n_types + 1)
        # 标签种类个数
        self.n_types = n_types
        # 设置标签集合
        self.label_set = np.repeat(np.arange(1, self.n_types + 1), int(self.n_vars / self.n_types))
        # 设置随机种子
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        # 随机生成聚类问题数据
        self.points = rng.uniform(0, 1, size=(self.n_vars, 2))

    def calc_objs_mat(self, xs):
        """计算目标值矩阵"""
        n_sols = len(xs)
        objs = np.zeros(n_sols)
        types = np.unique(xs[0])
        points = np.repeat(self.points[np.newaxis, :, :], repeats=n_sols, axis=0)
        for t in types:
            # 得到该类中的所有点
            this_type_points = points[np.where(xs == t)].reshape(n_sols, -1, self.points.shape[-1])
            # 得到该类的中心点坐标
            centroids = np.mean(this_type_points, axis=1)
            # 计算与中心点之间的距离
            distances = np.linalg.norm(this_type_points - centroids[:, np.newaxis, :], axis=-1)
            objs += np.sum(distances, axis=1)
        return objs

    def plot_by_problem(self, n_iter=None, best=None, **kwargs):
        """绘制当前最优聚类结果"""
        if best is None:
            warn_once("Missing required argument: 'best'. "
                      "Please call plot_by_problem(best=your_solution).")
            return None
        frame = Frame()
        frame.add_scatter(x=self.points[:, 0], y=self.points[:, 1], c=best.x, cmap='rainbow')
        frame.set_grid(True, alpha=0.5)
        # 设置标题
        if n_iter is None:
            frame.set_title("FixedSizeCluster")
        else:
            frame.set_title(f"FixedSizeCluster (Iteration {n_iter})")
        return frame
