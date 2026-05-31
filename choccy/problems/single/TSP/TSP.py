# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
import networkx as nx
from typing import Optional
from ...problem import Problem
from ....core import warn_once
from scipy.spatial import distance_matrix
from ....utilities.visualization import Frame


class TSP(Problem):
    def __init__(self,
                 n_vars: int = 30,
                 locations: Optional[np.ndarray] = None,
                 dist_mat: Optional[np.ndarray] = None,
                 round_dist: bool = False,
                 seed: Optional[int] = 42):
        """
        旅行商问题
        :param n_vars: 决策变量个数(城市点的数量)
        :param locations: 给定城市点的位置坐标
        :param dist_mat: 给定城市点之间的距离矩阵
        :param round_dist: 是否将距离矩阵进行取整操作
        :param seed: 随机给定数据时使用的随机种子
        """
        super().__init__(self.PMU, n_vars, n_objs=1, l_bounds=0, u_bounds=n_vars)
        # 设置随机种子
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        # 初始化locations和dist_mat
        self.locations = locations
        self.dist_mat = dist_mat

        # 情况1：都没有提供，随机生成locations，并根据locations计算dist_mat
        if locations is None and dist_mat is None:
            self.locations = rng.uniform(0, 1, size=(self.n_vars, 2))
            self.dist_mat = distance_matrix(self.locations, self.locations)

        # 情况2：只提供了locations
        elif locations is not None and dist_mat is None:
            # 验证坐标数据
            if self.locations.shape[0] != n_vars:
                raise ValueError(f"Number of locations ({self.locations.shape[0]}) does not match n_vars ({n_vars})")
            # 从坐标计算距离矩阵
            self.dist_mat = distance_matrix(self.locations, self.locations)

        # 情况3：只提供了dist_mat
        elif locations is None and dist_mat is not None:
            # 验证距离矩阵
            self._validate_distance_matrix(dist_mat, n_vars)
            self.dist_mat = dist_mat

        # 情况4：同时提供了locations和dist_mat
        elif locations is not None and dist_mat is not None:
            # 验证坐标数据
            if self.locations.shape[0] != n_vars:
                raise ValueError(f"Number of locations ({self.locations.shape[0]}) does not match n_vars ({n_vars})")
            # 验证距离矩阵
            self._validate_distance_matrix(dist_mat, n_vars)
            self.dist_mat = dist_mat

        # 处理距离取整（如果需要）
        if round_dist:
            self.dist_mat = (self.dist_mat + 0.5).astype(int).astype(np.float64)

    @staticmethod
    def _validate_distance_matrix(dist_mat: np.ndarray, n_vars: int):
        """验证距离矩阵的有效性"""
        # 验证距离矩阵形状
        if dist_mat.shape != (n_vars, n_vars):
            raise ValueError(
                f"Distance matrix shape should be ({n_vars}, {n_vars}), but got {dist_mat.shape}")
        # 验证是否是对称矩阵
        if not np.allclose(dist_mat, dist_mat.T):
            raise ValueError("Distance matrix must be symmetric")
        # 验证对角线是否是全0
        if not np.all(np.diag(dist_mat) == 0):
            raise ValueError("Diagonal of distance matrix must be all zeros")
        # 验证距离矩阵是否非负
        if np.any(dist_mat < 0):
            raise ValueError("Distance matrix cannot contain negative values")

    def calc_objs_mat(self, xs):
        """计算目标值矩阵"""
        # 计算TSP的目标值矩阵：一行代码实现！
        objs = np.sum(self.dist_mat[xs.astype(int), np.roll(xs.astype(int), shift=-1, axis=1)], axis=1)
        return objs

    def plot_by_problem(self, n_iter=None, best=None, **kwargs):
        """绘制当前最优TSP回路"""
        if best is None:
            warn_once("Missing required argument: 'best'. "
                      "Please call plot_by_problem(best=your_solution).")
            return None
        if self.locations is None:
            warn_once("Locations not set during initialization. "
                      "Create problem with TSP(locations=your_data).")
            return None
        # 初始化绘制的帧
        frame = Frame()
        # 创建要绘制的图
        graph = nx.Graph()
        # 计算节点个数
        num_nodes = len(self.locations)
        # 提取最优路线
        best_route = best.x.astype(int)
        graph.add_nodes_from(np.arange(num_nodes))
        graph.add_edges_from(zip(best_route, np.roll(best_route, -1)))
        pos = dict(zip(range(num_nodes), self.locations))
        # 控制点的大小
        node_size = 100 if num_nodes < 100 else 50 / (num_nodes // 50)
        # 绘制TSP结果图
        frame.add_nx_graph(graph, pos, node_size=node_size, with_labels=False)
        # 设置标题
        if n_iter is None:
            frame.set_title("TSP Solution")
        else:
            frame.set_title(f"TSP Solution (Iteration {n_iter})")
        return frame
