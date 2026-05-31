# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ...problem import Problem
from ....core import warn_once
from ....utilities.visualization import Frame


class Clustering(Problem):
    def __init__(self,
                 points: Optional[np.ndarray] = None,
                 n_points: int = 120,
                 n_features: Optional[int] = None,
                 n_clusters: int = 3,
                 l_bounds: float = 0.0,
                 u_bounds: float = 1.0,
                 seed: int = 42):
        """
        聚类问题：将数据点划分为指定数量的簇。

        :param points: 待聚类的数据点集合，形状为 (n_points, n_features)
        :param n_points: 待聚类点的个数。仅在 points 为 None 时生效
        :param n_features: 数据点的维度。如果 points 为 None 且未提供该参数，则默认为 2
        :param n_clusters: 聚类后簇的个数（即聚类中心的数量）
        :param l_bounds: 随机生成数据点的下界（边界值）
        :param u_bounds: 随机生成数据点的上界（边界值）
        :param seed: 随机种子，用于保证结果可重复。设为 None 时使用系统随机源
        """
        # 处理 points 参数
        if points is not None:
            self.points = np.asarray(points)
            self.n_points, self.n_features = self.points.shape
            # 如果用户同时传入了 points 和显式的 n_features，给予警告
            if n_features is not None and n_features != self.n_features:
                warn_once(
                    f"The provided n_features ({n_features}) does not match the dimensionality of points ({self.n_features}). "
                    f"Using n_features = {self.n_features} from points instead.",
                )
        else:
            # 未提供 points，使用随机生成
            self.n_points = n_points
            self.n_features = n_features if n_features is not None else 2
            self.points = None  # 将在后面生成

        # 保存基本参数
        self.n_clusters = n_clusters

        # 设置随机种子并生成随机数据（如果需要）
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        if self.points is None:
            self.points = rng.uniform(l_bounds, u_bounds, size=(self.n_points, self.n_features))

        # 继承并初始化父类参数
        # 决策变量：n_clusters 个聚类中心，每个中心有 n_features 维
        n_vars = self.n_clusters * self.n_features
        super().__init__(Problem.REAL, n_vars, n_objs=1, l_bounds=l_bounds, u_bounds=u_bounds)


    def calc_obj(self, x: np.ndarray):
        """计算目标值"""
        # 将决策变量重整为聚类中心
        centers = x.reshape(-1, self.n_features)
        # 找到每个数据点最近的聚类中心
        labels = self.assign_clusters(self.points, centers)  # type: ignore
        # 计算总距离（目标值）
        return np.sum((self.points - centers[labels]) ** 2)

    @staticmethod
    def assign_clusters(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        将每个数据点分配到最近的聚类中心。

        对于给定的数据点和聚类中心，计算每个点到所有中心的欧氏距离，
        并将每个点分配给距离最近的中心，返回对应的簇标签。
        :param points: 数据点集合，形状为 (n_samples, n_features)
        :param centers: 聚类中心，形状为 (n_clusters, n_features)
        :return: 簇标签数组，形状为 (n_samples,)，每个元素是 [0, n_clusters-1] 范围内的整数，表示对应数据点所属的簇索引。
        """
        # 计算每个数据点到每个聚类中心的距离 (每个数据归类到最近的聚类中心的那一类)
        distances = np.linalg.norm(points[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=-1)
        # 找到每个数据点最近的聚类中心
        return np.argmin(distances, axis=1)

    def plot_by_problem(self, n_iter=None, best=None, **kwargs):
        """绘制当前最优聚类结果"""
        if best is None:
            warn_once("Missing required argument: 'best'. "
                      "Please call plot_by_problem(best=your_solution).")
            return None

        # 将决策变量重整为聚类中心
        centers = best.x.reshape(-1, self.n_features)
        # 找到每个数据点最近的聚类中心
        labels = self.assign_clusters(self.points, centers)  # type: ignore

        if self.n_features == 2:
            frame = Frame()
            frame.add_scatter(x=centers[:, 0], y=centers[:, 1], c='black', marker='x')
            frame.add_scatter(x=self.points[:, 0], y=self.points[:, 1], c=labels, cmap='rainbow')
            frame.set_grid(True, alpha=0.5)
        elif self.n_features == 3:
            frame = Frame(is_3d=True)
            frame.add_scatter(x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], c='black', marker='x')
            frame.add_scatter(x=self.points[:, 0], y=self.points[:, 1], z=self.points[:, 2], c=labels, cmap='rainbow')
        else:
            warn_once(f"Cannot visualize: n_features must be 2 or 3, but got {self.n_features}.")
            return None
        # 设置标题
        if n_iter is None:
            frame.set_title("Clustering")
        else:
            frame.set_title(f"Clustering (Iteration {n_iter})")
        return frame
