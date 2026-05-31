# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from ...problem import Problem
from ....core import warn_once
from ....utilities.visualization import Frame


class Regression(Problem):
    def __init__(self, n_vars: int = 2, data_size: int = 100, seed: int = 42):
        """
        线性回归问题
        :param n_vars: 决策变量个数
        :param data_size: 随机数据集大小
        :param seed: 随机种子
        """
        if n_vars < 2:
            raise ValueError(f"n_vars must be ≥ 2, got {n_vars}. "
                             f"Classification problem needs at least 2 decision variables.")
        # 继承并初始化父类参数
        super().__init__(Problem.REAL, n_vars, n_objs=1, l_bounds=-10, u_bounds=10)
        # 生成的数据集大小
        self.data_size = data_size
        # 设置随机种子
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        # 随机生成回归问题数据
        self.x_data = rng.uniform(0, 10, size=(self.data_size, self.n_vars - 1))
        # 随机生成回归问题权重
        self.real_weights = rng.uniform(-10, 10, size=(self.n_vars, 1))
        # 在数据最后一列添加一列单位矩阵作为偏置b
        self.x_data_b = np.concatenate((self.x_data, np.ones((self.data_size, 1))), axis=1)
        # 得到对应y数据
        self.y_data = self.x_data_b @ self.real_weights
        # 对数据进行扰动
        self.y_data += np.random.normal(0, 1, size=self.y_data.shape)

    def calc_objs_mat(self, xs):
        objs = np.sum((xs @ self.x_data_b.T - self.y_data.T) ** 2, axis=1) / (2 * self.data_size)
        return objs

    def calc_objs_grad_mat(self, xs):
        objs_grad = (xs @ self.x_data_b.T - self.y_data.T) @ self.x_data_b / self.data_size
        return objs_grad

    def plot_by_problem(self, n_iter=None, best=None, **kwargs):
        """绘制当前最优回归结果"""
        if best is None:
            warn_once("Missing required argument: 'best'. "
                      "Please call plot_by_problem(best=your_solution).")
            return None
        if self.n_vars > 3:
            # 只允许绘制 决策变量个数 等于 3 的数据
            warn_once(f"Cannot plot: number of decision variables ({self.n_vars}) exceeds 3. "
                      f"This plotting function only supports visualization for problems with <= 3 variables.")
            return None
        if self.n_vars == 2:
            # 初始化绘制的帧
            frame = Frame()
            # 绘制数据点
            frame.add_scatter(x=self.x_data, y=self.y_data, c='blue')
            # 创建绘制的线的采样
            x_plot = np.linspace(0, 10, 100).reshape(-1, 1)
            # 创建带偏置数据的采样
            xb_plot = np.concatenate((x_plot, np.ones((len(x_plot), 1))), axis=1)
            # 绘制回归结果的直线
            frame.add_line(x_plot, xb_plot @ best.x, c='red')
            # 设置xy轴标签
            frame.set_labels(xlabel='x', ylabel='y')
            # 添加网格
            frame.set_grid(kwargs.get('show_grid', True), alpha=0.5)
        elif self.n_vars == 3:
            # 初始化绘制的帧
            frame = Frame(is_3d=True)
            # 绘制面
            x1_grid, x2_grid = np.meshgrid(np.linspace(0, 10, 100),
                                           np.linspace(0, 10, 100))
            xb_grid = np.stack((x1_grid, x2_grid, np.ones_like(x1_grid)), axis=-1)
            frame.add_surface(x1_grid, x2_grid, xb_grid @ best.x, alpha=0.3, cmap='viridis')
            # 绘制点
            frame.add_scatter(self.x_data[:, 0], self.x_data[:, 1], self.y_data[:, 0], marker="o", c="blue")
            # 设置xyz轴标签
            frame.set_labels(xlabel='x', ylabel='y', zlabel='z')
            frame.set_ticklabel_format(axis='both', style='sci')
        else:
            return None
        # 设置标题
        if n_iter is None:
            frame.set_title("Regression")
        else:
            if self.n_vars == 3:
                frame.set_view(elev=30, azim=(n_iter * 2) % 360)
            frame.set_title(f"Regression (Iteration {n_iter})")
        return frame
