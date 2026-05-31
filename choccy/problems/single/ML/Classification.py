# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from ...problem import Problem
from ....core import warn_once
from ....utilities.commons import sigmoid
from ....utilities.visualization import Frame


class Classification(Problem):
    def __init__(self, n_vars: int = 3, data_size: int = 100, seed: int = 42):
        """
        线性分类问题 (使用逻辑回归算法)
        :param n_vars: 决策变量个数
        :param data_size: 随机数据集大小
        :param seed: 随机种子
        """
        if n_vars < 3:
            raise ValueError(f"n_vars must be ≥ 3, got {n_vars}. "
                             f"Classification problem needs at least 3 decision variables.")
        # 继承并初始化父类参数
        super().__init__(Problem.REAL, n_vars, n_objs=1, l_bounds=-10, u_bounds=10)
        # 生成的数据集大小
        self.data_size = data_size
        # 设置随机种子
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        # 随机生成分类问题数据
        self.x_data = rng.uniform(-1, 1, size=(self.data_size, self.n_vars - 1))
        # 得到生成数据的中值
        x_mid = (np.max(self.x_data, axis=0) + np.min(self.x_data, axis=0)) / 2
        # 随机生成分类权重
        self.real_weights = rng.uniform(-1, 1, size=(self.n_vars - 1, 1))
        # 得到真实偏置
        bias = -x_mid.reshape(1, -1) @ self.real_weights
        # 将权重和偏置进行合并
        self.real_weights = np.concatenate((self.real_weights, bias), axis=0)
        # 在数据最后一列添加一列单位矩阵作为偏置b
        self.x_data_b = np.concatenate((self.x_data, np.ones((self.data_size, 1))), axis=1)
        # 初始化y数据
        self.y_data = np.ones((len(self.x_data), 1))
        # 标记出负例
        self.y_data[self.x_data_b @ self.real_weights < 0] = 0

    def calc_objs_mat(self, xs):
        y_predict = sigmoid(xs @ self.x_data_b.T)
        y_predict = np.clip(y_predict, 1.e-15, 1 - 1.e-15)  # 防止数值下溢
        objs = -np.mean(self.y_data.T * np.log(y_predict) + (1 - self.y_data).T * np.log(1 - y_predict), axis=1)
        return objs

    def calc_objs_grad_mat(self, xs):
        y_predict = sigmoid(xs @ self.x_data_b.T)
        y_predict = np.clip(y_predict, 1.e-15, 1 - 1.e-15)  # 防止数值下溢
        objs_grad = (y_predict - self.y_data.T) @ self.x_data_b / self.data_size
        return objs_grad

    def plot_by_problem(self, n_iter=None, best=None, **kwargs):
        """绘制当前最优分类结果"""
        if best is None:
            warn_once("Missing required argument: 'best'. "
                      "Please call plot_by_problem(best=your_solution).")
            return None
        if self.n_vars > 3:
            # 只允许绘制 决策变量个数 等于 3 的数据
            warn_once(f"Cannot plot: number of decision variables ({self.n_vars}) exceeds 3. "
                      f"This plotting function only supports visualization for problems with 3 variables.")
            return None
        # 初始化绘制的帧
        frame = Frame(is_3d=True)
        # 绘制面
        x1_grid, x2_grid = np.meshgrid(np.linspace(-1, 1, 100),
                                       np.linspace(-1, 1, 100))
        xb_grid = np.stack((x1_grid, x2_grid, np.ones_like(x1_grid)), axis=-1)
        frame.add_surface(x1_grid, x2_grid, sigmoid(xb_grid @ best.x), alpha=0.3, cmap='viridis')
        # 绘制点
        positive, negative = np.array(self.y_data == 1).flatten(), np.array(self.y_data == 0).flatten()
        frame.add_scatter(self.x_data[positive, 0], self.x_data[positive, 1], self.y_data[positive, 0],
                          marker="o", c="red")
        frame.add_scatter(self.x_data[negative, 0], self.x_data[negative, 1], self.y_data[negative, 0],
                          marker="o", c="blue")
        # 设置xyz轴标签
        frame.set_labels(xlabel='x', ylabel='y', zlabel='z')
        frame.set_ticklabel_format(axis='both', style='sci')
        # 设置标题
        if n_iter is None:
            frame.set_title("Classification")
        else:
            frame.set_view(elev=30, azim=(n_iter * 2) % 360)  # 更新图形的旋转角度
            frame.set_title(f"Classification (Iteration {n_iter})")
        return frame
