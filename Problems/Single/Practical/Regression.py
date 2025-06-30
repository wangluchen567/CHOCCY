"""
Copyright (c) 2024 LuChen Wang
CHOCCY is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
import numpy as np
from Problems import PROBLEM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Regression(PROBLEM):
    def __init__(self, num_dec=2, data_size=100):
        if num_dec < 2:
            raise ValueError("The number of decision variables in Regression cannot be less than 2")
        # 继承并初始化父类参数
        super().__init__(PROBLEM.REAL, num_dec, num_obj=1, lower=-100, upper=100)
        # 生成的数据集大小
        self.data_size = data_size
        # 随机生成回归问题数据
        self.x_data = np.random.uniform(0, 10, size=(self.data_size, self.num_dec - 1))
        # 随机生成回归权重
        self.real_weights = np.random.uniform(-10, 10, size=(self.num_dec, 1))
        # 在数据最后一列添加一列单位矩阵作为偏置b
        self.x_data_b = np.concatenate((self.x_data, np.ones((self.data_size, 1))), axis=1)
        # 得到对应y数据
        self.y_data = self.x_data_b @ self.real_weights
        # 对数据进行扰动
        self.y_data += np.random.normal(0, 1, size=self.y_data.shape)

    def _cal_objs(self, X):
        objs = np.sum((X @ self.x_data_b.T - self.y_data.T) ** 2, axis=1) / (2 * self.data_size)
        return objs

    def _cal_objs_grad(self, X):
        objs_grad = (X @ self.x_data_b.T - self.y_data.T) @ self.x_data_b / self.data_size
        return objs_grad

    def plot(self, best, n_iter=None, pause=False, pause_time=0.06):
        if self.num_dec > 3:  # 若决策变量超过3个则不进行绘制
            return
        plt.clf()
        if self.num_dec == 2:
            plt.scatter(self.x_data, self.y_data, c='blue')
            x_plot = np.linspace(0, 10, 100).reshape(-1, 1)
            x_plot_b = np.concatenate((x_plot, np.ones((len(x_plot), 1))), axis=1)
            plt.plot(x_plot, x_plot_b @ best, c='red')
            plt.xlabel('x')
            plt.ylabel('y')
        elif self.num_dec == 3:
            ax = plt.subplot(111, projection='3d')
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
            ax.scatter(self.x_data[:, 0], self.x_data[:, 1], self.y_data[:, 0], marker="o", c="blue")
            x1_grid, x2_grid = np.meshgrid(np.linspace(0, 10, 10), np.linspace(0, 10, 10))
            x_grid_b = np.concatenate((x1_grid[:, :, np.newaxis], x2_grid[:, :, np.newaxis],
                                       np.ones((x1_grid.shape[0], x1_grid.shape[1], 1))), axis=-1)
            ax.plot_surface(x1_grid, x2_grid, x_grid_b @ best, alpha=0.2, color='red')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        else:
            raise ValueError("The number of decision variables exceeds 3 and cannot be plotted")
        # 画图
        if n_iter is not None:
            plt.title("iter: " + str(n_iter))
        if pause:
            plt.pause(pause_time)
        else:
            plt.show()
