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


class Classification(PROBLEM):
    def __init__(self, num_dec=3, data_size=100):
        """
        线性分类问题 (逻辑回归)
        :param num_dec: 决策变量个数
        :param data_size: 随机的数据集大小
        """
        if num_dec < 3:
            raise ValueError("The number of decision variables in Regression cannot be less than 3")
        # 继承并初始化父类参数
        super().__init__(PROBLEM.REAL, num_dec, num_obj=1, lower=-10, upper=10)
        # 生成的数据集大小
        self.data_size = data_size
        # 随机生成分类问题数据
        self.x_data = np.random.uniform(-1, 1, size=(self.data_size, self.num_dec - 1))
        # 得到生成数据的中值
        x_mid = (np.max(self.x_data, axis=0) + np.min(self.x_data, axis=0)) / 2
        # 随机生成分类权重
        self.real_weights = np.random.uniform(-1, 1, size=(self.num_dec - 1, 1))
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
        # 初始化一个极小值防止指数溢出
        self.eps = 1.e-15

    def _cal_objs(self, X):
        y_predict = self.sigmoid(X @ self.x_data_b.T)
        y_predict = np.clip(y_predict, 1.e-15, 1 - 1.e-15)  # 防止数值下溢
        objs = -np.mean(self.y_data.T * np.log(y_predict) + (1 - self.y_data).T * np.log(1 - y_predict), axis=1)
        return objs

    def _cal_objs_grad(self, X):
        y_predict = self.sigmoid(X @ self.x_data_b.T)
        y_predict = np.clip(y_predict, 1.e-15, 1 - 1.e-15)  # 防止数值下溢
        objs_grad = (y_predict - self.y_data.T) @ self.x_data_b / self.data_size
        return objs_grad

    @staticmethod
    def sigmoid(x):
        """sigmoid函数"""
        # 防止指数溢出
        indices_pos = np.nonzero(x >= 0)
        indices_neg = np.nonzero(x < 0)
        y = np.zeros_like(x)
        # y = 1 / (1 + exp(-x)), x >= 0
        # y = exp(x) / (1 + exp(x)), x < 0
        y[indices_pos] = 1 / (1 + np.exp(-x[indices_pos]))
        y[indices_neg] = np.exp(x[indices_neg]) / (1 + np.exp(x[indices_neg]))
        return y

    def plot(self, best, n_iter=None, pause=False, pause_time=0.06):
        if self.num_dec > 3:  # 若决策变量超过3个则不进行绘制
            return
        plt.clf()
        ax = plt.subplot(111, projection='3d')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
        # 绘制面
        x1_grid, x2_grid = np.meshgrid(np.linspace(-1, 1, 100),
                                       np.linspace(-1, 1, 100))
        x_grid_b = np.stack((x1_grid, x2_grid, np.ones_like(x1_grid)), axis=-1)
        ax.plot_surface(x1_grid, x2_grid, self.sigmoid(x_grid_b @ best), alpha=0.3, cmap='viridis')
        # 绘制点
        positive, negative = np.array(self.y_data == 1).flatten(), np.array(self.y_data == 0).flatten()
        ax.scatter(self.x_data[positive, 0], self.x_data[positive, 1], self.y_data[positive, 0], marker="o", c="red")
        ax.scatter(self.x_data[negative, 0], self.x_data[negative, 1], self.y_data[negative, 0], marker="o", c="blue")
        ax.view_init(elev=30, azim=(n_iter * 2) % 360)  # 更新图形的旋转角度
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # 画图
        if n_iter is not None:
            plt.title("iter: " + str(n_iter))
        if pause:
            plt.pause(pause_time)
        else:
            plt.show()
