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
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Problems import PROBLEM
from scipy.spatial import distance_matrix


class TSP(PROBLEM):
    def __init__(self, num_dec=30, data=None, is_dist_mat=False, round_to_int=False):
        """
        旅行商问题
        :param num_dec: 决策变量个数
        :param data: 给定数据(城市点的坐标位置/各点之间的距离矩阵)
        :param is_dist_mat: 给定的data是否是 各点之间的距离矩阵
        :param round_to_int: 是否将距离矩阵进行取整操作
        """
        problem_type = PROBLEM.PMU
        num_obj = 1
        lower = 0
        upper = num_dec
        super().__init__(problem_type, num_dec, num_obj, lower, upper)
        # 检查给定数据是否为空
        if data is None:
            # 若给定数据为空，则需要先检查是否有数据集
            # 得到项目的根目录
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 2))
            # 保存到Datasets中
            file_name = project_root + "\\Datasets\\Single\\TSP-" + str(self.num_dec) + ".txt"
            if os.path.isfile(file_name):
                self.data = np.loadtxt(file_name, delimiter=',')
            else:
                # 若没有数据集则随机生成数据集并保存
                self.data = np.random.uniform(0, 1, size=(num_dec, 2))
                np.savetxt(file_name, self.data, fmt="%.6e", delimiter=',')
            # 默认数据为点的数据
            is_dist_mat = False
        else:
            self.data = data
        # 给定的data是否是 各点之间的距离矩阵
        if is_dist_mat:
            if self.data.shape[0] != self.data.shape[1]:
                raise ValueError("The given dataset is not a matrix")
            self.points = None
            self.dist_mat = self.data
        else:
            self.points = self.data
            # self.dist_mat = np.linalg.norm(self.points[:, None] - self.points, axis=-1)
            self.dist_mat = distance_matrix(self.data, self.data)
        # 是否将距离矩阵进行取整操作
        if round_to_int:
            self.dist_mat = (self.dist_mat + 0.5).astype(int).astype(np.float64)

    def _cal_objs(self, X):
        # 一行代码得到所有解的目标值
        objs = np.sum(self.dist_mat[X.astype(int), np.roll(X.astype(int), shift=-1, axis=1)], axis=1)
        return objs

    def plot(self, best, n_iter=None, pause=False, pause_time=0.06):
        plt.clf()
        if self.points is None:
            raise ValueError("Not given the position of each point")
        num_points = len(self.points)
        plt.clf()
        graph = nx.Graph()
        graph.add_nodes_from(np.arange(num_points))
        graph.add_edges_from(zip(best, np.roll(best, -1)))
        pos = dict(zip(range(num_points), self.points))
        # 控制点的大小
        if num_points >= 100:
            node_size = 50 / (num_points // 50)
        else:
            node_size = 100
        # 画图
        if n_iter is not None:
            plt.title("iter: " + str(n_iter))
        nx.draw(graph, pos, node_size=node_size)
        if pause:
            plt.pause(pause_time)
        else:
            plt.show()

