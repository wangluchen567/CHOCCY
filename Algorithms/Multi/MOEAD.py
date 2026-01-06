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
from typing import Optional
from Algorithms import ALGORITHM
from scipy.spatial import distance_matrix
from Algorithms.Utility.SupportUtils import get_uniform_vectors


class MOEAD(ALGORITHM):
    PBI = 1  # 基于惩罚边界的聚合方法
    TCH = 2  # 切比雪夫聚合方法
    WSM = 3  # 线性聚合方法

    def __init__(self,
                 pop_size: Optional[int] = None,
                 max_iter: Optional[int] = None,
                 cross_prob: Optional[float] = None,
                 mutate_prob: Optional[float] = None,
                 agg_type: int = PBI,
                 num_near: Optional[int] = None,
                 show_mode: Optional[str] = None):
        """
        一种基于分解的多目标进化算法

        References:
            MOEA/D: A multi-objective evolutionary algorithm based on decomposition,
            Q. Zhang and H. Li
        Code Maintainer:
            Luchen Wang
        :param pop_size: 种群大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param agg_type: 聚合函数类型
        :param num_near: 最近邻居的数量
        :param show_mode: 绘图模式
        """
        # 初始化相关参数(调用父类初始化)
        super().__init__(pop_size, max_iter, cross_prob, mutate_prob, None, show_mode)
        # 聚合函数类型
        self.agg_type = agg_type
        self.num_near = num_near
        self.vectors = None
        self.indexes = None
        self.ref = None

    @ALGORITHM.record_time
    def init_algorithm(self, problem):
        """初始化算法"""
        # 选择的最近邻居的数量
        self.num_near = self.num_near if self.num_near else int(np.ceil(self.pop_size / 10))
        # 均匀生成权重向量
        self.vectors = get_uniform_vectors(self.pop_size, problem.num_obj)
        # 获取每个权重向量的前T个邻居向量的下标
        self.indexes = self.get_neighbor_index(self.vectors, self.num_near)
        # 根据权重向量个数重新确定种群大小(必须匹配)
        self.pop_size = len(self.vectors)
        # 调用父类的初始化函数
        super().init_algorithm(problem)

    def init_and_eval_pop(self, pop=None):
        super().init_and_eval_pop(pop)
        # 初始化参考点
        self.ref = np.min(self.objs, axis=0)

    @staticmethod
    def get_neighbor_index(weights, t):
        """
        获取每个权重向量的前T个邻居向量的下标
        :param weights: 权重向量
        :param t: 最近邻居的数量
        :return: 前t个邻居向量的下标
        """
        # 计算欧式距离矩阵
        dist_mat = distance_matrix(weights, weights)
        # 获取前T个最近的邻居的下标
        return np.argsort(dist_mat, axis=1)[:, :t]

    @ALGORITHM.record_time
    def run_step(self, i):
        """运行算法单步"""
        for j in range(self.pop_size):
            # 选择阶段：随机选择两个个体作为父代个体
            parent_indices = self.get_pair_indices(j)
            # 衍生阶段：对选择的个体应用交叉和变异生成子代
            offspring = self.apply_operator(parent_indices)
            # 环境选择阶段：父代与子代竞争选择下一代幸存者
            self.survival_selection(offspring[0], j)
        # 记录每步状态
        self.record()

    def get_pair_indices(self, j):
        """随机选择两个个体作为父代"""
        return np.random.choice(self.indexes[j], size=2, replace=False)

    def survival_selection(self, offspring, j):
        """进行环境选择(分解式)"""
        # 计算该子代的目标值
        offspring_obj = self.cal_objs(offspring)
        offspring_con = self.cal_cons(offspring)
        # 更新参考点
        self.ref = np.min((offspring_obj.flatten(), self.ref), axis=0)
        # 对新解的所有邻居解进行更新
        neighbors = self.indexes[j]
        np.random.shuffle(neighbors)  # 打乱邻居解
        # 使用指定聚合函数计算后选择更优的个体
        better = (self.aggregate(self.vectors[neighbors], offspring_obj) <=
                  self.aggregate(self.vectors[neighbors], self.objs[neighbors]))
        # 更新种群和目标值
        self.pop[neighbors[better]] = offspring
        self.objs[neighbors[better]] = offspring_obj
        self.cons[neighbors[better]] = offspring_con

    def aggregate(self, vectors, objs):
        """
        聚合函数
        :param vectors: 聚合向量
        :param objs: 聚合目标
        :return: 聚合结果
        """
        # 改变形状方便矩阵运算
        if objs.ndim == 1:
            objs = objs.reshape(1, -1)
        if self.agg_type == self.PBI:
            # 基于惩罚边界的聚合方法
            theta = 5  # 设置超参数
            if len(objs) == 1:
                # 若是单个个体则直接求
                d1 = np.abs(np.dot(objs - self.ref, vectors.T)).flatten() / np.linalg.norm(vectors, axis=1)
                d2 = np.linalg.norm(objs - (self.ref + d1.reshape(-1, 1) * vectors), axis=1)
            else:
                # 若是需要对整个邻居目标值则使用取对角线方法
                d1 = np.abs(np.diag(np.dot(objs - self.ref, vectors.T))) / np.linalg.norm(vectors, axis=1)
                d2 = np.linalg.norm(objs - (self.ref + d1.reshape(-1, 1) * vectors), axis=1)
            return d1 + theta * d2
        elif self.agg_type == self.TCH:
            # 切比雪夫聚合方法
            return np.max(vectors * np.abs(objs - self.ref), axis=1)
        elif self.agg_type == self.WSM:
            # 线性聚合方法
            if len(objs) == 1:
                # 若是单个个体则直接求点积
                return np.dot(objs, vectors.T).flatten()
            else:
                # 若是需要对整个邻居目标值则使用 取对角线方法
                return np.diag(np.dot(objs, vectors.T)).flatten()
                # 使用爱因斯坦求和约定对矩阵逐行求点积
                # return np.einsum('ij,ij->i', objs, vectors).flatten()
        else:
            raise ValueError("There is no such aggregate function type")

    def get_params_info(self):
        """获取参数信息"""
        info = super().get_params_info()
        types = ['', 'Penalty Boundary', 'Tchebycheff', 'Weighted Sum']
        info['agg_type'] = types[self.agg_type]
        return info
