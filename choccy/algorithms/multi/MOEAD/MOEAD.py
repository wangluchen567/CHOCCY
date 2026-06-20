# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional, Union
from ...algorithm import Algorithm
from ....types import AggregationMethod
from scipy.spatial import distance_matrix
from ....utilities.commons import aggregate, generate_uniform_weights, calc_penalized_objs


class MOEAD(Algorithm):
    # 枚举聚合方法类型
    PBI = AggregationMethod.PBI  # 基于惩罚边界的聚合方法
    TCH = AggregationMethod.TCH  # 切比雪夫聚合方法
    WSM = AggregationMethod.WSM  # 线性聚合方法

    def __init__(self,
                 n_sols: int = 100,
                 max_iter: int = 100,
                 cross_prob: Optional[float] = None,
                 mutate_prob: Optional[float] = None,
                 agg_method: Union[str, AggregationMethod] = PBI,
                 n_neighbors: Optional[int] = None,
                 visual_mode: Optional[str] = None):
        """
        一种基于分解的多目标进化算法

        References:
            MOEA/D: A multi-objective evolutionary algorithm based on decomposition,
            Q. Zhang and H. Li
        Code Maintainer:
            LuChen Wang
        :param n_sols: 种群大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param agg_method: 聚合方法类型
        :param n_neighbors: 最近邻居个数
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols, max_iter, cross_prob, mutate_prob, visual_mode)
        self.agg_method = agg_method  # 聚合方法类型
        self.n_neighbors = n_neighbors  # 最近邻居个数
        self.weights = None  # 权重向量
        self.neighbor_indices = None  # 邻居向量的下标
        self.ref_point = None  # 参考点向量
        self.max_point = None  # 最大值向量(用于处理约束问题)

    def init_parameters(self):
        """初始化额外参数"""
        super().init_parameters()
        # 选择的最近邻居的数量
        self.n_neighbors = self.n_neighbors if self.n_neighbors else int(np.ceil(self.n_sols / 10))
        # 均匀生成权重向量
        self.weights = generate_uniform_weights(self.n_sols, self.problem.n_objs)
        # 获取每个权重向量的前 n_neighbors 个邻居向量的下标
        self.neighbor_indices = self.find_nearest_neighbors(self.weights, self.n_neighbors)
        # 根据权重向量个数重新确定种群大小(必须匹配)
        self.n_sols = len(self.weights)

    def prepare(self):
        # 初始化参考点
        self.ref_point = np.min(self.sols.objs, axis=0)
        # 初始化最大值点
        self.max_point = np.max(self.sols.objs, axis=0)

    @staticmethod
    def find_nearest_neighbors(weights, t):
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

    def run_step(self, i):
        """运行算法单步"""
        for j in range(self.n_sols):
            # 选择阶段：随机选择两个个体作为父代个体
            parent_indices = self.get_pair_indices(j)
            # 衍生阶段：对选择的个体应用交叉和变异生成子代
            offspring_sols = self.apply_operator(parent_indices)
            # 环境选择阶段：父代与子代竞争选择下一代幸存者
            self.survival_selection(offspring_sols[0], j)
        # 更新解集的最优解信息
        self.update_best()

    def get_pair_indices(self, j):
        """随机选择两个个体作为父代"""
        return np.asarray(np.random.choice(self.neighbor_indices[j], size=2, replace=False))

    def survival_selection(self, offspring, j):
        """进行竞争式环境选择"""
        # 对子代解集进行评估（基础信息）
        offspring.evaluate()
        # 更新参考点
        self.ref_point = np.min((offspring.objs.flatten(), self.ref_point), axis=0)
        # 更新最大值点
        self.max_point = np.max((offspring.objs.flatten(), self.max_point), axis=0)
        # 对新解的所有邻居解进行更新
        neighbors = self.neighbor_indices[j]
        # 将目标值转换为带约束处理后的目标值（无约束则会自动跳过）
        offspring_objs = calc_penalized_objs(offspring.objs,
                                             offspring.cons,
                                             max_objs=self.max_point)
        neighbors_objs = calc_penalized_objs(self.sols.objs[neighbors],
                                             self.sols.cons[neighbors],
                                             max_objs=self.max_point)
        # 使用指定聚合函数计算后选择更优的个体
        better = (aggregate(offspring_objs, self.weights[neighbors], self.ref_point, self.agg_method) <=
                  aggregate(neighbors_objs, self.weights[neighbors], self.ref_point, self.agg_method))
        # 更新种群
        self.sols[neighbors[better]] = offspring

    def get_config(self) -> dict:
        """获取算法的完整配置"""
        config = super().get_config()
        config['aggregation_method'] = self.agg_method
        config['num_neighbors'] = self.n_neighbors
        return config
