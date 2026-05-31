# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ....core import record_time
from ...algorithm import Algorithm
from ....solutions import Solutions
from ....utilities.strategies import operator_real
from ....utilities.commons import fast_nd_sort, crowding_dist, composite_rank, leaky_relu, step


class NNDREA(Algorithm):
    def __init__(self,
                 n_sols: int = 100,
                 max_iter: int = 100,
                 cross_prob: Optional[float] = None,
                 mutate_prob: Optional[float] = None,
                 structure: Optional[list] = None,
                 search_range: Optional[tuple] = None,
                 search_ratio: float = 0.5,
                 visual_mode: Optional[str] = None):
        """
        基于神经网络降维的大规模二进制优化算法

        References:
            Neural Network-Based Dimensionality Reduction for LargeScale Binary Optimization With Millions of Variables,
            Ye Tian, Luchen Wang, Shangshang Yang, Jinliang Ding, Yaochu Jin, Xingyi Zhang
        Code Maintainer:
            Luchen Wang
        :param n_sols: 种群大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param structure: 神经网络结构
        :param search_range: 权重搜索范围
        :param search_ratio: 第一阶段搜索占比
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols, max_iter, cross_prob, mutate_prob, visual_mode)
        # 仅支持二进制问题
        self.supported_var_types = [self.BIN]
        self.structure = structure
        self.search_range = search_range
        self.search_ratio = search_ratio
        # 初始化参数
        self.instance = None
        self.weight_sols = None
        self.search_iters = None
        self.layers = []
        self.n_weights = 0
        self.weight_lower_bounds = None
        self.weight_upper_bounds = None
        self.weight_cross_prob = None
        self.weight_mutate_prob = None

    def init_parameters(self):
        """初始化算法参数"""
        super().init_parameters()
        # 验证并获取问题实例
        if not hasattr(self.problem, 'instance'):
            raise AttributeError(
                f"Problem '{type(self.problem).__name__}' missing required 'instance' attribute. "
                f"This algorithm requires a problem instance with an 'instance' attribute."
            )
        # 获取问题实例
        self.instance = self.problem.instance
        # 验证实例维度：至少有一个维度与问题变量数匹配
        if self.instance.shape[0] != self.problem.n_vars and self.instance.shape[1] != self.problem.n_vars:
            raise ValueError(
                f"Instance shape {self.instance.shape} must have at least one dimension "
                f"equal to problem variables count {self.problem.n_vars}."
            )
        # 统一为行优先格式（shape[0] = n_vars）以优化矩阵运算
        if self.instance.shape[1] == self.problem.n_vars:
            self.instance = self.instance.T
        # 对实例中相同的数据进行扰动防止相同数据输出同一个值
        # self.instance += np.random.normal(0, 0.1, self.instance.shape)
        # 获取神经网络的结构信息(若为空则默认为[D, 4, 1])
        if self.structure is None:
            self.structure = [self.instance.shape[1], 4, 1]
        # 根据神经网络结构信息得到神经网络层级列表和权重数量以方便计算
        self.layers = []
        self.n_weights = 0
        for i in range(len(self.structure) - 1):
            self.layers.append([self.structure[i], self.structure[i + 1]])
            self.layers.append([self.structure[i + 1]])
            self.n_weights += self.structure[i] * self.structure[i + 1]
            self.n_weights += self.structure[i + 1]
        # 初始化求解神经网络连续问题的第一阶段算法相关参数
        if self.search_range is None:
            self.search_range = (-1.0, 1.0)
        self.weight_lower_bounds = self.search_range[0] + np.zeros(self.n_weights)
        self.weight_upper_bounds = self.search_range[1] + np.zeros(self.n_weights)
        # 初始化神经网络权重的交叉和变异概率
        self.weight_cross_prob = 1.0
        self.weight_mutate_prob = 1 / self.n_weights
        # 按照第一阶段搜索占比将迭代次数分为两个阶段
        self.search_iters = self.search_ratio * self.max_iter

    def init_solutions(self,
                       seeds: Optional[np.ndarray] = None,
                       shuffle: bool = False):
        """初始化种群"""
        # 初始化神经网络权重解集
        self.weight_sols = Solutions(
            np.random.uniform(self.weight_lower_bounds,
                              self.weight_upper_bounds,
                              size=(self.n_sols, self.n_weights)).astype(np.float64)
        )
        # 使用神经网络计算与映射得到真实二进制解集作为初始种群
        super().init_solutions(seeds=self.mlp_forward(self.weight_sols.xs).astype(int))

    @record_time
    def run_step(self, i: int):
        """运行算法单步"""
        # 首阶段使用神经网络在连续空间中进行搜索
        if i <= self.search_iters:
            # 选择阶段：从当前种群中选择父代个体组成配对池
            parent_indices = self.get_mating_indices()
            # 衍生阶段：对配对池中个体应用交叉和变异生成子代
            offspring_weight_sols = self.apply_operator_weights(parent_indices)
            # 环境选择阶段：合并父代与子代，选择下一代种群
            self.environmental_selection_weights(offspring_weight_sols)
        else:  # 末阶段在原始二进制离散空间中搜索
            # 选择阶段：从当前种群中选择父代个体组成配对池
            parent_indices = self.get_mating_indices()
            # 衍生阶段：对配对池中个体应用交叉和变异生成子代
            offspring_sols = self.apply_operator(parent_indices)
            # 环境选择阶段：合并父代与子代，选择下一代种群
            self.environmental_selection(offspring_sols)

    def apply_operator_weights(self, mating_indices):
        """
        执行算子操作进行生成新一代解集（神经网络权重）
        :param mating_indices: 配对池索引
        :return: 新一代解集（必须由原解集生成）
        """
        # 根据配对池索引创建新解集
        new_weight_sols = self.weight_sols[mating_indices]
        new_weight_sols.xs = operator_real(new_weight_sols.xs,
                                           self.weight_lower_bounds,
                                           self.weight_upper_bounds,
                                           self.weight_cross_prob,
                                           self.weight_mutate_prob)
        return new_weight_sols

    def environmental_selection_weights(self, new_weight_sols: Solutions):
        """
        进行环境选择（神经网络权重）
        :param new_weight_sols: 下一代神经网络权重解集
        """
        # 使用神经网络计算与映射得到真实二进制解集作为新种群
        new_sols = self.sols.create_new_with(
            self.mlp_forward(new_weight_sols.xs).astype(int)
        )
        # 进行全局竞争选择
        better = self.global_selection(new_sols)
        # 将原始解集与新一代解集合并（神经网络权重）
        self.weight_sols = self.weight_sols.concat(new_weight_sols, ignore_warn=True)
        # 全局竞争后的优胜者进入下一代解集（神经网络权重）
        self.weight_sols = self.weight_sols[better]

    def eval_fits(self, sols: Solutions):
        """覆写评估解集的适应度值向量函数"""
        # 若是多目标则使用非支配排序与拥挤度距离计算适应度
        if self.problem.n_objs > 1:
            # 计算带约束惩罚的目标值(若有约束)
            objs = self.eval_penalized_objs(sols)
            # 使用非支配排序与拥挤度距离情况代替原始适应度计算
            fronts, ranks = fast_nd_sort(objs)
            crowd_dist = crowding_dist(objs, fronts)
            return composite_rank(ranks, crowd_dist)
        # 若是单目标则使用经过约束惩罚处理后的目标值计算适应度
        return self.eval_penalized_objs(sols).flatten()

    def mlp_forward(self, weights: np.ndarray):
        """全连接神经网络前向传播函数"""
        n_weights = len(weights)
        ins_size = len(self.instance)
        output = np.array([self.instance]).repeat(n_weights, 0)
        pointer = 0
        for i in range(len(self.layers)):
            if len(self.layers[i]) > 1:
                weight = weights[:, pointer: pointer + self.layers[i][0] * self.layers[i][1]]
                pointer = pointer + self.layers[i][0] * self.layers[i][1]
                output = np.matmul(output, weight.reshape(n_weights, self.layers[i][0], self.layers[i][1]))
            else:
                bias = weights[:, pointer: pointer + self.layers[i][0]]
                pointer = pointer + self.layers[i][0]
                output = output + bias.reshape(n_weights, 1, -1).repeat(ins_size, 1)
                if i == len(self.layers) - 1:
                    output = step(output)
                else:
                    output = leaky_relu(output)
        return output.squeeze()

    def get_config(self) -> dict:
        """获取算法的完整配置"""
        config = super().get_config()
        config['structure'] = self.structure
        config['search_range'] = self.search_range
        config['search_ratio'] = self.search_ratio
        return config
