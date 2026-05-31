# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ...algorithm import Algorithm
from ....utilities.strategies import search_2opt


class HGATSP(Algorithm):
    def __init__(self,
                 n_sols: int = 100,
                 max_iter: int = 100,
                 cross_prob: Optional[float] = None,
                 mutate_prob: Optional[float] = None,
                 educate_prob: Optional[float] = None,
                 visual_mode: Optional[str] = None):
        """
        混合遗传算法(求解TSP问题)

        Code Maintainer: LuChen Wang
        :param n_sols: 种群大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols, max_iter, cross_prob, mutate_prob, visual_mode)
        self.educate_prob = educate_prob
        self.dist_mat = None  # 距离矩阵
        self.symmetric = True  # 是否是对称矩阵
        # 混合遗传算法仅支持 单目标的 TSP问题优化
        self.single_obj_only = True
        self.supported_var_types = [self.PMU]

    def init_parameters(self):
        super().init_parameters()
        # 初始化教育概率
        if self.educate_prob is None:
            self.educate_prob = 0.5
        # 定义需要的属性
        if hasattr(self.problem, 'dist_mat'):
            # 所有必需属性都存在
            self.dist_mat = self.problem.dist_mat
        else:
            raise AttributeError(
                f"Problem '{type(self.problem).__name__}' missing required 'dist_mat' attribute. "
                f"This algorithm requires a distance matrix. "
            )
        # 检查是否是非对称矩阵
        if hasattr(self.problem, 'symmetric'):
            self.symmetric = self.problem.symmetric

    def run_step(self, i):
        """运行算法单步"""
        # 选择阶段：从当前种群中选择父代个体组成配对池
        parent_indices = self.get_mating_indices()
        # 衍生阶段：对配对池中个体应用交叉和变异生成子代
        offspring_sols = self.apply_operator(parent_indices)
        # 教育阶段：对子代进行教育（等价于局部搜索操作）
        offspring_educated = self.apply_education(offspring_sols)
        # 环境选择阶段：合并父代与子代，选择下一代种群
        self.environmental_selection(offspring_educated)

    def apply_education(self, offspring_sols):
        """对子代进行教育"""
        # 浅拷贝，防止原数据被修改
        new_xs = offspring_sols.xs.copy().astype(int)
        # 逐个按概率对子代进行教育
        for i in range(len(new_xs)):
            if np.random.rand() < self.educate_prob:
                new_xs[i], _, _ = search_2opt(new_xs[i], self.dist_mat, symmetric=self.symmetric)
        return offspring_sols.create_new_with(new_xs)
