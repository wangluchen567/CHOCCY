# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

from typing import Optional
from ...algorithm import Algorithm
from ....solutions import Solutions
from ....utilities.commons import fast_nd_sort, crowding_dist, composite_rank


class NSGAII(Algorithm):
    def __init__(self,
                 n_sols: int = 100,
                 max_iter: int = 100,
                 cross_prob: Optional[float] = None,
                 mutate_prob: Optional[float] = None,
                 visual_mode: Optional[str] = None):
        """
        基于快速非支配排序的多目标遗传算法

        References:
            A fast and elitist multi-objective genetic algorithm: NSGA-II,
            K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan
        Code Maintainer:
            LuChen Wang
        :param n_sols: 种群大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols, max_iter, cross_prob, mutate_prob, visual_mode)

    def run_step(self, i):
        """运行算法单步"""
        # 选择阶段：从当前种群中选择父代个体组成配对池
        parent_indices = self.get_mating_indices()
        # 衍生阶段：对配对池中个体应用交叉和变异生成子代
        offspring_sols = self.apply_operator(parent_indices)
        # 环境选择阶段：合并父代与子代，选择下一代种群
        self.environmental_selection(offspring_sols)

    def eval_fits(self, sols: Solutions):
        """覆写评估解集的适应度值向量函数"""
        # 计算带约束惩罚的目标值(若有约束)
        objs = self.eval_penalized_objs(sols)
        # 使用非支配排序与拥挤度距离情况代替原始适应度计算
        fronts, ranks = fast_nd_sort(objs)
        crowd_dist = crowding_dist(objs, fronts)
        fits = composite_rank(ranks, crowd_dist)
        return fits
