# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

from typing import Optional
from ...algorithm import Algorithm


class GA(Algorithm):
    def __init__(self,
                 n_sols: int = 100,
                 max_iter: int = 100,
                 cross_prob: Optional[float] = None,
                 mutate_prob: Optional[float] = None,
                 visual_mode: Optional[str] = None):
        """
        遗传算法

        Code Maintainer: LuChen Wang
        :param n_sols: 种群大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param mutate_prob: 变异概率
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols, max_iter, cross_prob, mutate_prob, visual_mode)
        # 遗传算法仅支持 单目标的 问题优化
        self.single_obj_only = True

    def run_step(self, i):
        """运行算法单步"""
        # 选择阶段：从当前种群中选择父代个体组成配对池
        parent_indices = self.get_mating_indices()
        # 衍生阶段：对配对池中个体应用交叉和变异生成子代
        offspring_sols = self.apply_operator(parent_indices)
        # 环境选择阶段：合并父代与子代，选择下一代种群
        self.environmental_selection(offspring_sols)
