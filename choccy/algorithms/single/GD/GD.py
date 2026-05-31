# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

from typing import Optional
from ...algorithm import Algorithm


class GradientDecent(Algorithm):
    def __init__(self,
                 max_iter: int = 100,
                 learning_rate: float = 0.01,
                 visual_mode: Optional[str] = None):
        """
        梯度下降算法

        Code Maintainer: LuChen Wang
        :param max_iter: 迭代次数
        :param learning_rate: 学习率
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols=1, max_iter=max_iter, visual_mode=visual_mode)
        self.learning_rate = learning_rate
        self.single_obj_only = True
        self.supported_var_types = [self.REAL]

    def run_step(self, i):
        # 计算梯度
        grad = self.eval_grad(self.sols)
        # 根据梯度与学习率 更新解
        self.sols.xs -= self.learning_rate * grad
        # 对新解进行重新评估并更新最优解信息
        self.evaluate_and_update()
