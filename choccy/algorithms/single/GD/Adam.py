# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ...algorithm import Algorithm


class Adam(Algorithm):
    def __init__(self,
                 max_iter: int = 100,
                 learning_rate: float = 0.01,
                 beta_1: float = 0.9,
                 beta_2: float = 0.99,
                 visual_mode: Optional[str] = None):
        """
        梯度下降算法

        Code Maintainer: LuChen Wang
        :param max_iter: 迭代次数
        :param learning_rate: 学习率
        :param beta_1: 历史梯度衰减系数
        :param beta_2: 历史梯度各分量平方衰减系数
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols=1, max_iter=max_iter, visual_mode=visual_mode)
        self.learning_rate = learning_rate
        self.beta_1, self.beta_2 = beta_1, beta_2
        if not (0.0 < self.beta_1 < 1.0):
            raise ValueError("The range of beta_1 must be between 0 and 1")
        if not (0.0 < self.beta_2 < 1.0):
            raise ValueError("The range of beta_2 must be between 0 and 1")
        self.single_obj_only = True
        self.supported_var_types = [self.REAL]
        self.v_, self.s_ = None, None

    def prepare(self):
        self.v_ = np.zeros_like(self.sols.xs)  # 初始化一阶矩估计
        self.s_ = np.zeros_like(self.sols.xs)  # 初始化二阶矩估计

    def run_step(self, i):
        # 计算梯度
        grad = self.eval_grad(self.sols)
        # 更新一阶矩估计
        self.v_ = self.beta_1 * self.v_ + (1 - self.beta_1) * grad
        # 更新二阶矩估计
        self.s_ = self.beta_2 * self.s_ + (1 - self.beta_2) * grad * grad
        # 进行偏差校正
        v_corr = self.v_ / (1 - self.beta_1 ** (i + 1))
        s_corr = self.s_ / (1 - self.beta_2 ** (i + 1))
        # 更新解
        self.sols.xs -= self.learning_rate * v_corr / np.sqrt(s_corr + 1.e-8)
        # 对新解进行重新评估并更新最优解信息
        self.evaluate_and_update()
