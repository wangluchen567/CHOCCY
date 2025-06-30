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

from Algorithms import ALGORITHM


class Adam(ALGORITHM):
    def __init__(self, max_iter=100, learning_rate=0.1, beta_1=0.9, beta_2=0.99, show_mode=0):
        """
        Adam 梯度下降算法

        Code Author: Luchen Wang
        :param max_iter: 迭代次数
        :param learning_rate: 学习率
        :param beta_1: 历史梯度衰减系数
        :param beta_2: 历史梯度各分量平方衰减系数
        :param show_mode: 绘图模式
        """
        super().__init__(pop_size=1, max_iter=max_iter, show_mode=show_mode)
        self.only_solve_single = True
        self.solvable_type = [self.REAL]
        self.learning_rate = learning_rate
        self.beta_1, self.beta_2 = beta_1, beta_2
        if not (0.0 < self.beta_1 < 1.0):
            raise ValueError("The range of beta_1 must be between 0 and 1")
        if not (0.0 < self.beta_2 < 1.0):
            raise ValueError("The range of beta_2 must be between 0 and 1")
        self.v_ = np.zeros_like(self.pop)  # 初始化一阶矩估计
        self.s_ = np.zeros_like(self.pop)  # 初始化二阶矩估计


    @ALGORITHM.record_time
    def run_step(self, i):
        """运行算法单步"""
        # 计算梯度
        grad = self.cal_grad(self.pop)
        # 更新一阶矩估计
        self.v_ = self.beta_1 * self.v_ + (1 - self.beta_1) * grad
        # 更新二阶矩估计
        self.s_ = self.beta_2 * self.s_ + (1 - self.beta_2) * grad * grad
        # 进行偏差校正
        v_corr = self.v_ / (1 - self.beta_1 ** (i + 1))
        s_corr = self.s_ / (1 - self.beta_2 ** (i + 1))
        # 更新种群解
        self.pop = self.pop - self.learning_rate * v_corr / np.sqrt(s_corr + 1.e-8)
        # 更新种群相关参数
        self.eval_and_update(self.pop)
        # 记录每步状态
        self.record()

