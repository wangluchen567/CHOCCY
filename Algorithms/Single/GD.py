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
from Algorithms import ALGORITHM


class GD(ALGORITHM):
    def __init__(self, max_iter=100, learning_rate=0.01, show_mode=0):
        """
        梯度下降算法

        Code Maintainer: Luchen Wang
        :param max_iter: 迭代次数
        :param learning_rate: 学习率
        :param show_mode: 绘图模式
        """
        super().__init__(pop_size=1, max_iter=max_iter, show_mode=show_mode)
        self.only_solve_single = True
        self.solvable_type = [self.REAL]
        self.learning_rate = learning_rate

    @ALGORITHM.record_time
    def run_step(self, i):
        """运行算法单步"""
        # 计算梯度
        grad = self.cal_grad(self.pop)
        # 更新种群解
        self.pop = self.pop - self.learning_rate * grad
        # 更新种群相关参数
        self.eval_and_update(self.pop)
        # 记录每步状态
        self.record()
