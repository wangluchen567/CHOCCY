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


class GreedyKP(ALGORITHM):
    def __init__(self, show_mode=0):
        """
        贪婪算法求解背包问题(KP)

        Code Maintainer: Luchen Wang
        :param show_mode: 绘图模式
        """
        super().__init__(pop_size=1, max_iter=None, show_mode=show_mode)
        self.only_solve_single = True
        self.solvable_type = [self.BIN]
        self.weights = None
        self.values = None
        self.capacity = None

    @ALGORITHM.record_time
    def init_algorithm(self, problem, pop=None):
        super().init_algorithm(problem, pop)
        # 初始化迭代次数
        self.max_iter = self.num_dec
        # 问题必须为背包问题
        if hasattr(self.problem, 'weights') and hasattr(self.problem, 'values') and hasattr(self.problem, 'capacity'):
            self.weights = self.problem.weights.flatten()
            self.values = self.problem.values.flatten()
            self.capacity = self.problem.capacity
        else:
            raise ValueError("This method can only solve knapsack problems")

    @ALGORITHM.record_time
    def run(self):
        # 计算每单位重量物品的价值
        cost = self.values / self.weights
        # 将价值从大到小进行排序，得到排序下标
        sort_indices = np.argsort(-cost)
        # 选择排序下标中求和不超出背包容量的部分
        chosen = sort_indices[np.cumsum(self.weights[sort_indices]) <= self.capacity]
        # 初始化解
        solution = np.zeros(len(self.weights), dtype=int)
        # 根据选择的下标为解进行赋值
        solution[chosen] = 1
        self.pop = np.array([solution])
        self.eval_and_update(self.pop)
        # 清空所有记录后重新记录
        self.clear_record()
        self.record()

    def get_current_best(self):
        self.best, self.best_obj, self.best_con = self.pop[0], self.objs[0], self.cons[0]
