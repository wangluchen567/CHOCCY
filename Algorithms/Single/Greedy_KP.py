"""
Copyright (c) 2024 LuChen Wang
[Software Name] is licensed under Mulan PSL v2.
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
        *Code Author: Luchen Wang
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
            self.weights = self.problem.weights
            self.values = self.problem.values
            self.capacity = self.problem.capacity
        else:
            raise ValueError("This method can only solve knapsack problems")

    @ALGORITHM.record_time
    def run(self):
        cost = (self.values / self.weights).flatten()
        cost_sort = np.argsort(-cost)
        sum_weight = 0
        chosen = []
        for i in self.get_iterator():
            if sum_weight == self.capacity:
                break
            if sum_weight > self.capacity:
                last = chosen.pop()
                sum_weight -= self.weights[last]
                break
            chosen.append(cost_sort[i])
            sum_weight += self.weights[cost_sort[i]]
        solution = np.zeros(len(self.weights), dtype=int)
        solution[chosen] = 1
        self.pop = np.array([solution])
        self.objs = self.cal_objs(self.pop)
        self.cons = self.cal_cons(self.pop)
        # 清空所有记录后重新记录
        self.clear_record()
        self.record()

    def get_current_best(self):
        self.best, self.best_obj, self.best_con = self.pop[0], self.objs[0], self.cons[0]
