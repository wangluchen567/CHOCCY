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
import warnings
import numpy as np
from Algorithms import ALGORITHM


class DPKP(ALGORITHM):
    def __init__(self, show_mode=0):
        """
        动态规划求解背包问题(KP)
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
            self.weights = self.problem.weights.astype(int).flatten()
            self.values = self.problem.values.astype(int).flatten()
            self.capacity = int(self.problem.capacity)
            # 问题中的数据必须是整数
            if not ('int' in self.problem.weights.dtype.name and
                    'int' in self.problem.values.dtype.name and
                    isinstance(self.problem.capacity, int)):
                warnings.warn(
                    "This method can only solve integer type knapsack problems, "
                    "and the provided dataset will be forcibly converted to integer type")
        else:
            raise ValueError("This method can only solve knapsack problems")

    @ALGORITHM.record_time
    def run(self):
        num_items = len(self.weights)
        dp = [0] * (self.capacity + 1)
        # selected 用于记录在每个dp状态下选择的物品
        selected = [[0] * num_items for _ in range(self.capacity + 1)]
        for i in self.get_iterator():
            for j in range(self.capacity, self.weights[i] - 1, -1):
                if dp[j] < dp[j - self.weights[i]] + self.values[i]:
                    dp[j] = dp[j - self.weights[i]] + self.values[i]
                    # 更新选择状态
                    selected[j] = selected[j - self.weights[i]].copy()
                    selected[j][i] = 1
        self.pop = np.array([selected[self.capacity]])
        self.objs = self.cal_objs(self.pop)
        self.cons = self.cal_cons(self.pop)
        # 清空所有记录后重新记录
        self.clear_record()
        self.record()

    def get_current_best(self):
        self.best, self.best_obj, self.best_con = self.pop[0], self.objs[0], self.cons[0]
