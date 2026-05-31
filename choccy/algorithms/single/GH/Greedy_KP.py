# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ...algorithm import Algorithm
from ....solutions import Solutions


class GreedyKP(Algorithm):
    def __init__(self, visual_mode: Optional[str] = None):
        """
        贪婪算法求解0-1背包问题(BinaryKP)

        Code Maintainer: LuChen Wang
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols=1, max_iter=0, visual_mode=visual_mode)
        self.single_obj_only = True
        self.supported_var_types = [self.BIN]
        self.weights = None
        self.profits = None
        self.capacity = None

    def init_parameters(self):
        super().init_parameters()
        # 定义需要的属性
        required_attrs = ['weights', 'profits', 'capacity']
        missing_attrs = [attr for attr in required_attrs if not hasattr(self.problem, attr)]
        if not missing_attrs:
            # 所有必需属性都存在
            self.weights = self.problem.weights.flatten()
            self.profits = self.problem.profits.flatten()
            self.capacity = self.problem.capacity
        else:
            raise ValueError(
                f"This algorithm requires a knapsack problem instance. "
                f"Missing attributes: {', '.join(missing_attrs)}. "
                f"Problem type: {type(self.problem).__name__}"
            )

    def init_solutions(self, *args, **kwargs):
        # 计算每单位重量物品的价值
        unit_cost = self.profits / self.weights
        # 将价值从大到小进行排序，得到排序下标
        sort_indices = np.argsort(-unit_cost)
        # 选择排序下标中求和不超出背包容量的部分
        chosen = sort_indices[np.cumsum(self.weights[sort_indices]) <= self.capacity]
        # 得到贪心算法的解
        self.sols = Solutions(decs=np.where(np.isin(np.arange(len(self.weights)), chosen), 1, 0).astype(int))
        # 设置解集的评估函数
        self.set_evaluate_funcs()
        # 对初始解集进行评估并更新最优解
        self.evaluate_and_update()
