# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ....core import warn_once
from ...algorithm import Algorithm
from ....solutions import Solutions


class DPKP(Algorithm):
    def __init__(self, visual_mode: Optional[str] = None):
        """
        动态规划求解0-1背包问题(BinaryKP)

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
            # 检查数据是否都是整数，若不是则进行取整并报警告
            non_integer = any(
                not np.allclose(data, np.round(data)) for data in [self.weights, self.profits, self.capacity])
            if non_integer:
                warn_once("Non-integer values detected. Rounding all data to integers for DP algorithm.")
                self.weights = np.round(self.weights).astype(int)
                self.profits = np.round(self.profits).astype(int)
                self.capacity = int(round(self.capacity))
        else:
            raise ValueError(
                f"This algorithm requires a knapsack problem instance. "
                f"Missing attributes: {', '.join(missing_attrs)}. "
                f"Problem type: {type(self.problem).__name__}"
            )

    def init_solutions(self, *args, **kwargs):
        # 计算物品数量
        num_items = len(self.weights)
        dp = np.zeros(self.capacity + 1, dtype=int)
        # selected 用于记录在每个dp状态下选择的物品
        selected = np.zeros((self.capacity + 1, num_items), dtype=int)
        for item_idx in range(num_items):
            weight = self.weights[item_idx]
            profit = self.profits[item_idx]
            for capacity in range(self.capacity, weight - 1, -1):
                # 不选当前物品的价值收益
                profit_without_current = dp[capacity]
                # 选当前物品的价值收益
                profit_with_current = dp[capacity - weight] + profit
                if profit_without_current < profit_with_current:
                    dp[capacity] = profit_with_current
                    # 复制之前的选择状态并更新
                    selected[capacity] = selected[capacity - weight].copy()
                    selected[capacity, item_idx] = 1
        # 得到动态规划算法的解
        self.sols = Solutions(decs=np.array([selected[self.capacity]], dtype=int))
        # 设置解集的评估函数
        self.set_evaluate_funcs()
        # 对初始解集进行评估并更新最优解
        self.evaluate_and_update()
