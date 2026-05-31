# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ...problem import Problem


class MOKP(Problem):
    def __init__(self,
                 n_vars: int = 100,
                 n_objs: int = 2,
                 weights: Optional[np.ndarray] = None,
                 profits: Optional[np.ndarray] = None,
                 capacity: Optional[np.ndarray] = None,
                 seed: Optional[int] = 42):
        """
        背包问题
        :param n_vars: 决策变量个数
        :param weights: 每个物品的重量(若为空则随机给定)
        :param profits: 每个物品的价值(若为空则随机给定)
        :param capacity: 背包的容量(若为空则指定为物品总重量的一半)
        :param seed: 随机给定数据时使用的随机种子
        """
        # 设置随机种子
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        # 参数验证与初始化
        if weights is not None or profits is not None or capacity is not None:
            # 若给定参数均非空，则使用指定参数
            self.weights = weights
            self.profits = profits
            self.capacity = capacity
        elif weights is None and profits is None and capacity is None:
            # 若所有参数均为None，则随机生成
            self.weights = rng.integers(10, 100, size=n_vars)
            self.profits = rng.integers(10, 100, size=(n_vars, n_objs))
            self.capacity = int(self.weights.sum() / 2)
        else:
            provided = []
            if weights is not None:
                provided.append("weights")
            if profits is not None:
                provided.append("profits")
            if capacity is not None:
                provided.append("capacity")
            raise ValueError(
                f"Invalid parameter combination. All three parameters must be either provided or omitted. "
                f"Provided: {', '.join(provided) if provided else 'none'}. "
                f"Missing: {', '.join(['weights', 'profits', 'capacity']) if len(provided) < 3 else 'none'}."
            )
        assert self.weights is not None
        assert self.profits is not None
        assert self.capacity is not None
        # 若给定的数据集不是纵向排布的则进行转换
        if self.weights.ndim == 1:
            self.weights = self.weights.reshape(-1, 1)
        if self.profits.ndim == 1:
            self.profits = self.profits.reshape(-1, 1)
        # 储存实例数据集以便某些特殊算法使用
        self.instance = np.hstack((self.weights, self.profits))
        super().__init__(self.BIN, n_vars, n_objs=n_objs, l_bounds=0, u_bounds=1)

    def calc_objs_mat(self, xs):
        assert self.profits is not None
        objs = np.sum(self.profits, axis=0) - xs.dot(self.profits)
        return objs

    def calc_cons_mat(self, xs):
        assert self.weights is not None
        cons = xs.dot(self.weights) - self.capacity
        return cons

    def get_optimums(self):
        """返回参考点（不是理论最优解）"""
        assert self.profits is not None
        return np.sum(self.profits, axis=0).reshape(1, -1)
