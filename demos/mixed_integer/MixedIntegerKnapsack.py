# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from choccy.problems import Problem


class MixedIntegerKnapsack(Problem):
    def __init__(self,
                 n_real: int = 10,
                 n_int: int = 10,
                 n_bin: int = 10,
                 seed: int = 42):
        """
        混合整数背包问题 (MIP-KP)

        场景：工厂采购原材料
            - 原材料A（连续）：可以买任意实数吨，价格与购买量成正比
            - 原材料B（整数）：必须买整数箱（每箱固定规格）
            - 原材料C（二进制）：要么不买，要么买一个固定批次（可能有折扣）
        :param n_real: 连续变量个数
        :param n_int: 整数变量个数
        :param n_bin: 二进制变量个数
        :param seed: 随机给定数据时使用的随机种子
        """
        self.n_real = n_real
        self.n_int = n_int
        self.n_bin = n_bin
        n_vars = self.n_real + self.n_int + self.n_bin
        var_types = np.hstack([[self.REAL] * self.n_real,
                              [self.INT] * self.n_int,
                              [self.BIN] * self.n_bin])
        # 设置随机种子
        rng = np.random.default_rng(seed)
        # 生成价值和重量以及容量
        self.profits = rng.uniform(10, 100, n_vars)
        self.weights = rng.uniform(10, 100, n_vars)
        self.capacity = int(self.weights.sum() / 2)
        u_bounds = np.concatenate([
            np.full(self.n_real, self.capacity / (2 * self.weights[:self.n_real].mean() + 1e-8)),
            np.full(self.n_int, int(self.capacity / self.weights.mean()) if np.mean(self.weights) > 0 else 10),
            np.ones(self.n_bin)
        ])
        super().__init__(var_types, n_vars, n_objs=1, l_bounds=0, u_bounds=u_bounds)
        # 储存实例数据集以便某些特殊算法使用
        self.instance = np.hstack((self.weights, self.profits))

    def calc_objs_mat(self, xs):
        objs = np.sum(self.u_bounds * self.profits) - xs.dot(self.profits)
        return objs

    def calc_cons_mat(self, xs):
        cons = xs.dot(self.weights) - self.capacity
        return cons
