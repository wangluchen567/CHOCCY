# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
权重分解函数集
"""

import itertools
import numpy as np


def generate_uniform_weights(n_sols: int, n_dims: int) -> np.ndarray:
    """
    获取分解后的均匀分布的权重向量
    :param n_sols: 解数量
    :param n_dims: 维度数
    :return: 权重向量
    """
    ht = 1
    while len(list(itertools.combinations(range(ht + n_dims), n_dims - 1))) <= n_sols:
        ht = ht + 1
    wt = np.array(list(itertools.combinations(range(ht + n_dims - 1), n_dims - 1)))
    wt = wt - np.repeat(np.array([range(n_dims - 1)]), len(wt), axis=0)
    wt = (np.hstack((wt, np.zeros((len(wt), 1)) + ht)) - np.hstack((np.zeros((len(wt), 1)), wt))) / ht
    # 确保所有权重分量严格大于 0
    wt = np.maximum(wt, 1.e-16)
    # 重新归一化使每行和为 1（加 epsilon 后需重新归一）
    wt = wt / np.sum(wt, axis=1, keepdims=True)
    return wt
