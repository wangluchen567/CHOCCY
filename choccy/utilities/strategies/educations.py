# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
教育算子函数集
"""

import numpy as np
from .searching import two_opt


def educate_tsp(dist_mat: np.ndarray,
                offspring: np.ndarray,
                educate_prob: float) -> np.ndarray:
    """
    针对指定问题(旅行商, tsp)对子代进行教育
    :param dist_mat: 距离矩阵，形状: (n_vars, n_vars)
    :param offspring: 需要教育的子代种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param educate_prob: 对子代教育的概率，范围: [0, 1]
    :return: 教育后的子代种群(决策变量矩阵)，形状: (n_sols, n_vars)
    """
    # 逐个按概率对子代进行教育
    for i in range(len(offspring)):
        if np.random.rand() < educate_prob:
            offspring[i], _ = two_opt(offspring[i], dist_mat)
    return offspring
