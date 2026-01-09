"""
教育操作(用于混合算法)
Educations Operator

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
from Algorithms.Utility.SupportUtils import two_opt


def educate_tsp(dist_mat: np.ndarray,
                population: np.ndarray,
                educate_prob: float) -> np.ndarray:
    """
    针对指定问题(旅行商, tsp)对子代进行教育
    :param dist_mat: 距离矩阵，形状: (num_dec, num_dec)
    :param population: 原始种群(解集)，形状: (pop_size, num_dec)
    :param educate_prob: 对子代教育的概率，范围: [0, 1]
    :return: 教育后的子代，形状: (pop_size, num_dec)
    """
    # 浅拷贝，防止原数据被修改
    offspring = population.copy()
    # 逐个按概率对子代进行教育
    for i in range(len(offspring)):
        if np.random.rand() < educate_prob:
            offspring[i], _ = two_opt(offspring[i], dist_mat)
    return offspring
