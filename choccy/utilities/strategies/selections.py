# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
选择策略函数集
"""

import numpy as np
from typing import Optional


def select_by_elitism(fitness: np.ndarray,
                      next_size: Optional[int] = None) -> np.ndarray:
    """
    精英选择策略
    :param fitness: 种群的适应度值向量(最小化)
    :param next_size: 进入下一步操作的个体数量
    :return: 选择的优秀个体进入下一步操作
    """
    if next_size is None:
        next_size = len(fitness)
    best_indices = np.argsort(fitness)[:next_size]
    return best_indices


def select_by_tournament(fitness: np.ndarray,
                         next_size: Optional[int] = None,
                         k: int = 2) -> np.ndarray:
    """
    k元锦标赛选择
    :param fitness: 种群的适应度值向量(最小化)
    :param next_size: 进入下一步操作的个体数量
    :param k: 参数k(默认值为2)
    :return: 选择的优秀个体进入下一步操作
    """
    if next_size is None:
        next_size = len(fitness)
    indices = np.random.randint(0, len(fitness), (next_size, k))
    best = np.argmin(fitness.flatten()[indices], axis=1)
    best_indices = indices[range(next_size), best]
    return best_indices


def select_by_roulette(fitness: np.ndarray,
                       next_size: Optional[int] = None,
                       replace: bool = True) -> np.ndarray:
    """
    轮盘选择法

    适应度值必须为正数，函数内部取倒数后按概率选择（原值越小被选中的概率越大）
    :param fitness: 种群的适应度值向量(最小化)
    :param next_size: 进入下一步操作的个体数量
    :param replace: 是否可以重复抽取选择
    :return: 选择的优秀个体进入下一步操作
    """
    if next_size is None:
        next_size = len(fitness)
    # 轮盘选择要求适应度值均为正数
    if np.any(fitness <= 0):
        raise ValueError(
            f"Roulette selection requires all fitness values to be positive, "
            f"but got min={np.min(fitness):.6e}, max={np.max(fitness):.6e}. "
            f"Ensure fitness values are positive (e.g., via scaling/shifting) before calling this function."
        )
    # 对适应度取倒数并进行概率化
    recip_fitness = 1 / fitness
    prob = recip_fitness / np.sum(recip_fitness)
    best_indices = np.random.choice(np.arange(len(fitness)), size=next_size,
                                    replace=replace, p=prob.flatten())
    return np.asarray(best_indices)
