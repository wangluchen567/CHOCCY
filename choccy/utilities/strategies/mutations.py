# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
变异算子函数集
"""

import numpy as np
from typing import Union


def polynomial_mutation(offspring: np.ndarray,
                        l_bounds: np.ndarray,
                        u_bounds: np.ndarray,
                        mutate_prob: float,
                        eta: float = 20.0) -> np.ndarray:
    """
    多项式变异(实数问题)
    :param offspring: 需要变异的(子代)种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param mutate_prob: 变异概率，范围: [0, 1]
    :param eta: 分布指数，控制变异强度，越大变异越小
    :return: 变异后的子代种群，形状: (n_sols, n_vars)
    """
    n_sols, n_vars = offspring.shape
    # 将边界数组转为边界矩阵
    lbs = np.broadcast_to(l_bounds, (n_sols, l_bounds.size))
    ubs = np.broadcast_to(u_bounds, (n_sols, u_bounds.size))
    # 变异掩码与随机数生成
    mask = np.random.random((n_sols, n_vars)) < mutate_prob
    mu = np.random.random((n_sols, n_vars))
    # 情况1：mu <= 0.5
    t = mask & (mu <= 0.5)
    offspring[t] += (ubs[t] - lbs[t]) * (
            (2 * mu[t] + (1 - 2 * mu[t]) * (1 - (offspring[t] - lbs[t]) / (ubs[t] - lbs[t]))
             ** (eta + 1))
            ** (1 / (eta + 1)) - 1)
    # 情况2：mu > 0.5
    t = mask & (mu > 0.5)
    offspring[t] += (ubs[t] - lbs[t]) * (
            1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (ubs[t] - offspring[t]) / (ubs[t] - lbs[t]))
                 ** (eta + 1))
            ** (1 / (eta + 1)))
    return offspring


def differential_mutation(parents: np.ndarray, scale_factor: Union[np.ndarray, float]) -> np.ndarray:
    """
    差分变异(实数问题)(用于差分进化算法)
    :param parents: 差分变异的种群(目标向量)(多个种群的决策变量矩阵)
    :param scale_factor: 缩放因子(差分变异的超参数)(标量/数组)，默认值为 0.5
    :return: 变异后的子代种群，形状: (n_sols, n_vars)
    """
    if parents.shape[0] == 3:
        return parents[0] + scale_factor * (parents[1] - parents[2])
    elif parents.shape[0] == 5:
        return parents[0] + scale_factor * (parents[1] - parents[2]) + scale_factor * (parents[3] - parents[4])
    else:
        raise ValueError("The given number of parent populations does not match the required number")


def bit_mutation(offspring: np.ndarray, mutate_prob: float) -> np.ndarray:
    """
    位翻转变异(二进制问题)
    :param offspring: 需要变异的(子代)种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param mutate_prob: 变异概率，范围: [0, 1]
    :return: 变异后的子代种群，形状: (n_sols, n_vars)
    """
    n_sols, n_vars = offspring.shape
    mask = np.random.rand(n_sols, n_vars) < mutate_prob
    offspring[mask] = 1 - offspring[mask]
    return offspring


def exchange_mutation(offspring: np.ndarray, mutate_prob: float) -> np.ndarray:
    """
    换位变异(序列问题)
    :param offspring: 需要变异的(子代)种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param mutate_prob: 变异概率，范围: [0, 1]
    :return: 变异后的子代种群，形状: (n_sols, n_vars)
    """
    n_sols, n_vars = offspring.shape
    # 为每个个体生成两个要交换的下标
    exchanges = np.random.randint(n_vars, size=(n_sols, 2))
    # 要满足变异概率才可变异
    mask = np.asarray(np.random.rand(n_sols) < mutate_prob)
    exchanges = exchanges * mask.reshape(-1, 1).repeat(2, axis=1)
    offspring[np.arange(n_sols), exchanges[:, 0]], offspring[np.arange(n_sols), exchanges[:, 1]] \
        = offspring[np.arange(n_sols), exchanges[:, 1]], offspring[np.arange(n_sols), exchanges[:, 0]]
    return offspring


def flip_mutation(offspring: np.ndarray, mutate_prob: float) -> np.ndarray:
    """
    翻转变异(序列问题)
    :param offspring: 需要变异的(子代)种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param mutate_prob: 变异概率，范围: [0, 1]
    :return: 变异后的子代种群，形状: (n_sols, n_vars)
    """
    n_sols, n_vars = offspring.shape
    # 生成随机的起始和结束索引
    starts = np.random.randint(0, n_vars, size=n_sols)
    ends = np.random.randint(0, n_vars, size=n_sols)
    # 确保start <= end
    starts, ends = np.minimum(starts, ends), np.maximum(starts, ends)
    # 生成列索引网格
    cols = np.arange(n_vars).reshape(1, -1)
    # 计算需要倒置的区域掩码
    mask = (cols >= starts.reshape(-1, 1)) & (cols <= ends.reshape(-1, 1))
    # 计算倒置后的索引
    reversed_indices = starts.reshape(-1, 1) + ends.reshape(-1, 1) - cols
    # 组合索引：在掩码位置使用倒置索引，否则使用原索引
    indices = np.where(mask, reversed_indices, cols)
    # 得到部分片段倒置后的结果
    offspring_ = offspring[np.arange(n_sols).reshape(-1, 1), indices]
    # 要满足变异概率才可变异
    mask = np.random.rand(n_sols) < mutate_prob
    offspring[mask] = offspring_[mask]
    return offspring


def fix_label_mutation(offspring: np.ndarray, mutate_prob: float) -> np.ndarray:
    """
    固定类型数的标签的交换式变异(固定类型数的标签问题)
    :param offspring: 需要变异的(子代)种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param mutate_prob: 变异概率，范围: [0, 1]
    :return: 变异后的子代种群，形状: (n_sols, n_vars)
    """
    n_sols, n_vars = offspring.shape
    # 确定哪些个体需要变异（每个个体一个概率）
    mutate_flags = np.random.rand(n_sols) < mutate_prob
    need_mutate = np.where(mutate_flags)[0]
    if len(need_mutate) == 0:
        return offspring
    # 只为需要变异的个体生成交换位置
    points = np.random.randint(n_vars, size=(len(need_mutate), 2))
    # 执行交换
    offspring[need_mutate, points[:, 0]], offspring[need_mutate, points[:, 1]] = \
        offspring[need_mutate, points[:, 1]], offspring[need_mutate, points[:, 0]]
    return offspring
