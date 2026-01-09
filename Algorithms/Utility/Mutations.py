"""
变异算子
Mutation Operator

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


def polynomial_mutation(offspring: np.ndarray,
                        lower: np.ndarray,
                        upper: np.ndarray,
                        mutate_prob: float,
                        eta: float = 20.0):
    """
    多项式变异(实数问题)
    :param offspring: 需要变异的(子代)种群(解集)，形状: (pop_size, num_dec)
    :param lower: 取值范围的下界(数组)，可以是1D或2D数组
    :param upper: 取值范围的上界(数组)，可以是1D或2D数组
    :param mutate_prob: 变异概率，范围: [0, 1]
    :param eta: 分布指数，控制变异强度，越大变异越小
    :return: 变异后的子代种群，形状: (pop_size, num_dec)
    """
    pop_size, num_dec = offspring.shape
    # 将边界数组转为边界矩阵
    lowers = lower.reshape(1, -1).repeat(pop_size, 0)
    uppers = upper.reshape(1, -1).repeat(pop_size, 0)
    # 变异掩码与随机数生成
    mask = np.random.random((pop_size, num_dec)) < mutate_prob
    mu = np.random.random((pop_size, num_dec))
    # 情况1：mu <= 0.5
    t = mask * (mu <= 0.5)
    offspring[t] += (uppers[t] - lowers[t]) * (
            (2 * mu[t] + (1 - 2 * mu[t]) * (1 - (offspring[t] - lowers[t]) / (uppers[t] - lowers[t]))
             ** (eta + 1))
            ** (1 / (eta + 1)) - 1)
    # 情况2：mu > 0.5
    t = mask * (mu > 0.5)
    offspring[t] += (uppers[t] - lowers[t]) * (
            1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (uppers[t] - offspring[t]) / (uppers[t] - lowers[t]))
                 ** (eta + 1))
            ** (1 / (eta + 1)))
    return offspring


def differential_mutation(parents: np.ndarray, factor: float):
    """
    差分变异(实数问题)(用于差分进化算法)
    :param parents: 差分变异的种群(目标向量)(多个种群解集)
    :param factor: 缩放因子(差分变异的超参数)，默认值为 0.5
    :return: 变异后的子代种群，形状: (pop_size, num_dec)
    """
    if parents.shape[0] == 3:
        return parents[0] + factor * (parents[1] - parents[2])
    elif parents.shape[0] == 5:
        return parents[0] + factor * (parents[1] - parents[2]) + factor * (parents[3] - parents[4])
    else:
        raise ValueError("The given number of parent populations does not match the required number")


def bit_mutation(offspring: np.ndarray, mutate_prob: float):
    """
    位翻转变异(二进制问题)
    :param offspring: 需要变异的(子代)种群(解集)，形状: (pop_size, num_dec)
    :param mutate_prob: 变异概率，范围: [0, 1]
    :return: 变异后的子代种群，形状: (pop_size, num_dec)
    """
    pop_size, num_dec = offspring.shape
    mask = np.random.rand(pop_size, num_dec) < mutate_prob
    offspring[mask] = 1 - offspring[mask]
    return offspring


def exchange_mutation(offspring: np.ndarray, mutate_prob: float):
    """
    换位变异(序列问题)
    :param offspring: 需要变异的(子代)种群(解集)，形状: (pop_size, num_dec)
    :param mutate_prob: 变异概率，范围: [0, 1]
    :return: 变异后的子代种群，形状: (pop_size, num_dec)
    """
    pop_size, num_dec = offspring.shape
    # 为每个个体生成两个要交换的下标
    exchanges = np.random.randint(num_dec, size=(pop_size, 2))
    # 要满足变异概率才可变异
    mask = np.asarray(np.random.rand(pop_size) < mutate_prob)
    exchanges = exchanges * mask.reshape(-1, 1).repeat(2, axis=1)
    offspring[np.arange(pop_size), exchanges[:, 0]], offspring[np.arange(pop_size), exchanges[:, 1]] \
        = offspring[np.arange(pop_size), exchanges[:, 1]], offspring[np.arange(pop_size), exchanges[:, 0]]
    return offspring


def flip_mutation(offspring: np.ndarray, mutate_prob: float):
    """
    翻转变异(序列问题)
    :param offspring: 需要变异的(子代)种群(解集)，形状: (pop_size, num_dec)
    :param mutate_prob: 变异概率，范围: [0, 1]
    :return: 变异后的子代种群，形状: (pop_size, num_dec)
    """
    pop_size, num_dec = offspring.shape
    # 生成随机的起始和结束索引
    starts = np.random.randint(0, num_dec, size=pop_size)
    ends = np.random.randint(0, num_dec, size=pop_size)
    # 确保start <= end
    starts, ends = np.minimum(starts, ends), np.maximum(starts, ends)
    # 生成列索引网格
    cols = np.arange(num_dec).reshape(1, -1)
    # 计算需要倒置的区域掩码
    mask = (cols >= starts.reshape(-1, 1)) & (cols <= ends.reshape(-1, 1))
    # 计算倒置后的索引
    reversed_indices = starts.reshape(-1, 1) + ends.reshape(-1, 1) - cols
    # 组合索引：在掩码位置使用倒置索引，否则使用原索引
    indices = np.where(mask, reversed_indices, cols)
    # 得到部分片段倒置后的结果
    offspring_ = offspring[np.arange(pop_size).reshape(-1, 1), indices]
    # 要满足变异概率才可变异
    mask = np.random.rand(pop_size) < mutate_prob
    offspring[mask] = offspring_[mask]
    return offspring


def fix_label_mutation(offspring: np.ndarray, mutate_prob: float):
    """
    固定类型数的标签的交换式变异(固定类型数的标签问题)
    :param offspring: 需要变异的(子代)种群(解集)，形状: (pop_size, num_dec)
    :param mutate_prob: 变异概率，范围: [0, 1]
    :return: 变异后的子代种群，形状: (pop_size, num_dec)
    """
    pop_size, num_dec = offspring.shape
    mask = np.random.rand(pop_size, num_dec) < mutate_prob
    need_mutate = np.where(np.sum(mask, axis=1) > 0)[0]
    points = np.random.randint(num_dec, size=(pop_size, 2))
    # 进行交换
    offspring[need_mutate, points[need_mutate, 0]], offspring[need_mutate, points[need_mutate, 1]] \
        = offspring[need_mutate, points[need_mutate, 1]], offspring[need_mutate, points[need_mutate, 0]]
    return offspring
