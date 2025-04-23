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


def polynomial_mutation(offspring, lower, upper, mutate_prob, eta=20):
    """
    多项式变异(实数问题)
    :param offspring: 交叉得到的子代种群
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param mutate_prob: 变异概率
    :param eta: 超参数
    :return: 变异后的子代种群
    """
    o_size, num_dec = offspring.shape
    if isinstance(lower, int) or isinstance(lower, float):
        lowers = np.ones((o_size, num_dec)) * lower
        uppers = np.ones((o_size, num_dec)) * upper
    else:
        lowers = lower.reshape(1, -1).repeat(o_size, 0)
        uppers = upper.reshape(1, -1).repeat(o_size, 0)
    site = np.random.random((o_size, num_dec)) < mutate_prob
    mu = np.random.random((o_size, num_dec))
    temp = site * (mu <= 0.5)
    offspring[temp] = offspring[temp] + (uppers[temp] - lowers[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (
                1 - (offspring[temp] - lowers[temp]) / (uppers[temp] - lowers[temp])) ** (eta + 1)) ** (
                                                                                     1 / (eta + 1)) - 1)
    temp = site * (mu > 0.5)
    offspring[temp] = offspring[temp] + (uppers[temp] - lowers[temp]) * (1 - (
                2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (
                    1 - (uppers[temp] - offspring[temp]) / (uppers[temp] - lowers[temp])) ** (eta + 1)) ** (
                                                                                     1 / (eta + 1)))
    return offspring


def bit_mutation(offspring, mutate_prob):
    """
    位翻转变异(二进制问题)
    :param offspring: 交叉得到的子代种群
    :param mutate_prob: 变异概率
    :return: 变异后的子代种群
    """
    o_size, num_dec = offspring.shape
    mask = np.random.rand(o_size, num_dec) < mutate_prob
    offspring[mask] = 1 - offspring[mask]
    return offspring


def exchange_mutation(offspring, mutate_prob):
    """
    换位变异(序列问题)
    :param offspring: 交叉得到的子代种群
    :param mutate_prob: 变异概率
    :return: 变异后的子代种群
    """
    o_size, num_dec = offspring.shape
    # 为每个个体生成两个要交换的下标
    exchanges = np.random.randint(num_dec, size=(o_size, 2))
    # 要满足变异概率才可变异
    mask = np.random.rand(o_size) < mutate_prob
    exchanges = exchanges * mask.reshape(-1, 1).repeat(2, axis=1)
    offspring[np.arange(o_size), exchanges[:, 0]], offspring[np.arange(o_size), exchanges[:, 1]] = offspring[
        np.arange(o_size), exchanges[:, 1]], offspring[np.arange(o_size), exchanges[:, 0]]
    return offspring


def flip_mutation(offspring, mutate_prob):
    """
    翻转变异(序列问题)
    :param offspring: 交叉得到的子代种群
    :param mutate_prob: 变异概率
    :return: 变异后的子代种群
    """
    o_size, num_dec = offspring.shape
    # 生成随机的起始和结束索引
    starts = np.random.randint(0, num_dec, size=o_size)
    ends = np.random.randint(0, num_dec, size=o_size)
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
    offspring_ = offspring[np.arange(o_size).reshape(-1, 1), indices]
    # 要满足变异概率才可变异
    mask = np.random.rand(o_size) < mutate_prob
    offspring[mask] = offspring_[mask]
    return offspring


def fix_label_mutation(offspring, mutate_prob):
    """
    固定类型数的标签的交换式变异(固定类型数的标签问题)
    :param offspring: 交叉得到的子代种群
    :param mutate_prob: 变异概率
    :return: 变异后的子代种群
    """
    o_size, num_dec = offspring.shape
    mask = np.random.rand(o_size, num_dec) < mutate_prob
    need_mutate = np.where(np.sum(mask, axis=1) > 0)[0]
    points = np.random.randint(num_dec, size=(o_size, 2))
    # 进行交换
    offspring[need_mutate, points[need_mutate, 0]], offspring[need_mutate, points[need_mutate, 1]] = offspring[
        need_mutate, points[need_mutate, 1]], offspring[need_mutate, points[need_mutate, 0]]
    return offspring
