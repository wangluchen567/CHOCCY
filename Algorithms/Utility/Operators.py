"""
算子调用
Operators

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
from Algorithms.Utility.Crossovers import *
from Algorithms.Utility.Mutations import *


def operator_real(pop, lower, upper, cross_prob=None, mutate_prob=None):
    """
    对实数问题进行交叉和变异操作(算子)
    :param pop: 要交叉变异的种群
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param mutate_prob: 变异概率
    :return: 交叉变异得到的子代
    """
    # 计算种群一半的数量(奇数向下取整)
    num_half = int(len(pop) / 2)
    # 设置默认交叉与变异概率
    cross_prob = 1.0 if cross_prob is None else cross_prob
    mutate_prob = 1 / pop.shape[1] if mutate_prob is None else mutate_prob
    # 将种群均分为两个父代种群(防止修改原数据)
    parents1 = pop[:num_half].copy()
    parents2 = pop[num_half:num_half * 2].copy()
    # 进行模拟二进制交叉
    offspring = simulated_binary_crossover(parents1, parents2, lower, upper, cross_prob)
    # 进行多项式变异
    offspring = polynomial_mutation(offspring, lower, upper, mutate_prob)
    return offspring


def operator_diff(base_pop, parents, lower, upper, cross_prob=None, factor=None):
    """
    差分进化算子
    :param base_pop: 原始种群
    :param parents: 要差分的父代
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param factor: 缩放因子
    :return: 变异交叉得到的子代
    """
    # 设置默认交叉概率和缩放因子
    cross_prob = 0.2 if cross_prob is None else cross_prob
    factor = 0.5 if factor is None else factor
    # 进行差分变异得到变异子代
    vari_pop = diff_mutation(parents, factor)
    # 进行差分交叉得到最终子代
    offspring = diff_crossover(base_pop, vari_pop, lower, upper, cross_prob)
    return offspring


def operator_binary(pop, lower=None, upper=None, cross_prob=None, mutate_prob=None):
    """
    对二进制问题进行交叉和变异操作(算子)
    :param pop: 要交叉变异的种群
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param mutate_prob: 变异概率
    :return: 交叉变异得到的子代
    """
    # 计算种群一半的数量(奇数向下取整)
    num_half = int(len(pop) / 2)
    # 设置默认交叉与变异概率
    cross_prob = 1.0 if cross_prob is None else cross_prob
    mutate_prob = 1 / pop.shape[1] if mutate_prob is None else mutate_prob
    # 将种群均分为两个父代种群(防止修改原数据)
    parents1 = pop[:num_half].copy()
    parents2 = pop[num_half:num_half * 2].copy()
    # 均匀二进制交叉
    offspring = binary_crossover(parents1, parents2, cross_prob)
    # 位翻转变异
    offspring = bit_mutation(offspring, mutate_prob)
    return offspring


def operator_permutation(pop, lower=None, upper=None, cross_prob=None, mutate_prob=None):
    """
    对序列问题进行交叉和变异操作(算子)
    :param pop: 要交叉变异的种群
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param mutate_prob: 变异概率
    :return: 交叉变异得到的子代
    """
    # 计算种群一半的数量(奇数向下取整)
    num_half = int(len(pop) / 2)
    # 设置默认交叉与变异概率
    cross_prob = 1.0 if cross_prob is None else cross_prob
    mutate_prob = 1 / pop.shape[1] if mutate_prob is None else mutate_prob
    # 将种群均分为两个父代种群(防止修改原数据)
    parents1 = pop[:num_half].copy()
    parents2 = pop[num_half:num_half * 2].copy()
    # 顺序交叉
    offspring = order_crossover(parents1, parents2, cross_prob)
    # # 交换式变异
    # offspring = exchange_mutation(offspring, mutate_prob)
    # 翻转式变异
    offspring = flip_mutation(offspring, mutate_prob)
    return offspring


def operator_fix_label(pop, lower=None, upper=None, cross_prob=None, mutate_prob=None):
    """
    对固定类型数的标签问题进行交叉和变异操作(算子)
    :param pop: 要交叉变异的种群
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param mutate_prob: 变异概率
    :return: 交叉变异得到的子代
    """
    # 计算种群一半的数量(奇数向下取整)
    num_half = int(len(pop) / 2)
    # 设置默认交叉与变异概率
    cross_prob = 1.0 if cross_prob is None else cross_prob
    mutate_prob = 1 / pop.shape[1] if mutate_prob is None else mutate_prob
    # 将种群均分为两个父代种群(防止修改原数据)
    parents1 = pop[:num_half].copy()
    parents2 = pop[num_half:num_half * 2].copy()
    # 固定类型数的标签的均匀交叉
    offspring = fix_label_crossover(parents1, parents2, cross_prob)
    # 固定类型数的标签的交换式变异
    offspring = fix_label_mutation(offspring, mutate_prob)
    return offspring
