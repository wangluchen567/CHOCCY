"""
算子调用
Operators
"""
from .Crossovers import *
from .Mutations import *


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
    num_pop = len(pop)
    if num_pop % 2 == 1:
        # 种群中个体数量必须是偶数
        raise ValueError("The number of individuals in the population must be even")
    offspring = pop.copy()  # 防止影响原数据
    if cross_prob is None:
        cross_prob = 1
    if mutate_prob is None:
        mutate_prob = 1 / offspring.shape[1]
    # 将种群均分为两个父代种群
    parents1 = offspring[:int(num_pop / 2)]
    parents2 = offspring[int(num_pop / 2):]
    # 进行模拟二进制交叉
    offspring = simulated_binary_crossover(parents1, parents2, lower, upper, cross_prob)
    # 进行多项式变异
    offspring = polynomial_mutation(offspring, lower, upper, mutate_prob)
    return offspring


def operator_binary(pop, lower, upper, cross_prob=None, mutate_prob=None):
    """
    对二进制问题进行交叉和变异操作(算子)
    :param pop: 要交叉变异的种群
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param mutate_prob: 变异概率
    :return: 交叉变异得到的子代
    """
    num_pop = len(pop)
    if num_pop % 2 == 1:
        # 种群中个体数量必须是偶数
        raise ValueError("The number of individuals in the population must be even")
    offspring = pop.copy()  # 防止影响原数据
    if cross_prob is None:
        cross_prob = 1
    if mutate_prob is None:
        mutate_prob = 1 / offspring.shape[1]
    # 将种群均分为两个父代种群
    parents1 = offspring[:int(num_pop / 2)]
    parents2 = offspring[int(num_pop / 2):]
    # 均匀二进制交叉
    offspring = binary_crossover(parents1, parents2, cross_prob)
    # 位翻转变异
    offspring = bit_mutation(offspring, mutate_prob)
    return offspring


def operator_permutation(pop, lower, upper, cross_prob=None, mutate_prob=None):
    """
    对序列问题进行交叉和变异操作(算子)
    :param pop: 要交叉变异的种群
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param mutate_prob: 变异概率
    :return: 交叉变异得到的子代
    """
    num_pop = len(pop)
    if num_pop % 2 == 1:
        # 种群中个体数量必须是偶数
        raise ValueError("The number of individuals in the population must be even")
    offspring = pop.copy()  # 防止影响原数据
    if cross_prob is None:
        cross_prob = 1
    if mutate_prob is None:
        mutate_prob = 1 / offspring.shape[1]
    # 将种群均分为两个父代种群
    parents1 = offspring[:int(num_pop / 2)]
    parents2 = offspring[int(num_pop / 2):]
    # 顺序交叉
    offspring = order_crossover(parents1, parents2, cross_prob)
    # 交换式变异
    # offspring = exchange_mutation(offspring, mutate_prob)
    # 翻转式变异
    offspring = flip_mutation(offspring, mutate_prob)
    return offspring


def operator_fix_label(pop, lower, upper, cross_prob=None, mutate_prob=None):
    """
    对固定类型数的标签问题进行交叉和变异操作(算子)
    :param pop: 要交叉变异的种群
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param mutate_prob: 变异概率
    :return: 交叉变异得到的子代
    """
    num_pop = len(pop)
    if num_pop % 2 == 1:
        # 种群中个体数量必须是偶数
        raise ValueError("The number of individuals in the population must be even")
    offspring = pop.copy()  # 防止影响原数据
    if cross_prob is None:
        cross_prob = 1
    if mutate_prob is None:
        mutate_prob = 1 / offspring.shape[1]
    # 将种群均分为两个父代种群
    parents1 = offspring[:int(num_pop / 2)]
    parents2 = offspring[int(num_pop / 2):]
    # 固定类型数的标签的均匀交叉
    offspring = fix_label_crossover(parents1, parents2, cross_prob)
    # 固定类型数的标签的交换式变异
    offspring = fix_label_mutation(offspring, mutate_prob)
    return offspring
