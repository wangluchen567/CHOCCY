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


def operator_rand_de(parents, lower, upper, cross_prob=None, mutate_prob=None, factor=None):
    """
    差分进化算子（随机差分）
    :param parents: 要差分的父代
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param mutate_prob: 变异概率
    :param factor: 缩放因子
    :return: 交叉变异得到的子代
    """
    base_parent = parents[0].copy()  # 防止影响原数据
    if cross_prob is None:
        cross_prob = 0.9
    if mutate_prob is None:
        mutate_prob = 1 / base_parent.shape[1]
    if factor is None:
        factor = 0.5
    # 进行差分进化交叉
    if len(parents) == 3:
        # 一个基向量加一对差分向量
        offspring = de_rand_1(base_parent, parents[1], parents[2], lower, upper, cross_prob, factor)
    elif len(parents) == 5:
        # 一个基向量加两对差分向量
        offspring = de_rand_2(base_parent, parents[1], parents[2], parents[3], parents[4],
                              lower, upper, cross_prob, factor)
    else:
        raise ValueError("The number of parents is incorrect")
    # 进行多项式变异
    offspring = polynomial_mutation(offspring, lower, upper, mutate_prob)
    return offspring


def operator_best_de(parents, best, lower, upper, cross_prob=None, mutate_prob=None, factor=None):
    """
    差分进化算子（最优个体差分）
    :param parents: 要差分的父代
    :param best: 父代中最优个体
    :param lower: 取值范围的下界
    :param upper: 取值范围的上界
    :param cross_prob: 交叉概率
    :param mutate_prob: 变异概率
    :param factor: 缩放因子
    :return: 交叉变异得到的子代
    """
    best_ = best.copy()  # 防止影响原数据
    if cross_prob is None:
        cross_prob = 0.9
    if mutate_prob is None:
        mutate_prob = 1 / best_.shape[1]
    if factor is None:
        factor = 0.5
    # 进行差分进化交叉
    if len(parents) == 2:
        # 最优个体加一对差分向量
        offspring = de_best_1(best_, parents[0], parents[1], lower, upper, cross_prob, factor)
    elif len(parents) == 4:
        # 最优个体加两对差分向量
        offspring = de_best_2(best_, parents[0], parents[1], parents[2], parents[3], lower, upper, cross_prob, factor)
    else:
        raise ValueError("The number of parents is incorrect")
    # 进行多项式变异
    offspring = polynomial_mutation(offspring, lower, upper, mutate_prob)
    return offspring
