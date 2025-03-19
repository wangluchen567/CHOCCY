"""
选择操作
Selections
Copyright (c) 2024 LuChen Wang
"""
import numpy as np


def elitist_selection(fitness, num_next=None):
    """
    精英选择策略
    :param fitness: 种群的适应度值(最小化)
    :param num_next: 进入下一步操作的个体数量
    :return: 选择的优秀个体进入下一步操作
    """
    if num_next is None:
        num_next = len(fitness)
    best_indices = np.argsort(fitness)[:num_next]
    return best_indices


def tournament_selection(fitness, num_next=None, k=2):
    """
    k元锦标赛选择
    :param fitness: 种群的适应度值(最小化)
    :param num_next: 进入下一步操作的个体数量
    :param k: 参数k(默认值为2)
    :return: 选择的优秀个体进入下一步操作
    """
    if num_next is None:
        num_next = len(fitness)
    indices = np.random.randint(0, len(fitness), (num_next, k))
    best = np.argmin(fitness.flatten()[indices], axis=1)
    best_indices = indices[range(num_next), best]
    return best_indices


def roulette_selection(fitness, num_next=None, replace=True):
    """
    轮盘赌选择
    :param fitness: 种群的适应度值(最小化)
    :param num_next: 进入下一步操作的个体数量
    :param replace: 是否可以重复抽取选择
    :return: 选择的优秀个体进入下一步操作
    """
    if num_next is None:
        num_next = len(fitness)
    # 对适应度取倒数并进行概率化
    fits = 1 / (fitness + 1e-9)
    prob = fits / np.sum(fits)
    best_indices = np.random.choice(np.arange(len(fitness)), size=num_next,
                                    replace=replace, p=prob.flatten())
    return best_indices
