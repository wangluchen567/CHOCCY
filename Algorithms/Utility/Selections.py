"""
选择操作
Selections
"""
import numpy as np


def tournament_selection(objs, k=2):
    """
    k元锦标赛选择(单目标)
    :param objs: 种群的目标值向量
    :param k: 参数k(默认值为2)
    :return: 选择的优秀个体组成的交配池
    """
    num_pop = len(objs)
    indices = np.random.randint(0, num_pop, (num_pop, k))
    best = np.argmin(objs.flatten()[indices], axis=1)
    mating_pool = indices[range(num_pop), best]
    return mating_pool

