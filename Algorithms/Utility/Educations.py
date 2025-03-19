"""
教育操作(用于混合算法)
Educations Operator
Copyright (c) 2024 LuChen Wang
"""
import numpy as np
from Algorithms.Utility.Utils import two_opt


def educate_tsp(problem, offspring, educate_prob):
    """
    根据指定问题(tsp)对子代进行教育
    :param problem: 问题对象
    :param offspring: 子代
    :return: 教育后的子代
    """
    if not hasattr(problem, 'dist_mat'):
        raise ValueError("The problem must provide the distance matrix")
    for i in range(len(offspring)):
        if np.random.rand() < educate_prob:
            offspring[i], _ = two_opt(offspring[i], problem.dist_mat)
    return offspring
