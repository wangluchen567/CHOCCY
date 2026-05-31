# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
收敛指标计算函数集
"""

import numpy as np
from ...core import warn_once
from scipy.spatial.distance import cdist


def calc_gd(objs: np.ndarray, optimums: np.ndarray) -> float:
    """
    计算代际距离指标(Generational Distance)
    :param objs: 目标值
    :param optimums: 理论最优目标值
    :return: 代际距离指标值
    """
    if objs.shape[1] != optimums.shape[1]:
        raise ValueError("The objs does not match the dimension of the optimal targets")
    if len(optimums) == 1:
        warn_once("Only one theoretical optimal solution has been provided, "
                  "which may be a reference point. Please use HV to calculate the score")
    # 计算给定目标值中每一行与最优目标值中每一行之间的欧式距离
    distance_matrix = cdist(objs, optimums, metric='euclidean')
    # 按行取最小值，得到每个点到最近最优点的距离
    distance = np.min(distance_matrix, axis=1)
    # 计算得到分数值
    score = np.mean(distance)
    return score


def calc_igd(objs: np.ndarray, optimums: np.ndarray) -> float:
    """
    计算逆代际距离指标(Inverted Generational Distance)
    :param objs: 目标值
    :param optimums: 理论最优目标值
    :return: 逆代际距离指标值
    """
    if objs.shape[1] != optimums.shape[1]:
        raise ValueError("The objs does not match the dimension of the optimal targets")
    if len(optimums) == 1:
        warn_once("Only one theoretical optimal solution has been provided, "
                  "which may be a reference point. Please use HV to calculate the score")
    # 计算给定目标值中每一行与最优目标值中每一行之间的欧式距离
    distance_matrix = cdist(objs, optimums, metric='euclidean')
    # 按列取最小值，得到每个最优点到最近点的距离
    min_distances = np.min(distance_matrix, axis=0)
    # 计算最小值的均值
    score = np.mean(min_distances)
    return score


def distance_plus(x, y):
    """
    自定义距离函数：
    计算两个向量之间的
    逐元素差值的最大值的平方和的平方根
    """
    # 取逐元素差值的最大值（与零比较）
    diff = np.maximum(x - y, 0)
    # 计算其平方和的平方根
    return np.sqrt(np.sum(diff ** 2))


def calc_gd_plus(objs: np.ndarray, optimums: np.ndarray) -> float:
    """
    计算代际距离+指标(Generational Distance Plus)
    :param objs: 目标值
    :param optimums: 理论最优目标值
    :return: 代际距离+指标值
    """
    if objs.shape[1] != optimums.shape[1]:
        raise ValueError("The objs does not match the dimension of the optimal targets")
    if len(optimums) == 1:
        warn_once("Only one theoretical optimal solution has been provided, "
                  "which may be a reference point. Please use HV to calculate the score")
    # 计算给定目标值中每一行与最优目标值中每一行之间的自定义plus距离
    distance_matrix = cdist(objs, optimums, metric=distance_plus)
    # 按行取最小值，得到每个点到最近最优点的距离
    distance = np.min(distance_matrix, axis=1)
    # 计算得到分数值
    score = np.mean(distance)
    return score


def calc_igd_plus(objs: np.ndarray, optimums: np.ndarray) -> float:
    """
    计算逆代际距离+指标(Inverted Generational Distance Plus)
    :param objs: 目标值
    :param optimums: 理论最优目标值
    :return: 逆代际距离+指标值
    """
    if objs.shape[1] != optimums.shape[1]:
        raise ValueError("The objs does not match the dimension of the optimal targets")
    if len(optimums) == 1:
        warn_once("Only one theoretical optimal solution has been provided, "
                  "which may be a reference point. Please use HV to calculate the score")
    # 计算给定目标值中每一行与最优目标值中每一行之间的自定义plus距离
    distance_matrix = cdist(objs, optimums, metric=distance_plus)
    # 按列取最小值，得到每个最优点到最近点的距离
    distance = np.min(distance_matrix, axis=0)
    # 计算得到分数值
    score = np.mean(distance)
    return score
