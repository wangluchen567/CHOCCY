# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
约束处理相关函数集
"""

import numpy as np
from typing import Optional


def calc_penalized_objs(objs: np.ndarray,
                        cons: Optional[np.ndarray] = None,
                        coef: Optional[float] = None,
                        max_objs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    计算约束惩罚后的目标值（Penalized Objectives）
    :param objs: 目标值矩阵
    :param cons: 约束值矩阵
    :param coef: 惩罚参数
                - None: 使用相对惩罚（种群内比较）
                - float: 使用静态惩罚（penalized = obj + coef * violation）
    :param max_objs: 全局最大目标值（可选）
                     - None: 使用当前种群的最大目标值
                     - np.ndarray: 使用指定的全局最大目标值
    :return: 基于约束惩罚后的目标值矩阵
    """
    # 处理约束为空的情况
    if cons is None or cons.size == 0 or cons.shape[1] == 0:
        return objs  # 无约束，直接返回原目标值
    # 找出所有不满足约束的个体
    not_feas = np.any(cons > 0, axis=1)
    # 如果没有不可行解，直接返回原目标值
    if not np.any(not_feas):
        return objs  # 无约束，直接返回原目标值
    # 创建约束惩罚后的目标值矩阵
    penalized_objs = objs.copy()
    # 计算不满足约束的个体的不满足约束的程度值
    violation = np.sum(np.maximum(cons[not_feas], 0), axis=1)
    # 若指定了惩罚超参数则使用静态惩罚
    if coef is not None:  # 静态惩罚模式
        penalized_objs[not_feas] = objs[not_feas] + coef * violation.reshape(-1, 1)
    else:  # 动态惩罚模式
        # 使用全局最大目标值或当前种群的最大目标值
        if max_objs is None:
            # 计算当前种群中每个目标的最大值
            max_objs = np.max(objs, axis=0)
        # 利用广播机制更新不满足约束的个体的目标函数值
        penalized_objs[not_feas] = max_objs + violation.reshape(-1, 1)
    # 返回约束惩罚后的目标值矩阵
    return penalized_objs
