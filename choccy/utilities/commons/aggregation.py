# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
聚合函数集
"""

import numpy as np
from typing import Union
from ...types import AggregationMethod


def aggregate(objs: np.ndarray,
              weights: np.ndarray,
              ref_point: np.ndarray,
              method: Union[str, AggregationMethod] = 'pbi') -> np.ndarray:
    """
    聚合函数
    :param objs: 聚合目标
    :param weights: 权重向量
    :param ref_point: 参考点
    :param method: 聚合类型
    :return: 聚合结果
    """
    if AggregationMethod.parse(method) == AggregationMethod.PBI:
        # 基于惩罚边界的聚合方法
        theta = 5  # 设置超参数
        if len(objs) == 1:
            # 若是单个个体则直接求
            d1 = np.abs(np.dot(objs - ref_point, weights.T)).flatten() / np.linalg.norm(weights, axis=1)
            d2 = np.linalg.norm(objs - (ref_point + d1.reshape(-1, 1) * weights), axis=1)
        else:
            # 若是需要对整个邻居目标值则使用取对角线方法
            d1 = np.abs(np.diag(np.dot(objs - ref_point, weights.T))) / np.linalg.norm(weights, axis=1)
            d2 = np.linalg.norm(objs - (ref_point + d1.reshape(-1, 1) * weights), axis=1)
        return d1 + theta * d2
    elif AggregationMethod.parse(method) == AggregationMethod.TCH:
        # 切比雪夫聚合方法
        return np.max(weights * np.abs(objs - ref_point), axis=1)
    elif AggregationMethod.parse(method) == AggregationMethod.WSM:
        # 线性聚合方法
        if len(objs) == 1:
            # 若是单个个体则直接求点积
            return np.dot(objs, weights.T).flatten()
        else:
            # 若是需要对整个邻居目标值则使用 取对角线方法
            return np.diag(np.dot(objs, weights.T)).flatten()
            # 使用爱因斯坦求和约定对矩阵逐行求点积
            # return np.einsum('ij,ij->i', objs, weights).flatten()
    else:
        raise ValueError("There is no such aggregate function type")
