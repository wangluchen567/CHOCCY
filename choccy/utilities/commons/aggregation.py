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
              method: Union[str, AggregationMethod] = 'pbi',
              theta: float = 5.0) -> np.ndarray:
    """
    聚合函数
    :param objs: 聚合目标
    :param weights: 权重向量
    :param ref_point: 参考点
    :param method: 聚合类型
    :param theta: PBI方法的超参数
    :return: 聚合结果
    """
    if AggregationMethod.parse(method) == AggregationMethod.PBI:
        # 基于惩罚边界的聚合方法
        norm_w = np.linalg.norm(weights, axis=1)
        unit_w = weights / (norm_w.reshape(-1, 1) + 1e-12)  # 归一化成单位向量
        proj = np.sum((objs - ref_point) * weights, axis=1)
        d1 = np.abs(proj) / norm_w
        d2 = np.linalg.norm(objs - (ref_point + d1.reshape(-1, 1) * unit_w), axis=1)
        return d1 + theta * d2
    elif AggregationMethod.parse(method) == AggregationMethod.TCH:
        # 切比雪夫聚合方法
        return np.max(weights * np.abs(objs - ref_point), axis=1)
    elif AggregationMethod.parse(method) == AggregationMethod.WSM:
        # 线性聚合方法
        return np.sum(objs * weights, axis=1).flatten()
    else:
        raise ValueError("There is no such aggregate function type")
