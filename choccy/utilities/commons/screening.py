# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
多属性决策筛选方法
"""

import numpy as np
from typing import Optional, Union


def select_by_cosine(objs: np.ndarray,
                     weights: Union[list, np.ndarray],
                     n_objs: int) -> int:
    """
    使用余弦相似度从候选解中挑选最接近权重偏好的解

    此方法仅考虑方向一致性，不考虑解的绝对好坏。

    :param objs: 目标值矩阵，形状 (n_sols, n_objs)
    :param weights: 偏好权重向量，长度需等于目标数
    :param n_objs: 目标个数
    :return: 选中的解索引
    """
    weights = np.array(weights[:n_objs], dtype=float)
    weight_norm = np.linalg.norm(weights)
    if weight_norm == 0:
        weights = np.ones(n_objs) / n_objs
        weight_norm = 1.0
    weight_unit = weights / weight_norm

    # L2 归一化每个解的目标向量，保留方向信息
    obj_norms = np.linalg.norm(objs, axis=1, keepdims=True)
    obj_norms = np.maximum(obj_norms, 1.e-9)
    objs_normed = objs / obj_norms

    # 计算每个解与权重方向的余弦相似度
    similarities = objs_normed @ weight_unit
    return int(np.argmax(similarities))


def select_by_topsis(objs: np.ndarray,
                     weights: Optional[Union[list, np.ndarray]] = None) -> int:
    """
    使用 TOPSIS 从多个解中选一个最均衡的解

    :param objs: 目标值矩阵，形状 (n_sols, n_objs)，假设为最小化问题
    :param weights: 权重向量，形状 (n_objs,)，为 None 时使用等权重
    :return: 选中的解索引
    """
    n_sols, n_objs = objs.shape

    if weights is None:
        weights = np.ones(n_objs) / n_objs
    else:
        weights = np.asarray(weights, dtype=float).flatten()
        if len(weights) != n_objs:
            raise ValueError(f"Weights length {len(weights)} must match n_objs {n_objs}")

    # Step 1: 向量归一化（按列）
    norm = np.sqrt(np.sum(objs ** 2, axis=0))
    norm = np.maximum(norm, 1e-12)
    r = objs / norm

    # Step 2: 加权
    v = r * weights

    # Step 3: 理想解 A⁺ 和负理想解 A⁻（最小化问题）
    ideal = np.min(v, axis=0)
    neg_ideal = np.max(v, axis=0)

    # Step 4: 到理想解和负理想解的距离
    s_plus = np.sqrt(np.sum((v - ideal) ** 2, axis=1))
    s_minus = np.sqrt(np.sum((v - neg_ideal) ** 2, axis=1))

    # Step 5: 综合评分
    cc = s_minus / (s_plus + s_minus + 1e-12)
    return int(np.argmax(cc))


def select_by_vikor(objs: np.ndarray,
                    weights: Optional[Union[list, np.ndarray]] = None,
                    v: float = 0.5) -> int:
    """
    使用 VIKOR 从多个解中选一个折衷解

    VIKOR 在"最大群体效益"（S 最小）和"最小个人遗憾"（R 最小）之间折中，
    适用于更复杂的多属性决策场景。

    :param objs: 目标值矩阵，形状 (n_sols, n_objs)，假设为最小化问题
    :param weights: 权重向量，形状 (n_objs,)，为 None 时使用等权重
    :param v: 决策机制系数，v=0.5 时均衡折中，v>0.5 偏向群体效益，v<0.5 偏向个人遗憾
    :return: 选中的解索引
    """
    n_sols, n_objs = objs.shape

    if weights is None:
        weights = np.ones(n_objs) / n_objs
    else:
        weights = np.asarray(weights, dtype=float).flatten()
        if len(weights) != n_objs:
            raise ValueError(f"Weights length {len(weights)} must match n_objs {n_objs}")

    # Step 1: 理想解 f* 和负理想解 f^（最小化问题）
    best = np.min(objs, axis=0)
    worst = np.max(objs, axis=0)
    diff = worst - best
    diff = np.maximum(diff, 1e-12)

    # Step 2: 归一化偏差并加权
    normalized = (objs - best) / diff  # (n_sols, n_objs)，0 表示达到理想值

    # Step 3: S 群体效益（L1 聚合），R 个人遗憾（L∞ 聚合）
    s = np.sum(normalized * weights, axis=1)  # (n_sols,)
    r = np.max(normalized * weights, axis=1)  # (n_sols,)

    # Step 4: 计算 Q 值
    s_min, s_max = np.min(s), np.max(s)
    r_min, r_max = np.min(r), np.max(r)
    s_range = s_max - s_min
    r_range = r_max - r_min
    s_range = max(s_range, 1e-12)
    r_range = max(r_range, 1e-12)

    q = v * (s - s_min) / s_range + (1 - v) * (r - r_min) / r_range

    # Step 5: 返回 Q 最小的解（越小越好）
    return int(np.argmin(q))
