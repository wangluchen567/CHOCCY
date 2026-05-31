# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
邻域搜索函数集
"""

import numpy as np


def gaussian_perturb(solutions: np.ndarray,
                     l_bounds: np.ndarray,
                     u_bounds: np.ndarray,
                     scale: float = 0.1) -> np.ndarray:
    """
    高斯扰动 + 自动裁剪 (实数问题)
    :param solutions: 当前解，形状: (n_sols, n_vars)
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param scale: 扰动程度（标准差）
    :return: 扰动后的新解，形状: (n_sols, n_vars)
    """
    noise = np.random.normal(0, scale, size=solutions.shape)
    new_solutions = solutions + noise
    new_solutions = np.clip(new_solutions, l_bounds, u_bounds)
    return new_solutions


def uniform_perturb(solutions: np.ndarray,
                    l_bounds: np.ndarray,
                    u_bounds: np.ndarray,
                    half_range: float = 0.1,) -> np.ndarray:
    """
    均匀扰动 + 自动裁剪 (实数问题)
    :param solutions: 当前解，形状: (n_sols, n_vars)
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param half_range: 扰动范围的一半（在 [-half_range, half_range] 内均匀随机）
    :return: 扰动后的新解，形状: (n_sols, n_vars)
    """
    noise = np.random.uniform(-half_range, half_range, size=solutions.shape)
    new_solutions = solutions + noise
    new_solutions = np.clip(new_solutions, l_bounds, u_bounds)
    return new_solutions


def cauchy_perturb(solutions: np.ndarray,
                   l_bounds: np.ndarray,
                   u_bounds: np.ndarray,
                   scale: float = 1.0) -> np.ndarray:
    """
    柯西扰动（厚尾分布）+ 自动裁剪 (实数问题)
    :param solutions: 当前解，形状: (n_sols, n_vars)
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param scale: 扰动尺度，柯西扰动的缩放尺度
    :return: 扰动后的新解，形状: (n_sols, n_vars)
    """
    noise = np.random.standard_cauchy(size=solutions.shape) * scale
    new_solutions = solutions + noise
    new_solutions = np.clip(new_solutions, l_bounds, u_bounds)
    return new_solutions


def polynomial_perturb(solutions: np.ndarray,
                       l_bounds: np.ndarray,
                       u_bounds: np.ndarray,
                       perturb_rate: float,
                       eta: float = 20.0) -> np.ndarray:
    """
    多项式变异扰动 (实数问题)
    :param solutions:当前解，形状: (n_sols, n_vars)
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param perturb_rate: 扰动变异概率，范围: [0, 1]
    :param eta: 分布指数，控制变异强度，越大变异越小
    :return: 扰动后的新解，形状: (n_sols, n_vars)
    """
    n_sols, n_vars = solutions.shape
    # 将边界数组转为边界矩阵
    lbs = np.broadcast_to(l_bounds, (n_sols, l_bounds.size))
    ubs = np.broadcast_to(u_bounds, (n_sols, u_bounds.size))
    # 变异掩码与随机数生成
    mask = np.random.random((n_sols, n_vars)) < perturb_rate
    mu = np.random.random((n_sols, n_vars))
    # 情况1：mu <= 0.5
    t = mask * (mu <= 0.5)
    solutions[t] += (ubs[t] - lbs[t]) * (
            (2 * mu[t] + (1 - 2 * mu[t]) * (1 - (solutions[t] - lbs[t]) / (ubs[t] - lbs[t]))
             ** (eta + 1))
            ** (1 / (eta + 1)) - 1)
    # 情况2：mu > 0.5
    t = mask * (mu > 0.5)
    solutions[t] += (ubs[t] - lbs[t]) * (
            1 - (2 * (1 - mu[t]) + 2 * (mu[t] - 0.5) * (1 - (ubs[t] - solutions[t]) / (ubs[t] - lbs[t]))
                 ** (eta + 1))
            ** (1 / (eta + 1)))
    return solutions


def swap_perturb(solutions: np.ndarray) -> np.ndarray:
    """
    换位扰动 (序列问题)
    :param solutions: 当前解，形状: (n_sols, n_vars)
    :return: 扰动后的新解，形状: (n_sols, n_vars)
    """
    n_sols, n_vars = solutions.shape
    exchanges = np.random.randint(n_vars, size=(n_sols, 2))
    solutions[np.arange(n_sols), exchanges[:, 0]], solutions[np.arange(n_sols), exchanges[:, 1]] \
        = solutions[np.arange(n_sols), exchanges[:, 1]], solutions[np.arange(n_sols), exchanges[:, 0]]
    return solutions


def flip_perturb(solutions: np.ndarray) -> np.ndarray:
    """
    翻转扰动 (序列问题)
    :param solutions: 当前解，形状: (n_sols, n_vars)
    :return: 扰动后的新解，形状: (n_sols, n_vars)
    """
    n_sols, n_vars = solutions.shape
    # 生成随机的起始和结束索引
    starts = np.random.randint(0, n_vars, size=n_sols)
    ends = np.random.randint(0, n_vars, size=n_sols)
    # 确保start <= end
    starts, ends = np.minimum(starts, ends), np.maximum(starts, ends)
    # 生成列索引网格
    cols = np.arange(n_vars).reshape(1, -1)
    # 计算需要倒置的区域掩码
    mask = (cols >= starts.reshape(-1, 1)) & (cols <= ends.reshape(-1, 1))
    # 计算倒置后的索引
    reversed_indices = starts.reshape(-1, 1) + ends.reshape(-1, 1) - cols
    # 组合索引：在掩码位置使用倒置索引，否则使用原索引
    indices = np.where(mask, reversed_indices, cols)
    # 得到部分片段倒置后的结果
    return solutions[np.arange(n_sols).reshape(-1, 1), indices]


def bit_perturb(solutions: np.ndarray, perturb_rate: float) -> np.ndarray:
    """
    位翻转扰动 (二进制问题)
    :param solutions: 当前解，形状: (n_sols, n_vars)
    :param perturb_rate: 扰动比率，范围: [0, 1]
    :return: 扰动后的新解，形状: (n_sols, n_vars)
    """
    n_sols, n_vars = solutions.shape
    mask = np.random.rand(n_sols, n_vars) < perturb_rate
    solutions[mask] = 1 - solutions[mask]
    return solutions


def fix_label_perturb(solutions: np.ndarray, perturb_rate: float) -> np.ndarray:
    """
    固定类型数的标签的交换式扰动 (固定类型数的标签问题)
    :param solutions: 当前解，形状: (n_sols, n_vars)
    :param perturb_rate: 扰动比率，范围: [0, 1]
    :return: 扰动后的新解，形状: (n_sols, n_vars)
    """
    n_sols, n_vars = solutions.shape
    # 确定哪些个体需要扰动（每个个体一个概率）
    perturb_flags = np.random.rand(n_sols) < perturb_rate
    need_perturb = np.where(perturb_flags)[0]
    if len(need_perturb) == 0:
        return solutions
    # 只为需要扰动的个体生成交换位置
    points = np.random.randint(n_vars, size=(len(need_perturb), 2))
    # 执行交换
    solutions[need_perturb, points[:, 0]], solutions[need_perturb, points[:, 1]] = \
        solutions[need_perturb, points[:, 1]], solutions[need_perturb, points[:, 0]]
    return solutions
