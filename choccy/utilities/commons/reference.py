# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
参考点生成函数集
"""

import math
import itertools
import numpy as np


def generate_uniform_weights(n_sols: int, n_dims: int, method: str = 'nbi') -> np.ndarray:
    """
    生成均匀分布在单位超平面上的参考点

    :param n_sols: 期望的参考点数
    :param n_dims: 目标维数 M
    :param method: 生成方法
                   'nbi' - NBI 方法（默认），点数 ≈ C(H+M-1, M-1)
                   'mud' - MUD 方法，点数 = n_sols（精确）
    :return: 参考点矩阵，形状 (N_actual, M)
    """
    if method == 'nbi':
        return uniform_weights_nbi(n_sols, n_dims)
    elif method == 'mud':
        return uniform_weights_mud(n_sols, n_dims)
    else:
        raise ValueError(f"Unknown method '{method}'. Available: 'nbi', 'mud'")


def uniform_weights_nbi(n_sols: int, n_dims: int) -> np.ndarray:
    """
    使用 NBI (Normal-Boundary Intersection) 方法生成均匀分布的参考点

    采用 Das & Dennis 的格点生成方法，支持两层结构：
    当 H < M 且单层点数不够时，自动在超平面内部补充第二层。

    Code References:
        PlatEMO(https://github.com/BIMK/PlatEMO)

    :param n_sols: 期望的参考点数（近似值）
    :param n_dims: 目标维数 M
    :return: 参考点矩阵，形状 (N_actual, M)
    """
    # ---- 第一层：找最大 H 使 C(H+M-1, M-1) ≤ N ----
    ht = 1
    while len(list(itertools.combinations(range(ht + n_dims), n_dims - 1))) <= n_sols:
        ht = ht + 1
    # 生成第一层参考点矩阵
    wt = _nbi_weights(ht, n_dims)
    n_weights = len(wt)  # 当前参考点总数

    # ---- 第二层：当 H < M 且当前点数不足时，在超平面内部补充 ----
    if ht < n_dims:
        ht2 = 0
        n_layer2 = len(list(itertools.combinations(range(ht2 + n_dims), n_dims - 1)))
        while n_weights + n_layer2 <= n_sols:
            ht2 = ht2 + 1
            n_layer2 = len(list(itertools.combinations(range(ht2 + n_dims), n_dims - 1)))
        if ht2 > 0:
            # 生成第二层参考点，并缩放到超平面内部
            wt2 = _nbi_weights(ht2, n_dims)
            # 将内层参考点映射到超平面中心区域
            wt2 = wt2 / 2 + 1 / (2 * n_dims)
            wt = np.vstack((wt, wt2))

    # 确保所有分量大于 0，避免 Tchebycheff 聚合时忽略某些目标
    wt = np.maximum(wt, 1e-16)
    return wt


def _nbi_weights(ht: int, n_dims: int) -> np.ndarray:
    """
    使用 Das & Dennis 方法生成单层权重

    对 (M-1) 个整数变量取组合，映射到 M 维超平面 Σw_i = 1。
    例如 3 维时，从 {0,...,H-1} 取两个数 (i,j) 并排序 i≤j，
    得到权重 [i/H, (j-i)/H, (H-j)/H]。

    :param ht: 每维等分份数 H
    :param n_dims: 目标维数 M
    :return: 权重矩阵，形状 (C(H+M-1, M-1), M)
    """
    wt = np.array(list(itertools.combinations(range(ht + n_dims - 1), n_dims - 1)))
    if len(wt) == 0:
        return np.empty((0, n_dims))
    wt = wt - np.arange(n_dims - 1)
    return np.asarray((np.hstack((wt, np.zeros((len(wt), 1)) + ht)) -
                       np.hstack((np.zeros((len(wt), 1)), wt))) / ht)


def uniform_weights_mud(n_sols: int, n_dims: int) -> np.ndarray:
    """
    使用 MUD (Mixture Uniform Design) 方法生成均匀分布的参考点

    通过 Good Lattice Point 在低维超立方体中生成均匀点，
    再通过非线性变换映射到单位超平面上，可精确生成 N 个参考点。

    References:
        Sampling reference points on the Pareto fronts of benchmark multi-objective optimization problems,
        Y. Tian, X. Xiang, X. Zhang, R. Cheng, and Y. Jin
    Code References:
        PlatEMO(https://github.com/BIMK/PlatEMO)

    :param n_sols: 期望的参考点数（精确值）
    :param n_dims: 目标维数 M
    :return: 参考点矩阵，形状 (n_sols, n_dims)
    """
    # 在 (M-1) 维超立方体中生成 N 个均匀点
    x = _good_lattice_point(n_sols, n_dims - 1)
    # 对各列施加非线性变换，使分布映射到单纯形
    x = x ** (1.0 / np.arange(n_dims - 1, 0, -1))
    x = np.maximum(x, 1e-6)

    # 变换到单纯形
    cum_prod = np.cumprod(x, axis=1)
    wt = np.zeros((n_sols, n_dims))
    wt[:, :-1] = (1 - x) * cum_prod / x
    wt[:, -1] = np.prod(x, axis=1)

    wt = np.maximum(wt, 1e-16)
    return wt


def _good_lattice_point(n_sols: int, n_dims: int) -> np.ndarray:
    """
    使用 Good Lattice Point 方法在 [0,1]^n_dims 中生成 n_sols 个均匀点

    通过寻找最优生成向量，使生成的点集具有最小的 CD2（中心化 L2 偏差）。

    :param n_sols: 点数
    :param n_dims: 维度
    :return: 点集矩阵，形状 (n_sols, n_dims)
    """
    # 找到 [1, N) 中与 N 互质的数（候选生成元）
    generators = [h for h in range(1, n_sols) if math.gcd(h, n_sols) == 1]
    # 生成候选矩阵
    udt = np.mod(np.arange(1, n_sols + 1).reshape(-1, 1) * np.array(generators), n_sols)
    udt[udt == 0] = n_sols

    n_gen = len(generators)
    n_comb = math.comb(n_gen, n_dims) if hasattr(math, 'comb') else len(
        list(itertools.combinations(range(n_gen), n_dims)))

    if n_comb < 10000:
        # 组合数较少时，穷举所有组合，选 CD2 最小的
        best_cd2 = float('inf')
        best_data = None
        for comb in itertools.combinations(range(n_gen), n_dims):
            ut = udt[:, list(comb)]
            cd2 = _cal_cd2(ut)
            if cd2 < best_cd2:
                best_cd2 = cd2
                best_data = ut
        data = best_data
    else:
        # 组合数过多时，逐个使用每个候选生成元作为幂基
        best_cd2 = float('inf')
        best_data = None
        for i in range(1, n_sols):
            ut = np.mod(np.arange(1, n_sols + 1).reshape(-1, 1) *
                        np.array([i ** k for k in range(n_dims)]), n_sols)
            ut[ut == 0] = n_sols
            cd2 = _cal_cd2(ut)
            if cd2 < best_cd2:
                best_cd2 = cd2
                best_data = ut
        data = best_data

    return np.asarray((data - 1) / (n_sols - 1))


def _cal_cd2(ut: np.ndarray) -> float:
    """
    计算中心化 L2 偏差 (Centered L2-discrepancy)，
    用于评估 GoodLatticePoint 生成的点集的均匀性。

    :param ut: 点集矩阵，形状 (N, S)
    :return: CD2 值（越小表示点集越均匀）
    """
    n, s = ut.shape
    x = (2 * ut - 1) / (2 * n)
    # CS1: 单点贡献
    cs1 = np.sum(np.prod(2 + np.abs(x - 0.5) - (x - 0.5) ** 2, axis=1))
    # CS2: 点对贡献
    cs2 = 0.0
    for i in range(n):
        cs2 += np.sum(np.prod(
            1 + 0.5 * np.abs(x[i] - 0.5) + 0.5 * np.abs(x - 0.5) -
            0.5 * np.abs(x[i] - x), axis=1))
    return (13 / 12) ** s - 2 ** (1 - s) / n * cs1 + cs2 / (n ** 2)
