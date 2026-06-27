# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
截断选择函数
"""

import numpy as np
from typing import Tuple
from scipy.spatial import distance_matrix


def truncation(objs: np.ndarray, k: int) -> np.ndarray:
    """
    SPEA2 截断选择：从目标值矩阵中删除 k 个最拥挤的个体

    算法流程：
      1. 计算种群中所有个体间的欧氏距离矩阵
      2. 第一次循环做完整排序，建立排序状态
      3. 后续循环增量维护排序矩阵（删除已移除个体对应的行列）
      4. 每次用字典序淘汰法找到最拥挤个体，标记删除

    :param objs: 种群的目标值矩阵，形状 (n, n_objs)
    :param k: 需要删除的个体数量
    :return: 布尔标志向量，形状 (n,)，True 表示该个体被删除
    """
    n_pop = objs.shape[0]

    # ---- 计算距离矩阵 ----
    dist_mat = distance_matrix(objs, objs)
    np.fill_diagonal(dist_mat, np.inf)  # 自身距离为 inf（避免自匹配）

    # ---- 初始化状态 ----
    del_flag = np.zeros(n_pop, dtype=bool)
    remain = np.arange(n_pop)   # 当前剩余个体的全局索引
    n_del = 0

    # ---- 第一次循环：完整排序，建立排序状态 ----
    # raw 是当前剩余个体间的距离子矩阵
    raw = dist_mat[np.ix_(remain, remain)]
    # sort_idx[i, j] = 个体 i 的第 j 近邻在原始子矩阵中的列索引
    sort_idx = np.argsort(raw, axis=1)
    # sorted_vals[i, j] = 个体 i 到其第 j 近邻的距离值
    sorted_vals = np.take_along_axis(raw, sort_idx, axis=1)

    # ---- 循环删除最拥挤个体 ----
    while n_del < k:
        # 在已排序的距离矩阵中，找到字典序最小的行
        # 字典序最小 = 最近邻距离最小 → 次近邻距离最小 → ……
        order = _arglexmin(sorted_vals)
        r = order[0]  # 当前剩余集合中的本地索引（最拥挤个体）

        # 标记删除（转回全局索引）
        del_flag[remain[r]] = True
        n_del += 1

        # 还需继续删除 → 增量更新排序矩阵
        if n_del < k:
            remain, sorted_vals, sort_idx = _remove_individual(
                remain, sorted_vals, sort_idx, r
            )

    return del_flag


def _remove_individual(remain: np.ndarray,
                       sorted_vals: np.ndarray,
                       sort_idx: np.ndarray,
                       r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从排序状态中增量删除个体 r

    sorted_vals 和 sort_idx 是当前的(m × m)排序矩阵，
    每次只需删掉第 r 行，以及其余每行中对应 r 的那一列，
    即可得到新的(m-1 × m-1)排序矩阵，无需重新排序。

    :param remain: 剩余个体的全局索引数组
    :param sorted_vals: 每行按升序排列的距离值矩阵
    :param sort_idx: 每行 argsort 索引矩阵
    :param r: 要删除的个体在当前集合中的本地索引
    :return: (新的 remain, sorted_vals, sort_idx)
    """
    m = sorted_vals.shape[0]

    # 找到每行中 r（即被删除个体）所在的列位置
    # 每行有且仅有一列的 sort_idx == r（个体 i 到个体 r 的距离）
    col_of_r = np.argmax(sort_idx == r, axis=1)  # (m,)

    # ---- 删掉第 r 行 ----
    row_mask = np.ones(m, dtype=bool)
    row_mask[r] = False
    remaining_rows = np.where(row_mask)[0]

    # ---- 向量化删掉剩余每行中不同位置的列 ----
    # 将(m-1 × m)矩阵展平为 1D，
    # 在展平的 1D 数组中，第 i 行要删的列位置 = i * m + col_of_r[i]
    col_of_r_rem = col_of_r[remaining_rows]
    flat_vals = sorted_vals[remaining_rows].ravel()
    flat_idx = sort_idx[remaining_rows].ravel()

    del_pos = np.arange(m - 1) * m + col_of_r_rem
    keep = np.ones((m - 1) * m, dtype=bool)
    keep[del_pos] = False

    sorted_vals = flat_vals[keep].reshape(m - 1, m - 1)
    sort_idx = flat_idx[keep].reshape(m - 1, m - 1)

    # 矩阵缩小了一位（原索引指向的列位置变了），
    # 所有大于 r 的索引值减 1
    sort_idx[sort_idx > r] -= 1
    remain = np.delete(remain, r)

    return remain, sorted_vals, sort_idx


def _arglexmin(sorted_dists: np.ndarray) -> np.ndarray:
    """
    逐列淘汰法找字典序最小行

    等价于 np.lexsort(np.fliplr(sorted_dists).T)，但避免了 structured
    dtype 的创建和 O(m²) 的逐列拷贝开销，平均只需 1-2 轮比较。

    原理：
      要找按(列0, 列1, ..., 列m-1)字典序最小的行，不需要对全行排序。
      只需逐列递进：先找列0最小值的行 → 若多行并列则在列1中再比 → ……
      直到只剩最后一行。

    :param sorted_dists: 已按升序排列的距离矩阵，形状 (m, m)
    :return: 单元素数组，[0] 为字典序最小行的索引
             （兼容原有调用习惯：result[0] 取结果）
    """
    m = sorted_dists.shape[0]
    cand = np.arange(m)
    for col in range(m):
        vals = sorted_dists[cand, col]
        cand = cand[vals == vals.min()]
        if len(cand) == 1:
            break
    return np.array([cand[0]])
