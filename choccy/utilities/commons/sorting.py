# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
排序相关函数集
"""

import numpy as np
from typing import Tuple
from ...core import warn_once, PerformanceWarning


def is_dom(p_objs: np.ndarray, q_objs: np.ndarray) -> bool:
    """
    定义支配关系
    :param p_objs: p 的目标值向量
    :param q_objs: q 的目标值向量
    :return: p 是否支配 q
    """
    # 将输入转换为 NumPy 数组
    p_objs = np.asarray(p_objs)
    q_objs = np.asarray(q_objs)
    # 条件1: 对所有子目标， p 不比 q 差
    condition1 = np.all(p_objs <= q_objs)
    # 条件2: 至少存在一个子目标， p 比 q 好
    condition2 = np.any(p_objs < q_objs)
    # 满足以上两个条件则说明 p 支配 q
    return bool(condition1 and condition2)


def _dom_matrix(objs: np.ndarray) -> np.ndarray:
    """
    得到每对解的支配关系矩阵(使用numpy实现)
    :param objs: 目标向量组成的矩阵
    :return: 每对解的支配关系
    """
    n = len(objs)
    dom_mat = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if is_dom(objs[i], objs[j]):
                dom_mat[i, j] = True
    return dom_mat


def _fast_nd_sort(objs: np.ndarray) -> Tuple[list, np.ndarray]:
    """
    快速非支配排序(使用numpy实现)
    :param objs: 目标向量组成的矩阵
    :return: 前沿面信息，前沿面排名信息
    """
    pop_size = len(objs)  # 获取种群数量
    objs = np.array(objs)  # 将目标转换为numpy数组
    fronts = []  # 初始化各前沿面列表
    ranks = np.zeros(pop_size, dtype=int)  # 每个个体所在的前沿数
    num_dom = np.zeros(pop_size, dtype=int)  # 每一个解被支配的次数初始化为0
    sol_dom = [[] for _ in range(pop_size)]  # 每一个解所支配的解列表
    # 创建比较矩阵以确定支配关系
    for i in range(pop_size):
        # 判断解 i 是否支配其他解
        dominates = np.all(objs[i] <= objs, axis=1) & np.any(objs[i] < objs, axis=1)
        # 记录被解 i 支配的解
        sol_dom[i] = np.where(dominates)[0].tolist()
        # 更新每一个解被支配的次数
        num_dom += dominates.astype(int)
    # 找出第一前沿面
    first_front = np.where(num_dom == 0)[0].tolist()
    ranks[num_dom == 0] = 1  # 将第一前沿面的排序序号设置为1
    fronts.append(first_front)  # 将第一前沿面添加到fronts列表中
    i = 0
    # 迭代处理每一个前沿面
    while i < len(fronts):
        next_front = []  # 初始化下一个前沿面
        for p in fronts[i]:  # 遍历当前前沿面的每一个解
            for q in sol_dom[p]:  # 遍历每一个被当前解支配的解
                num_dom[q] -= 1  # 被支配的次数减1
                if num_dom[q] == 0:  # 如果支配次数减为0，表示该解进入下一前沿面
                    ranks[q] = i + 2  # 排序序号更新
                    next_front.append(q)  # 将解添加到下一前沿面
        if next_front:  # 如果下一个前沿面不为空
            fronts.append(next_front)  # 将下一个前沿面添加到fronts列表中
        i += 1  # 处理下一个前沿面
    return fronts, ranks


def crowding_dist(objs: np.ndarray, fronts: list) -> np.ndarray:
    """
    计算拥挤度距离
    :param objs: 目标向量组成的矩阵
    :param fronts: 前沿面信息
    :return: 拥挤度距离
    """
    pop_size, num_dim = objs.shape
    crowd_dist = np.zeros(pop_size)
    for f in fronts:
        # 获取当前前沿面中解的目标值
        objs_f = objs[f, :]
        # 求最大与最小值
        f_max = np.max(objs_f, axis=0)
        f_min = np.min(objs_f, axis=0)
        # 求最大与最小的差，方便归一化
        f_range = f_max - f_min
        f_range[f_range == 0] = np.finfo(np.float32).tiny  # 避免除零
        # 排序索引矩阵
        sorted_indices = np.argsort(objs_f, axis=0)
        f_sorted = np.array(f)[sorted_indices]
        # 设置边界个体的距离为无穷大
        crowd_dist[f_sorted[0, np.arange(num_dim)]] = float('inf')
        crowd_dist[f_sorted[-1, np.arange(num_dim)]] = float('inf')
        # 计算中间个体的拥挤度增量
        dist_increments = (
            objs_f[sorted_indices[2:], np.arange(num_dim)] -
            objs_f[sorted_indices[:-2], np.arange(num_dim)]
        ) / f_range
        # 累加增量到距离
        np.add.at(crowd_dist, f_sorted[1:-1], dist_increments)
    # 返回计算得到的拥挤度距离
    return crowd_dist


def composite_rank(ranks: np.ndarray, crowd_dist: np.ndarray) -> np.ndarray:
    """
    根据支配前沿数和拥挤度距离计算个体综合排名
    :param ranks: 前沿面排名信息
    :param crowd_dist: 拥挤度距离
    :return: 最终排名
    """
    # 初始化排序后的种群索引
    indicator = np.hstack((ranks.reshape(-1, 1), -crowd_dist.reshape(-1, 1)))
    # 使用 np.lexsort 对两列指标进行排序
    indices = np.lexsort((indicator[:, 1], indicator[:, 0]))
    # 获取排序下标
    ranking = np.argsort(indices)
    return ranking


try:
    # 尝试导入numba
    from numba import jit, boolean


    @jit(nopython=True, cache=True)
    def dom_matrix_jit(objs: np.ndarray) -> np.ndarray:
        """得到每对解的支配关系矩阵"""
        n = len(objs)
        dom_mat = np.zeros((n, n), dtype=boolean)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(objs[i] <= objs[j]) and np.any(objs[i] < objs[j]):
                    dom_mat[i, j] = True
        return dom_mat


    @jit(nopython=True, cache=True)
    def dominates_loop(objs: np.ndarray, i: int) -> np.ndarray:
        n = objs.shape[0]
        m = objs.shape[1]
        dominates = np.zeros(n, dtype=np.bool_)
        for j in range(n):
            all_less_equal = True
            any_less = False
            for k in range(m):
                if objs[i, k] > objs[j, k]:
                    all_less_equal = False
                    break
                elif objs[i, k] < objs[j, k]:
                    any_less = True
            dominates[j] = all_less_equal and any_less
        return dominates


    @jit(nopython=True, cache=True)
    def fast_nd_sort_jit(objs: np.ndarray):
        """快速非支配排序(使用numba加速)"""
        pop_size = objs.shape[0]  # 获取种群数量
        fronts = np.zeros((pop_size, pop_size), dtype=np.int16)  # 初始化各前沿面的索引数组
        fronts_trunc = np.zeros(pop_size, dtype=np.int16)  # 各前沿面的索引数组的索引截断
        ranks = np.zeros(pop_size, dtype=np.int16)  # 每个个体所在的前沿数
        num_dom = np.zeros(pop_size, dtype=np.int16)  # 每一个解被支配的次数初始化为0
        sol_dom = np.zeros((pop_size, pop_size), dtype=np.int16)  # 每一个解所支配的解的索引数组
        sol_trunc = np.zeros(pop_size, dtype=np.int16)  # 所支配解的索引截断
        # 创建比较矩阵以确定支配关系
        for i in range(pop_size):
            # 判断解 i 是否支配其他解
            dominates = dominates_loop(objs, i)
            # 得到被解 i 支配的解的索引
            indices = np.where(dominates)[0]
            # 得到索引数组的截断
            sol_trunc[i] = len(indices)
            # 记录被解 i 支配的解的索引
            sol_dom[i][:len(indices)] = indices
            # 更新每一个解被支配的次数
            num_dom += dominates.astype(np.int16)
        # 找出第一前沿面
        first_front = np.where(num_dom == 0)[0]
        ranks[num_dom == 0] = 1  # 将第一前沿面的排序序号设置为1
        fronts[0][:len(first_front)] = first_front  # 将第一前沿面添加到数组中
        front_count = len(first_front)  # 第一前沿面的解数量
        fronts_trunc[0] = front_count  # 记录截断数据
        i = 0
        # 迭代处理每一个前沿面
        while True:
            next_front = np.zeros(pop_size, dtype=np.int16)  # 初始化下一个前沿面
            next_count = 0
            for p in fronts[i][:front_count]:  # 遍历当前前沿面的每一个解
                for q in sol_dom[p][:sol_trunc[p]]:  # 遍历每一个被当前解支配的解
                    num_dom[q] -= 1  # 被支配的次数减1
                    if num_dom[q] == 0:  # 如果支配次数减为0，表示该解进入下一前沿面
                        ranks[q] = i + 2  # 排序序号更新
                        next_front[next_count] = q  # 将解添加到下一前沿面
                        next_count += 1
            if next_count > 0:  # 如果下一个前沿面不为空
                fronts[i + 1][:next_count] = next_front[:next_count]
                front_count = next_count
                fronts_trunc[i + 1] = front_count  # 记录截断数据
                i += 1  # 处理下一个前沿面
            else:  # 若下一个前沿面为空则返回
                break
        return fronts, fronts_trunc, ranks


    def dom_matrix(objs: np.ndarray) -> np.ndarray:
        """
        得到每对解的支配关系(默认使用numba加速)
        :param objs: 目标向量组成的矩阵
        :return: 每对解的支配关系
        """
        return dom_matrix_jit(objs)


    def fast_nd_sort(objs: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        快速非支配排序(默认使用numba加速)
        :param objs: 目标向量组成的矩阵
        :return: 前沿面信息，前沿面排名信息
        """
        fronts_mat, fronts_trunc, ranks = fast_nd_sort_jit(objs)
        fronts_trunc = fronts_trunc[fronts_trunc > 0]
        fronts = [fronts_mat[i][:fronts_trunc[i]].tolist() for i in range(len(fronts_trunc))]
        return fronts, ranks

except ImportError:
    # 如果导入numba加速库失败，使用原始的函数
    warn_once("Numba acceleration unavailable - "
              "falling back to slower implementation",
              warning_class=PerformanceWarning)
    dom_matrix = _dom_matrix
    fast_nd_sort = _fast_nd_sort

# 只允许外部调用以下函数：
__all__ = [
    'is_dom',
    'dom_matrix',
    'fast_nd_sort',
    'crowding_dist',
    'composite_rank',
]
