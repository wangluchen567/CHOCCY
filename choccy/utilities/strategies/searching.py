# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
搜索策略函数集
"""

import numpy as np
from typing import Tuple, Optional
from ...core import warn_once, PerformanceWarning


def eval_2opt_move(cost_mat: np.ndarray, a: int, b: int, c: int, d: int,
                   pena_mat: Optional[np.ndarray] = None,
                   pena_coef: float = 0.0,
                   segment: Optional[np.ndarray] = None) -> float:
    """
    评估2-opt移动，返回成本变化（负值为改进）

    断开边 (a, b), (c, d) 连接边  (a, c), (b, d)
    :param cost_mat: 成本矩阵
    :param a: 第一条断开的边的起点
    :param b: 第一条断开的边的终点
    :param c: 第二条断开的边的起点
    :param d: 第二条断开的边的终点
    :param pena_mat: 惩罚矩阵（可选）
    :param pena_coef: 惩罚参数
    :param segment: 需要翻转的片段序列（不含a和d）
    :return: 成本变化量
    """
    # 提前计算标志位
    use_penalty = pena_mat is not None
    use_segment = segment is not None
    # 初始化变化量
    cost_delta, pena_delta = 0.0, 0.0
    # 外部边变化：删除(a,b) (c,d)，添加(a,c) (b,d)
    cost_delta = cost_mat[a, c] + cost_mat[b, d] - cost_mat[a, b] - cost_mat[c, d]
    if use_penalty:
        pena_delta = pena_mat[a, c] + pena_mat[b, d] - pena_mat[a, b] - pena_mat[c, d]
    # 检查是否使用翻转片段
    if use_segment:
        # 非对称TSP：计算片段翻转的内部成本变化
        seg_len = len(segment)
        for i in range(seg_len - 1):
            # 减去原始边
            cost_delta -= cost_mat[segment[i], segment[i + 1]]
            # 加上翻转后的边
            cost_delta += cost_mat[segment[seg_len - 1 - i], segment[seg_len - 2 - i]]
            # 检查是否使用惩罚矩阵
            if use_penalty:
                # 减去原始边
                pena_delta -= pena_mat[segment[i], segment[i + 1]]
                # 加上翻转后的边
                pena_delta += pena_mat[segment[seg_len - 1 - i], segment[seg_len - 2 - i]]
    # 检查是否使用惩罚矩阵
    if use_penalty:
        return cost_delta + pena_coef * pena_delta
    else:
        return cost_delta


def search_2opt_from_node(start_node: int,
                          route: np.ndarray,
                          cost_mat: np.ndarray,
                          pena_mat: Optional[np.ndarray] = None,
                          pena_coef: float = 0.0,
                          symmetric: bool = True) -> Tuple[np.ndarray, bool, np.ndarray]:
    """
    从指定节点开始执行2-opt局部搜索（单次改进）
    :param start_node: 搜索起始节点
    :param route: 当前路线
    :param cost_mat: 成本矩阵
    :param pena_mat: 惩罚矩阵（可选）
    :param pena_coef: 惩罚参数
    :param symmetric: 是否是对称矩阵
    :return: (改进后的路线, 是否找到改进, 改进的端点集合)
    """
    improved = False
    # 将 start_node 旋转到 route 首位
    pos = np.where(route == start_node)[0][0]
    route = np.concatenate((route[pos:], route[:pos]))
    # 存储改进操作的四个端点
    endpoints = np.zeros(4)
    # 搜索可能的 2-opt 交换，j 为第二个断点的位置
    for j in range(3, len(route)):
        # 2-opt 的四个端点（按路径顺序）
        a, b, c, d = route[0], route[1], route[j - 1], route[j]
        # 判断是否是对称矩阵
        if symmetric:
            # 对称矩阵
            cost_delta = eval_2opt_move(cost_mat, a, b, c, d, pena_mat, pena_coef)
        else:
            # 非对称矩阵：需要传入翻转片段计算内部成本
            cost_delta = eval_2opt_move(cost_mat, a, b, c, d, pena_mat, pena_coef, route[1:j])
        # 负值表示成本降低，找到改进
        if cost_delta < -1.e-9:
            # 执行 2-opt 翻转：反转 b 到 c 之间的片段
            route[1:j] = route[j - 1:0:-1]
            improved = True
            endpoints = np.array([a, b, c, d])
            return route, improved, endpoints

    return route, improved, endpoints


def _search_2opt(route: np.ndarray,
                 cost_mat: np.ndarray,
                 pena_mat: Optional[np.ndarray] = None,
                 pena_coef: float = 0.0,
                 symmetric: bool = True) -> Tuple[np.ndarray, bool, np.ndarray]:
    """
    2-opt局部搜索 (First-Improvement) (使用numpy实现)
    :param route: 当前路线
    :param cost_mat: 成本矩阵
    :param pena_mat: 惩罚矩阵（可选）
    :param pena_coef: 惩罚参数
    :param symmetric: 是否是对称矩阵
    :return: (改进后的路线, 是否找到改进, 改进的端点集合)
    """
    improved = False
    # 存储改进操作的四个端点
    endpoints = np.zeros(4)
    # 遍历每个节点作为搜索起点
    for i in range(len(route)):
        route, improved, endpoints \
            = search_2opt_from_node(i, route, cost_mat, pena_mat,
                                    pena_coef, symmetric)
        # 找到改进立即返回（first-improvement）
        if improved:
            return route, improved, endpoints
    return route, improved, endpoints


def _local_search_2opt(route: np.ndarray,
                       cost_mat: np.ndarray,
                       pena_mat: Optional[np.ndarray] = None,
                       pena_coef: float = 0.0,
                       symmetric: bool = True) -> np.ndarray:
    """
    使用2-opt算子进行局部搜索（多重改进策略） (使用numpy实现)

    反复扫描所有节点，直到无法找到任何改进，达到局部最优。
    采用多重改进（multiple-improvement）策略，每轮完整扫描应用多个改进。
    :param route: 当前路线
    :param cost_mat: 成本矩阵
    :param pena_mat: 惩罚矩阵（可选）
    :param pena_coef: 惩罚系数
    :param symmetric: 是否是对称矩阵
    :return: 改进后的路线
    """
    # 本轮扫描是否有改进
    has_improve = True
    # 反复扫描，直到无法找到任何改进（达到局部最优）
    while has_improve:
        # 重置本轮改进标志
        has_improve = False
        # 遍历所有节点作为搜索起点
        for i in range(len(route)):
            # 从节点i开始执行2-opt搜索
            route, found_improve, _ = _search_2opt(
                route, cost_mat, pena_mat, pena_coef, symmetric
            )
            # 如果找到改进，则标记本轮有改进
            if found_improve:
                has_improve = True
    return route


def _fast_local_search_2opt(bits: np.ndarray,
                            route: np.ndarray,
                            cost_mat: np.ndarray,
                            pena_mat: Optional[np.ndarray] = None,
                            pena_coef: float = 0.0,
                            symmetric: bool = True) -> np.ndarray:
    """
    使用2-opt算子进行快速局部搜索（基于激活集策略） (使用numpy实现)

    采用动态激活/冻结机制，只搜索可能产生改进的邻域，
    避免重复搜索已收敛的区域，加速收敛过程。
    :param bits: 激活标记数组
    :param route: 当前路线
    :param cost_mat: 成本矩阵
    :param pena_mat: 惩罚矩阵（可选）
    :param pena_coef: 惩罚系数
    :param symmetric: 是否是对称矩阵
    :return: 改进后的路线
    """
    # 持续搜索，直到所有激活节点都被处理（无可改进的邻域）
    while np.any(bits):
        # 遍历所有节点，检查并搜索被激活的邻域
        for i in range(len(route)):
            if bits[i]:  # 仅当节点i的邻域被激活时才进行搜索
                # 从节点i开始执行2-opt局部搜索
                route, improved, active_set = _search_2opt(
                    route, cost_mat, pena_mat, pena_coef, symmetric
                )
                if improved:
                    # 找到改进：激活改进操作涉及的四个端点
                    bits[active_set] = True
                else:
                    # 未找到改进：冻结当前节点
                    bits[i] = False
    return route


try:
    # 尝试导入numba
    from numba import jit


    @jit(nopython=True, cache=True)
    def eval_2opt_move_jit(cost_mat: np.ndarray, a: int, b: int, c: int, d: int,
                           pena_mat: Optional[np.ndarray] = None,
                           pena_coef: float = 0.0,
                           segment: Optional[np.ndarray] = None) -> float:
        """
        评估2-opt移动，返回成本变化（负值为改进）(使用numba加速)

        断开边 (a, b), (c, d) 连接边  (a, c), (b, d)
        :param cost_mat: 成本矩阵
        :param a: 第一条断开的边的起点
        :param b: 第一条断开的边的终点
        :param c: 第二条断开的边的起点
        :param d: 第二条断开的边的终点
        :param pena_mat: 惩罚矩阵（可选）
        :param pena_coef: 惩罚参数
        :param segment: 需要翻转的片段序列（不含a和d）
        :return: 成本变化量
        """
        # 提前计算标志位
        use_penalty = pena_mat is not None
        use_segment = segment is not None
        # 初始化变化量
        cost_delta, pena_delta = 0.0, 0.0
        # 外部边变化：删除(a,b) (c,d)，添加(a,c) (b,d)
        cost_delta = cost_mat[a, c] + cost_mat[b, d] - cost_mat[a, b] - cost_mat[c, d]
        if use_penalty:
            pena_delta = pena_mat[a, c] + pena_mat[b, d] - pena_mat[a, b] - pena_mat[c, d]
        # 检查是否使用翻转片段
        if use_segment:
            # 非对称TSP：计算片段翻转的内部成本变化
            seg_len = len(segment)
            for i in range(seg_len - 1):
                # 减去原始边
                cost_delta -= cost_mat[segment[i], segment[i + 1]]
                # 加上翻转后的边
                cost_delta += cost_mat[segment[seg_len - 1 - i], segment[seg_len - 2 - i]]
                # 检查是否使用惩罚矩阵
                if use_penalty:
                    # 减去原始边
                    pena_delta -= pena_mat[segment[i], segment[i + 1]]
                    # 加上翻转后的边
                    pena_delta += pena_mat[segment[seg_len - 1 - i], segment[seg_len - 2 - i]]
        # 检查是否使用惩罚矩阵
        if use_penalty:
            return cost_delta + pena_coef * pena_delta
        else:
            return cost_delta


    @jit(nopython=True, cache=True)
    def search_2opt_from_node_jit(start_node: int,
                                  route: np.ndarray,
                                  cost_mat: np.ndarray,
                                  pena_mat: Optional[np.ndarray] = None,
                                  pena_coef: float = 0.0,
                                  symmetric: bool = True) -> Tuple[np.ndarray, bool, np.ndarray]:
        """
        从指定节点开始执行2-opt局部搜索（单次改进）(使用numba加速)
        :param start_node: 搜索起始节点
        :param route: 当前路线
        :param cost_mat: 成本矩阵
        :param pena_mat: 惩罚矩阵（可选）
        :param pena_coef: 惩罚参数
        :param symmetric: 是否是对称矩阵
        :return: (改进后的路线, 是否找到改进, 改进的端点集合)
        """
        improved = False
        # 将 start_node 旋转到 route 首位
        pos = np.where(route == start_node)[0][0]
        route = np.concatenate((route[pos:], route[:pos]))
        # 存储改进操作的四个端点
        endpoints = np.zeros(4, dtype=np.int32)
        # 搜索可能的 2-opt 交换，j 为第二个断点的位置
        for j in range(3, len(route)):
            # 2-opt 的四个端点（按路径顺序）
            a, b, c, d = route[0], route[1], route[j - 1], route[j]
            # 判断是否是对称矩阵
            if symmetric:
                # 对称矩阵
                cost_delta = eval_2opt_move_jit(cost_mat, a, b, c, d, pena_mat, pena_coef)
            else:
                # 非对称矩阵：需要传入翻转片段计算内部成本
                cost_delta = eval_2opt_move_jit(cost_mat, a, b, c, d, pena_mat, pena_coef, route[1:j])
            # 负值表示成本降低，找到改进
            if cost_delta < -1.e-9:
                # 执行 2-opt 翻转：反转 b 到 c 之间的片段
                route[1:j] = route[j - 1:0:-1]
                improved = True
                endpoints = np.array([a, b, c, d], dtype=np.int32)
                return route, improved, endpoints

        return route, improved, endpoints


    @jit(nopython=True, cache=True)
    def search_2opt_jit(route: np.ndarray,
                        cost_mat: np.ndarray,
                        pena_mat: Optional[np.ndarray] = None,
                        pena_coef: float = 0.0,
                        symmetric: bool = True) -> Tuple[np.ndarray, bool, np.ndarray]:
        """
        2-opt局部搜索 (First-Improvement) (使用numba加速)
        :param route: 当前路线
        :param cost_mat: 成本矩阵
        :param pena_mat: 惩罚矩阵（可选）
        :param pena_coef: 惩罚参数
        :param symmetric: 是否是对称矩阵
        :return: (改进后的路线, 是否找到改进, 改进的端点集合)
        """
        improved = False
        # 存储改进操作的四个端点
        endpoints = np.zeros(4, dtype=np.int32)
        # 遍历每个节点作为搜索起点
        for i in range(len(route)):
            route, improved, endpoints \
                = search_2opt_from_node_jit(i, route, cost_mat, pena_mat,
                                            pena_coef, symmetric)
            # 找到改进立即返回（first-improvement）
            if improved:
                return route, improved, endpoints
        return route, improved, endpoints


    @jit(nopython=True, cache=True)
    def local_search_2opt_jit(route: np.ndarray,
                              cost_mat: np.ndarray,
                              pena_mat: Optional[np.ndarray] = None,
                              pena_coef: float = 0.0,
                              symmetric: bool = True) -> np.ndarray:
        """
        使用2-opt算子进行局部搜索（多重改进策略）(使用numba加速)

        反复扫描所有节点，直到无法找到任何改进，达到局部最优。
        采用多重改进（multiple-improvement）策略，每轮完整扫描应用多个改进。
        :param route: 当前路线
        :param cost_mat: 成本矩阵
        :param pena_mat: 惩罚矩阵（可选）
        :param pena_coef: 惩罚系数
        :param symmetric: 是否是对称矩阵
        :return: 改进后的路线
        """
        # 本轮扫描是否有改进
        has_improve = True
        # 反复扫描，直到无法找到任何改进（达到局部最优）
        while has_improve:
            # 重置本轮改进标志
            has_improve = False
            # 遍历所有节点作为搜索起点
            for i in range(len(route)):
                # 从节点i开始执行2-opt搜索
                route, found_improve, _ = search_2opt_jit(
                    route, cost_mat, pena_mat, pena_coef, symmetric
                )
                # 如果找到改进，则标记本轮有改进
                if found_improve:
                    has_improve = True
        return route


    @jit(nopython=True, cache=True)
    def fast_local_search_2opt_jit(bits: np.ndarray,
                                   route: np.ndarray,
                                   cost_mat: np.ndarray,
                                   pena_mat: Optional[np.ndarray] = None,
                                   pena_coef: float = 0.0,
                                   symmetric: bool = True) -> np.ndarray:
        """
        使用2-opt算子进行快速局部搜索（基于激活集策略）(使用numba加速)

        采用动态激活/冻结机制，只搜索可能产生改进的邻域，
        避免重复搜索已收敛的区域，加速收敛过程。
        :param bits: 激活标记数组
        :param route: 当前路线
        :param cost_mat: 成本矩阵
        :param pena_mat: 惩罚矩阵（可选）
        :param pena_coef: 惩罚系数
        :param symmetric: 是否是对称矩阵
        :return: 改进后的路线
        """
        # 持续搜索，直到所有激活节点都被处理（无可改进的邻域）
        while np.any(bits):
            # 遍历所有节点，检查并搜索被激活的邻域
            for i in range(len(route)):
                if bits[i]:  # 仅当节点i的邻域被激活时才进行搜索
                    # 从节点i开始执行2-opt局部搜索
                    route, improved, active_set = search_2opt_jit(
                        route, cost_mat, pena_mat, pena_coef, symmetric
                    )
                    if improved:
                        # 找到改进：激活改进操作涉及的四个端点
                        bits[active_set] = True
                    else:
                        # 未找到改进：冻结当前节点
                        bits[i] = False
        return route


    def search_2opt(route: np.ndarray,
                    cost_mat: np.ndarray,
                    pena_mat: Optional[np.ndarray] = None,
                    pena_coef: float = 0.0,
                    symmetric: bool = True) -> Tuple[np.ndarray, bool, np.ndarray]:
        """
        2-opt局部搜索 (默认使用numba加速)
        :param route: 当前路线
        :param cost_mat: 成本矩阵
        :param pena_mat: 惩罚矩阵（可选）
        :param pena_coef: 惩罚参数
        :param symmetric: 是否是对称矩阵
        :return: (改进后的路线, 是否找到改进, 改进的端点集合)
        """
        return search_2opt_jit(route, cost_mat, pena_mat, pena_coef, symmetric)


    def local_search_2opt(route: np.ndarray,
                          cost_mat: np.ndarray,
                          pena_mat: Optional[np.ndarray] = None,
                          pena_coef: float = 0.0,
                          symmetric: bool = True) -> np.ndarray:
        """
        使用2-opt算子进行局部搜索（多重改进策略）(默认使用numba加速)

        反复扫描所有节点，直到无法找到任何改进，达到局部最优。
        采用多重改进（multiple-improvement）策略，每轮完整扫描应用多个改进。

        算法逻辑：
            1. 初始化本轮改进标志
            2. 遍历所有节点作为搜索起点
            3. 如果任一节点找到改进，则设置本轮改进标志
            4. 重复完整扫描，直到某轮扫描中没有任何改进
            5. 此时达到局部最优，返回路线
        :param route: 当前路线
        :param cost_mat: 成本矩阵
        :param pena_mat: 惩罚矩阵（可选）
        :param pena_coef: 惩罚系数
        :param symmetric: 是否是对称矩阵
        :return: 改进后的路线
        """
        return local_search_2opt_jit(route, cost_mat, pena_mat, pena_coef, symmetric)


    def fast_local_search_2opt(bits: np.ndarray,
                               route: np.ndarray,
                               cost_mat: np.ndarray,
                               pena_mat: Optional[np.ndarray] = None,
                               pena_coef: float = 0.0,
                               symmetric: bool = True) -> np.ndarray:
        """
        使用2-opt算子进行快速局部搜索（基于激活集策略）(默认使用numba加速)

        采用动态激活/冻结机制，只搜索可能产生改进的邻域，
        避免重复搜索已收敛的区域，加速收敛过程。

        算法逻辑：
            1. 只要还有激活的节点，就继续搜索
            2. 遍历所有节点，如果该节点被激活，则从其开始搜索
            3. 如果找到改进，则激活改进涉及的四个端点的邻域（它们可能产生更多改进）
            4. 如果未找到改进，则冻结当前节点（其邻域已收敛）
            5. 重复直到所有节点都被冻结（bits全为False）
        :param bits: 激活标记数组
        :param route: 当前路线
        :param cost_mat: 成本矩阵
        :param pena_mat: 惩罚矩阵（可选）
        :param pena_coef: 惩罚系数
        :param symmetric: 是否是对称矩阵
        :return: 改进后的路线
        """
        return fast_local_search_2opt_jit(bits, route, cost_mat, pena_mat, pena_coef, symmetric)

except ImportError:
    # 如果导入numba加速库失败，使用原始的函数
    warn_once("Numba acceleration unavailable - "
              "falling back to slower implementation",
              warning_class=PerformanceWarning)
    search_2opt = _search_2opt
    local_search_2opt = _local_search_2opt
    fast_local_search_2opt = _fast_local_search_2opt

# 只允许外部调用以下函数
__all__ = ['search_2opt', 'local_search_2opt', 'fast_local_search_2opt']
