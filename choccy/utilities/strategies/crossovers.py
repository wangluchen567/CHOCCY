# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
交叉算子函数集
"""

import numpy as np
from typing import Union
from ...core import warn_once, PerformanceWarning


def simulated_binary_crossover(parents1: np.ndarray,
                               parents2: np.ndarray,
                               l_bounds: np.ndarray,
                               u_bounds: np.ndarray,
                               cross_prob: float,
                               eta: float = 20.0) -> np.ndarray:
    """
    模拟二进制交叉(实数问题)
    :param parents1: 父代种群1(决策变量矩阵)，形状: (n_sols, n_vars)
    :param parents2: 父代种群2(决策变量矩阵)，形状: (n_sols, n_vars)
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param cross_prob: 交叉概率，范围: [0, 1]
    :param eta: 分布指数，控制子代与父代的接近程度，越大子代越接近父代
    :return: 交叉后的子代种群(决策变量矩阵)，形状: (2 * n_sols, n_vars)
    """
    if parents1.shape != parents2.shape:
        raise ValueError(f"Parent populations must have same shape, "
                         f"got {parents1.shape} and {parents2.shape}")
    # 父代形状信息
    n_sols, n_vars = parents1.shape
    # 生成beta值
    mu = np.random.random((n_sols, n_vars))
    beta = np.where(mu <= 0.5,
                    (2 * mu) ** (1 / (eta + 1)),
                    (2 - 2 * mu) ** (-1 / (eta + 1)))
    # 随机正负号
    beta *= np.where(np.random.randint(0, 2, (n_sols, n_vars)), 1, -1)
    # 应用交叉概率
    mask = np.random.random((n_sols, 1)) < cross_prob
    beta = np.where(mask, beta, 1)
    # 计算子代
    center = (parents1 + parents2) / 2  # 父代中点，交叉中心
    spread = beta * (parents1 - parents2) / 2  # 扩散距离，受beta控制的偏移量
    # 生成两个子代：中心 ± 扩散
    offspring = np.vstack([
        center + spread,  # 子代1: 中点正向偏移
        center - spread  # 子代2: 中点负向偏移
    ])
    # 边界裁剪
    offspring = np.clip(offspring, l_bounds, u_bounds)
    return offspring


def differential_crossover(population: np.ndarray,
                           variation: np.ndarray,
                           l_bounds: np.ndarray,
                           u_bounds: np.ndarray,
                           cross_probs: Union[np.ndarray, float]) -> np.ndarray:
    """
    差分交叉(实数问题)(用于差分进化算法)
    :param population: 原始种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param variation: 变异后种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param cross_probs: 交叉概率(标量/数组)，范围 [0, 1]
    :return: 交叉后的子代种群(决策变量矩阵)，形状: (n_sols, n_vars)
    """
    # 获取种群形状
    n_sols, n_vars = population.shape
    # 根据概率创建交叉掩码
    mask = np.asarray(np.random.random((n_sols, n_vars)) < cross_probs)
    # 强制至少有一个变量维度交叉
    random_dims = np.random.randint(0, n_vars, n_sols)
    # 创建索引矩阵来设置强制交叉位
    row_indices = np.arange(n_sols)[:, np.newaxis]
    col_indices = random_dims[:, np.newaxis]
    # 使用花式索引设置强制交叉位
    mask[row_indices, col_indices] = True
    # 交叉得到子代种群
    offspring = np.where(mask, variation, population)
    # 按照上下界对超出部分进行裁剪
    offspring = np.clip(offspring, l_bounds, u_bounds)
    return offspring


def binary_crossover(parents1: np.ndarray,
                     parents2: np.ndarray,
                     cross_prob: float) -> np.ndarray:
    """
    二进制均匀交叉(二进制问题)
    :param parents1: 父代种群1(决策变量矩阵)，形状: (n_sols, n_vars)
    :param parents2: 父代种群2(决策变量矩阵)，形状: (n_sols, n_vars)
    :param cross_prob: 交叉概率，范围: [0, 1]
    :return: 交叉后的子代种群(决策变量矩阵)，形状: (2 * n_sols, n_vars)
    """
    if parents1.shape != parents2.shape:
        raise ValueError(f"Parent populations must have same shape, "
                         f"got {parents1.shape} and {parents2.shape}")
    n_sols, n_vars = parents1.shape
    # 变量维度方面均匀交叉，个数方面按照交叉概率交叉
    mask = (np.random.rand(n_sols, n_vars) < 0.5) & (np.random.rand(n_sols, 1) < cross_prob)
    # 若mask为true则取第一个矩阵元素,否则取第二个矩阵中元素
    offspring1 = np.where(mask, parents2, parents1)
    offspring2 = np.where(mask, parents1, parents2)
    offspring = np.vstack((offspring1, offspring2))
    return offspring


def order_crossover(parents1: np.ndarray,
                    parents2: np.ndarray,
                    cross_prob: float) -> np.ndarray:
    """
    顺序交叉(序列问题)
    :param parents1: 父代种群1(决策变量矩阵)，形状: (n_sols, n_vars)
    :param parents2: 父代种群2(决策变量矩阵)，形状: (n_sols, n_vars)
    :param cross_prob: 交叉概率，范围: [0, 1]
    :return: 交叉后的子代种群(决策变量矩阵)，形状: (2 * n_sols, n_vars)
    """
    if parents1.shape != parents2.shape:
        raise ValueError(f"Parent populations must have same shape, "
                         f"got {parents1.shape} and {parents2.shape}")
    n_sols, n_vars = parents1.shape
    # 初始化子代
    offspring1 = np.zeros_like(parents1)
    offspring2 = np.zeros_like(parents2)
    # 生成所有需要的随机数
    crossover_mask = np.asarray(np.random.random(n_sols) < cross_prob)
    starts1 = np.random.randint(0, n_vars - 1, size=n_sols)  # 确保区间不为空
    ends1 = np.random.randint(starts1 + 1, n_vars + 1, size=n_sols)
    starts2 = np.random.randint(0, n_vars - 1, size=n_sols)  # 确保区间不为空
    ends2 = np.random.randint(starts2 + 1, n_vars + 1, size=n_sols)
    for i in range(n_sols):
        if crossover_mask[i]:
            # 进行顺序交叉
            offspring1_ = list(dict.fromkeys(np.concatenate((parents1[i][starts1[i]:ends1[i]],
                                                             np.roll(parents2[i], -starts2[i])))))
            offspring1[i] = np.roll(np.array(offspring1_), starts1[i])
            offspring2_ = list(dict.fromkeys(np.concatenate((parents2[i][starts2[i]:ends2[i]],
                                                             np.roll(parents1[i], -starts1[i])))))
            offspring2[i] = np.roll(np.array(offspring2_), starts2[i])
    offspring = np.vstack((offspring1, offspring2))
    return offspring


def fix_label_crossover(parents1: np.ndarray,
                        parents2: np.ndarray,
                        cross_prob: float) -> np.ndarray:
    """
    固定类型数的标签的均匀交叉(固定类型数标签问题)
    :param parents1: 父代种群1(决策变量矩阵)，形状: (n_sols, n_vars)
    :param parents2: 父代种群2(决策变量矩阵)，形状: (n_sols, n_vars)
    :param cross_prob: 交叉概率，范围: [0, 1]
    :return: 交叉后的子代种群(决策变量矩阵)，形状: (2 * n_sols, n_vars)
    """
    if parents1.shape != parents2.shape:
        raise ValueError(f"Parent populations must have same shape, "
                         f"got {parents1.shape} and {parents2.shape}")
    # 得到每种标签的类型和数量
    labels_type, labels_num = np.unique(parents1[0], return_counts=True)
    offspring = fix_label_cx(parents1, parents2, labels_type, labels_num, cross_prob)
    return offspring


def _fix_label_cx(parents1: np.ndarray,
                  parents2: np.ndarray,
                  labels_type: np.ndarray,
                  labels_num: np.ndarray,
                  cross_prob: float) -> np.ndarray:
    """
    固定类型数的标签的均匀交叉(子函数)(使用 numpy 实现)
    :param parents1: 父代种群1(决策变量矩阵)，形状: (n_sols, n_vars)
    :param parents2: 父代种群2(决策变量矩阵)，形状: (n_sols, n_vars)
    :param labels_type: 每种标签的类型(1D数组)
    :param labels_num: 每种标签的数量(1D数组)
    :param cross_prob: 交叉概率，范围: [0, 1]
    :return: 交叉后的子代种群(决策变量矩阵)，形状: (2 * n_sols, n_vars)
    """
    n_sols, n_vars = parents1.shape
    # 初始化子代
    offspring1 = np.zeros_like(parents1)
    offspring2 = np.zeros_like(parents2)
    # 两父代相同位保持不变，不同位均匀交叉，并且需要保证标签等量约束
    equals = np.array(parents1 == parents2, dtype=bool)
    offspring1[equals] = parents1[equals]
    offspring2[equals] = parents2[equals]
    # 这里需要遍历以满足固定数量的约束
    for i in range(n_sols):
        # 统计剩余标签数量
        last_labels1 = labels_num.copy()
        last_labels2 = labels_num.copy()
        for j in range(len(labels_type)):
            last_labels1[j] -= np.sum(offspring1[i] == labels_type[j])
            last_labels2[j] -= np.sum(offspring2[i] == labels_type[j])
        # 根据现存数量在考虑约束的情况下得到子代
        for j in range(n_vars):
            if equals[i][j]:
                pass
            else:
                # 随机从父代中选择继承点
                r1 = (parents1[i][j] if np.random.random() < 0.5 else parents2[i][j]) \
                    if np.random.random() < cross_prob else offspring1[i][j]
                r2 = (parents2[i][j] if np.random.random() < 0.5 else parents1[i][j]) \
                    if np.random.random() < cross_prob else offspring2[i][j]
                k1, k2 = np.where(labels_type == r1)[0], np.where(labels_type == r2)[0]
                # 判断是否可继承，若无法继承，则直接随机从剩余的类型中选择一个
                if last_labels1[k1] <= 0:
                    k1 = np.random.choice(np.where(last_labels1 > 0)[0])
                    r1 = labels_type[k1]
                offspring1[i][j] = r1
                last_labels1[k1] -= 1
                # 判断是否可继承，若无法继承，则直接随机从剩余的类型中选择一个
                if last_labels2[k2] <= 0:
                    k2 = np.random.choice(np.where(last_labels2 > 0)[0])
                    r2 = labels_type[k2]
                offspring2[i][j] = r2
                last_labels2[k2] -= 1
    offspring = np.vstack((offspring1, offspring2))
    return offspring


try:
    # 尝试导入numba
    from numba import jit


    @jit(nopython=True, cache=True)
    def fix_label_cx_jit(parents1: np.ndarray,
                         parents2: np.ndarray,
                         labels_type: np.ndarray,
                         labels_num: np.ndarray,
                         cross_prob: float) -> np.ndarray:
        """
        固定类型数的标签的均匀交叉(子函数)(使用 numba 加速版本)
        :param parents1: 父代种群1(决策变量矩阵)，形状: (n_sols, n_vars)
        :param parents2: 父代种群2(决策变量矩阵)，形状: (n_sols, n_vars)
        :param labels_type: 每种标签的类型(1D数组)
        :param labels_num: 每种标签的数量(1D数组)
        :param cross_prob: 交叉概率，范围: [0, 1]
        :return: 交叉后的子代种群(决策变量矩阵)，形状: (2 * n_sols, n_vars)
        """
        n_sols, n_vars = parents1.shape
        # 初始化子代
        offspring1 = np.zeros_like(parents1, dtype=np.int32)
        offspring2 = np.zeros_like(parents2, dtype=np.int32)
        # 两父代相同位保持不变，不同位均匀交叉，并且需要保证标签等量约束
        # 使用显式循环代替布尔数组索引
        for i in range(n_sols):
            for j in range(n_vars):
                if parents1[i, j] == parents2[i, j]:
                    offspring1[i, j] = parents1[i, j]
                    offspring2[i, j] = parents2[i, j]

        # 这里需要遍历以满足固定数量的约束
        for i in range(n_sols):
            # 统计剩余标签数量
            last_labels1 = labels_num.copy()
            last_labels2 = labels_num.copy()
            for j in range(len(labels_type)):
                count1 = 0
                count2 = 0
                for k in range(n_vars):
                    if offspring1[i, k] == labels_type[j]:
                        count1 += 1
                    if offspring2[i, k] == labels_type[j]:
                        count2 += 1
                last_labels1[j] -= count1
                last_labels2[j] -= count2

            # 根据现存数量在考虑约束的情况下得到子代
            for j in range(n_vars):
                if parents1[i, j] != parents2[i, j]:
                    # 随机从父代中选择继承点
                    if np.random.random() < cross_prob:
                        r1 = parents1[i, j] if np.random.random() < 0.5 else parents2[i, j]
                    else:
                        r1 = offspring1[i, j]

                    if np.random.random() < cross_prob:
                        r2 = parents2[i, j] if np.random.random() < 0.5 else parents1[i, j]
                    else:
                        r2 = offspring2[i, j]

                    # 找到对应的标签类型索引
                    k1 = -1
                    k2 = -1
                    for idx in range(len(labels_type)):
                        if labels_type[idx] == r1:
                            k1 = idx
                        if labels_type[idx] == r2:
                            k2 = idx

                    # 判断是否可继承，若无法继承，则直接随机从剩余的类型中选择一个
                    if last_labels1[k1] <= 0:
                        available = np.where(last_labels1 > 0)[0]
                        if len(available) > 0:
                            k1 = np.random.choice(available)
                            r1 = labels_type[k1]

                    offspring1[i, j] = r1
                    last_labels1[k1] -= 1

                    # 判断是否可继承，若无法继承，则直接随机从剩余的类型中选择一个
                    if last_labels2[k2] <= 0:
                        available = np.where(last_labels2 > 0)[0]
                        if len(available) > 0:
                            k2 = np.random.choice(available)
                            r2 = labels_type[k2]

                    offspring2[i, j] = r2
                    last_labels2[k2] -= 1

        offspring = np.vstack((offspring1, offspring2))
        return offspring


    def fix_label_cx(parents1: np.ndarray,
                     parents2: np.ndarray,
                     labels_type: np.ndarray,
                     labels_num: np.ndarray,
                     cross_prob: float) -> np.ndarray:
        """
        固定类型数的标签的均匀交叉(子函数)(默认使用numba加速)
        :param parents1: 父代种群1(决策变量矩阵)，形状: (n_sols, n_vars)
        :param parents2: 父代种群2(决策变量矩阵)，形状: (n_sols, n_vars)
        :param labels_type: 每种标签的类型(1D数组)
        :param labels_num: 每种标签的数量(1D数组)
        :param cross_prob: 交叉概率，范围: [0, 1]
        :return: 交叉后的子代种群(决策变量矩阵)，形状: (2 * n_sols, n_vars)
        """
        # 确保输入数组是Numba兼容的类型
        parents1 = np.asarray(parents1, dtype=np.int32)
        parents2 = np.asarray(parents2, dtype=np.int32)
        labels_type = np.asarray(labels_type, dtype=np.int32)
        labels_num = np.asarray(labels_num, dtype=np.int32)

        return fix_label_cx_jit(parents1, parents2, labels_type, labels_num, cross_prob)


except ImportError:
    # 如果导入numba加速库失败，使用原始的函数
    warn_once("Numba acceleration unavailable - "
              "falling back to slower implementation",
              warning_class=PerformanceWarning)
    fix_label_cx = _fix_label_cx

# 只允许外部调用以下函数：
__all__ = [
    'simulated_binary_crossover',
    'differential_crossover',
    'binary_crossover',
    'order_crossover',
    'fix_label_crossover',
]
