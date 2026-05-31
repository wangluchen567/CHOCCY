# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

"""
算子策略函数集
"""

import numpy as np
from typing import Optional, Union
from .crossovers import simulated_binary_crossover, differential_crossover, binary_crossover, order_crossover, \
    fix_label_crossover
from .mutations import polynomial_mutation, differential_mutation, bit_mutation, flip_mutation, fix_label_mutation
from ..commons import sigmoid


def operator_real(population: np.ndarray,
                  l_bounds: np.ndarray,
                  u_bounds: np.ndarray,
                  cross_prob: Optional[float] = None,
                  mutate_prob: Optional[float] = None) -> np.ndarray:
    """
    对实数/整数问题进行交叉和变异操作(遗传算子)
    :param population: 要交叉变异的种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param cross_prob: 交叉概率，范围 [0, 1]
    :param mutate_prob: 变异概率，范围 [0, 1]
    :return: 交叉变异得到的子代种群(决策变量矩阵)，形状: (n_sols, n_vars)
    """
    # 计算种群一半的数量(奇数向下取整)
    num_half = int(len(population) / 2)
    # 设置默认交叉与变异概率
    cross_prob = 1.0 if cross_prob is None else cross_prob
    mutate_prob = 1.0 / population.shape[1] if mutate_prob is None else mutate_prob
    # 将种群均分为两个父代种群(防止修改原数据)
    parents1 = population[:num_half].copy()
    parents2 = population[num_half:num_half * 2].copy()
    # 进行模拟二进制交叉
    offspring = simulated_binary_crossover(parents1, parents2, l_bounds, u_bounds, cross_prob)
    # 进行多项式变异
    offspring = polynomial_mutation(offspring, l_bounds, u_bounds, mutate_prob)
    return offspring


def operator_binary(population: np.ndarray,
                    l_bounds: Optional[np.ndarray] = None,
                    u_bounds: Optional[np.ndarray] = None,
                    cross_prob: Optional[float] = None,
                    mutate_prob: Optional[float] = None) -> np.ndarray:
    """
    对二进制问题进行交叉和变异操作(遗传算子)
    :param population: 要交叉变异的种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param cross_prob: 交叉概率，范围 [0, 1]
    :param mutate_prob: 变异概率，范围 [0, 1]
    :return: 交叉变异得到的子代种群(决策变量矩阵)，形状: (n_sols, n_vars)
    """
    # 计算种群一半的数量(奇数向下取整)
    num_half = int(len(population) / 2)
    # 设置默认交叉与变异概率
    cross_prob = 1.0 if cross_prob is None else cross_prob
    mutate_prob = 1.0 / population.shape[1] if mutate_prob is None else mutate_prob
    # 将种群均分为两个父代种群(防止修改原数据)
    parents1 = population[:num_half].copy()
    parents2 = population[num_half:num_half * 2].copy()
    # 均匀二进制交叉
    offspring = binary_crossover(parents1, parents2, cross_prob)
    # 位翻转变异
    offspring = bit_mutation(offspring, mutate_prob)
    return offspring


def operator_permutation(population: np.ndarray,
                         l_bounds: Optional[np.ndarray] = None,
                         u_bounds: Optional[np.ndarray] = None,
                         cross_prob: Optional[float] = None,
                         mutate_prob: Optional[float] = None) -> np.ndarray:
    """
    对序列问题进行交叉和变异操作(遗传算子)
    :param population: 要交叉变异的种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param cross_prob: 交叉概率，范围 [0, 1]
    :param mutate_prob: 变异概率，范围 [0, 1]
    :return: 交叉变异得到的子代种群(决策变量矩阵)，形状: (n_sols, n_vars)
    """
    # 计算种群一半的数量(奇数向下取整)
    num_half = int(len(population) / 2)
    # 设置默认交叉与变异概率
    cross_prob = 1.0 if cross_prob is None else cross_prob
    mutate_prob = 1.0 / population.shape[1] if mutate_prob is None else mutate_prob
    # 将种群均分为两个父代种群(防止修改原数据)
    parents1 = population[:num_half].copy()
    parents2 = population[num_half:num_half * 2].copy()
    # 顺序交叉
    offspring = order_crossover(parents1, parents2, cross_prob)
    # # 交换式变异
    # offspring = exchange_mutation(offspring, mutate_prob)
    # 翻转式变异
    offspring = flip_mutation(offspring, mutate_prob)
    return offspring


def operator_fix_label(population: np.ndarray,
                       l_bounds: Optional[np.ndarray] = None,
                       u_bounds: Optional[np.ndarray] = None,
                       cross_prob: Optional[float] = None,
                       mutate_prob: Optional[float] = None) -> np.ndarray:
    """
    对固定类型数的标签问题进行交叉和变异操作(遗传算子)
    :param population: 要交叉变异的种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param cross_prob: 交叉概率，范围 [0, 1]
    :param mutate_prob: 变异概率，范围 [0, 1]
    :return: 交叉变异得到的子代种群(决策变量矩阵)，形状: (n_sols, n_vars)
    """
    # 计算种群一半的数量(奇数向下取整)
    num_half = int(len(population) / 2)
    # 设置默认交叉与变异概率
    cross_prob = 1.0 if cross_prob is None else cross_prob
    mutate_prob = 1.0 / population.shape[1] if mutate_prob is None else mutate_prob
    # 将种群均分为两个父代种群(防止修改原数据)
    parents1 = population[:num_half].copy()
    parents2 = population[num_half:num_half * 2].copy()
    # 固定类型数的标签的均匀交叉
    offspring = fix_label_crossover(parents1, parents2, cross_prob)
    # 固定类型数的标签的交换式变异
    offspring = fix_label_mutation(offspring, mutate_prob)
    return offspring


def operator_pso_real(particles: np.ndarray,
                      personal_best: np.ndarray,
                      global_best: np.ndarray,
                      velocities: np.ndarray,
                      l_bounds: np.ndarray,
                      u_bounds: np.ndarray,
                      v_min: np.ndarray,
                      v_max: np.ndarray,
                      inertia_weight: float = 0.729,
                      personal_factor: float = 1.494,
                      global_factor: float = 1.494) -> tuple:
    """
    实数/整数 粒子群优化算子
    :param particles: 粒子群位置，形状: (n_sols, n_vars)
    :param personal_best: 单个粒子找到的最优位置，形状: (n_sols, n_vars)
    :param global_best: 整个粒子群找到的最优位置，形状: (1, n_vars)
    :param velocities: 粒子群速度，形状: (n_sols, n_vars)
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param v_min: 粒子群速度的的下界(数组)，可以是1D或2D数组
    :param v_max: 粒子群速度的的上界(数组)，可以是1D或2D数组
    :param inertia_weight: 惯性权重(w)
    :param personal_factor: 个体学习因子/认知系数(c1)
    :param global_factor: 全局学习因子/社会系数(c2)
    :return: 下一代粒子群位置, 裁剪后的粒子群速度
    """
    # 创建两个随机矩阵以引入随机性（学习因子）
    random_factor1 = np.random.uniform(size=particles.shape)
    random_factor2 = np.random.uniform(size=particles.shape)
    # 计算下一代粒子群速度
    velocities = (inertia_weight * velocities +
                  random_factor1 * personal_factor * (personal_best - particles) +
                  random_factor2 * global_factor * (global_best - particles))
    # 对粒子群速度进行裁剪
    velocities = np.clip(velocities, v_min, v_max)
    # 得到下一代粒子群位置
    next_particles = np.clip(particles + velocities, l_bounds, u_bounds)
    # 返回下一代粒子群位置 与 裁剪后的粒子群速度
    return next_particles, velocities


def operator_pso_binary(particles: np.ndarray,
                        personal_best: np.ndarray,
                        global_best: np.ndarray,
                        velocities: np.ndarray,
                        l_bounds: np.ndarray,
                        u_bounds: np.ndarray,
                        v_min: np.ndarray,
                        v_max: np.ndarray,
                        inertia_weight: float = 1.0,
                        personal_factor: float = 1.494,
                        global_factor: float = 1.494) -> tuple:
    """
    二进制 粒子群优化算子
    :param particles: 粒子群位置，形状: (n_sols, n_vars)
    :param personal_best: 单个粒子找到的最优位置，形状: (n_sols, n_vars)
    :param global_best: 整个粒子群找到的最优位置，形状: (1, n_vars)
    :param velocities: 粒子群速度，形状: (n_sols, n_vars)
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param v_min: 粒子群速度的的下界(数组)，可以是1D或2D数组
    :param v_max: 粒子群速度的的上界(数组)，可以是1D或2D数组
    :param inertia_weight: 惯性权重(w) BPSO 默认为 1.0
    :param personal_factor: 个体学习因子/认知系数(c1)
    :param global_factor: 全局学习因子/社会系数(c2)
    :return: 下一代粒子群位置, 裁剪后的粒子群速度
    """
    # 创建两个随机矩阵以引入随机性（学习因子）
    random_factor1 = np.random.uniform(size=particles.shape)
    random_factor2 = np.random.uniform(size=particles.shape)
    # 计算下一代粒子群速度
    velocities = (velocities +
                  random_factor1 * personal_factor * (personal_best - particles) +
                  random_factor2 * global_factor * (global_best - particles))
    # 对粒子群速度进行裁剪
    velocities = np.clip(velocities, v_min, v_max)
    # 将速度转换为概率值
    aux_probs = sigmoid(velocities)
    # 得到下一代粒子群位置
    next_particles = (np.random.uniform(size=aux_probs.shape) < aux_probs).astype(int)
    # 返回下一代粒子群位置 与 裁剪后的粒子群速度
    return next_particles, velocities


def operator_differential(population: np.ndarray,
                          parents: np.ndarray,
                          l_bounds: np.ndarray,
                          u_bounds: np.ndarray,
                          cross_probs: Union[np.ndarray, float] = 0.5,
                          scale_factor: Union[np.ndarray, float] = 0.5) -> np.ndarray:
    """
    差分进化算子
    :param population: 原始种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param parents: 要差分的父代(决策变量矩阵)，形状: (n_parents, n_sols, n_vars)
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param cross_probs: 交叉概率(标量/数组)，范围 [0, 1]
    :param scale_factor: 缩放因子(标量/数组)，范围 [0, 1]
    :return: 变异交叉得到的子代种群(决策变量矩阵)，形状: (n_sols, n_vars)
    """
    # 进行差分变异得到变异子代
    variation = differential_mutation(parents, scale_factor)
    # 进行差分交叉得到最终子代
    offspring = differential_crossover(population, variation, l_bounds, u_bounds, cross_probs)
    return offspring


def operator_de_real(population: np.ndarray,
                     auxiliaries: np.ndarray,
                     parent_indices: list,
                     l_bounds: np.ndarray,
                     u_bounds: np.ndarray,
                     cross_probs: Union[np.ndarray, float] = 0.5,
                     scale_factor: Union[np.ndarray, float] = 0.5) -> tuple:
    """
    实数/整数 差分进化算子
    :param population: 原始种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param auxiliaries: 辅助种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param parent_indices: 要差分的父代下标集合
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param cross_probs: 交叉概率(标量/数组)，范围 [0, 1]
    :param scale_factor: 缩放因子(标量/数组)，范围 [0, 1]
    :return: 差分进化后的 子代种群/辅助种群
    """
    # 获取要差分的父代
    parents = np.array([population[indices] for indices in parent_indices])
    # 差分进化得到子代
    offspring = operator_differential(population, parents, l_bounds, u_bounds, cross_probs, scale_factor)
    # 返回结果
    return offspring, auxiliaries


def operator_de_binary(population: np.ndarray,
                       auxiliaries: np.ndarray,
                       parent_indices: list,
                       l_bounds: np.ndarray,
                       u_bounds: np.ndarray,
                       cross_probs: Union[np.ndarray, float] = 0.5,
                       scale_factor: Union[np.ndarray, float] = 0.5) -> tuple:
    """
    二进制 差分进化算子
    :param population: 原始种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param auxiliaries: 辅助种群(决策变量矩阵)，形状: (n_sols, n_vars)
    :param parent_indices: 要差分的父代下标集合
    :param l_bounds: 取值范围的下界(数组)，可以是1D或2D数组
    :param u_bounds: 取值范围的上界(数组)，可以是1D或2D数组
    :param cross_probs: 交叉概率(标量/数组)，范围 [0, 1]
    :param scale_factor: 缩放因子(标量/数组)，范围 [0, 1]
    :return: 差分进化后的 子代种群/辅助种群
    """
    # 获取辅助种群形状
    n_sols, n_vars = auxiliaries.shape
    # 获取要差分的父代（辅助种群）
    aux_parents = np.array([auxiliaries[indices] for indices in parent_indices])
    # 进行差分变异得到变异子代辅助种群
    aux_variation = differential_mutation(aux_parents, scale_factor)
    # 根据概率创建交叉掩码
    mask = np.asarray(np.random.random((n_sols, n_vars)) < cross_probs)
    # 强制至少有一个变量维度交叉
    random_dims = np.random.randint(0, n_vars, n_sols)
    # 创建索引矩阵来设置强制交叉位
    row_indices = np.arange(n_sols)[:, np.newaxis]
    col_indices = random_dims[:, np.newaxis]
    # 使用花式索引设置强制交叉位
    mask[row_indices, col_indices] = True
    # 按照上下界对超出部分进行裁剪
    aux_variation = np.clip(aux_variation, l_bounds, u_bounds)
    # 得到变异子代种群
    variation = (sigmoid(aux_variation) >= 0.5).astype(int)
    # 交叉得到子代辅助种群
    aux_offspring = np.where(mask, aux_variation, auxiliaries)
    # 交叉得到子代种群
    offspring = np.where(mask, variation, population)
    # 返回结果
    return offspring, aux_offspring
