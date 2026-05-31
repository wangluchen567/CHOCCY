# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional
from ....types import VarType
from ...algorithm import Algorithm
from ....solutions import Solutions
from ....utilities.commons import calc_penalized_objs
from ....utilities.strategies.perturbers import (polynomial_perturb, bit_perturb,
                                                 flip_perturb, fix_label_perturb)


class SA(Algorithm):
    def __init__(self,
                 n_sols: int = 1,
                 max_iter: int = 10000,
                 init_temp: float = 1.e4,
                 final_temp: float = 1.e-200,
                 cooling_rate: float = 0.99,
                 perturb_rate: Optional[float] = None,
                 penalty_coef: float = 1.e6,
                 visual_mode: Optional[str] = None):
        """
        模拟退火算法

        Code Maintainer: Luchen Wang
        :param n_sols: 解个数
        :param max_iter: 最大迭代次数
        :param init_temp: 初始温度
        :param final_temp: 终止温度
        :param cooling_rate: 降温系数(alpha，降维比率)
        :param perturb_rate: 扰动比率(控制每次扰动比例)
        :param penalty_coef: 惩罚系数(用于带约束问题)
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols, max_iter, None, None, visual_mode)
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.perturb_rate = perturb_rate
        self.penalty_coef = penalty_coef
        self.single_obj_only = True
        self.sol = None  # 初始解
        self.temp = self.init_temp

    def prepare(self):
        # 创建初始解
        self.sol = self.sols[0]
        # 初始化扰动比率
        if self.perturb_rate is None:
            self.perturb_rate = 1 / self.problem.n_vars

    def run_step(self, i):
        """运行算法单步"""
        # 若温度到达终止温度则停止优化
        if self.temp <= self.final_temp:
            return
        # 逐个解进行扰动与优化
        for j in range(len(self.sols)):
            new_sol = self.apply_perturb(self.sol)
            new_sol.evaluate()  # 评估新解
            # 使用 metrospolis 接受准则选择是否接受解
            if self.metrospolis(self.sol.t, new_sol.t, self.temp):
                # 若接受解则替换原有解
                self.sol = new_sol.copy()
            # 更新并记录解
            self.sols[j] = self.sol
            # 更新温度 (逐渐降温)
            self.temp *= self.cooling_rate
        # 更新最优解
        self.update_best()

    def eval_penalized_obj(self, sols: Solutions) -> float:
        """评估解集的约束惩罚后的最优目标值"""
        # 返回约束惩罚后的最优目标值（作为分数）
        return calc_penalized_objs(sols.objs, sols.cons, self.penalty_coef).flat[0]

    def eval_penalized_objs(self, sols: Solutions) -> np.ndarray:
        """评估计算约束惩罚后的目标值矩阵"""
        # 返回约束惩罚后的最优目标值矩阵（作为带约束的目标值矩阵）
        return calc_penalized_objs(sols.objs, sols.cons, self.penalty_coef)

    @staticmethod
    def metrospolis(old, new, temp):
        """
        使用 metrospolis 接受准则接受解
        :param old: 扰动前旧的解(适应度值)
        :param new: 扰动得到的新解(适应度值)
        :param temp: 当前温度
        :return: 是否接受新解
        """
        # 计算能量差
        delta_e = new - old
        if delta_e < 0:
            # 若新解比旧解更好则直接接受新解
            return True
        else:
            # 若新解比旧解更差则以一定概率接受新解
            return np.random.rand() < np.exp(-delta_e / temp)

    def apply_perturb(self, solutions: Solutions) -> Solutions:
        """
        执行扰动操作得到新解集
        :param solutions: 指定解集
        :return: 新解集
        """
        new_sols = solutions.copy()
        # 按类型操作各个部分
        for var_type in self.problem.unique_types:
            indices = self.problem.type_to_indices[var_type]  # 该类型的变量索引
            new_sols.xs[:, indices] \
                = self.perturb_funcs(var_type, new_sols.xs[:, indices],
                                     self.problem.l_bounds[indices],
                                     self.problem.u_bounds[indices])
        return new_sols

    def perturb_funcs(self,
                      var_type: int,
                      solutions: np.ndarray,
                      l_bounds: np.ndarray,
                      u_bounds: np.ndarray) -> np.ndarray:
        """
        扰动函数映射函数
        :param var_type: 变量类型
        :param solutions: 解集数组
        :param l_bounds: 问题下界
        :param u_bounds: 问题上界
        :return: 扰动后的解集数组
        """
        if var_type == self.REAL:
            return polynomial_perturb(solutions, l_bounds, u_bounds, self.perturb_rate)
        elif var_type == self.INT:
            return polynomial_perturb(solutions, l_bounds, u_bounds, self.perturb_rate)
        elif var_type == self.BIN:
            return bit_perturb(solutions, self.perturb_rate)
        elif var_type == self.PMU:
            return flip_perturb(solutions)
        elif var_type == self.FIX:
            return fix_label_perturb(solutions, self.perturb_rate)
        else:
            # 收集有效的类型信息
            valid_types = [
                f"{member.name} ({member.value})"
                for member in VarType
            ]
            raise ValueError(
                f"Invalid variable type: {var_type}\n"
                f"Expected one of the following:\n"
                f"  " + "\n  ".join(valid_types)
            )
