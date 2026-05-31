# Copyright (c) 2024 LuChen Wang
# SPDX-License-Identifier: MulanPSL-2.0

import numpy as np
from typing import Optional, Union
from ...algorithm import Algorithm
from ....utilities.commons import sigmoid


class DE(Algorithm):
    # 枚举差分进化算子的类型
    RAND_1 = 'DE/rand/1'
    RAND_2 = 'DE/rand/2'
    BEST_1 = 'DE/best/1'
    BEST_2 = 'DE/best/2'
    RAND_BEST_1 = 'DE/rand-to-best/1'
    CURRENT_BEST_1 = 'DE/current-to-best/1'

    def __init__(self,
                 n_sols: int = 50,
                 max_iter: int = 200,
                 cross_probs: Union[np.ndarray, float] = 0.5,
                 scale_factor: Union[np.ndarray, float] = 0.5,
                 operator_type: str = RAND_1,
                 quick_indices: bool = True,
                 visual_mode: Optional[str] = None):
        """
        差分进化算法

        Code Maintainer: LuChen Wang
        :param n_sols: 种群大小
        :param max_iter: 迭代次数
        :param cross_probs: 交叉概率(CR)
        :param scale_factor: 缩放因子(F)
        :param operator_type: 算子类型
        :param quick_indices: 是否快速生成索引
        :param visual_mode: 可视化模式
        """
        super().__init__(n_sols, max_iter, None, None, visual_mode)
        self.operator_type = operator_type
        self.quick_indices = quick_indices
        self.cross_probs = cross_probs
        self.scale_factor = scale_factor
        # 初始化辅助决策变量（用于求解二进制问题）
        self.aux_xs = self.sols.xs.copy()
        # 初始化问题的上下界（用于辅助决策变量）
        self.l_bounds, self.u_bounds = None, None
        # 对于二进制问题，辅助决策变量上下界设置为[-5.0, 5.0]
        self.l_bounds_binary, self.u_bounds_binary = -5.0, 5.0
        # 差分进化算法仅支持 单目标的 实数与整数 问题优化
        self.single_obj_only = True
        self.supported_var_types = [self.REAL, self.INT, self.BIN]

    def prepare(self):
        # 设置辅助决策变量的值
        self.aux_xs = np.zeros_like(self.sols.xs)
        # 取得问题的上下界
        self.l_bounds = self.problem.l_bounds.copy()
        self.u_bounds = self.problem.u_bounds.copy()
        # 将差分交叉与缩放概率格式化为数组
        self.cross_probs = self.problem.format_to_arr(self.cross_probs, 'cross_probs')
        self.scale_factor = self.problem.format_to_arr(self.scale_factor, 'scale_factor')
        # 检查问题是否存在二进制决策变量
        if self.BIN in self.problem.unique_types:
            indices_binary = self.problem.type_to_indices[self.BIN]
            # 对于二进制问题，重新设置辅助决策变量上下界
            self.l_bounds[indices_binary], self.u_bounds[indices_binary] \
                = self.l_bounds_binary, self.u_bounds_binary
            # 设置辅助决策变量的值
            self.aux_xs[:, indices_binary] = np.random.uniform(self.l_bounds_binary, self.u_bounds_binary,
                                                               self.aux_xs[:, indices_binary].shape)
            # 更新初始解对应的二进制决策变量值
            self.sols.xs[:, indices_binary] = (sigmoid(self.aux_xs[:, indices_binary]) >= 0.5).astype(int)

    def run_step(self, i):
        """运行算法单步"""
        # 获取差分索引作为匹配池
        parent_indices = self.generate_differential_indices()
        # 交叉变异生成子代
        offspring_sols, offspring_aux = self.apply_operator_de(parent_indices)
        # 进行环境选择
        self.environmental_selection_de(offspring_sols, offspring_aux)

    def apply_operator_de(self, parent_indices):
        """差分进化算子"""
        # 创建新子代的决策变量矩阵
        new_xs = np.zeros_like(self.sols.xs)
        new_aux_xs = np.zeros_like(self.aux_xs)
        # 按类型操作各个部分
        for var_type in self.problem.unique_types:
            indices = self.problem.type_to_indices[var_type]  # 该类型的变量索引
            new_xs[:, indices], new_aux_xs[:, indices] \
                = self.operator_de_funcs[var_type](self.sols.xs[:, indices], self.aux_xs[:, indices], parent_indices,
                                                   self.l_bounds[indices], self.u_bounds[indices],
                                                   self.cross_probs[indices], self.scale_factor[indices])
        new_sols = self.sols.create_new_with(new_xs)
        return new_sols, new_aux_xs

    def environmental_selection_de(self, new_sols, new_aux_xs):
        """差分进化环境选择(使用一对一的局部竞争选择)"""
        # 更新主种群
        better = self.local_selection(new_sols)
        # 更新辅助种群
        self.aux_xs = np.concatenate((self.aux_xs, new_aux_xs))[better]

    def generate_differential_indices(self):
        """根据算子设置生成所需要差分的个体索引"""
        if self.operator_type == self.RAND_1:
            # V = X_{r1} + F * (X_{r2} - X_{r3})
            indices = self.generate_indices(num_indices=3)
        elif self.operator_type == self.RAND_2:
            # V = X_{r1} + F * (X_{r2} - X_{r3}) + F * (X_{r4} - X_{r5})
            indices = self.generate_indices(num_indices=5)
        elif self.operator_type == self.BEST_1:
            # V = X_{best} + F * (X_{r1} - X_{r2})
            # 根据适应度获取最优解索引
            best_index = np.argmin(self.sols.fits)
            indices = self.generate_indices(num_indices=2)
            indices.insert(0, np.array([best_index] * self.n_sols))
        elif self.operator_type == self.BEST_2:
            # V = X_{best} + F * (X_{r1} - X_{r2}) + F * (X_{r3} - X_{r4})
            # 根据适应度获取最优解索引
            best_index = np.argmin(self.sols.fits)
            indices = self.generate_indices(num_indices=4)
            indices.insert(0, np.array([best_index] * self.n_sols))
        elif self.operator_type == self.RAND_BEST_1:
            # V = X_{r1} + F * (X_{best} - X_{r1}) + F * (X_{r2} - X_{r3})
            # 根据适应度获取最优解索引
            best_index = np.argmin(self.sols.fits)
            indices = self.generate_indices(num_indices=3)
            indices.insert(0, indices[0])
            indices.insert(1, np.array([best_index] * self.n_sols))
        elif self.operator_type == self.CURRENT_BEST_1:
            # V = X_i + F * (X_{best} - X_i) + F * (X_{r1} - X_{r2})
            # 根据适应度获取最优解索引
            best_index = np.argmin(self.sols.fits)
            indices = self.generate_indices(num_indices=2)
            indices.insert(0, np.arange(self.n_sols))
            indices.insert(0, np.arange(self.n_sols))
            indices.insert(1, np.array([best_index] * self.n_sols))
        else:
            raise ValueError(f"There is no such operator type: {self.operator_type}")
        return indices

    def generate_indices(self, num_indices):
        """生成随机不重复索引(可选择性能生成或者严格生成)"""
        if self.quick_indices:
            return self.generate_quick_indices(num_indices)
        else:
            return self.generate_strict_indices(num_indices)

    def generate_quick_indices(self, num_indices):
        """
        生成随机不重复索引(高性能但无法完全保证不重复)
        :param num_indices: 生成不重复索引的个数
        :return: 不重复索引结果(list)
        """
        indices = []
        for _ in range(num_indices):
            indices.append(np.random.permutation(self.n_sols))
        return indices

    def generate_strict_indices(self, num_indices, strategy='no_self'):
        """
        生成随机不重复索引(性能差但能保证完全不存在重复)
        :param num_indices: 生成不重复索引的个数
        :param strategy: 生成不重复索引的策略
        :return: 不重复索引结果(list)
        """
        # 初始化随机不重复索引矩阵
        indices_matrix = np.zeros((self.n_sols, num_indices), dtype=int)
        # 普通版本：允许包含自身，只要求多个索引互异
        if strategy == 'standard':
            for i in range(self.n_sols):
                indices_matrix[i] = np.random.choice(self.n_sols, num_indices, replace=False)
        # 严格版本：排除自身索引i，并保证
        elif strategy == 'no_self':
            for i in range(self.n_sols):
                candidates = np.delete(np.arange(self.n_sols), i)
                indices_matrix[i] = np.random.choice(candidates, num_indices, replace=False)
        else:
            raise ValueError(f"There is no specified strategy: {strategy}")
        # 将结果转换为list
        return [indices_matrix[:, col] for col in range(num_indices)]

    def get_config(self) -> dict:
        """获取算法的完整配置"""
        config = super().get_config()
        # 算子类型
        config['operator_type'] = self.operator_type
        # 交叉概率(CR)
        config['cross_probs'] = self.cross_probs
        # 缩放比例(F)
        config['scale_factor'] = self.scale_factor
        return config
