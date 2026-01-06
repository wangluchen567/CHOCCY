"""
Copyright (c) 2024 LuChen Wang
CHOCCY is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
import numpy as np
from typing import Optional
from Algorithms import ALGORITHM
from Algorithms.Utility.Operators import operator_diff


class DE(ALGORITHM):
    RAND_1 = 'DE/rand/1'
    RAND_2 = 'DE/rand/2'
    BEST_1 = 'DE/best/1'
    BEST_2 = 'DE/best/2'
    RAND_BEST_1 = 'DE/rand-to-best/1'
    CURRENT_BEST_1 = 'DE/current-to-best/1'

    def __init__(self,
                 pop_size: Optional[int] = None,
                 max_iter: Optional[int] = None,
                 cross_prob: Optional[float] = None,
                 factor: Optional[float] = None,
                 operator_type: str = RAND_1,
                 quick_indices: bool = True,
                 show_mode: Optional[str] = None):
        """
        差分进化算法

        Code Maintainer: Luchen Wang
        :param pop_size: 种群大小
        :param max_iter: 迭代次数
        :param cross_prob: 交叉概率
        :param factor: 缩放因子
        :param operator_type: 算子类型
        :param quick_indices: 是否快速生成索引
        :param show_mode: 绘图模式
        """
        super().__init__(pop_size, max_iter, cross_prob, None, None, show_mode)
        self.only_solve_single = True
        self.solvable_type = [self.REAL, self.INT]
        self.factor = factor
        self.operator_type = operator_type
        self.quick_indices = quick_indices
        self.cross_prob = 0.5 if cross_prob is None else cross_prob
        self.factor = 0.5 if factor is None else factor

    @ALGORITHM.record_time
    def run_step(self, i):
        """运行算法单步"""
        # 获取匹配池
        parent_indices = self.generate_mating_indices()
        # 交叉变异生成子代
        offspring = self.apply_operator(parent_indices)
        # 进行环境选择
        self.environmental_selection(offspring)
        # 记录每步状态
        self.record()

    def apply_operator(self, parent_indices):
        """重写算子为差分进化算子"""
        parents = np.array([self.pop[indices] for indices in parent_indices])
        return operator_diff(self.pop, parents, self.lower, self.upper, self.cross_prob, self.factor)

    def environmental_selection(self, offspring):
        """差分进化环境选择(使用一对一的局部竞争选择)"""
        self.local_selection(offspring)

    def generate_mating_indices(self):
        """根据算子设置生成所需要的个体索引"""
        if self.operator_type == self.RAND_1:
            # V = X_{r1} + F * (X_{r2} - X_{r3})
            indices = self.generate_indices(num_indices=3)
        elif self.operator_type == self.RAND_2:
            # V = X_{r1} + F * (X_{r2} - X_{r3}) + F * (X_{r4} - X_{r5})
            indices = self.generate_indices(num_indices=5)
        elif self.operator_type == self.BEST_1:
            # V = X_{best} + F * (X_{r1} - X_{r2})
            best_index = np.argmin(self.fits)  # 获取最优解索引
            indices = self.generate_indices(num_indices=2)
            indices.insert(0, np.array([best_index] * self.pop_size))
        elif self.operator_type == self.BEST_2:
            # V = X_{best} + F * (X_{r1} - X_{r2}) + F * (X_{r3} - X_{r4})
            best_index = np.argmin(self.fits)  # 获取最优解索引
            indices = self.generate_indices(num_indices=4)
            indices.insert(0, np.array([best_index] * self.pop_size))
        elif self.operator_type == self.RAND_BEST_1:
            # V = X_{r1} + F * (X_{best} - X_{r1}) + F * (X_{r2} - X_{r3})
            best_index = np.argmin(self.fits)  # 获取最优解索引
            indices = self.generate_indices(num_indices=3)
            indices.insert(0, indices[0])
            indices.insert(1, np.array([best_index] * self.pop_size))
        elif self.operator_type == self.CURRENT_BEST_1:
            # V = X_i + F * (X_{best} - X_i) + F * (X_{r1} - X_{r2})
            best_index = np.argmin(self.fits)  # 获取最优解索引
            indices = self.generate_indices(num_indices=2)
            indices.insert(0, np.arange(self.pop_size))
            indices.insert(0, np.arange(self.pop_size))
            indices.insert(1, np.array([best_index] * self.pop_size))
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
            indices.append(np.random.permutation(self.pop_size))
        return indices

    def generate_strict_indices(self, num_indices, strategy='no_self'):
        """
        生成随机不重复索引(性能差但能保证完全不存在重复)
        :param num_indices: 生成不重复索引的个数
        :param strategy: 生成不重复索引的策略
        :return: 不重复索引结果(list)
        """
        # 初始化随机不重复索引矩阵
        indices_matrix = np.zeros((self.pop_size, num_indices), dtype=int)
        # 普通版本：允许包含自身，只要求多个索引互异
        if strategy == 'standard':
            for i in range(self.pop_size):
                indices_matrix[i] = np.random.choice(self.pop_size, num_indices, replace=False)
        # 严格版本：排除自身索引i，并保证
        elif strategy == 'no_self':
            for i in range(self.pop_size):
                candidates = np.delete(np.arange(self.pop_size), i)
                indices_matrix[i] = np.random.choice(candidates, num_indices, replace=False)
        else:
            raise ValueError(f"There is no specified strategy: {strategy}")
        # 将结果转换为list
        return [indices_matrix[:, col] for col in range(num_indices)]

    def get_params_info(self):
        """获取参数信息"""
        info = super().get_params_info()
        info['factor'] = self.factor
        info['operator_type'] = self.operator_type
        return info
